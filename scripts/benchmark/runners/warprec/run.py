import copy
import json
import math
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from collections import deque
from contextlib import contextmanager
from importlib import metadata, util
from pathlib import Path

import pandas as pd
import psutil
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.bench import Terminated, start
from common.datasets import dataset_info, dataset_path

MODELS = {
    "BPR": "interaction",
    "LightGCN": "interaction",
    "LightCCF": "interaction",
    "TwoTower": "interaction",
    "SASRec": "prefix",
    "eSASRec": "prefix",
    "DiffRec": "user",
}
CUTOFF = 10
WARMUP_EPOCHS = 1

DATA_PREPARATION = "Data Preparation Time"
SETUP = "Best Trial Setup Time (s)"
TRAIN_ONLY = "Best Trial Train-Only Time (s)"
VALIDATION = "Best Trial Validation Time (s)"
EPOCHS = "Best Trial Epochs"
EXPLORATION = "Hyperparameter Exploration Time"
AVERAGE_TRIAL = "Average Trial Time (s)"
EVALUATION = "Evaluation Time"
TOTAL = "Total Time"
DATA_STEPS = (
    "Data Reading Time (s)",
    "Filtering Time (s)",
    "Splitting Time (s)",
    "Dataset Building Time (s)",
)
EVALUATION_DATA = "Evaluation Data Building Time (s)"

EPOCH_KEYS = ("epoch_train_s", "epoch_validation_s", "epoch_samples", "setup_s", "report_epochs")

DATA_READY = re.compile(r"Data preparation completed in ([0-9.]+) seconds")
DATA_BREAKDOWN = re.compile(r"Data preparation breakdown: (.*?)(?:\x1b\[[0-9;]*m)?\s*$")
EVALUATING = "Delegating evaluation of"

GPU_OOM = ("CUDA out of memory", "CUBLAS_STATUS_ALLOC_FAILED")
RAM_OOM = re.compile(
    r"running low on memory|Unable to allocate|std::bad_alloc|(?<![A-Za-z])MemoryError"
)

MARKER = "BENCH_RAY_CLUSTER"
WORKER_PORTS = (10000, 32000, 200)


def main():
    run = start()
    run.execute(body)


def body(run):
    model_name = run.model_name
    if model_name not in MODELS:
        raise ValueError(
            "The benchmark has no model {}: one of {}.".format(model_name, ", ".join(MODELS))
        )
    example = MODELS[model_name]
    package, version = warprec_source()
    print("WarpRec {} from {}".format(version, package), flush=True)
    run.versions(
        warprec=version,
        **{name: _version(name) for name in ("ray", "torch", "lightning", "polars")}
    )

    cpus = available_cpus()
    gpus = visible_gpus()
    if cpus < 2:
        raise RuntimeError(
            "WarpRec needs at least 2 CPUs: Ray Tune's trial driver takes a slice "
            "of one besides the CPUs of the trial."
        )
    model_config = run.config["models"][model_name]
    epochs = run.epochs()
    experiment_dir = run.work_dir / "experiment" / run.dataset

    cluster = RayCluster(run, cpus, gpus)
    progress = Progress()
    failure = None
    try:
        config = fill_config(run, model_name, epochs, cpus - 1, cluster.address)
        config_path = run.output_dir / "warprec-config.yml"
        with open(config_path, "w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        cluster.start()
        exit_code = run_warprec(run, config_path, cluster, progress)
        outcome = read_outcome(experiment_dir, model_name, config)
        if outcome is None:
            raise progress.failure(exit_code)
        if exit_code != 0:
            run.note(
                "WarpRec exited with code {} after writing its reports.".format(exit_code)
            )
        record(run, outcome, config, example, epochs, cluster, experiment_dir)
    except BaseException as exc:
        failure = exc
        with deferred_sigterm():
            record_failure(run, progress, experiment_dir, model_config, example)
        raise
    finally:
        with deferred_sigterm() as received:
            progress.stop()
            leftovers = cluster.stop(
                keep_logs=None if failure is None else run.output_dir / "ray-logs"
            )
        if leftovers:
            run.note(
                "{} processes of the Ray cluster survived its stop: {}.".format(
                    len(leftovers), ", ".join(leftovers)
                )
            )
        if received and failure is None:
            raise Terminated("Received SIGTERM: the time limit was reached.")


def fill_config(run, model_name, epochs, cpu_per_trial, ray_address):
    config = copy.deepcopy(run.config)
    reader = config["reader"]
    info = dataset_info(run.dataset)
    reader["local_path"] = str(dataset_path(run.repo_root, run.dataset))
    reader["sep"] = info["sep"]
    reader["labels"] = {
        "user_id_label": info["user"],
        "item_id_label": info["item"],
        "timestamp_label": info["time"],
    }
    dtypes = reader.setdefault("dtypes", {})
    if info["time_format"] in ("date", "datetime"):
        dtypes["timestamp_type"] = "datetime"
    if info.get("item_ids") == "str":
        dtypes["item_id_type"] = "str"
    config["writer"]["dataset_name"] = run.dataset
    config["writer"]["local_experiment_path"] = str(run.work_dir / "experiment")
    config.setdefault("general", {})["ray_address"] = ray_address

    model = config["models"][model_name]
    model["epochs"] = epochs
    optimization = model.setdefault("optimization", {})
    optimization.setdefault("cpu_per_trial", cpu_per_trial)
    return config


def run_warprec(run, config_path, cluster, progress):
    environment = dict(
        cluster.environment,
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    command = [sys.executable, "-m", "warprec.run", "-c", str(config_path), "-p", "train"]
    print("Running: {}".format(" ".join(command)), flush=True)
    progress.process = subprocess.Popen(
        command,
        cwd=str(run.work_dir),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    for line in progress.process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        progress.read(line)
    return progress.process.wait()


def read_outcome(experiment_dir, model_name, config):
    evaluation_dir = experiment_dir / "evaluation"
    report = _model_row(evaluation_dir, "Time_Report_*.tsv", "\t", "Model Name", model_name)
    if report is None:
        return None
    results_format = config["writer"].get("results", {})
    results = _model_row(
        evaluation_dir,
        "Overall_Results_*" + results_format.get("ext", ".tsv"),
        results_format.get("sep", "\t"),
        "Model",
        model_name,
        cutoff=CUTOFF,
    )
    return {
        "report": report,
        "results": results,
        "trials": read_trials(experiment_dir),
        "files": sorted(evaluation_dir.glob("*")),
    }


def record(run, outcome, config, example, epochs, cluster, experiment_dir):
    report, results, trials = outcome["report"], outcome["results"], outcome["trials"]
    missing = [
        column for column in (DATA_STEPS[0], EVALUATION, TOTAL) if report.get(column) is None
    ]
    if missing:
        raise RuntimeError("WarpRec's Time Report has no value for {}.".format(missing))
    if results is None:
        raise RuntimeError("WarpRec wrote no test results at top-{}.".format(CUTOFF))
    if not trials:
        raise RuntimeError("WarpRec's trials reported no epoch timing to Ray Tune.")

    model_config = config["models"][run.model_name]
    world_size = trial_world_size(model_config)
    for trial in trials:
        trial["stats"] = epoch_stats(trial["epochs"], world_size, model_config["batch_size"])
    run.result["trials"] = [
        {key: value for key, value in trial.items() if key != "folder"} for trial in trials
    ]
    search = len(trials) > 1
    if search:
        training, validation = report.get(EXPLORATION), 0.0
        if training is None:
            raise RuntimeError("WarpRec's Time Report has no value for {}.".format(EXPLORATION))
        record_search(run, trials, report, model_config)
    else:
        trial = trials[0]
        training = report.get(TRAIN_ONLY)
        if training is None:
            raise RuntimeError("WarpRec's Time Report has no value for {}.".format(TRAIN_ONLY))
        validation = report.get(VALIDATION) or 0.0
        record_epochs(run, trial["epochs"])
        stats = trial["stats"]
        run.result["epoch_stats"] = stats
        run.training_work(stats["steps_per_epoch"], stats["examples_per_epoch"], example)
        if report.get(EPOCHS) is not None and int(report[EPOCHS]) != epochs:
            run.note("WarpRec's trial ran {} epochs, not {}.".format(report[EPOCHS], epochs))

    metrics = {"ndcg@10": results["nDCGRendle2020"], "recall@10": results["Recall"]}
    run.framework_report(
        {
            "time_report": report,
            "test_results": results,
            "resources": {
                "ray_cpus": cluster.cpus,
                "ray_gpus": cluster.gpus,
                "cpu_per_trial": model_config["optimization"].get("cpu_per_trial"),
                "gpu_per_trial": model_config["optimization"].get("gpu_per_trial"),
                "world_size": world_size,
            },
        }
    )
    copy_reports(outcome["files"], run.output_dir / "warprec")
    record_emissions(run, experiment_dir)

    preprocessing = data_work(report)
    candidates = evaluation_data(report)
    run.add_time("preprocessing", preprocessing)
    run.add_time("evaluation_data", candidates)
    run.add_time("training", training)
    run.add_time("validation", validation)
    run.add_time("evaluation", report[EVALUATION])
    overhead = (
        report[TOTAL] - preprocessing - candidates - training - validation - report[EVALUATION]
    )
    if overhead >= 0:
        run.add_time("overhead", overhead)
    else:
        run.note("No overhead recorded: the rest of {} is {:.3f} s.".format(TOTAL, overhead))
    run.set_metrics(metrics, source="native")

    run.note(
        "Preprocessing: WarpRec's work on the data, {}, but its {} (the candidates of "
        "the sampled evaluation), recorded apart as evaluation_data. The rest of its {} "
        "(connecting to Ray, the Ray task, moving the datasets) and the trials' setup "
        "(starting the Ray Train workers, fetching the datasets, building the model on the "
        "GPU, its optimizer and loaders) are in the overhead.".format(
            " + ".join(DATA_STEPS), EVALUATION_DATA, DATA_PREPARATION
        )
    )
    if not search:
        run.note(
            "Validation: WarpRec validated {} time(s), on the test set (there is no "
            "validation split), in {:.3f} s ({}).".format(
                sum(1 for row in trials[0]["epochs"] if (row.get("epoch_validation_s") or 0) > 0),
                validation,
                VALIDATION,
            )
        )


def record_search(run, trials, report, model_config):
    started = [trial["started_at"] for trial in trials if trial.get("started_at") is not None]
    ended = [trial["ended_at"] for trial in trials if trial.get("ended_at") is not None]
    run.result["search"] = {
        "trials": len(trials),
        "exploration_s": report.get(EXPLORATION),
        "trials_span_s": (max(ended) - min(started)) if started and ended else None,
        "average_trial_s": report.get(AVERAGE_TRIAL),
        "max_concurrent_trials": model_config["optimization"].get("max_concurrent_trials"),
        "gpu_per_trial": model_config["optimization"].get("gpu_per_trial"),
        "cpu_per_trial": model_config["optimization"].get("cpu_per_trial"),
        "num_workers": model_config["optimization"].get("num_workers"),
    }


def record_failure(run, progress, experiment_dir, model_config, example):
    try:
        run.result["failed_phase"] = progress.phase
        if run.result["times_s"]:
            return
        trials = read_trials(experiment_dir)
        world_size = trial_world_size(model_config)
        for trial in trials:
            trial["stats"] = epoch_stats(trial["epochs"], world_size, model_config["batch_size"])
        run.result["trials"] = [
            {key: value for key, value in trial.items() if key != "folder"} for trial in trials
        ]
        trial = trials[0] if len(trials) == 1 else None
        setup = (trial["epochs"][-1].get("setup_s") or 0.0) if trial and trial["epochs"] else 0.0
        if progress.data_preparation_s is not None:
            preprocessing = data_work(progress.data_steps)
            candidates = evaluation_data(progress.data_steps)
            run.add_time("preprocessing", preprocessing)
            run.add_time("evaluation_data", candidates)
            run.add_time(
                "overhead",
                max(progress.data_preparation_s - preprocessing - candidates, 0.0) + setup,
            )
        if trial and trial["epochs"]:
            record_epochs(run, trial["epochs"])
            rows = trial["epochs"]
            run.add_time("training", sum(row.get("epoch_train_s") or 0.0 for row in rows))
            run.add_time("validation", sum(row.get("epoch_validation_s") or 0.0 for row in rows))
            run.result["epoch_stats"] = trial["stats"]
            stats = trial["stats"]
            run.training_work(stats["steps_per_epoch"], stats["examples_per_epoch"], example)
    except Exception as exc:
        _say("Could not read how far WarpRec got: {}".format(exc))


def data_work(steps):
    seconds = [steps.get(step) for step in DATA_STEPS]
    total = sum(float(value) for value in seconds if value is not None and value == value)
    return total - evaluation_data(steps)


def evaluation_data(steps):
    value = steps.get(EVALUATION_DATA)
    return float(value) if value is not None and value == value else 0.0


def trial_world_size(model_config):
    gpus = float(model_config.get("optimization", {}).get("gpu_per_trial", 1))
    return max(1, int(gpus))


def record_epochs(run, epochs):
    ends, elapsed, completed = [], 0.0, 0
    for row in epochs:
        elapsed += row.get("epoch_train_s") or 0.0
        ends.append(elapsed)
        completed += int(row.get("report_epochs") or 1)
    run.epochs_measured(ends, completed=completed)


def epoch_samples(row):
    return int(round((row["epoch_samples"] or 0) / (row.get("report_epochs") or 1)))


def epoch_stats(epochs, world_size, batch_size):
    rows = []
    for row in epochs:
        count = int(row.get("report_epochs") or 1)
        for _ in range(count):
            rows.append(
                {
                    "train_s": (row.get("epoch_train_s") or 0.0) / count,
                    "validation_s": (row.get("epoch_validation_s") or 0.0) / count,
                    "examples": epoch_samples(row),
                }
            )
    if not rows:
        return {"epochs": 0}
    warmup = WARMUP_EPOCHS if len(rows) > WARMUP_EPOCHS else 0
    measured = rows[warmup:]
    rates = [row["examples"] / row["train_s"] for row in measured if row["train_s"] > 0]
    examples = rows[0]["examples"]
    return {
        "epochs": len(rows),
        "warmup_epochs": warmup,
        "mean_epoch_train_s": _mean([row["train_s"] for row in measured]),
        "mean_epoch_validation_s": _mean([row["validation_s"] for row in measured]),
        "mean_examples_per_s": _mean(rates),
        "examples_per_epoch": examples,
        "steps_per_epoch": math.ceil(examples / (world_size * batch_size)),
        "world_size": world_size,
        "epoch_train_s": [row["train_s"] for row in rows],
        "epoch_validation_s": [row["validation_s"] for row in rows],
    }


def epoch_metrics(row):
    return {
        key: value
        for key, value in row.items()
        if "@" in key and not key.startswith("best_") and isinstance(value, (int, float))
    }


def read_trials(experiment_dir):
    trials = []
    for path in sorted(Path(experiment_dir, "ray_results").glob("**/result.json")):
        reports = []
        for line in path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("epoch_train_s") is not None:
                reports.append(row)
        if not reports:
            continue
        epochs = {}
        for row in reports:
            epoch = row.get("epoch", len(epochs))
            epochs[epoch] = {
                "epoch": epoch,
                **{key: row.get(key) for key in EPOCH_KEYS},
                "metrics": epoch_metrics(row),
            }
        last = reports[-1]
        params_path = path.parent / "params.json"
        try:
            params = json.loads(params_path.read_text()) if params_path.is_file() else {}
        except ValueError:
            params = {}
        ended, total = last.get("timestamp"), last.get("time_total_s")
        trials.append(
            {
                "trial_id": last.get("trial_id"),
                "folder": str(path.parent),
                "params": params,
                "epochs": [epochs[epoch] for epoch in sorted(epochs)],
                "time_total_s": total,
                "started_at": (ended - total) if ended is not None and total is not None else None,
                "ended_at": ended,
            }
        )
    unique = {}
    for trial in trials:
        key = trial["trial_id"] or trial["folder"]
        if key not in unique or len(trial["epochs"]) > len(unique[key]["epochs"]):
            unique[key] = trial
    return list(unique.values())


def record_emissions(run, experiment_dir):
    path = Path(experiment_dir, "ray_results", "emissions.csv")
    if not path.is_file():
        return
    shutil.copy2(str(path), str(run.output_dir / "emissions.csv"))
    frame = pd.read_csv(path)

    def total(column):
        return float(frame[column].sum()) if column in frame else None

    run.result["emissions"] = {
        "rows": int(len(frame)),
        "energy_kwh": total("energy_consumed"),
        "cpu_energy_kwh": total("cpu_energy"),
        "gpu_energy_kwh": total("gpu_energy"),
        "ram_energy_kwh": total("ram_energy"),
        "emissions_kg": total("emissions"),
        "duration_s": total("duration"),
        "country_iso_code": (
            str(frame["country_iso_code"].iloc[0]) if "country_iso_code" in frame else None
        ),
        "tracking_mode": (
            str(frame["tracking_mode"].iloc[0]) if "tracking_mode" in frame else None
        ),
    }


def _mean(values):
    return sum(values) / len(values) if values else None


def copy_reports(files, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for path in files:
        if path.is_file():
            shutil.copy2(str(path), str(destination / path.name))


class Progress:

    def __init__(self):
        self.process = None
        self.phase = "preprocessing"
        self.data_preparation_s = None
        self.data_steps = {}
        self.out_of_memory = None
        self.tail = deque(maxlen=40)

    def read(self, line):
        self.tail.append(line.rstrip())
        ready = DATA_READY.search(line)
        breakdown = DATA_BREAKDOWN.search(line)
        if ready:
            self.data_preparation_s = float(ready.group(1))
            self.phase = "training"
        elif breakdown:
            for item in breakdown.group(1).split(", "):
                column, _, seconds = item.rpartition(": ")
                self.data_steps[column] = float(seconds)
        elif EVALUATING in line:
            self.phase = "evaluation"
        if self.out_of_memory is None:
            if any(marker in line for marker in GPU_OOM):
                self.out_of_memory = "gpu"
            elif RAM_OOM.search(line):
                self.out_of_memory = "ram"

    def failure(self, exit_code):
        if exit_code == -signal.SIGKILL:
            what = "was killed with SIGKILL, in a job usually by the out-of-memory killer"
        elif exit_code < 0:
            what = "was killed by signal {}".format(-exit_code)
        elif exit_code != 0:
            what = "exited with code {}".format(exit_code)
        else:
            what = "ended without a Time Report row for the model: its trial failed"
        text = "WarpRec {}. Last lines of its output:\n{}".format(what, "\n".join(self.tail))
        if self.out_of_memory == "gpu":
            return RuntimeError("CUDA out of memory: " + text)
        if self.out_of_memory == "ram" or exit_code == -signal.SIGKILL:
            return MemoryError(text)
        return RuntimeError(text)

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


class RayCluster:

    def __init__(self, run, cpus, gpus):
        self.cpus = cpus
        self.gpus = gpus
        self.work_dir = run.work_dir
        job = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        temp_root = Path(os.environ.get("BENCH_RAY_TMPDIR") or tempfile.gettempdir())
        spill_root = Path(os.environ.get("BENCH_RAY_SPILL_DIR") or run.work_dir)
        temp_root.mkdir(parents=True, exist_ok=True)
        spill_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ray-{}-".format(job), dir=str(temp_root)))
        self.spill_dir = Path(
            tempfile.mkdtemp(prefix="ray-spill-{}-".format(job), dir=str(spill_root))
        )
        self.ports = _free_ports(4)
        first = WORKER_PORTS[0] + WORKER_PORTS[2] * random.SystemRandom().randrange(
            (WORKER_PORTS[1] - WORKER_PORTS[0]) // WORKER_PORTS[2]
        )
        self.worker_ports = (first, first + WORKER_PORTS[2] - 1)
        self.address = "127.0.0.1:{}".format(self.ports[0])
        self.environment = dict(
            os.environ,
            RAY_ADDRESS=self.address,
            RAY_USAGE_STATS_ENABLED="0",
            **{MARKER: str(self.temp_dir)}
        )

    def start(self):
        command = [
            _ray_executable(),
            "start",
            "--head",
            "--node-ip-address=127.0.0.1",
            "--port={}".format(self.ports[0]),
            "--ray-client-server-port={}".format(self.ports[1]),
            "--dashboard-agent-listen-port={}".format(self.ports[2]),
            "--dashboard-port={}".format(self.ports[3]),
            "--min-worker-port={}".format(self.worker_ports[0]),
            "--max-worker-port={}".format(self.worker_ports[1]),
            "--num-cpus={}".format(self.cpus),
            "--num-gpus={}".format(self.gpus),
            "--disable-usage-stats",
            "--temp-dir={}".format(self.temp_dir),
            "--object-spilling-directory={}".format(self.spill_dir),
        ]
        print("Starting Ray: {}".format(" ".join(command)), flush=True)
        completed = subprocess.run(command, cwd=str(self.work_dir), env=self.environment)
        if completed.returncode != 0:
            raise RuntimeError("ray start exited with code {}.".format(completed.returncode))

    def stop(self, keep_logs=None):
        processes = self._processes()
        for process in processes:
            try:
                process.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(processes, timeout=10)
        for process in alive:
            try:
                process.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(alive, timeout=5)
        logs = self.temp_dir / "session_latest" / "logs"
        if keep_logs is not None and logs.is_dir():
            shutil.copytree(str(logs), str(keep_logs), dirs_exist_ok=True)
        shutil.rmtree(str(self.temp_dir), ignore_errors=True)
        shutil.rmtree(str(self.spill_dir), ignore_errors=True)
        leftovers = ["{} {}".format(p.pid, _name(p)) for p in self._processes()]
        _say("Stopped the Ray cluster: {} processes, {} left.".format(len(processes), len(leftovers)))
        return leftovers

    def _processes(self):
        me = os.getpid()
        uid = os.getuid()
        temp_dir = str(self.temp_dir)
        found = {}
        for process in psutil.process_iter(["pid", "uids", "cmdline"]):
            try:
                uids = process.info["uids"]
                if process.pid == me or uids is None or uids.real != uid:
                    continue
                mine = temp_dir in " ".join(process.info["cmdline"] or [])
                if not mine:
                    mine = process.environ().get(MARKER) == temp_dir
                if mine:
                    found[process.pid] = process
                    for child in process.children(recursive=True):
                        found[child.pid] = child
            except psutil.Error:
                continue
        found.pop(me, None)
        return list(found.values())


@contextmanager
def deferred_sigterm():
    received = []
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: received.append(signum))
    try:
        yield received
    finally:
        signal.signal(signal.SIGTERM, previous)


def available_cpus():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def visible_gpus():
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return len([device for device in devices.split(",") if device.strip()])


def warprec_source():
    spec = util.find_spec("warprec")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("WarpRec is not importable.")
    package = Path(spec.origin).resolve().parent
    pyproject = package.parent / "pyproject.toml"
    if pyproject.is_file():
        found = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.MULTILINE)
        if found:
            return package, found.group(1)
    return package, _version("warprec")


def _model_row(directory, pattern, sep, model_column, model_name, cutoff=None):
    paths = sorted(directory.glob(pattern))
    if not paths:
        return None
    frame = pd.read_csv(paths[-1], sep=sep)
    rows = frame[frame[model_column] == model_name]
    if cutoff is not None:
        rows = rows[rows["Top@k"] == cutoff]
    if rows.empty:
        return None
    return {column: _plain(value) for column, value in rows.iloc[0].items()}


def _plain(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _free_ports(count):
    sockets = []
    try:
        for _ in range(count):
            handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            handle.bind(("127.0.0.1", 0))
            sockets.append(handle)
        return [handle.getsockname()[1] for handle in sockets]
    finally:
        for handle in sockets:
            handle.close()


def _ray_executable():
    beside = Path(sys.executable).with_name("ray")
    if beside.is_file():
        return str(beside)
    found = shutil.which("ray")
    if found is None:
        raise FileNotFoundError("No ray command in the environment.")
    return found


def _say(text):
    try:
        print(text, flush=True)
    except OSError:
        pass


def _name(process):
    try:
        return process.name()
    except psutil.Error:
        return "?"


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


if __name__ == "__main__":
    main()
