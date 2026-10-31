import argparse
import json
import os
import platform
import resource
import shutil
import signal
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASES = ("preprocessing", "evaluation_data", "training", "validation", "evaluation", "overhead")


class Terminated(Exception):
    pass


def start():
    parser = argparse.ArgumentParser(description="Benchmark run of WarpRec.")
    parser.add_argument(
        "--config",
        required=True,
        help="The run's configuration, configs/<experiment>/<name>.yml.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. ml-1m.")
    parser.add_argument(
        "--output-dir",
        help="Where result.json goes. Defaults to "
        "results/<experiment>/<dataset>/<name>/<date-time>.",
    )
    parser.add_argument(
        "--work-dir",
        help="Where WarpRec's outputs go. Defaults to "
        "<output-dir>/work, removed at the end unless BENCH_KEEP_WORK=1.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Train for this many epochs instead of the configured ones (quick tests).",
    )
    args = parser.parse_args()
    return BenchmarkRun(args)


class BenchmarkRun:

    def __init__(self, args):
        self.dataset = args.dataset
        self.repo_root = REPO_ROOT

        self.config_path = Path(args.config).resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError("Missing configuration {}.".format(self.config_path))
        with open(self.config_path) as handle:
            self.config = yaml.safe_load(handle) or {}
        models = self.config.get("models") or {}
        if len(models) != 1:
            raise ValueError(
                "{} must configure one model, not {}.".format(self.config_path, len(models))
            )
        self.model_name = next(iter(models))
        self.model = self.model_name.lower()
        self.experiment = self.config_path.parent.name
        self.label = self.config_path.stem

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = Path(
            args.output_dir
            or REPO_ROOT / "results" / self.experiment / args.dataset / self.label / stamp
        ).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = Path(args.work_dir or self.output_dir / "work").resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.epochs_override = args.epochs
        self._open_phases = []

        self.result = {
            "experiment": self.experiment,
            "run": self.label,
            "model": self.model,
            "dataset": self.dataset,
            "config": _relative(self.config_path),
            "status": "running",
            "error": None,
            "epochs": None,
            "epochs_completed": 0,
            "training_s_at_epoch_end": [],
            "quick_test": args.epochs is not None,
            "times_s": {},
            "metrics": {},
            "metrics_source": None,
            "peak_ram_gb": {},
            "peak_vram_gb": {},
            "notes": [],
            "environment": {
                "host": socket.gethostname(),
                "python": platform.python_version(),
                "packages": {},
            },
            "command": " ".join(sys.argv),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
        }
        self._write()


    def epochs(self):
        configured = self.config["models"][self.model_name].get("epochs")
        epochs = self.epochs_override if self.epochs_override is not None else configured
        if epochs is None:
            raise ValueError("{} sets no epochs for {}.".format(self.config_path, self.model_name))
        self.result["epochs"] = int(epochs)
        if self.epochs_override is not None:
            self.note("Epochs overridden from the command line: {}.".format(epochs))
        return int(epochs)

    def epochs_measured(self, training_s_at_epoch_end, completed=None):
        self.result["epochs_completed"] = (
            len(training_s_at_epoch_end) if completed is None else int(completed)
        )
        self.result["training_s_at_epoch_end"] = [float(s) for s in training_s_at_epoch_end]
        self._write()

    def training_work(self, steps_per_epoch, examples_per_epoch, example):
        self.result["training_work"] = {
            "steps_per_epoch": int(steps_per_epoch),
            "examples_per_epoch": int(examples_per_epoch),
            "example": example,
        }
        self._write()

    def _record_throughput(self):
        work = self.result.get("training_work")
        seconds = self.result["times_s"].get("training")
        epochs = self.result.get("epochs")
        if self.result["status"] != "completed" or not work or not seconds or not epochs:
            return
        self.result["throughput"] = {
            "steps_per_s": work["steps_per_epoch"] * epochs / seconds,
            "examples_per_s": work["examples_per_epoch"] * epochs / seconds,
        }

    def set_metrics(self, metrics, source):
        self.result["metrics"] = {name: float(value) for name, value in metrics.items()}
        self.result["metrics_source"] = source
        self._write()

    def note(self, text):
        self.result["notes"].append(text)
        self._write()

    def versions(self, **packages):
        self.result["environment"]["packages"].update(
            {name: str(version) for name, version in packages.items()}
        )


    @contextmanager
    def phase(self, name):
        if name not in PHASES:
            raise ValueError("Unknown phase '{}', expected one of {}.".format(name, PHASES))
        _synchronize_gpu()
        self.result["current_phase"] = name
        self._write()
        now = time.perf_counter()
        if self._open_phases:
            outer = self._open_phases[-1]
            self._add_time(outer[0], now - outer[1])
        self._open_phases.append([name, now])
        try:
            yield
        except BaseException:
            self.result.setdefault("failed_phase", name)
            raise
        finally:
            _synchronize_gpu()
            now = time.perf_counter()
            current = self._open_phases.pop()
            self._add_time(current[0], now - current[1])
            if self._open_phases:
                self._open_phases[-1][1] = now
            self.result["current_phase"] = (
                self._open_phases[-1][0] if self._open_phases else None
            )
            self._write()

    def add_time(self, name, seconds):
        if name not in PHASES:
            raise ValueError("Unknown phase '{}', expected one of {}.".format(name, PHASES))
        self._add_time(name, float(seconds))
        self._write()

    def framework_report(self, report):
        self.result["framework_report"] = report
        self._write()

    def _add_time(self, name, seconds):
        times = self.result["times_s"]
        times[name] = times.get(name, 0.0) + seconds
        self._update_totals()

    def _update_totals(self):
        times = self.result["times_s"]
        times["total"] = sum(times.get(name, 0.0) for name in PHASES)


    def execute(self, body):
        terminated = []

        def on_sigterm(signum, frame):
            if not terminated:
                terminated.append(True)
                raise Terminated("Received SIGTERM: the time limit was reached.")

        previous = signal.signal(signal.SIGTERM, on_sigterm)
        exit_code = 0
        try:
            body(self)
            self.result["status"] = "completed"
        except BaseException as exc:
            self.result["status"] = "timeout" if terminated else classify(exc)
            self.result["error"] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )[-8000:]
            message = (str(exc).strip().splitlines() or [""])[0]
            self.result["error_message"] = "{}.{}: {}".format(
                type(exc).__module__, type(exc).__qualname__, message
            )[:300]
            exit_code = 124 if terminated else 1
            traceback.print_exc()
        finally:
            signal.signal(signal.SIGTERM, previous)
            self.result["finished_at"] = datetime.now().isoformat(timespec="seconds")
            self._record_throughput()
            self._record_memory()
            self._record_packages()
            self._write()
            if os.environ.get("BENCH_KEEP_WORK") != "1":
                shutil.rmtree(str(self.work_dir), ignore_errors=True)
        print("Benchmark result: {}".format(self.output_dir / "result.json"))
        sys.exit(exit_code)

    def _record_memory(self):
        to_gb = 1024.0**2
        if sys.platform == "darwin":
            to_gb = 1024.0**3
        self.result["peak_ram_gb"]["process"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / to_gb
        )
        self.result["peak_ram_gb"]["largest_child"] = (
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / to_gb
        )
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            self.result["peak_vram_gb"]["torch_allocated"] = (
                torch.cuda.max_memory_allocated() / 1024.0**3
            )
            self.result["environment"]["gpu"] = torch.cuda.get_device_name(0)

    def _record_packages(self):
        packages = self.result["environment"]["packages"]
        for name in ("numpy", "pandas", "scipy", "torch"):
            module = sys.modules.get(name)
            if module is not None and name not in packages:
                packages[name] = str(getattr(module, "__version__", "unknown"))

    def _write(self):
        path = self.output_dir / "result.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as handle:
            json.dump(self.result, handle, indent=2)
        os.replace(str(tmp), str(path))


def classify(exc):
    if isinstance(exc, Terminated):
        return "timeout"
    if isinstance(exc, MemoryError):
        return "oom_ram"
    name = type(exc).__name__
    text = str(exc)
    if name == "OutOfMemoryError" or any(
        marker in text for marker in ("CUDA out of memory", "CUBLAS_STATUS_ALLOC_FAILED")
    ):
        return "oom_gpu"
    if any(
        marker in text
        for marker in (
            "Unable to allocate",
            "std::bad_alloc",
            "can't allocate memory",
            "Cannot allocate memory",
        )
    ):
        return "oom_ram"
    if "out of memory" in text.lower():
        return "oom_gpu" if any(word in text for word in ("CUDA", "cuda", "GPU")) else "oom_ram"
    return "error"


def _synchronize_gpu():
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _relative(path):
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
