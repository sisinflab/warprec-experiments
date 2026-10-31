#!/usr/bin/env python3
import argparse
import csv
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.gpus import gpus_needed

EXP1, EXP2, EXP3 = "exp1-data-scalability", "exp2-vertical-scalability", "exp3-horizontal-scalability"
DATASETS = [
    ("ml-1m", "MovieLens-1M"),
    ("netflix", "Netflix-Prize 100M"),
    ("ml-32m", "MovieLens-32M"),
    ("spotify-mpd", "Spotify MPD"),
    ("lfm2b", "LastFM-2b"),
]
MODELS = [
    ("bpr", "BPR"),
    ("twotower", "TwoTower"),
    ("lightgcn", "LightGCN"),
    ("diffrec", "DiffRec"),
    ("lightccf", "LightCCF"),
    ("sasrec", "SASRec"),
    ("esasrec", "eSASRec"),
]
FAILURES = {
    "oom_gpu": "OOM (GPU)",
    "oom_ram": "OOM (RAM)",
    "killed": "OOM (RAM)",
    "timeout": "OOT",
    "error": "Error",
}
GPU_MEMORY_ERRORS = ("CUDA_ERROR_ILLEGAL_ADDRESS",)
EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Terminated|Interrupt|Exit)\b")

EXP1_COLUMNS = [
    "dataset", "model", "status", "epochs", "epochs_completed", "preprocessing_s",
    "evaluation_data_s", "mean_epoch_train_s", "mean_epoch_validation_s", "mean_examples_per_s", "example",
    "examples_per_epoch", "steps_per_epoch", "peak_vram_gb", "peak_ram_gb", "evaluation_s",
    "total_s", "ndcg@10", "recall@10", "oom_gpu_needed_gb", "gpu_capacity_gb", "error", "node", "run",
]
EXP2_COLUMNS = [
    "model", "gpus", "status", "epochs_completed", "mean_examples_per_s", "scaling_efficiency",
    "mean_epoch_train_s", "steps_per_epoch", "peak_vram_gb", "peak_vram_total_gb",
    "peak_ram_gb", "total_s", "error", "node", "run",
]
EXP3_COLUMNS = [
    "dataset", "model", "run_name", "gpus", "gpu_per_trial", "max_concurrent_trials", "cpu_per_trial",
    "num_workers", "status", "trials", "exploration_s", "trials_span_s", "average_trial_s",
    "speedup", "energy_kwh", "emissions_kg", "codecarbon_energy_kwh", "codecarbon_emissions_kg", "peak_vram_gb", "peak_ram_gb", "total_s",
    "ndcg@10", "error", "node", "run",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--output-dir", help="Where the CSV files go (default: the results folder).")
    parser.add_argument("--include-quick-tests", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir or results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = load_latest(results_dir, args.include_quick_tests)
    latest.update(queued_runs())

    exp1 = exp1_rows(latest, results_dir)
    exp2 = exp2_rows(latest, results_dir)
    exp3 = exp3_rows(latest, results_dir)
    for name, columns, rows in ((EXP1, EXP1_COLUMNS, exp1), (EXP2, EXP2_COLUMNS, exp2), (EXP3, EXP3_COLUMNS, exp3)):
        path = output_dir / "{}.csv".format(name)
        with open(str(path), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print("## {} ({} runs, {})\n".format(name, len(rows), path))
        print_markdown(rows, [column for column in columns if column not in ("error", "run", "node")])
        print()


def load_latest(results_dir, include_quick_tests):
    latest = {}
    for path in sorted(results_dir.glob("exp*/*/*/*/result.json")):
        with open(str(path)) as handle:
            result = json.load(handle)
        if result.get("quick_test") and not include_quick_tests:
            continue
        result["status"] = effective_status(result)
        oom_gpu_memory(result, path.parent)
        experiment, dataset, name = path.parts[-5], path.parts[-4], path.parts[-3]
        latest[(experiment, dataset, name)] = (path.parent, result)
    return latest


def queued_runs():
    if not shutil.which("squeue"):
        return {}
    try:
        names = subprocess.run(
            ["squeue", "-h", "-u", getpass.getuser(), "-t", "PENDING", "-o", "%j"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=60,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return {}
    import yaml

    experiments = {experiment.split("-")[0]: experiment for experiment in (EXP1, EXP2, EXP3)}
    datasets = sorted((key for key, _ in DATASETS), key=len, reverse=True)
    queued = {}
    for job in names:
        prefix, _, rest = job.partition("-")
        experiment = experiments.get(prefix)
        dataset = next((key for key in datasets if rest.startswith(key + "-")), None)
        if experiment is None or dataset is None:
            continue
        name = rest[len(dataset) + 1:]
        config = os.path.join("configs", experiment, name + ".yml")
        try:
            with open(str(REPO_ROOT / config)) as handle:
                model_name, params = next(iter(yaml.safe_load(handle)["models"].items()))
        except Exception:
            continue
        optimization = (params or {}).get("optimization") or {}
        result = {
            "status": "queued",
            "model": model_name.lower(),
            "config": config,
            "search": {
                key: optimization.get(key)
                for key in ("gpu_per_trial", "max_concurrent_trials", "cpu_per_trial", "num_workers")
            },
        }
        queued[(experiment, dataset, name)] = (None, result)
    return queued


CUDA_OOM = re.compile(
    r"Tried to allocate ([0-9.]+) (GiB|MiB|KiB|B)\.\s+GPU \d+ has a total capacity of"
    r" ([0-9.]+) (GiB|MiB|KiB|B) of which [0-9.]+ (?:GiB|MiB|KiB|B) is free\.\s+Including"
    r" non-PyTorch memory, this process has ([0-9.]+) (GiB|MiB|KiB|B) memory in use"
)
GIB = {"GiB": 1.0, "MiB": 1.0 / 1024, "KiB": 1.0 / 1024 ** 2, "B": 1.0 / 1024 ** 3}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def oom_gpu_memory(result, run_dir):
    if result.get("status") != "oom_gpu":
        return
    try:
        text = ANSI.sub("", (run_dir / "run.log").read_text(errors="replace"))
    except OSError:
        text = ""
    found = CUDA_OOM.findall(text + "\n" + (result.get("error") or ""))
    if not found:
        return
    asked, asked_unit, total, total_unit, held, held_unit = found[-1]
    result["oom_gpu_needed_gb"] = round(float(held) * GIB[held_unit] + float(asked) * GIB[asked_unit], 1)
    result["gpu_capacity_gb"] = round(float(total) * GIB[total_unit], 1)


def effective_status(result):
    status = result.get("status")
    if status == "error" and any(marker in (result.get("error") or "") for marker in GPU_MEMORY_ERRORS):
        return "oom_gpu"
    return status


def common_fields(run_dir, result, results_dir):
    times = result.get("times_s", {})
    metrics = result.get("metrics", {})
    vram = result.get("peak_vram_gb", {})
    ram = result.get("peak_ram_gb", {})
    return {
        "status": result.get("status"),
        "epochs": result.get("epochs"),
        "epochs_completed": result.get("epochs_completed"),
        "preprocessing_s": _round(times.get("preprocessing")),
        "evaluation_data_s": _round(times.get("evaluation_data")),
        "evaluation_s": _round(times.get("evaluation")),
        "total_s": _round(times.get("total")),
        "peak_vram_gb": _round(vram.get("nvidia_smi", vram.get("torch_allocated"))),
        "peak_vram_total_gb": _round(vram.get("nvidia_smi_total", vram.get("nvidia_smi"))),
        "peak_ram_gb": _round(ram.get("node", max(ram.values()) if ram else None)),
        "ndcg@10": _round(metrics.get("ndcg@10"), 4),
        "recall@10": _round(metrics.get("recall@10"), 4),
        "oom_gpu_needed_gb": result.get("oom_gpu_needed_gb"),
        "gpu_capacity_gb": result.get("gpu_capacity_gb"),
        "error": _error_line(result),
        "node": (result.get("environment", {}).get("host") or "").split(".")[0] or None,
        "run": os.path.relpath(str(run_dir), str(results_dir)) if run_dir else None,
    }


def epoch_fields(result):
    stats = result.get("epoch_stats") or {}
    work = result.get("training_work") or {}
    return {
        "mean_epoch_train_s": _round(stats.get("mean_epoch_train_s")),
        "mean_epoch_validation_s": _round(stats.get("mean_epoch_validation_s")),
        "mean_examples_per_s": _round(stats.get("mean_examples_per_s")),
        "example": work.get("example"),
        "examples_per_epoch": work.get("examples_per_epoch"),
        "steps_per_epoch": work.get("steps_per_epoch"),
    }


def exp1_rows(latest, results_dir):
    rows = []
    for (experiment, dataset, name), (run_dir, result) in latest.items():
        if experiment != EXP1:
            continue
        row = {"dataset": dataset, "model": result.get("model")}
        row.update(common_fields(run_dir, result, results_dir))
        row.update(epoch_fields(result))
        rows.append(row)
    recent = {}
    for row in rows:
        key = (row["dataset"], row["model"])
        if key not in recent or _recency(row) > _recency(recent[key]):
            recent[key] = row
    rows = list(recent.values())
    order = {key: index for index, (key, _) in enumerate(DATASETS)}
    models = {key: index for index, (key, _) in enumerate(MODELS)}
    rows.sort(key=lambda row: (order.get(row["dataset"], 99), models.get(row["model"], 99)))
    return rows


def _recency(row):
    if row["run"] is None:
        return (2, 0, "")
    return (1, int(row["status"] == "completed"), os.path.basename(row["run"]))


def exp2_rows(latest, results_dir):
    rows = []
    for (experiment, _dataset, name), (run_dir, result) in latest.items():
        if experiment != EXP2:
            continue
        row = {"model": result.get("model"), "gpus": run_gpus(result)}
        row.update(common_fields(run_dir, result, results_dir))
        row.update(epoch_fields(result))
        rows.append(row)
    single = {row["model"]: row["mean_examples_per_s"] for row in rows if row["gpus"] == 1}
    for row in rows:
        base = single.get(row["model"])
        if base and row["mean_examples_per_s"] and row["gpus"]:
            row["scaling_efficiency"] = round(row["mean_examples_per_s"] / (row["gpus"] * base), 3)
    models = {key: index for index, (key, _) in enumerate(MODELS)}
    rows.sort(key=lambda row: (models.get(row["model"], 99), row["gpus"] or 0))
    return rows


CARBON_INTENSITY_KG_PER_KWH = 0.330718


def search_energy_kwh(emissions, gpu_per_trial):
    total = emissions.get("energy_kwh")
    gpu = emissions.get("gpu_energy_kwh")
    if total is None or gpu is None or not gpu_per_trial or gpu_per_trial >= 1:
        return total
    return total - gpu * (1 - gpu_per_trial)


def exp3_rows(latest, results_dir):
    rows = []
    for (experiment, dataset, name), (run_dir, result) in latest.items():
        if experiment != EXP3:
            continue
        search = result.get("search") or _config_optimization(result)
        emissions = result.get("emissions") or {}
        energy = search_energy_kwh(emissions, search.get("gpu_per_trial"))
        row = {
            "dataset": dataset,
            "model": result.get("model"),
            "run_name": name,
            "gpus": run_gpus(result),
            "gpu_per_trial": search.get("gpu_per_trial"),
            "max_concurrent_trials": search.get("max_concurrent_trials"),
            "cpu_per_trial": search.get("cpu_per_trial"),
            "num_workers": search.get("num_workers"),
            "trials": search.get("trials", len(result.get("trials") or [])),
            "exploration_s": _round(search.get("exploration_s")),
            "trials_span_s": _round(search.get("trials_span_s")),
            "average_trial_s": _round(search.get("average_trial_s")),
            "energy_kwh": _round(energy, 4),
            "emissions_kg": None if energy is None else round(energy * CARBON_INTENSITY_KG_PER_KWH, 4),
            "codecarbon_energy_kwh": _round(emissions.get("energy_kwh"), 4),
            "codecarbon_emissions_kg": _round(emissions.get("emissions_kg"), 4),
        }
        row.update(common_fields(run_dir, result, results_dir))
        rows.append(row)
    order = {key: index for index, (key, _) in enumerate(DATASETS)}
    models = {key: index for index, (key, _) in enumerate(MODELS)}
    rows.sort(key=lambda row: (order.get(row["dataset"], 99), models.get(row["model"], 99), row["run_name"]))
    for row in rows:
        serial = next(
            (other for other in rows
             if (other["dataset"], other["model"]) == (row["dataset"], row["model"])
             and other["gpus"] == 1 and other["exploration_s"]),
            None,
        )
        if serial and row["exploration_s"]:
            row["speedup"] = round(serial["exploration_s"] / row["exploration_s"], 2)
    return rows


def _config_optimization(result):
    try:
        import yaml

        with open(str(REPO_ROOT / result["config"])) as handle:
            models = yaml.safe_load(handle)["models"]
        return dict(next(iter(models.values())).get("optimization") or {})
    except Exception:
        return {}


def run_gpus(result):
    resources = (result.get("framework_report") or {}).get("resources") or {}
    if resources.get("ray_gpus"):
        return int(resources["ray_gpus"])
    try:
        import yaml

        with open(str(REPO_ROOT / result["config"])) as handle:
            return gpus_needed(yaml.safe_load(handle))
    except Exception:
        return None


def print_markdown(rows, columns):
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for row in rows:
        print("| " + " | ".join("" if row.get(column) is None else str(row[column]) for column in columns) + " |")


def _error_line(result):
    if result.get("error_message"):
        return result["error_message"]
    lines = [line.strip() for line in (result.get("error") or "").splitlines()]
    named = [line for line in lines if EXCEPTION_LINE.match(line)]
    return named[-1][:300] if named else None


def _round(value, digits=1):
    return None if value is None else round(value, digits)


if __name__ == "__main__":
    main()
