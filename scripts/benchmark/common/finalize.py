import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--elapsed", type=float, help="Seconds the runner ran.")
    parser.add_argument("--time-limit", type=float, help="The runner's time limit, in seconds.")
    args = parser.parse_args()

    run_dir = Path(args.output_dir)
    path = run_dir / "result.json"
    result = {"status": "running", "notes": [], "peak_vram_gb": {}}
    if path.is_file():
        with open(path) as handle:
            result = json.load(handle)

    result["exit_code"] = args.exit_code
    if result.get("status") == "running":
        if result.get("current_phase"):
            result["failed_phase"] = result["current_phase"]
        if args.exit_code == 124:
            result["status"] = "timeout"
        elif args.exit_code == 143:
            result["status"] = "timeout"
            result.setdefault("notes", []).append(
                "Stopped by SIGTERM from outside: the scheduler's time limit, or a "
                "cancellation of the job."
            )
        elif args.exit_code in (137, -9):
            if args.elapsed is not None and args.time_limit and args.elapsed >= args.time_limit:
                result["status"] = "timeout"
            else:
                result["status"] = "oom_ram"
                result.setdefault("notes", []).append(
                    "Killed with SIGKILL before the time limit: the kernel's out-of-memory "
                    "killer. {}".format(oom_kills_note())
                )
        elif args.exit_code == 0:
            result["status"] = "completed"
        else:
            result["status"] = "error"
        result["finished_at"] = datetime.now().isoformat(timespec="seconds")

    gpu_peaks = gpu_memory_peaks(run_dir / "gpu-memory.log")
    if gpu_peaks:
        result.setdefault("peak_vram_gb", {}).update(gpu_peaks)
    ram_peak = node_ram_peak_gb(run_dir / "ram.log")
    if ram_peak is not None:
        result.setdefault("peak_ram_gb", {})["node"] = ram_peak

    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as handle:
        json.dump(result, handle, indent=2)
    os.replace(str(tmp), str(path))
    print("{}: {}".format(path, result["status"]))


def gpu_memory_peaks(path):
    if not path.is_file():
        return {}
    samples = []
    with open(str(path)) as handle:
        for line in handle:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2 or not all(field.isdigit() for field in fields):
                continue
            index, used = int(fields[0]), int(fields[1])
            if not samples or index in samples[-1]:
                samples.append({})
            samples[-1][index] = used
    if not samples:
        return {}
    per_gpu = {}
    for sample in samples:
        for index, used in sample.items():
            per_gpu[index] = max(per_gpu.get(index, 0), used)
    return {
        "nvidia_smi": max(per_gpu.values()) / 1024.0,
        "nvidia_smi_per_gpu": {str(index): used / 1024.0 for index, used in sorted(per_gpu.items())},
        "nvidia_smi_total": max(sum(sample.values()) for sample in samples) / 1024.0,
    }


def node_ram_peak_gb(path):
    if not path.is_file():
        return None
    used = []
    with open(str(path)) as handle:
        for line in handle:
            line = line.strip()
            if line.isdigit():
                used.append(int(line))
    if len(used) < 2:
        return None
    return (max(used) - used[0]) / 1024.0**2


def oom_kills_note():
    try:
        with open("/proc/self/cgroup") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return "No cgroup information."
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":
            candidates = [Path("/sys/fs/cgroup") / path.lstrip("/") / "memory.events"]
        elif "memory" in controllers.split(","):
            candidates = [Path("/sys/fs/cgroup/memory") / path.lstrip("/") / "memory.oom_control"]
        else:
            continue
        candidates += [candidate.parent.parent / candidate.name for candidate in candidates]
        for candidate in candidates:
            try:
                with open(str(candidate)) as handle:
                    for entry in handle.read().splitlines():
                        key, _, value = entry.partition(" ")
                        if key == "oom_kill":
                            return "The memory cgroup ({}) counts {} OOM kill(s).".format(
                                candidate, value
                            )
            except OSError:
                continue
    return "The memory cgroup gives no OOM count."


if __name__ == "__main__":
    main()
