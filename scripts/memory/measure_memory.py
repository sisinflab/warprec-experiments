#!/usr/bin/env python3
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

from common.datasets import dataset_info, dataset_path

GIB = 1024 ** 3
SKIPPED_PARAMS = ("meta", "optimization", "early_stopping")


def subsets_dir(dataset):
    return REPO_ROOT / "datasets" / "processed" / dataset / "item-subsets"


def subset_path(dataset, fraction):
    return subsets_dir(dataset) / "items-{:.3f}.npz".format(fraction)


def output_dir(config_path, dataset):
    return REPO_ROOT / "results" / "memory" / dataset / Path(config_path).stem


def read_pairs(dataset, max_rows):
    import polars as pl

    info = dataset_info(dataset)
    pairs = (
        pl.scan_csv(dataset_path(REPO_ROOT, dataset), separator=info["sep"], quote_char=None, n_rows=max_rows)
        .select([info["user"], info["item"]])
        .unique()
        .select(
            (pl.col(info["user"]).rank("dense").cast(pl.Int32) - 1).alias("user"),
            (pl.col(info["item"]).rank("dense").cast(pl.Int32) - 1).alias("item"),
        )
        .collect()
    )
    return pairs["user"].to_numpy(), pairs["item"].to_numpy()


def build_subsets(args):
    import scipy.sparse as sp

    folder = subsets_dir(args.dataset)
    folder.mkdir(parents=True, exist_ok=True)
    cache = {"users": folder / "pairs-users.npy", "items": folder / "pairs-items.npy", "full": folder / "full.json"}
    started = time.time()
    if all(path.exists() for path in cache.values()):
        with open(cache["full"]) as handle:
            full = json.load(handle)
        if full["max_rows"] != args.max_rows or full["seed"] != args.seed:
            raise SystemExit("{} was built with other options: remove it first.".format(folder))
        users, items = np.load(cache["users"]), np.load(cache["items"])
        print("{}: pairs loaded from {}".format(args.dataset, folder), flush=True)
    else:
        users, items = read_pairs(args.dataset, args.max_rows)
        full = {"n_users": int(users.max()) + 1, "n_items": int(items.max()) + 1, "pairs": len(users),
                "seed": args.seed, "max_rows": args.max_rows}
        np.save(cache["users"], users)
        np.save(cache["items"], items)
        with open(cache["full"], "w") as handle:
            json.dump(full, handle, indent=2)
    n_users, n_items = full["n_users"], full["n_items"]
    print("{}: {} users, {} items, {} pairs, ready in {:.0f} s".format(
        args.dataset, n_users, n_items, full["pairs"], time.time() - started), flush=True)

    rank = np.empty(n_items, dtype=np.int32)
    rank[np.random.default_rng(args.seed).permutation(n_items)] = np.arange(n_items, dtype=np.int32)
    item_rank = rank[items]
    del items, rank

    for fraction in sorted(args.fractions):
        path = subset_path(args.dataset, fraction)
        if path.exists():
            print("items {:.3f}: already built".format(fraction), flush=True)
            continue
        kept = int(round(fraction * n_items))
        mask = item_rank < kept
        matrix = sp.csr_matrix(
            (np.ones(int(mask.sum()), dtype=np.float32), (users[mask], item_rank[mask])),
            shape=(n_users, kept),
        )
        partial = path.with_name(path.stem + ".partial.npz")
        sp.save_npz(partial, matrix, compressed=False)
        partial.rename(path)
        print("items {:.3f}: {} items, {} pairs, {:.0f} s".format(
            fraction, kept, matrix.nnz, time.time() - started), flush=True)
        del matrix, mask


class Interactions:
    def __init__(self, matrix):
        self.matrix = matrix

    def get_sparse(self):
        return self.matrix


def model_params(name, section):
    from warprec.utils.registry import params_registry

    spec = params_registry.get(name, **copy.deepcopy(section))
    params = {}
    for key, value in spec.model_dump().items():
        if key in SKIPPED_PARAMS:
            continue
        if isinstance(value, list) and value:
            choices = value[1:]
            if len(choices) != 1:
                raise ValueError("{} has {} values for '{}': one is needed.".format(name, len(choices), key))
            value = choices[0]
        params[key] = value
    return params


def optimizer_of(model):
    optimizer = model.configure_optimizers()
    while isinstance(optimizer, (list, tuple)):
        optimizer = optimizer[0]
    if isinstance(optimizer, dict):
        optimizer = optimizer["optimizer"]
    return optimizer


def training_batch(model, matrix, batch_size, generator, device):
    import torch
    from torch.utils.data import DataLoader

    from warprec.data.entities.train_structures.interaction_structures import InteractionDataset

    if type(model).__name__ == "DiffRec":
        loader = DataLoader(InteractionDataset(matrix), batch_size=batch_size, shuffle=True, generator=generator)
        return [tensor.to(device) for tensor in next(iter(loader))]
    n_users, n_items = matrix.shape
    users = torch.randint(n_users, (batch_size,), generator=generator)
    positives = torch.randint(n_items, (batch_size,), generator=generator)
    negatives = torch.randint(n_items, (batch_size,), generator=generator)
    return [users.to(device), positives.to(device), negatives.to(device)]


def tensor_gib(tensors):
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors) / GIB


def memory_state(model, optimizer):
    import torch

    parameters = [parameter for parameter in model.parameters()]
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    states = [value for state in optimizer.state.values() for value in state.values()
              if torch.is_tensor(value) and value.is_cuda]
    allocated = torch.cuda.memory_allocated() / GIB
    accounted = tensor_gib(parameters) + tensor_gib(gradients) + tensor_gib(states)
    return {
        "allocated_gib": allocated,
        "parameters_gib": tensor_gib(parameters),
        "gradients_gib": tensor_gib(gradients),
        "optimizer_state_gib": tensor_gib(states),
        "other_gib": allocated - accounted,
    }


def measure_fraction(args, fraction):
    import scipy.sparse as sp
    import torch

    import warprec.recommenders
    from warprec.utils.registry import model_registry

    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    (name, section), = config["models"].items()
    evaluation = config.get("evaluation") or {}
    eval_batch = int(evaluation.get("batch_size", 1024))
    candidates = int(evaluation.get("num_negatives", 0)) + 1
    seed = int(((section.get("optimization") or {}).get("properties") or {}).get("seed", 42))
    params = model_params(name, section)

    matrix = sp.load_npz(subset_path(args.dataset, fraction)).tocsr()
    n_users, n_items = matrix.shape
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(seed)
    point = {
        "config": str(Path(args.config).resolve().relative_to(REPO_ROOT)),
        "dataset": args.dataset,
        "fraction": fraction,
        "n_users": n_users,
        "n_items": n_items,
        "pairs": int(matrix.nnz),
        "model": name,
        "params": params,
        "train_batch": params["batch_size"],
        "eval_batch": min(eval_batch, n_users),
        "eval_candidates": candidates,
        "steps": args.steps,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capacity_gib": torch.cuda.get_device_properties(0).total_memory / GIB,
        "torch": torch.__version__,
        "status": "completed",
        "baseline_gib": torch.cuda.memory_allocated() / GIB,
    }
    started = time.time()
    try:
        model = model_registry.get(name, params=params, interactions=Interactions(matrix),
                                   info={"n_users": n_users, "n_items": n_items}, seed=seed)
        model.log = lambda *args, **kwargs: None
        model.to(device)
        torch.cuda.synchronize()
        point["build_seconds"] = time.time() - started
        point["parameters"] = [
            {"name": parameter_name, "shape": list(parameter.shape), "dtype": str(parameter.dtype).replace("torch.", ""),
             "gib": tensor_gib([parameter])}
            for parameter_name, parameter in model.named_parameters()
        ]
        point["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        optimizer = optimizer_of(model)
        point["optimizer"] = type(optimizer).__name__
        point["built"] = memory_state(model, optimizer)

        model.train()
        point["train_steps"] = []
        for step in range(args.steps):
            torch.cuda.reset_peak_memory_stats()
            batch = training_batch(model, matrix, params["batch_size"], generator, device)
            record = {"step": step, "batch_shapes": [list(tensor.shape) for tensor in batch]}
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_step(batch, step)
            torch.cuda.synchronize()
            record["after_forward"] = memory_state(model, optimizer)
            loss.backward()
            torch.cuda.synchronize()
            record["after_backward"] = memory_state(model, optimizer)
            optimizer.step()
            torch.cuda.synchronize()
            record["after_optimizer_step"] = memory_state(model, optimizer)
            record["peak_gib"] = torch.cuda.max_memory_allocated() / GIB
            record["reserved_peak_gib"] = torch.cuda.max_memory_reserved() / GIB
            point["train_steps"].append(record)
            del batch, loss
        point["train_peak_gib"] = max(record["peak_gib"] for record in point["train_steps"])

        model.eval()
        torch.cuda.synchronize()
        point["eval_before"] = memory_state(model, optimizer)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            users = torch.randint(n_users, (point["eval_batch"],), generator=generator)
            items = torch.randint(n_items, (len(users), candidates), generator=generator)
            scores = model.predict(users.to(device), item_indices=items.to(device))
        torch.cuda.synchronize()
        point["eval_scores_shape"] = list(scores.shape)
        point["eval_peak_gib"] = torch.cuda.max_memory_allocated() / GIB
        point["eval_reserved_peak_gib"] = torch.cuda.max_memory_reserved() / GIB
        point["model_gib"] = point["built"]["allocated_gib"]
    except torch.cuda.OutOfMemoryError as error:
        point["status"] = "oom"
        point["error"] = str(error).splitlines()[0]
    point["seconds"] = time.time() - started

    folder = output_dir(args.config, args.dataset)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "items-{:.3f}.json".format(fraction), "w") as handle:
        json.dump(point, handle, indent=2)
    print(json.dumps({key: point.get(key) for key in (
        "model", "fraction", "n_items", "pairs", "status", "model_gib", "train_peak_gib", "eval_peak_gib",
        "seconds", "error")}), flush=True)


def measure(args):
    import subprocess

    fractions = sorted(args.fractions)
    if len(fractions) == 1:
        measure_fraction(args, fractions[0])
        return
    for fraction in fractions:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "measure", args.config, args.dataset,
                        str(fraction), "--steps", str(args.steps)], check=False)


def fit(args):
    folder = output_dir(args.config, args.dataset)
    with open(subsets_dir(args.dataset) / "full.json") as handle:
        full = json.load(handle)
    points = []
    for path in sorted(folder.glob("items-*.json")):
        with open(path) as handle:
            point = json.load(handle)
        if point["status"] == "completed":
            points.append(point)
    if len(points) < 2:
        raise SystemExit("At least two completed measurements are needed, found {}.".format(len(points)))

    items = np.array([point["n_items"] for point in points], dtype=float)
    pairs = np.array([point["pairs"] for point in points], dtype=float)
    quantities = {
        "baseline_gib": lambda point: point["baseline_gib"],
        "train_peak_gib": lambda point: point["train_peak_gib"] - point["baseline_gib"],
        "eval_peak_gib": lambda point: point["eval_peak_gib"] - point["baseline_gib"],
        "parameters_gib": lambda point: point["eval_before"]["parameters_gib"],
        "gradients_gib": lambda point: point["eval_before"]["gradients_gib"],
        "optimizer_state_gib": lambda point: point["eval_before"]["optimizer_state_gib"],
        "other_persistent_gib": lambda point: point["eval_before"]["other_gib"] - point["baseline_gib"],
        "parameter_count": lambda point: point["parameter_count"],
    }
    result = {
        "model": points[0]["model"],
        "config": points[0]["config"],
        "params": points[0]["params"],
        "gpu": points[0]["gpu"],
        "gpu_capacity_gib": points[0]["gpu_capacity_gib"],
        "full": full,
        "pairs_per_item": {"measured": (pairs / items).tolist(), "full": full["pairs"] / full["n_items"]},
        "measured": [dict({"fraction": point["fraction"], "n_items": point["n_items"], "pairs": point["pairs"]},
                          **{key: get(point) for key, get in quantities.items()}) for point in points],
        "extrapolated": {},
    }
    for key, get in quantities.items():
        values = np.array([get(point) for point in points], dtype=float)
        slope, intercept = np.polyfit(items, values, 1)
        predicted = slope * items + intercept
        spread = float(((values - values.mean()) ** 2).sum())
        r2 = 1.0 - float(((values - predicted) ** 2).sum()) / spread if spread else 1.0
        result["extrapolated"][key] = {"per_million_items": slope * 1e6, "intercept": intercept, "r2": r2,
                                       "full": slope * full["n_items"] + intercept}
    result["needed_gib"] = max(result["extrapolated"]["train_peak_gib"]["full"],
                               result["extrapolated"]["eval_peak_gib"]["full"])

    with open(folder / "fit.json", "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    subsets = commands.add_parser("subsets")
    subsets.add_argument("dataset")
    subsets.add_argument("fractions", type=float, nargs="+")
    subsets.add_argument("--seed", type=int, default=42)
    subsets.add_argument("--max-rows", type=int)

    one = commands.add_parser("measure")
    one.add_argument("config")
    one.add_argument("dataset")
    one.add_argument("fractions", type=float, nargs="+")
    one.add_argument("--steps", type=int, default=2)

    line = commands.add_parser("fit")
    line.add_argument("config")
    line.add_argument("dataset")

    args = parser.parse_args()
    {"subsets": build_subsets, "measure": measure, "fit": fit}[args.command](args)


if __name__ == "__main__":
    main()
