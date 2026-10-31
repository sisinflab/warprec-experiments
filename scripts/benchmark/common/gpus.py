import math
import sys

import yaml


def gpus_needed(config):
    models = config.get("models") or {}
    if len(models) != 1:
        raise ValueError("The configuration must have one model, not {}.".format(len(models)))
    optimization = next(iter(models.values())).get("optimization") or {}
    per_trial = float(optimization.get("gpu_per_trial", 1))
    concurrent = int(optimization.get("max_concurrent_trials") or 1)
    return max(1, math.ceil(per_trial * concurrent - 1e-9))


if __name__ == "__main__":
    with open(sys.argv[1]) as handle:
        print(gpus_needed(yaml.safe_load(handle)))
