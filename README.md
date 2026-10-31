# WarpRec experiments

Experiments run with **WarpRec 2.2.8** (`frameworks/warprec`, tag `v2.2.8`).

## 1. Clone

```bash
git clone --recurse-submodules https://github.com/sisinflab/warprec-experiments.git
cd warprec-experiments
```

If `frameworks/warprec` is empty:

```bash
git clone --branch v2.2.8 https://github.com/sisinflab/warprec.git frameworks/warprec
```

## 2. Environment

```bash
scripts/setup/install_env.sh warprec
conda activate ./envs/warprec
```

With micromamba, use `micromamba activate ./envs/warprec`.

## 3. Datasets

Download MovieLens-1M, MovieLens-32M and Netflix Prize:

```bash
scripts/download/datasets.sh
```

Place the other two archives here:

- `datasets/raw/spotify-mpd/spotify_million_playlist_dataset.zip` ([Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge))
- `datasets/raw/lfm2b/LFM-2b.zip`

Preprocess:

```bash
scripts/preprocessing/ml-1m.sh
scripts/preprocessing/netflix.sh
scripts/preprocessing/ml-32m.sh
scripts/preprocessing/spotify-mpd.sh
scripts/preprocessing/lfm2b.sh
```

On a SLURM cluster, the preprocessing can run as a job:

```bash
sbatch -A <account> -p <partition> hpc/preprocessing/<dataset>.sbatch
```

## 4. Run on a server (Jupyter)

Start JupyterLab from the repository folder, with the environment active:

```bash
conda activate ./envs/warprec
jupyter lab
```

Open a notebook from the `notebooks/` folder. Each code cell runs one step: select it and press `Shift+Enter`. The cell shows the run's output and finishes when the run ends. Run only the cells you need, one at a time.

| Order | Notebook | One cell for each |
|---|---|---|
| 1 | `notebooks/download/datasets.ipynb` | download of MovieLens-1M, MovieLens-32M and Netflix Prize (one cell) |
| 2 | `notebooks/preprocessing/<dataset>.ipynb` | dataset to preprocess (one notebook per dataset) |
| 3 | `notebooks/benchmark/exp1.ipynb` | model on a dataset, grouped by dataset |
| 4 | `notebooks/benchmark/exp2.ipynb` | model on 1, 2 or 4 GPUs, grouped by model |
| 5 | `notebooks/benchmark/exp3.ipynb` | search setting (a) to (d), grouped by model |
| 6 | `notebooks/benchmark/results.ipynb` | collection of the results (one cell) |

Notebooks 1 and 2 do the same as step 3, and notebook 6 does the same as step 6: skip what you have already run from the shell.

Each benchmark cell runs one command, for example:

```bash
!bash ../../scripts/benchmark/run.sh ../../configs/exp1-data-scalability/bpr.yml ml-1m
```

The same run from a shell, in the repository folder:

```bash
scripts/benchmark/run.sh configs/exp1-data-scalability/bpr.yml ml-1m
```

## 5. Run on a SLURM cluster

```bash
export BENCH_PARTITION=<gpu-partition>
export BENCH_SCRATCH_DIR=<scratch-directory>
```

`BENCH_SCRATCH_DIR` is optional. Each run is one job on a whole node:

```bash
hpc/benchmark/submit.sh <account> <config> <dataset> [sbatch options]
```

Experiment 1, data scalability:

```bash
for dataset in ml-1m netflix ml-32m spotify-mpd lfm2b; do
    for model in bpr twotower lightgcn diffrec lightccf sasrec esasrec; do
        hpc/benchmark/submit.sh <account> configs/exp1-data-scalability/$model.yml $dataset
    done
done
```

Experiment 2, vertical scalability:

```bash
for config in configs/exp2-vertical-scalability/*.yml; do
    hpc/benchmark/submit.sh <account> $config ml-32m
done
```

Experiment 3, horizontal scalability:

```bash
for config in configs/exp3-horizontal-scalability/*.yml; do
    hpc/benchmark/submit.sh <account> $config ml-32m
done
```

## 6. Results

```bash
python scripts/benchmark/collect.py
```

This writes one CSV file per experiment to `results/`. Each run also has its own `result.json` in `results/<experiment>/<dataset>/<config>/<timestamp>/`.

## Memory estimate on LastFM-2b

Separate from the experiments: it measures the peak GPU memory of DiffRec and LightGCN on random subsets of LastFM-2b's items and extrapolates it to the whole dataset.

Build the subsets (about 250 GB of RAM):

```bash
python scripts/memory/measure_memory.py subsets lfm2b 0.01 0.02 0.03 0.04 0.05 0.10 0.15 0.20 0.25
```

Measure on one GPU and extrapolate:

```bash
python scripts/memory/measure_memory.py measure configs/exp1-data-scalability/diffrec-lfm2b.yml lfm2b 0.01 0.02 0.03 0.04 0.05
python scripts/memory/measure_memory.py fit configs/exp1-data-scalability/diffrec-lfm2b.yml lfm2b
python scripts/memory/measure_memory.py measure configs/exp1-data-scalability/lightgcn-lfm2b.yml lfm2b 0.05 0.10 0.15 0.20 0.25
python scripts/memory/measure_memory.py fit configs/exp1-data-scalability/lightgcn-lfm2b.yml lfm2b
```

On a SLURM cluster:

```bash
sbatch -A <account> -p <partition> hpc/memory/subsets.sbatch lfm2b 0.01 0.02 0.03 0.04 0.05 0.10 0.15 0.20 0.25
sbatch -A <account> -p <gpu-partition> hpc/memory/measure.sbatch configs/exp1-data-scalability/diffrec-lfm2b.yml lfm2b 0.01 0.02 0.03 0.04 0.05
sbatch -A <account> -p <gpu-partition> hpc/memory/measure.sbatch configs/exp1-data-scalability/lightgcn-lfm2b.yml lfm2b 0.05 0.10 0.15 0.20 0.25
```

On a server, the same steps are the cells of `notebooks/memory/lfm2b.ipynb`. The estimate of each model is in `results/memory/lfm2b/<config>/fit.json`.
