from pathlib import Path

import pandas as pd

DATASETS = {
    "ml-1m": {
        "path": "datasets/processed/ml-1m/ratings.tsv",
        "sep": "\t",
        "user": "user_id",
        "item": "movie_id",
        "time": "timestamp",
        "time_format": "unix",
    },
    "ml-32m": {
        "path": "datasets/processed/ml-32m/ratings.csv",
        "sep": ",",
        "user": "userId",
        "item": "movieId",
        "time": "timestamp",
        "time_format": "unix",
    },
    "netflix": {
        "path": "datasets/processed/netflix/ratings.tsv",
        "sep": "\t",
        "user": "customer_id",
        "item": "movie_id",
        "time": "date",
        "time_format": "date",
    },
    "spotify-mpd": {
        "path": "datasets/processed/spotify-mpd/playlists.tsv",
        "sep": "\t",
        "user": "pid",
        "item": "track_uri",
        "item_ids": "str",
        "time": "pos",
        "time_format": "position",
    },
    "lfm2b": {
        "path": "datasets/processed/lfm2b/listening-events.tsv",
        "sep": "\t",
        "user": "user_id",
        "item": "track_id",
        "time": "timestamp",
        "time_format": "datetime",
    },
}

USER, ITEM, TIME = "user", "item", "timestamp"


def dataset_info(name):
    if name not in DATASETS:
        raise ValueError(
            "Unknown dataset '{}'. Known datasets: {}.".format(name, ", ".join(DATASETS))
        )
    return DATASETS[name]


def dataset_path(repo_root, name):
    path = Path(repo_root) / dataset_info(name)["path"]
    if not path.is_file():
        raise FileNotFoundError(
            "Missing {}: run scripts/preprocessing/{}.sh first.".format(path, name)
        )
    return path


def read_interactions(repo_root, name):
    info = dataset_info(name)
    frame = pd.read_csv(
        dataset_path(repo_root, name),
        sep=info["sep"],
        usecols=[info["user"], info["item"], info["time"]],
    )
    frame = frame.rename(
        columns={info["user"]: USER, info["item"]: ITEM, info["time"]: TIME}
    )[[USER, ITEM, TIME]]

    if info["time_format"] in ("date", "datetime"):
        seconds = pd.to_datetime(frame[TIME]).astype("int64") // 10**9
        frame[TIME] = seconds
    else:
        frame[TIME] = frame[TIME].astype("int64")
    return frame
