import importlib
import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA_PACKAGES = ("polars", "narwhals", "pandas", "numpy", "pyarrow", "scipy")
SUBMODULES = ("scipy.sparse",)
CHUNK = 1 << 20
THREADS = 64


def main():
    start = time.monotonic()
    packages = [name for name in DATA_PACKAGES if importlib.util.find_spec(name) is not None]
    for name in packages + [s for s in SUBMODULES if s.split(".")[0] in packages]:
        importlib.import_module(name)
    imported = time.monotonic()

    files = set()
    for name in packages:
        module = sys.modules[name]
        for folder in getattr(module, "__path__", []):
            files.update(path for path in Path(folder).rglob("*") if path.is_file())
        folder = Path(module.__file__).parent
        files.update(path for path in folder.parent.glob(name + ".libs/*") if path.is_file())
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and ".so" in fields[5]:
                files.add(Path(fields[5].strip()))
    listed = time.monotonic()

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        size = sum(pool.map(read, sorted(files)))
    end = time.monotonic()
    print(
        "Preloaded the data libraries ({}): {} files, {:.0f} MB, in {:.1f} s "
        "(imports {:.1f} s, listing {:.1f} s, reading {:.1f} s).".format(
            ", ".join(packages), len(files), size / 1e6, end - start,
            imported - start, listed - imported, end - listed,
        ),
        flush=True,
    )


def read(path):
    size = 0
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(CHUNK)
                if not chunk:
                    break
                size += len(chunk)
    except OSError:
        pass
    return size


if __name__ == "__main__":
    main()
