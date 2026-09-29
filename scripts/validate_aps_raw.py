"""Read-only checks for one original APS CSV against recorded public metadata."""
import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "aps-develop-20260927T135520Z-a6474c"
TRAIN_SHA256 = "cbbcc17b812feaff4433b4aac7fb1e94b8a95381e93a29e63069afd20782d5d8"
TEST_SHA256 = "d02335a04b3db9bdad06a617afcd607660b10cd2d6c63628831206af05149251"
TRAIN_BYTES = 44_668_322
TEST_BYTES = 11_942_686
AUDIT_FILE = f"aps-develop-20260927T135520Z-a6474c-feature-quality.json"


@dataclass(frozen=True)
class ExpectedFile:
    filename: str
    sha256: str
    size: int
    rows: int
    class_counts: dict[str, int]
    features: tuple[str, ...]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def public_profile(split, root=ROOT):
    """Read committed summaries for the selected split; never inspect the other CSV."""
    require(split in ("train", "test"), "Choose --split train or --split test.")
    root = Path(root)
    quality = json.loads((root / "results/rca" / AUDIT_FILE).read_text(encoding="utf-8"))
    require(quality.get("run_id") == RUN_ID, "Feature schema summary has the wrong run ID.")
    records = quality["feature_quality"]
    features = tuple(record["aps_feature"] for record in records)
    require(len(features) == 170 and len(set(features)) == 170 and
            [record["legacy_name"] for record in records] == [f"f_{i}" for i in range(170)],
            "Committed feature schema must contain 170 unique features in indexed order.")
    require(quality["verification"]["training_sha256"] == TRAIN_SHA256,
            "Feature schema summary disagrees with recorded training hash.")

    public_dir = root / "results/phase1" / RUN_ID
    if split == "train":
        development = json.loads((public_dir / "development.json").read_text(encoding="utf-8"))
        require(development.get("run_id") == RUN_ID and development["training_sha256"] == TRAIN_SHA256,
                "Development summary disagrees with recorded training input.")
        pieces = (development["data"]["train"], development["data"]["validation"])
        require(all(part["shape"][1] == 170 for part in pieces), "Development feature count mismatch.")
        rows = sum(part["shape"][0] for part in pieces)
        counts = {label: sum(part["label_counts"][label] for part in pieces)
                  for label in ("0", "1")}
        return ExpectedFile("aps_failure_training_set.csv", TRAIN_SHA256, TRAIN_BYTES,
                            rows, {"neg": counts["0"], "pos": counts["1"]}, features)

    public_test = json.loads((public_dir / "test.json").read_text(encoding="utf-8"))
    require(public_test.get("run_id") == RUN_ID and public_test.get("stage") == "official_test",
            "Official test summary has the wrong run/stage.")
    metadata = public_test["test_input"]
    require(metadata["sha256"] == TEST_SHA256 and metadata["bytes"] == TEST_BYTES,
            "Official test summary disagrees with recorded test input.")
    data = public_test["data"]
    require(data["shape"][1] == 170, "Official test feature count mismatch.")
    return ExpectedFile("aps_failure_test_set.csv", TEST_SHA256, TEST_BYTES,
                        data["shape"][0], {"neg": data["label_counts"]["0"],
                                            "pos": data["label_counts"]["1"]}, features)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw_file(path, expected):
    """Stream one CSV; return compact checks, never write or load the other split."""
    path = Path(path)
    require(path.name == expected.filename,
            f"Wrong filename: expected {expected.filename}, found {path.name}.")
    require(path.is_file(), f"Raw CSV missing: {path}.")
    require(len(expected.features) > 0 and len(expected.features) == len(set(expected.features)),
            "Expected feature schema is empty or duplicated.")
    size_before = path.stat().st_size
    rows = 0
    counts = {"neg": 0, "pos": 0}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        require(header and header[0] == "class", "CSV header must start with class.")
        require(len(header) == len(set(header)), "CSV header contains duplicate names.")
        require(len(header) - 1 == len(expected.features),
                f"Feature count mismatch: expected {len(expected.features)}, found {len(header) - 1}.")
        require(tuple(header[1:]) == expected.features,
                "Feature names/order differ from the recorded APS schema.")
        for line_number, record in enumerate(reader, start=2):
            require(len(record) == len(header),
                    f"CSV row {line_number} has {len(record)} fields; expected {len(header)}.")
            require(record[0] in counts,
                    f"CSV row {line_number} has invalid class {record[0]!r}; expected neg or pos.")
            for column, value in enumerate(record[1:], start=2):
                if value == "na":
                    continue
                try:
                    numeric = float(value)
                except ValueError as exc:
                    raise ValueError(f"CSV row {line_number}, column {column} is neither numeric nor na.") from exc
                require(math.isfinite(numeric),
                        f"CSV row {line_number}, column {column} has a nonfinite value.")
            counts[record[0]] += 1
            rows += 1
    require(rows == expected.rows, f"Row count mismatch: expected {expected.rows}, found {rows}.")
    require(counts == expected.class_counts,
            f"Class counts mismatch: expected {expected.class_counts}, found {counts}.")
    size_after = path.stat().st_size
    require(size_after == size_before, "Raw CSV changed while being read.")
    require(size_after == expected.size,
            f"Byte size mismatch: expected {expected.size}, found {size_after}.")
    observed_sha256 = sha256_file(path)
    require(observed_sha256 == expected.sha256,
            f"SHA-256 mismatch: expected {expected.sha256}, found {observed_sha256}.")
    require(path.stat().st_size == size_after, "Raw CSV changed during hashing.")
    return {"filename": expected.filename, "rows": rows, "features": len(expected.features),
            "class_counts": counts, "bytes": size_after, "sha256": observed_sha256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True,
                        help="Validate only the selected original CSV.")
    args = parser.parse_args()
    expected = public_profile(args.split)
    path = ROOT / "data/raw" / expected.filename
    try:
        result = validate_raw_file(path, expected)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"FAIL [{args.split}]: {exc}\n")
    print(json.dumps({"split": args.split, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
