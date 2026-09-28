"""Print, initially save, or compare the independent synthetic benchmark."""
import argparse
import json
from pathlib import Path

from src.synthetic_localization.benchmark import run_benchmark

OUTPUT = Path(__file__).resolve().parents[1] / "results/rca/synthetic-marginal-localization-v1.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["compare", "write", "print"], default="compare")
    args = parser.parse_args()
    result = run_benchmark()
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.mode == "write":
        with OUTPUT.open("x") as stream:
            stream.write(serialized)
        print(f"Saved {OUTPUT.relative_to(OUTPUT.parents[2])}")
    elif args.mode == "compare":
        if json.loads(serialized) != json.loads(OUTPUT.read_text()):
            raise SystemExit("FAIL: recomputed results/settings/versions/source hashes differ; saved JSON untouched.")
        print("PASS: in-memory benchmark exactly matches the saved JSON; no files written.")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
