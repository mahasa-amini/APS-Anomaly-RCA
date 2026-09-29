"""Audit existing validation artifacts; compare read-only or create one new summary."""
import argparse
import json
from pathlib import Path
from src.phase1_explanation_audit import audit, RUN_ID

OUTPUT = Path(__file__).resolve().parents[1] / 'results/phase1_explanations' / RUN_ID / 'audit.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['compare', 'write'], default='compare')
    args = parser.parse_args()
    if args.mode == 'write' and OUTPUT.exists():
        raise FileExistsError('Audit output exists; use --mode compare. Nothing overwritten.')
    result = audit()
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.mode == 'compare':
        if json.loads(serialized) != json.loads(OUTPUT.read_text()):
            raise SystemExit('FAIL: audit differs from saved summary; nothing overwritten.')
        print('PASS: verified validation audit matches saved summary; no output written.')
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open('x') as stream:
            stream.write(serialized)
        print('Created new aggregate validation audit; existing inputs/results untouched.')


if __name__ == '__main__':
    main()
