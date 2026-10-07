from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import yaml

from gems.ablation import build_spatial_ablation_commands, summarize_spatial_ablation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan or execute a reproducible spatial-fold ablation matrix"
    )
    parser.add_argument("--matrix", default="configs/ms_edge_matrix.yaml")
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--fold-map", required=True)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--summary-json",
        help="After execution, summarize paired fold scores to this JSON path.",
    )
    args = parser.parse_args()

    matrix = yaml.safe_load(Path(args.matrix).read_text())
    planned = build_spatial_ablation_commands(
        matrix,
        features=args.features,
        labels=args.labels,
        template=args.template,
        fold_map=args.fold_map,
        output_root=args.output_root,
    )

    for item in planned:
        command = item["command"]
        print(
            f"{item['variant']} fold={item['fold']}: "
            + " ".join(shlex.quote(value) for value in command)
        )
        if not args.execute:
            continue
        Path(item["prediction"]).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)

    if args.summary_json:
        if not args.execute:
            raise SystemExit("--summary-json requires --execute")
        summary = summarize_spatial_ablation(planned)
        output = Path(args.summary_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"wrote {output}")

    if not args.execute:
        print(f"planned {len(planned)} runs; add --execute to launch them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
