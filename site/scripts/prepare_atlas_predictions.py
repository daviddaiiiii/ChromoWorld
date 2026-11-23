"""Package the archived ChromoPix polygons for the exact fields in the 2D atlas."""
import argparse
import json
import re
import sys
from pathlib import Path


def read_predictions(path):
    polygons = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        category, score, *points = line.split()
        cls = int(category)
        confidence = float(score)
        coords = [float(value) for value in points]
        if not 0 <= cls < 24 or not 0 <= confidence <= 1:
            raise ValueError(f"Invalid class or confidence in {path.name}")
        if len(coords) < 6 or len(coords) % 2 or not all(0 <= v <= 1 for v in coords):
            # Degenerate contours have no displayable area.
            if len(coords) < 6 and len(coords) % 2 == 0:
                continue
            raise ValueError(f"Invalid polygon in {path.name}")
        polygons.append([cls, [round(value, 5) for value in coords]])
    return polygons


def build(index, predictions, destination):
    sources = {}
    for group in index.values():
        for rows in group.values():
            for row in rows:
                key = Path(row["field"]).stem
                source = row["source"]
                if not re.fullmatch(r"[a-zA-Z0-9_-]+", source) or not re.fullmatch(r"[a-f0-9]{12}", key):
                    raise ValueError("Invalid atlas field identifier")
                if sources.setdefault(key, source) != source:
                    raise ValueError(f"Multiple source fields for {key}")

    # Read all the inputs before replacing any browser assets.
    data = {key: read_predictions(predictions / (source + ".txt")) for key, source in sources.items()}
    destination.mkdir(parents=True, exist_ok=True)
    for key, polygons in data.items():
        (destination / (key + ".json")).write_text(json.dumps(polygons, separators=(",", ":")))
    print(f"Exported {sum(map(len, data.values()))} predictions for {len(data)} matched fields", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--index", type=Path, help="Atlas index JSON. Reads stdin if omitted.")
    args = parser.parse_args()
    manifest = json.loads(args.index.read_text()) if args.index else json.load(sys.stdin)
    build(manifest, args.predictions, args.output)
