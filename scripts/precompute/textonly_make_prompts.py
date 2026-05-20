#!/usr/bin/env python3
"""Convert text-only planning JSONL into SpaceDrive inference prompt format.

Reads the JSONL produced by textonly_build.py and writes a new JSONL where each
line is a JSON object with "Q" (prompt) and metadata fields, ready for
text-only trajectory prediction inference.

Usage
-----
    python textonly_make_prompts.py \
        --input workspace/textonly_planning_full.jsonl \
        --output workspace/textonly_prompts.jsonl
"""

import argparse
import json
import sys


PROMPT_TEMPLATE = (
    "system\n"
    "You are a helpful assistant.\n"
    "user\n"
    "{scene_text}\n\n"
    "You are driving in {city}. {mission}. "
    "Please provide the planning trajectory for the ego car without reasons.\n"
    "assistant\n"
)


def _strip_mission_section(prompt: str) -> str:
    """Remove the === MISSION === block from the scene text."""
    lines = prompt.split("\n")
    out = []
    skip = False
    for line in lines:
        if line.strip() == "=== MISSION ===":
            skip = True
            # Also drop the blank line before the section header
            if out and out[-1] == "":
                out.pop()
            continue
        if skip:
            if line.startswith("=== ") and line.endswith(" ==="):
                skip = False
            else:
                continue
        out.append(line)
    # Strip trailing blank lines
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def build_qa(record: dict) -> dict:
    city = record["location"].split("-")[0]
    mission = record["mission"].rstrip(".")
    scene_text = _strip_mission_section(record["prompt"])

    q = PROMPT_TEMPLATE.format(
        scene_text=scene_text,
        city=city,
        mission=mission,
    )

    return {
        "token": record["token"],
        "Q": q,
        "target_trajectory_xy": record["target_trajectory_xy"],
        "kinematic_baseline_xy": record["kinematic_baseline_xy"],
        "turn_severity": record["turn_severity"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", required=True, help="Input JSONL from textonly_build.py")
    p.add_argument("--output", "-o", required=True, help="Output JSONL with Q prompts")
    p.add_argument("--max-samples", type=int, default=0, help="Cap output (0=unlimited)")
    args = p.parse_args()

    written = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            out = build_qa(rec)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
            if args.max_samples > 0 and written >= args.max_samples:
                break

    print(f"Written {written} prompts → {args.output}")


if __name__ == "__main__":
    main()
