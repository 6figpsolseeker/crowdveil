import argparse
import sys
from pathlib import Path

import cv2

from crowdveil.faces import FaceFilter, crew_images
from crowdveil.video import redact_file


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="crowdveil", description="Blur bystanders' faces in video."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    redact = commands.add_parser("redact", help="redact a video file")
    redact.add_argument("input", type=Path)
    redact.add_argument("output", type=Path)
    redact.add_argument("--crew", type=Path, help="directory of photos of people to keep visible")
    redact.add_argument(
        "--radius", type=int, default=5, help="frames either side a detection is held for"
    )
    redact.add_argument(
        "--min-score", type=float, default=0.6, help="face detector confidence threshold"
    )

    args = parser.parse_args(argv)
    if args.radius < 0:
        parser.error("--radius must be non-negative")
    if not 0 < args.min_score <= 1:
        parser.error("--min-score must be in (0, 1]")
    if not args.input.is_file():
        parser.error(f"input not found: {args.input}")
    if args.crew and not args.crew.is_dir():
        parser.error(f"crew directory not found: {args.crew}")

    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

    try:
        faces = FaceFilter(min_score=args.min_score)
        if args.crew:
            faces.enroll(crew_images(args.crew))
        frames = redact_file(args.input, args.output, faces, radius=args.radius)
    except (OSError, ValueError) as e:
        sys.exit(f"crowdveil: {e}")
    print(f"wrote {frames} frames to {args.output}")
