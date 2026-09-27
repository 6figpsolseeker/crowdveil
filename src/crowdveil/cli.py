import argparse
import logging
import os
import sys
from pathlib import Path

import cv2

from crowdveil.faces import FaceFilter, crew_images
from crowdveil.live import serve
from crowdveil.video import redact_file


def main(argv: list[str] | None = None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--crew", type=Path, help="directory of photos of people to keep visible")
    common.add_argument(
        "--radius", type=int, default=5, help="frames either side a detection is held for"
    )
    common.add_argument(
        "--min-score", type=float, default=0.6, help="face detector confidence threshold"
    )

    parser = argparse.ArgumentParser(
        prog="crowdveil", description="Blur bystanders' faces in video."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    redact = commands.add_parser("redact", parents=[common], help="redact a video file")
    redact.add_argument("input", type=Path)
    redact.add_argument("output", type=Path)
    redact.add_argument(
        "--detect-width", type=int, help="downscale frames to this width for detection"
    )

    relay = commands.add_parser("serve", parents=[common], help="relay a live stream")
    relay.add_argument("input", help="stream URL to read, e.g. rtsp://localhost:8554/live")
    relay.add_argument(
        "output",
        nargs="?",
        default=os.environ.get("CROWDVEIL_OUTPUT"),
        help="stream URL to publish to (default: $CROWDVEIL_OUTPUT)",
    )
    relay.add_argument(
        "--detect-width",
        type=int,
        default=640,
        help="downscale frames to this width for detection (default: 640)",
    )
    relay.add_argument(
        "--detect-every",
        type=int,
        default=2,
        help="run detection on every Nth frame; --radius covers the rest (default: 2)",
    )
    relay.add_argument("--bitrate", type=int, default=6000, help="video bitrate in kbit/s")

    args = parser.parse_args(argv)
    if args.radius < 0:
        parser.error("--radius must be non-negative")
    if not 0 < args.min_score <= 1:
        parser.error("--min-score must be in (0, 1]")
    if args.detect_width is not None and args.detect_width < 32:
        parser.error("--detect-width must be at least 32")
    if args.crew and not args.crew.is_dir():
        parser.error(f"crew directory not found: {args.crew}")
    if args.command == "redact" and not args.input.is_file():
        parser.error(f"input not found: {args.input}")
    if args.command == "serve":
        if not args.output:
            parser.error("output is required (or set CROWDVEIL_OUTPUT)")
        if args.bitrate <= 0:
            parser.error("--bitrate must be positive")
        if not 1 <= args.detect_every <= args.radius + 1:
            parser.error("--detect-every must be between 1 and --radius + 1")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

    try:
        faces = FaceFilter(min_score=args.min_score, max_width=args.detect_width)
        if args.crew:
            faces.enroll(crew_images(args.crew))
        if args.command == "serve":
            serve(
                args.input,
                args.output,
                faces,
                radius=args.radius,
                bitrate=args.bitrate,
                detect_every=args.detect_every,
            )
        else:
            frames = redact_file(args.input, args.output, faces, radius=args.radius)
            print(f"wrote {frames} frames to {args.output}")
    except (OSError, ValueError) as e:
        sys.exit(f"crowdveil: {e}")
    except KeyboardInterrupt:
        sys.exit(130)
