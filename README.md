# crowdveil

[![CI](https://github.com/6figpsolseeker/crowdveil/actions/workflows/ci.yml/badge.svg)](https://github.com/6figpsolseeker/crowdveil/actions/workflows/ci.yml)

Blur the faces of people who didn't sign up to be on camera, while keeping the creator and
their crew visible.

## Install

Requires Python 3.10+.

```sh
pip install git+https://github.com/6figpsolseeker/crowdveil
```

Model weights (~40 MB) are downloaded and checksum-verified on first run, then cached in
`~/.cache/crowdveil`.

## Usage

```sh
crowdveil redact input.mp4 output.mp4 --crew crew/
```

`--crew` is a directory of photos, one person per photo. Faces that match anyone in it are
left visible; every other face is blurred. Without `--crew`, every face is blurred.

| Option | Default | Description |
| --- | --- | --- |
| `--crew DIR` | none | Photos of people to keep visible |
| `--radius N` | `5` | Frames either side a detection is held for, covering missed detections |
| `--min-score S` | `0.6` | Face detector confidence threshold |

Audio is copied through unchanged.

## Development

```sh
uv sync
uv run pytest
uv run ruff check && uv run ruff format --check
```

## License

[MIT](LICENSE)
