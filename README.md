# crowdveil

[![CI](https://github.com/6figpsolseeker/crowdveil/actions/workflows/ci.yml/badge.svg)](https://github.com/6figpsolseeker/crowdveil/actions/workflows/ci.yml)

Blur the faces of people who didn't sign up to be on camera, while keeping the creator and
their crew visible. Works on recorded video and as a relay for live streams.

## Install

Requires Python 3.10+.

```sh
pip install git+https://github.com/6figpsolseeker/crowdveil
```

Model weights (~40 MB) are downloaded and checksum-verified on first run, then cached in
`~/.cache/crowdveil`.

## Recorded video

```sh
crowdveil redact input.mp4 output.mp4 --crew crew/
```

`--crew` is a directory of photos, one person per photo. Faces that match anyone in it are
left visible; every other face is blurred. Without `--crew`, every face is blurred. Audio is
copied through unchanged.

## Live streams

crowdveil sits between your streaming app and the platform. A [MediaMTX](https://mediamtx.org)
server receives the stream from your phone; crowdveil reads it, blurs it, and publishes it on.

```
streaming app ──RTMP/SRT──▶ MediaMTX ──RTSP──▶ crowdveil ──RTMP──▶ Twitch / YouTube / Kick
```

1. Start MediaMTX ([install options](https://mediamtx.org/docs/kickoff/install)):

   ```sh
   docker run --rm -it -e MTX_RTSPTRANSPORTS=tcp \
     -p 8554:8554 -p 1935:1935 -p 8890:8890/udp bluenviron/mediamtx:1
   ```

   Before exposing it to the internet, require a password to publish
   ([authentication](https://mediamtx.org/docs/features/authentication)).

2. Point your streaming app at `rtmp://<server>/live`, or
   `srt://<server>:8890?streamid=publish:live` for SRT, which copes better with mobile networks.

3. Run crowdveil on the same machine:

   ```sh
   export CROWDVEIL_OUTPUT="rtmp://live.twitch.tv/app/<stream key>"
   crowdveil serve rtsp://localhost:8554/live --crew crew/
   ```

crowdveil waits for the stream, relays it until it ends, and waits for the next one. If face
detection fails or can't keep up, whole frames are blurred until it recovers; frames are never
passed through unblurred. The relay adds roughly 0.3 s of latency: faces are tracked a few
frames ahead so they are covered from the first frame they appear in.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--crew DIR` | none | Photos of people to keep visible |
| `--radius N` | `5` | Frames either side a detection is held for |
| `--min-score S` | `0.6` | Face detector confidence threshold |
| `--detect-width W` | full size; `640` for `serve` | Downscale frames to this width for detection |
| `--detect-every N` | `2` | `serve` only: run detection on every Nth frame |
| `--bitrate KBPS` | `6000` | `serve` only: output video bitrate |

A larger `--detect-width` finds smaller, more distant faces at a higher CPU cost.

## Development

```sh
uv sync
uv run pytest
uv run ruff check && uv run ruff format --check
```

## License

[MIT](LICENSE)
