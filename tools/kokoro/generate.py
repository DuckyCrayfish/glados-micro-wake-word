"""Generate wake word / phrase clips with Kokoro-82M, as 16 kHz 16-bit mono wavs.

Run from the project root with the Kokoro environment:

    uv run --project tools/kokoro python tools/kokoro/generate.py PHRASES \
        --count 40000 --output-dir data/generated_samples

PHRASES is a text file with one phrase per line (clip i says phrase i % n), or a
single phrase. Each clip uses a random English voice, sometimes a blend of two, and
a random speed, so the clips cover many different speakers.

Pronunciation: Kokoro's frontend has no dictionary entry for invented words like
"GLaDOS" and would guess. Spell out the pronunciation with misaki's inline syntax,
`[GLaDOS](/ɡlˈædˌOs/)`. Kokoro's phoneme set writes diphthongs as one letter
(A = eɪ, O = oʊ, I = aɪ, W = aʊ, Y = ɔɪ).

Resumable: clips that already exist are skipped, and each clip is seeded by its
index, so a rerun with the same arguments produces the same clips. Raising --count
only generates the missing ones.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch
from kokoro import KPipeline
from scipy.signal import resample_poly

KOKORO_RATE = 24_000
OUTPUT_RATE = 16_000
REPO_ID = "hexgrad/Kokoro-82M"

# American (a) and British (b) English, female (f) and male (m).
VOICES = [
    *(f"af_{n}" for n in "heart alloy aoede bella jessica kore nicole nova river sarah sky".split()),
    *(f"am_{n}" for n in "adam echo eric fenrir liam michael onyx puck santa".split()),
    *(f"bf_{n}" for n in "alice emma isabella lily".split()),
    *(f"bm_{n}" for n in "daniel fable george lewis".split()),
]


def read_phrases(source: str) -> list[str]:
    path = Path(source)
    if path.is_file():
        lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
        phrases = [line for line in lines if line and not line.startswith("#")]
    else:
        phrases = [source]
    if not phrases:
        sys.exit(f"No phrases in {source}")
    return phrases


def pick_voice(pipeline: KPipeline, rng: random.Random, blend_chance: float) -> torch.Tensor:
    """One voice, or with blend_chance a random mix of two."""
    first = pipeline.load_single_voice(rng.choice(VOICES))
    if rng.random() >= blend_chance:
        return first
    second = pipeline.load_single_voice(rng.choice(VOICES))
    return torch.lerp(first, second, rng.uniform(0.2, 0.8))


def trim_silence(audio: np.ndarray, margin_s: float, threshold: float = 0.02) -> np.ndarray:
    """Cut the silence Kokoro adds around the speech, keeping margin_s on each side.

    Piper's clips start and end almost immediately. Kokoro's have 0.2 to 0.6 s of padding,
    which would let the model tell the engines apart by where the word sits in the clip."""
    loud = np.flatnonzero(np.abs(audio) > threshold * np.abs(audio).max())
    margin = int(margin_s * OUTPUT_RATE)
    return audio[max(loud[0] - margin, 0) : loud[-1] + margin + 1]


def synthesize(pipeline: KPipeline, text: str, voice: torch.Tensor, speed: float) -> np.ndarray:
    chunks = [
        result.audio.detach().cpu().numpy()
        for result in pipeline(text, voice=voice, speed=speed, split_pattern=None)
        if result.audio is not None
    ]
    if not chunks:
        raise RuntimeError(f"Kokoro produced no audio for {text!r}")
    return np.concatenate(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("phrases", help="text file with one phrase per line, or a single phrase")
    parser.add_argument("--count", type=int, required=True, help="number of clips to generate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="kokoro_", help="filename prefix; Piper's clips are numbers only")
    # Piper runs at length scales 0.7 to 0.9 (about 1.1x to 1.4x speed), so speak at a similar pace.
    parser.add_argument("--speed-min", type=float, default=1.1)
    parser.add_argument("--speed-max", type=float, default=1.5)
    parser.add_argument("--blend-chance", type=float, default=0.5)
    parser.add_argument("--trim-margin", type=float, default=0.05, help="seconds of silence kept around the speech; negative keeps Kokoro's own padding")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    phrases = read_phrases(args.phrases)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(args.count) if not (out_dir / f"{args.prefix}{i:06d}.wav").exists()]
    print(f"{len(phrases)} phrases, {args.count} clips, {args.count - len(todo)} already exist")

    if not todo:
        return
    pipeline = KPipeline(lang_code="a", repo_id=REPO_ID)
    for done, i in enumerate(todo, 1):
        rng = random.Random(f"{args.seed}-{i}")
        voice = pick_voice(pipeline, rng, args.blend_chance)
        speed = rng.uniform(args.speed_min, args.speed_max)
        audio = synthesize(pipeline, phrases[i % len(phrases)], voice, speed)
        audio = resample_poly(audio, OUTPUT_RATE, KOKORO_RATE)
        if args.trim_margin >= 0:
            audio = trim_silence(audio, args.trim_margin)
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        # Write to a temporary name so an interrupted run never leaves a partial clip.
        path = out_dir / f"{args.prefix}{i:06d}.wav"
        tmp = path.with_suffix(".wav.part")
        with open(tmp, "wb") as f:
            scipy.io.wavfile.write(f, OUTPUT_RATE, pcm)
        os.replace(tmp, path)
        if done % 500 == 0 or done == len(todo):
            print(f"{done}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
