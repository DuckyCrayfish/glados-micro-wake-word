"""Count how often wake word models fire on sound-alike phrases.

Generates TTS clips of the phrases in near_miss_holdout.txt with the same Piper
voice and settings the notebook uses for the wake word, runs each streaming
TFLite model over them, and reports the fire rate at several cutoffs. A sample
of the real wake word clips is run as a control: if the control doesn't fire,
the numbers for the sound-alikes mean nothing.

The held-out phrases are never trained on (the notebook trains on
near_miss_phrases.txt), so the numbers say how a model does on sound-alikes it
has not seen. The script refuses to run if the two lists share a phrase.

Run from the project root with the main environment:

    uv run python tools/near_miss_check.py

Firing follows ESPHome: the probabilities of the last N inferences are averaged
(sliding window), and the model fires when that mean exceeds the cutoff.
"""

import argparse
import json
import math
import random
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from microwakeword.inference import Model

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16_000
CUTOFFS = [0.17, 0.5, 0.8, 0.9, 0.95]

# Same settings as the notebook's wake word generation (cell 3).
PIPER_MODEL = ROOT / "piper-models/en_US-libritts_r-medium.pt"

TRAIN_PHRASES = ROOT / "near_miss_phrases.txt"  # trained on, by the notebook
TEST_PHRASES = ROOT / "near_miss_holdout.txt"  # never trained on
LENGTH_SCALES = "0.8"
NOISE_SCALE_WS = "0.2"


GROUP_HEADER = re.compile(r"^#\s*---\s*(.+?)\s*---\s*$")


def read_phrases(path: Path) -> list[tuple[str, str]]:
    """(group, phrase) pairs. A "# --- Name ---" comment starts a new group."""
    group, pairs = "ungrouped", []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if header := GROUP_HEADER.match(line):
            group = header.group(1)
        elif line and not line.startswith("#"):
            pairs.append((group, line))
    return pairs


def generate_clips(phrases: list[str], per_phrase: int, work_dir: Path) -> list[Path]:
    """Clip i says phrases[i % len(phrases)]: Piper cycles through the text file."""
    clips_dir = work_dir / "clips"
    clean_file = work_dir / "phrases.txt"
    total = len(phrases) * per_phrase
    clean_text = "\n".join(phrases) + "\n"

    up_to_date = (
        clean_file.exists()
        and clean_file.read_text(encoding="utf-8") == clean_text
        and len(list(clips_dir.glob("*.wav"))) == total
    )
    if not up_to_date:
        work_dir.mkdir(parents=True, exist_ok=True)
        clips_dir.mkdir(exist_ok=True)
        for old in clips_dir.glob("*.wav"):
            old.unlink()
        clean_file.write_text(clean_text, encoding="utf-8")
        # piper-sample-generator runs in its own environment (tools/piper).
        subprocess.run(
            [
                "uv", "run", "--project", str(ROOT / "tools/piper"),
                "python", "-m", "piper_sample_generator", str(clean_file),
                "--length-scales", LENGTH_SCALES,
                "--noise-scale-ws", NOISE_SCALE_WS,
                "--max-samples", str(total),
                "--batch-size", "100",
                "--output-dir", str(clips_dir),
                "--model", str(PIPER_MODEL),
            ],
            check=True,
            cwd=ROOT,
        )
    return [clips_dir / f"{i}.wav" for i in range(total)]


def load_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if rate != SAMPLE_RATE:
        g = math.gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio.astype(np.float32), SAMPLE_RATE // g, rate // g)
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
    return audio


def clip_score(model_path: str, audio: np.ndarray, window: int) -> float:
    """Highest sliding-window mean probability over the clip."""
    pad = np.zeros(SAMPLE_RATE, dtype=np.int16)  # 1 s of silence either side
    # A fresh interpreter per clip: streaming models carry state between calls.
    probs = np.asarray(
        Model(model_path).predict_clip(np.concatenate([pad, audio, pad]), step_ms=10)
    )
    if len(probs) < window:
        return float(probs.mean())
    return float(np.convolve(probs, np.ones(window) / window, mode="valid").max())


def parse_model(spec: str) -> tuple[str, str, float]:
    """NAME=PATH@CUTOFF -> (name, path, reference cutoff)."""
    name, rest = spec.split("=", 1)
    path, cutoff = rest.rsplit("@", 1)
    return name, path, float(cutoff)


def main() -> None:
    deployed = json.loads((ROOT / "models/glados.json").read_text())["micro"]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phrases", type=Path, default=TEST_PHRASES)
    parser.add_argument("--per-phrase", type=int, default=10)
    parser.add_argument("--controls", type=int, default=200)
    parser.add_argument("--window", type=int, default=deployed["sliding_window_size"])
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/near_miss_check")
    parser.add_argument(
        "--model", action="append", metavar="NAME=PATH@CUTOFF",
        help="Model to test and the cutoff used for the per-phrase table. "
        "Defaults to the deployed model and the latest run.",
    )
    args = parser.parse_args()

    specs = args.model or [
        f"deployed={ROOT / 'models/glados.tflite'}@{deployed['probability_cutoff']}",
        f"latest run={ROOT / 'output/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite'}@0.9",
    ]
    models = [parse_model(s) for s in specs]

    grouped = read_phrases(args.phrases)
    phrases = [phrase for _, phrase in grouped]
    if args.phrases.resolve() != TRAIN_PHRASES.resolve() and TRAIN_PHRASES.exists():
        trained_on = {p.lower() for _, p in read_phrases(TRAIN_PHRASES)}
        shared = sorted(p for p in phrases if p.lower() in trained_on)
        if shared:
            raise SystemExit(
                f"These phrases are in {TRAIN_PHRASES.name} too, so they are not a fair test: {shared}"
            )
    print(f"{len(phrases)} phrases x {args.per_phrase} clips, window {args.window}")
    near_miss = generate_clips(phrases, args.per_phrase, args.work_dir)
    near_miss_group = [grouped[i % len(grouped)][0] for i in range(len(near_miss))]
    near_miss_phrase = [phrases[i % len(phrases)] for i in range(len(near_miss))]
    group_names = list(dict.fromkeys(near_miss_group))

    positives = sorted((ROOT / "data/generated_samples").glob("*.wav"))
    controls = random.Random(0).sample(positives, min(args.controls, len(positives)))

    near_audio = [load_16k(p) for p in near_miss]
    ctrl_audio = [load_16k(p) for p in controls]

    for name, path, ref_cutoff in models:
        near = np.array([clip_score(path, a, args.window) for a in near_audio])
        ctrl = np.array([clip_score(path, a, args.window) for a in ctrl_audio])

        print(f"\n=== {name} ({Path(path).name}) ===")
        print("share of clips that fire, by cutoff (window mean > cutoff)")
        print(f"{'':<42}" + "".join(f"{c:>7}" for c in CUTOFFS))
        print(f"{'wake word clips (should fire)':<42}" + "".join(f"{(ctrl > c).mean():>7.0%}" for c in CUTOFFS))
        for group in group_names:
            sel = near[[g == group for g in near_miss_group]]
            print(f"{group[:41]:<42}" + "".join(f"{(sel > c).mean():>7.0%}" for c in CUTOFFS))

        by_phrase = {}
        for phrase, s in zip(near_miss_phrase, near):
            by_phrase.setdefault(phrase, []).append(s > ref_cutoff)
        rows = sorted(((np.mean(v), p) for p, v in by_phrase.items()), reverse=True)
        print(f"\nper phrase at the reference cutoff {ref_cutoff} (share of {args.per_phrase} clips that fire):")
        for rate, phrase in rows:
            print(f"  {rate:>5.0%}  {phrase}")


if __name__ == "__main__":
    main()
