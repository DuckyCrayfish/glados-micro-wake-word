# /// script
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import os
    import re
    import shutil
    import signal
    import subprocess
    import time
    from pathlib import Path
    from urllib.parse import urlparse

    import marimo as mo
    import numpy as np
    import yaml


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Training a microWakeWord Model

    **The model generated will most likely not be usable for everyday use; it may be difficult to trigger or falsely activates too frequently. You will most likely have to experiment with many different settings to obtain a decent model!**

    At the end of this notebook, you will be able to download a tflite file. To use this in ESPHome, you need to write a model manifest JSON file. See the [ESPHome documentation](https://esphome.io/components/micro_wake_word) for the details and the [model repo](https://github.com/esphome/micro-wake-word-models/tree/main/models/v2) for examples.

    This notebook is set to `on_cell_change = "lazy"` (see the header of this file): changing a cell marks the cells after it as stale instead of re-running them, so editing a setting can't start a long download, generation or training run by itself. Run the cells you want, top to bottom.
    """)
    return


@app.cell
def _():
    # How the wake word is written for each voice model type. Each type has several variants
    # with different intonation, and the clips cycle through them, so the model doesn't only
    # hear one pitch contour. Piper takes espeak phonemes, where ↑ is a rising ending and
    # ↓ a falling one. Kokoro takes text, and its inflection comes from the punctuation.
    # Kokoro's frontend would guess "GLaDOS", so its pronunciation is spelled out in Kokoro's
    # own phoneme set (A = eɪ, O = oʊ).
    # target_word = 'okay_GLAD_ohss'
    target_words = {
        "piper": [
            "oʊˈkeɪ ɡlˈæ dˈoʊs↑",
            "oʊˈkeɪ ɡlˈæ dˈoʊs",
            "oʊˈkeɪ ɡlˈæ dˈoʊs↓",
            "oʊˈkeɪ, ɡlˈæ dˈoʊs.",
            "oʊˈkeɪ ɡlˈæ dˈoʊs?",
            "oʊˈkeɪ ɡlˈæ dˈoʊs!",
        ],
        "kokoro": [
            "okay [GLaDOS](/ɡlˈædˌOs/)",
            "okay [GLaDOS](/ɡlˈædˌOs/)?",
            "okay, [GLaDOS](/ɡlˈædˌOs/)!",
            "Okay [GLaDOS](/ɡlˈædˌOs/).",
            "okay, [GLaDOS](/ɡlˈædˌOs/)",
            "okay [GLaDOS](/ɡlˈædˌOs/)!",
        ],
    }
    piper_model_dir = "piper-models/"

    # The voice models that generate the wake word (and near-miss) clips, and how many wake
    # word clips each one makes. The augmented features take about 0.8 MB per clip, so the
    # sum of "samples" sets the disk space: to add a voice model without using more space,
    # take its clips from another model's "samples".
    # Different TTS engines keep the model from learning one engine's voice artifacts.
    # "name" is the model's folder under the samples folders (SAMPLES_DIR/<name>/), so it
    # must be unique. The same "type" can be used more than once, for example two Piper voices.
    voice_models = [
        {
            "name": "piper_libritts",
            "type": "piper",
            # PyTorch generator model (.pt). Unlike the .onnx voices, it generates in batches
            # on the GPU and blends speakers together for more voice variety.
            "model_url": "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt",
            "metadata_url": "https://raw.githubusercontent.com/rhasspy/piper-sample-generator/v3.2.0/models/en_US-libritts_r-medium.pt.json",
            "samples": 80_000,
        },
        {
            # Kokoro-82M (StyleTTS2, a different architecture from Piper's VITS). Its model
            # and voices download from Hugging Face the first time it runs.
            "name": "kokoro",
            "type": "kokoro",
            "samples": 20_000,
        },
    ]
    SAMPLES_DIR = "data/generated_samples"
    # Piper settings. Each is a list and every batch of clips uses one combination of them,
    # cycling through all combinations. Narrow lists make every clip sound alike.
    #   LENGTH_SCALES: speed (lower is faster)
    #   NOISE_SCALES: overall variability, including pitch contour
    #   NOISE_SCALE_WS: variability in how long each phoneme lasts
    LENGTH_SCALES = [0.7, 0.8, 0.9]
    NOISE_SCALES = [0.5, 0.667, 0.9]
    NOISE_SCALE_WS = [0.2, 0.5, 0.8]

    assert len({m["name"] for m in voice_models}) == len(voice_models), (
        "voice model names must be unique"
    )

    # Inside a voice model's folder, Piper clips are named <n>.wav and Kokoro clips kokoro_<n>.wav.
    CLIP_PREFIXES = {"piper": "", "kokoro": "kokoro_"}

    # Where each Piper voice model and its metadata are saved.
    for _voice_model in voice_models:
        if _voice_model["type"] != "piper":
            continue
        _model_path = os.path.join(
            piper_model_dir, os.path.basename(urlparse(_voice_model["model_url"]).path)
        )
        _voice_model["model_path"] = _model_path
        _voice_model["metadata_path"] = _model_path + ".json"
    return (
        CLIP_PREFIXES,
        LENGTH_SCALES,
        NOISE_SCALES,
        NOISE_SCALE_WS,
        SAMPLES_DIR,
        piper_model_dir,
        target_words,
        voice_models,
    )


@app.cell
def _(
    CLIP_PREFIXES,
    LENGTH_SCALES,
    NOISE_SCALES,
    NOISE_SCALE_WS,
    voice_models,
):
    def run(command):
        """Run a command, showing its output in the cell as it comes. Raises if it fails.
        Interrupting the cell stops the command too."""
        # In its own process group, so an interrupt reaches the Python process behind `uv run`.
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            while chunk := process.stdout.read1(4096):
                print(chunk.decode(errors="replace"), end="", flush=True)
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGINT)
            process.wait()
            raise
        if process.wait() != 0:
            raise RuntimeError(
                f"{command[0]} failed with exit code {process.returncode}"
            )

    def download(url, path):
        """Download `url` to `path`. It is saved as <path>.part until the download finishes,
        so an interrupted download is never mistaken for a complete file."""
        run(["wget", "-O", f"{path}.part", url])
        os.replace(f"{path}.part", path)

    def generate_clips(voice_model, text, count, output_dir, phonemes=False):
        """Generate `count` clips with one voice model. `text` is a phrase, a list of phrases
        or a file with one phrase per line. The clips cycle through the phrases. `phonemes`
        tells Piper that `text` is phonemes. Each generator runs in its own uv environment
        because their dependencies conflict."""
        if count <= 0:
            return
        if isinstance(text, list):
            phrases_path = Path("data/phrases") / f"{voice_model['name']}.txt"
            phrases_path.parent.mkdir(parents=True, exist_ok=True)
            phrases_path.write_text("\n".join(text) + "\n", encoding="utf-8")
            text = str(phrases_path)
        if voice_model["type"] == "piper":
            # piper-sample-generator pins audiomentations==0.33.0, which conflicts with
            # microWakeWord's newer version.
            run(
                [
                    "uv",
                    "run",
                    "--project",
                    "tools/piper",
                    "python",
                    "-m",
                    "piper_sample_generator",
                    text,
                ]
                + (["--phoneme-input"] if phonemes else [])
                + [
                    "--length-scales",
                    *map(str, LENGTH_SCALES),
                    "--noise-scales",
                    *map(str, NOISE_SCALES),
                    "--noise-scale-ws",
                    *map(str, NOISE_SCALE_WS),
                    "--max-samples",
                    str(count),
                    "--batch-size",
                    "100",
                    "--output-dir",
                    output_dir,
                    "--model",
                    voice_model["model_path"],
                ]
            )
        elif voice_model["type"] == "kokoro":
            # Clips that already exist are skipped.
            run(
                [
                    "uv",
                    "run",
                    "--project",
                    "tools/kokoro",
                    "python",
                    "tools/kokoro/generate.py",
                    text,
                    "--count",
                    str(count),
                    "--output-dir",
                    output_dir,
                ]
            )
        else:
            raise ValueError(f"unknown voice model type: {voice_model['type']}")

    def trim_clips(samples_dir, counts):
        """Delete clips past each voice model's count, left over from an earlier run with a
        bigger one. The generators number their clips from 0, so those extra clips would
        still be in the folder and trained on. `counts` maps a voice model name to its count."""
        removed = 0
        for voice_model in voice_models:
            pattern = re.compile(
                re.escape(CLIP_PREFIXES[voice_model["type"]]) + r"(\d+)"
            )
            for path in (Path(samples_dir) / voice_model["name"]).glob("*.wav"):
                match = pattern.fullmatch(path.stem)
                if match and int(match.group(1)) >= counts[voice_model["name"]]:
                    path.unlink()
                    removed += 1
        print(f"{samples_dir}: removed {removed} clips past the budget")

    def clips_dir(samples_dir, voice_model):
        return os.path.join(samples_dir, voice_model["name"])

    return clips_dir, download, generate_clips, run, trim_clips


@app.cell
def _(download, piper_model_dir, voice_models):
    # Downloads the Piper voice models that aren't in piper_model_dir yet.
    os.makedirs(piper_model_dir, exist_ok=True)
    for _voice_model in voice_models:
        if _voice_model["type"] != "piper":
            continue
        if not os.path.exists(_voice_model["model_path"]):
            download(_voice_model["model_url"], _voice_model["model_path"])
            download(_voice_model["metadata_url"], _voice_model["metadata_path"])
    return


@app.cell
def _(generate_clips, target_words, voice_models):
    # Generates 1 sample of the target word with each voice model for manual verification.
    # Each model's first wake word variant is used.
    # Kokoro's pronunciation is spelled out by hand, so listen to it before generating more.

    _players = []
    for _voice_model in voice_models:
        _check_dir = f"data/voice_check/{_voice_model['name']}"
        generate_clips(
            _voice_model,
            target_words[_voice_model["type"]],
            1,
            _check_dir,
            phonemes=True,
        )
        _players.append(
            mo.vstack(
                [
                    mo.md(f"**{_voice_model['name']}**"),
                    mo.audio(next(Path(_check_dir).glob("*.wav"))),
                ]
            )
        )
    mo.vstack(_players)
    return


@app.cell
def _(
    SAMPLES_DIR,
    clips_dir,
    generate_clips,
    target_words,
    trim_clips,
    voice_models,
):
    # Generates a larger amount of wake word samples.
    # Start here when trying to improve your model.
    # See https://github.com/rhasspy/piper-sample-generator for the full set of
    # parameters. In particular, experiment with noise-scales and noise-scale-ws,
    # generating negative samples similar to the wake word, and generating many more
    # wake word samples, possibly with different phonetic pronunciations.
    #
    # Each voice model makes the number of clips in its "samples" (see voice_models above),
    # in its own folder under SAMPLES_DIR.

    for _voice_model in voice_models:
        generate_clips(
            _voice_model,
            target_words[_voice_model["type"]],
            _voice_model["samples"],
            clips_dir(SAMPLES_DIR, _voice_model),
            phonemes=True,
        )

    trim_clips(SAMPLES_DIR, {m["name"]: m["samples"] for m in voice_models})
    return


@app.cell
def _():
    # The imports, and the sample rate, for the three augmentation downloads below.
    import datasets
    from scipy.io import wavfile
    from tqdm.notebook import tqdm

    SAMPLE_RATE = 16_000
    return SAMPLE_RATE, datasets, tqdm, wavfile


@app.cell
def _(SAMPLE_RATE, datasets, tqdm, wavfile):
    # Downloads audio data for augmentation. This can be slow!
    # Borrowed from openWakeWord's automatic_model_training.ipynb, accessed March 4, 2024

    # **Important note!** The data downloaded here has a mixture of difference
    # licenses and usage restrictions. As such, any custom models trained with this
    # data should be considered as appropriate for **non-commercial** personal use only.

    SOURCE_URL = "davidscripka/MIT_environmental_impulse_responses"
    MIT_RIRS_DIR = "data/augmentations/mit_rirs"
    SPLIT = "train"

    # Skipped if MIT_RIRS_DIR already has files. To force a fresh download, delete the folder.
    _staging_dir = MIT_RIRS_DIR + ".part"

    if os.path.isdir(MIT_RIRS_DIR) and os.listdir(MIT_RIRS_DIR):
        print(f"{MIT_RIRS_DIR}: already downloaded, skipping")
    else:
        shutil.rmtree(_staging_dir, ignore_errors=True)
        os.makedirs(_staging_dir)
        _builder = datasets.load_dataset_builder(SOURCE_URL)
        assert _builder.info.splits is not None
        _count = _builder.info.splits[SPLIT].num_examples
        _dataset = _builder.as_streaming_dataset(split=SPLIT)
        assert isinstance(_dataset, datasets.IterableDataset)
        _dataset = _dataset.decode(False)
        _audio_feature = datasets.Audio(sampling_rate=SAMPLE_RATE)

        # Save clips to 16-bit PCM wav files
        for _row in tqdm(_dataset, total=_count):
            _decoder = _audio_feature.decode_example(_row["audio"])
            _samples = _decoder.get_all_samples()
            _data = _samples.data.detach().cpu().numpy()
            _data = (
                np.mean(_data, axis=tuple(range(_data.ndim - 1)))
                if _data.ndim > 1
                else _data
            )
            _data = (_data * 32767).clip(-32768, 32767).astype(np.int16)
            _name = Path(_row["audio"]["path"]).with_suffix(".wav").name
            _path = os.path.join(_staging_dir, _name)
            wavfile.write(_path, _samples.sample_rate, _data)

        os.rename(_staging_dir, MIT_RIRS_DIR)
    return


@app.cell
def _(SAMPLE_RATE, datasets, tqdm, wavfile):
    # Download noise and background audio

    # Audioset Dataset (https://research.google.com/audioset/dataset/index.html)
    # Download shards of the balanced training split and convert to 16khz.
    # Each shard is a ~700 MB Parquet file of 10 s clips. Add more shard names (00 to
    # 37) to AUDIOSET_SHARDS for more background variety. For full-scale training,
    # it's recommended to use the entire dataset from
    # https://huggingface.co/datasets/agkphysics/AudioSet, and
    # even potentially combine it with other background noise datasets (e.g., FSD50k, Freesound, etc.)

    AUDIOSET_REPO = "agkphysics/AudioSet"
    AUDIOSET_REVISION = "0c609e8302cf139307f639c57652032af0a88041"
    AUDIOSET_SHARDS = [f"{i:02d}" for i in range(1, 10)]  # shards 01 to 09
    AUDIOSET_DIR = "data/augmentations/audioset_16k"

    # Each shard is skipped if its marker file (.done-bal_train-<shard>) is in AUDIOSET_DIR.
    # The marker is written after the shard's clips are saved, so an interrupted shard is
    # downloaded again and its clips are overwritten. Adding shard names to AUDIOSET_SHARDS
    # downloads only the new ones. To force a fresh download of a shard (for example after
    # changing AUDIOSET_REVISION), delete its marker file.
    #
    # A folder filled before this check existed has no markers, so every shard would be
    # downloaded again. If it is complete, mark its shards as done once with:
    #   touch data/augmentations/audioset_16k/.done-bal_train-{01..09}
    os.makedirs(AUDIOSET_DIR, exist_ok=True)
    _audio_feature = datasets.Audio(sampling_rate=SAMPLE_RATE)

    # Save clips to 16-bit PCM wav files
    for _shard in AUDIOSET_SHARDS:
        _marker = os.path.join(AUDIOSET_DIR, f".done-bal_train-{_shard}")
        if os.path.exists(_marker):
            print(f"bal_train/{_shard}: already downloaded, skipping")
            continue
        _dataset = datasets.load_dataset(
            AUDIOSET_REPO,
            data_files={"train": f"data/bal_train/{_shard}.parquet"},
            split="train",
            revision=AUDIOSET_REVISION,
            streaming=True,
        )
        assert isinstance(_dataset, datasets.IterableDataset)
        _dataset = _dataset.decode(False)
        for _row in tqdm(_dataset, desc=f"bal_train/{_shard}"):
            _samples = _audio_feature.decode_example(_row["audio"]).get_all_samples()
            _data = _samples.data.detach().cpu().numpy()
            _data = (
                np.mean(_data, axis=tuple(range(_data.ndim - 1)))
                if _data.ndim > 1
                else _data
            )
            _data = (_data * 32767).clip(-32768, 32767).astype(np.int16)
            _name = Path(_row["audio"]["path"]).with_suffix(".wav").name
            _path = os.path.join(AUDIOSET_DIR, _name)
            wavfile.write(_path, _samples.sample_rate, _data)
        Path(_marker).touch()
    return


@app.cell
def _(SAMPLE_RATE, datasets, download, run, tqdm, wavfile):
    # Download noise and background audio

    # Free Music Archive dataset
    # https://github.com/mdeff/fma (Third-party mchl914 extra small set)

    FMA_URL = (
        "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip"
    )
    FMA_TEMP_DIR = "data/augmentations/fma"
    FMA_DIR = "data/augmentations/fma_16k"

    _file_name = Path(urlparse(FMA_URL).path).name
    _temp_path = os.path.join(FMA_TEMP_DIR, _file_name)

    # Each step is skipped if its result is already on disk:
    #   - FMA_DIR has files -> skip the download, unzip and conversion
    #   - zip exists        -> skip the download, only unzip and convert
    # To force a fresh download, delete FMA_DIR and the zip in FMA_TEMP_DIR.
    # The zip is saved as .part until the download finishes, and the clips are saved to
    # FMA_DIR.part and the folder is renamed once all of them are saved, so an
    # interrupted run is never mistaken for a complete one.
    _staging_dir = FMA_DIR + ".part"

    if os.path.isdir(FMA_DIR) and os.listdir(FMA_DIR):
        print(f"{FMA_DIR}: already converted, skipping")
    else:
        os.makedirs(FMA_TEMP_DIR, exist_ok=True)
        if os.path.exists(_temp_path):
            print(f"{_file_name}: zip already downloaded, skipping download")
        else:
            download(FMA_URL, _temp_path)
        run(["unzip", "-oq", _temp_path, "-d", FMA_TEMP_DIR])

        # Save clips to 16-bit PCM wav files
        shutil.rmtree(_staging_dir, ignore_errors=True)
        os.makedirs(_staging_dir)
        _paths = [str(i) for i in Path(f"{FMA_TEMP_DIR}/fma_small").glob("**/*.mp3")]
        _dataset = datasets.Dataset.from_dict({"audio": _paths, "path": _paths})
        _dataset = _dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
        for _row in tqdm(_dataset, total=len(_paths)):
            _data = (
                (_row["audio"]["array"] * 32767).clip(-32768, 32767).astype(np.int16)
            )
            _name = Path(_row["path"]).with_suffix(".wav").name
            _path = os.path.join(_staging_dir, _name)
            wavfile.write(_path, SAMPLE_RATE, _data)

        os.rename(_staging_dir, FMA_DIR)
    return


@app.cell
def _(SAMPLES_DIR):
    # Sets up the augmentations.
    # To improve your model, experiment with these settings and use more sources of
    # background clips.

    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    clips = Clips(
        input_directory=SAMPLES_DIR,
        # One folder per voice model (SAMPLES_DIR/<name>/).
        file_pattern="*/*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )
    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.1,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5,
        },
        impulse_paths=["data/augmentations/mit_rirs"],
        background_paths=[
            "data/augmentations/fma_16k",
            "data/augmentations/audioset_16k",
        ],
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )
    return Clips, SpectrogramGeneration, augmenter, clips


@app.cell
def _(augmenter, clips):
    # Augment a random clip and play it back to verify it works well

    from microwakeword.audio.audio_utils import save_clip

    _random_clip = clips.get_random_clip()
    _augmented_clip = augmenter.augment_clip(_random_clip)
    save_clip(_augmented_clip, "data/augmented_clip.wav")

    mo.audio(Path("data/augmented_clip.wav"))
    return


@app.cell
def _(SpectrogramGeneration, augmenter, clips):
    # Augment samples and save the training, validation, and testing sets.
    # Validating and testing samples generated the same way can make the model
    # benchmark better than it performs in real-word use. Use real samples or TTS
    # samples generated with a different TTS engine to potentially get more accurate
    # benchmarks.
    # The wake word clips here are a mix of all the voice models, split at random, so the
    # validation and testing sets contain every engine.

    from typing import TypedDict
    from mmap_ninja.ragged import RaggedMmap

    class Split(TypedDict):
        name: str
        path: str
        repetition: int
        slide_frames: int
        step_ms: int

    FEATURES_DIR = "data/generated_augmented_features"
    splits: list[Split] = [
        {
            "name": "train",
            "path": "training",
            "repetition": 2,
            # Uses the same spectrogram repeatedly, just shifted over by one frame.
            # This simulates the streaming inferences while training/validating in nonstreaming mode.
            "slide_frames": 10,
            "step_ms": 10,
        },
        {
            "name": "validation",
            "path": "validation",
            "repetition": 1,
            "slide_frames": 10,
            "step_ms": 10,
        },
        {
            "name": "test",
            "path": "testing",
            "repetition": 1,
            # The testing set uses the streaming version of the model, so no artificial repetition is necessary
            "slide_frames": 1,
            "step_ms": 10,
        },
    ]

    for _split in splits:
        _output_path = os.path.join(FEATURES_DIR, _split["path"])
        os.makedirs(_output_path, exist_ok=True)

        _spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=_split["slide_frames"],
            step_ms=_split["step_ms"],
        )
        _generator = _spectrograms.spectrogram_generator(
            split=_split["name"], repeat=_split["repetition"]
        )

        RaggedMmap.from_generator(
            out_dir=os.path.join(_output_path, "wakeword_mmap"),
            sample_generator=_generator,
            batch_size=100,
            verbose=True,
        )
    return RaggedMmap, splits


@app.cell
def _(download, run):
    # Downloads pre-generated spectrogram features (made for microWakeWord in
    # particular) for various negative datasets. This can be slow!
    # Each step is skipped if its result is already on disk:
    #   - extracted folder exists -> skip download and unzip
    #   - zip exists              -> skip download, only unzip
    # To force a fresh download, delete the folder and its zip in NEGATIVE_DATASETS_DIR.

    NEGATIVE_DATASETS_DIR = "data/negative_datasets"
    os.makedirs(NEGATIVE_DATASETS_DIR, exist_ok=True)
    ROOT_URL = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
    FILE_NAMES = [
        "dinner_party.zip",
        "dinner_party_eval.zip",
        "no_speech.zip",
        "speech.zip",
    ]
    for _file_name in FILE_NAMES:
        _link = ROOT_URL + _file_name
        _zip_path = f"{NEGATIVE_DATASETS_DIR}/{_file_name}"
        # Each zip contains a single top-level folder named after the zip
        _extracted_dir = os.path.join(
            NEGATIVE_DATASETS_DIR, os.path.splitext(_file_name)[0]
        )

        if os.path.isdir(_extracted_dir) and os.listdir(_extracted_dir):
            print(f"{_file_name}: already extracted to {_extracted_dir}, skipping")
            continue

        if os.path.exists(_zip_path):
            print(f"{_file_name}: zip already downloaded, skipping download")
        else:
            # Download to a temporary name so an interrupted download is never mistaken for a complete zip
            download(_link, _zip_path)

        run(["unzip", "-q", _zip_path, "-d", NEGATIVE_DATASETS_DIR])
    return


@app.cell
def _(clips_dir, generate_clips, trim_clips, voice_models):
    # Generates near-miss negatives: everyday phrases that sound a bit like the wake
    # word (e.g. "okay thanks", "okay class", "glad to hear it"). They are trained on
    # as negatives so the model learns that "okay GLaDOS" is more than "okay ...".
    #
    # The phrases are in near_miss_phrases.txt. Phrases for testing live in
    # near_miss_holdout.txt and are never trained on. After training, check a model
    # against them with `uv run python tools/near_miss_check.py`.

    NEAR_MISS_PHRASES = "near_miss_phrases.txt"
    NEAR_MISS_SAMPLES_DIR = "data/near_miss_samples"
    # The features for these clips take about 8 GB per 10,000 clips (the wake word
    # clips take about 0.8 MB each). More clips add more voices per phrase, not more phrases.
    NEAR_MISS_SAMPLES = 10_000
    # Split between the voice models in the same proportions as their wake word "samples".
    near_miss_counts = {
        m["name"]: round(
            NEAR_MISS_SAMPLES * m["samples"] / sum(v["samples"] for v in voice_models)
        )
        for m in voice_models
    }
    # Rounding can leave the total off by one or two. The last voice model absorbs it.
    near_miss_counts[voice_models[-1]["name"]] += NEAR_MISS_SAMPLES - sum(
        near_miss_counts.values()
    )

    os.makedirs(NEAR_MISS_SAMPLES_DIR, exist_ok=True)
    # The generators read one phrase per line, so write a copy without comments and group headers.
    _phrases = [
        line.strip()
        for line in Path(NEAR_MISS_PHRASES).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    _phrases_file = "data/near_miss_phrases.txt"
    Path(_phrases_file).write_text("\n".join(_phrases) + "\n", encoding="utf-8")
    print(f"{len(_phrases)} phrases, {NEAR_MISS_SAMPLES} clips")

    # Same voices and settings as the wake word samples. Plain text, not phonemes.
    for _voice_model in voice_models:
        generate_clips(
            _voice_model,
            _phrases_file,
            near_miss_counts[_voice_model["name"]],
            clips_dir(NEAR_MISS_SAMPLES_DIR, _voice_model),
        )

    trim_clips(NEAR_MISS_SAMPLES_DIR, near_miss_counts)
    return (NEAR_MISS_SAMPLES_DIR,)


@app.cell
def _(
    Clips,
    NEAR_MISS_SAMPLES_DIR,
    RaggedMmap,
    SpectrogramGeneration,
    augmenter,
    splits: "list[Split]",
):
    # Augment the near-miss clips and save the training, validation and testing sets.
    # Same augmentation, splits and spectrogram settings as the wake word clips (the
    # `augmenter` and `splits` from the cells above), so the model cannot tell the
    # two apart by how they were made.

    NEAR_MISS_FEATURES_DIR = "data/near_miss_features"

    near_miss_clips = Clips(
        input_directory=NEAR_MISS_SAMPLES_DIR,
        file_pattern="*/*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )

    for _split in splits:
        _output_path = os.path.join(NEAR_MISS_FEATURES_DIR, _split["path"])
        os.makedirs(_output_path, exist_ok=True)

        _spectrograms = SpectrogramGeneration(
            clips=near_miss_clips,
            augmenter=augmenter,
            slide_frames=_split["slide_frames"],
            step_ms=_split["step_ms"],
        )
        _generator = _spectrograms.spectrogram_generator(
            split=_split["name"], repeat=_split["repetition"]
        )

        RaggedMmap.from_generator(
            out_dir=os.path.join(_output_path, "near_miss_mmap"),
            sample_generator=_generator,
            batch_size=100,
            verbose=True,
        )
    return


@app.cell
def _():
    # Save a yaml config that controls the training process
    # These hyperparamters can make a huge different in model quality.
    # Experiment with sampling and penalty weights and increasing the number of
    # training steps.

    # Name of this run. Its files go to output/<RUN_NAME>/. Training from scratch deletes
    # that folder if it already exists, so use a new name for every run to keep the old one.
    RUN_NAME = "05-fast-loader"

    config = {
        "window_step_ms": 10,
        "train_dir": f"output/{RUN_NAME}",
        # Each feature_dir should have at least one of the following folders with this structure:
        #  training/
        #    ragged_mmap_folders_ending_in_mmap
        #  testing/
        #    ragged_mmap_folders_ending_in_mmap
        #  testing_ambient/
        #    ragged_mmap_folders_ending_in_mmap
        #  validation/
        #    ragged_mmap_folders_ending_in_mmap
        #  validation_ambient/
        #    ragged_mmap_folders_ending_in_mmap
        #
        #  sampling_weight: Weight for choosing a spectrogram from this set in the batch
        #  penalty_weight: Penalizing weight for incorrect predictions from this set
        #  truth: Boolean whether this set has positive samples or negative samples
        #  truncation_strategy = If spectrograms in the set are longer than necessary for training, how are they truncated
        #       - random: choose a random portion of the entire spectrogram - useful for long negative samples
        #       - truncate_start: remove the start of the spectrogram
        #       - truncate_end: remove the end of the spectrogram
        #       - split: Split the longer spectrogram into separate spectrograms offset by 100 ms. Only for ambient sets
        "features": [
            {
                "features_dir": "data/generated_augmented_features",
                "sampling_weight": 3.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            {
                "features_dir": "data/negative_datasets/speech",
                "sampling_weight": 10.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": "data/negative_datasets/dinner_party",
                "sampling_weight": 10.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": "data/negative_datasets/no_speech",
                "sampling_weight": 5.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                # Everyday phrases that sound a bit like the wake word, trained on as negatives.
                # Weighted like the wake word clips, so batches hold as many near-misses as positives.
                "features_dir": "data/near_miss_features",
                "sampling_weight": 3.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            {
                # Only used for validation and testing
                "features_dir": "data/negative_datasets/dinner_party_eval",
                "sampling_weight": 0.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "split",
                "type": "mmap",
            },
        ],
        # Number of training steps in each iteration - various other settings are configured as lists that correspond to different steps
        # Three stages: a long run at the full learning rate, then two short ones at 10x and 100x
        # lower so the weights settle instead of jittering. The last two stages are 10% each.
        "training_steps": [160_000, 20_000, 20_000],
        # Penalizing weight for incorrect class predictions - lists that correspond to training steps
        # With CLASS_WEIGHTS = "upstream" in the training cell, these barely change training, so
        # comparing class weights needs CLASS_WEIGHTS = "per_sample" (see the training cell).
        "positive_class_weight": [1],
        # Changing the weights shifts scores, so re-pick the probability cutoff from the new ROC.
        # Negative class weight per training stage (80k / 10k / 10k steps). A/B test with everything
        # else held fixed, judged on real recordings rather than the synthetic validation set:
        #   [20, 20, 20]  control: flat 20, the upstream default (what runs 1 and 2 used)
        #   [20, 25, 30]  candidate: gentle ramp of about x1.25 per stage, in line with community
        #                 logs (e.g. malonestar/custom-micro-wake-word-model went 40->50 and 50->60,
        #                 and judged 60->75 too aggressive). Run 3's 20/40/80 was steeper than any usage found.
        "negative_class_weight": [20, 20, 20],
        # Learning rates for Adam optimizer - list that corresponds to training steps
        "learning_rates": [0.001, 0.0001, 0.00001],
        "batch_size": 128,
        # SpecAugment - list that corresponds to training steps
        "time_mask_max_size": [0],
        # SpecAugment - list that corresponds to training steps
        "time_mask_count": [0],
        # SpecAugment - list that corresponds to training steps
        "freq_mask_max_size": [0],
        # SpecAugment - list that corresponds to training steps
        "freq_mask_count": [0],
        # Test the validation sets after every this many steps
        "eval_step_interval": 500,
        # Maximum length of wake word that the streaming model will accept
        "clip_duration_ms": 1500,
        # The best model weights are chosen first by minimizing the specified minimization metric below the specified target_minimization
        # Once the target has been met, it chooses the maximum of the maximization metric. Set 'minimization_metric' to None to only maximize
        # Available metrics:
        #   - "loss" - cross entropy error on validation set
        #   - "accuracy" - accuracy of validation set
        #   - "recall" - recall of validation set
        #   - "precision" - precision of validation set
        #   - "false_positive_rate" - false positive rate of validation set
        #   - "false_negative_rate" - false negative rate of validation set
        #   - "ambient_false_positives" - count of false positives from the split validation_ambient set
        #   - "ambient_false_positives_per_hour" - estimated number of false positives per hour on the split validation_ambient set
        "target_minimization": 0.9,
        # Set to None to disable
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }

    with open("training_parameters.yaml", "w") as _file:
        yaml.dump(config, _file)
    return (config,)


@app.cell
def _(config, run):
    # Trains a model, then quantizes and converts the best weights to a streaming version
    # suitable for on-device detection, and tests it.
    # Interrupt the cell to stop it. It will resume if stopped and RESTORE_CHECKPOINT is True,
    # but it will start over at the configured training steps in the yaml file.
    # Set TRAIN = False to only convert and test the best-weighted model.
    #
    # Training runs tools/fast_train.py, which trains like microWakeWord's own
    # `python -m microwakeword.model_train_eval --train 1` but keeps the GPU busy: worker
    # processes build the training batches ahead of time, and the validation sets are read
    # once and kept in memory instead of being read from disk at every validation.
    # Every 100 steps it prints the training speed and how much of the time it waited for
    # batches, and each validation's log line says how long it took.
    # Converting and testing run microwakeword.model_train_eval itself.

    TRAIN = True
    RESTORE_CHECKPOINT = False
    # Processes that build the training batches. If the progress line shows the training loop
    # waiting for batches much of the time, use more. Each takes about 100 MB of memory.
    WORKERS = 16
    # How positive_class_weight and negative_class_weight are applied:
    #   "upstream"    The way microwakeword.train applies them, so runs compare with earlier
    #                 ones. It multiplies the penalty weights, shape (batch,), by the class
    #                 weights, shape (batch, 1), which broadcasts to a batch x batch matrix,
    #                 and Keras reduces that to mean(class weight) * mean(loss). So the whole
    #                 batch's loss is scaled by one number instead of each sample being
    #                 weighted, and Adam divides that scale out: the class weights have almost
    #                 no effect.
    #   "per_sample"  Each sample's loss is weighted by its class weight. A negative weight of
    #                 20 then really makes a false positive cost 20 times a missed wake word,
    #                 which moves the model's scores a lot, so choose the weights again.
    CLASS_WEIGHTS = "per_sample"

    # The model. Training and converting both take these.
    MODEL_ARGS = [
        "mixednet",
        "--pointwise_filters",
        "64,64,64,64",
        "--repeat_in_block",
        "1, 1, 1, 1",
        "--mixconv_kernel_sizes",
        "[5], [7,11], [9,15], [23]",
        "--residual_connection",
        "0,0,0,0",
        "--first_conv_filters",
        "32",
        "--first_conv_kernel_size",
        "5",
        "--stride",
        "3",
    ]

    # TensorFlow 2.21 does not find the pip-installed libcusolver on its own.
    _cusolver_lib = (
        Path.cwd() / ".venv/lib/python3.10/site-packages/nvidia/cusolver/lib"
    )
    os.environ["LD_LIBRARY_PATH"] = (
        f"{_cusolver_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )

    # Training from scratch needs an empty train_dir (the script refuses to start
    # otherwise), so a folder left by an earlier run with the same RUN_NAME is deleted
    # first. Give the run a new RUN_NAME in the config cell to keep the old one, or set
    # RESTORE_CHECKPOINT = True to resume from it. A folder written to in the last 15 minutes
    # is never deleted, because a training run may still be using it.
    if TRAIN and not RESTORE_CHECKPOINT:
        _train_dir = Path(config["train_dir"]).resolve()
        if _train_dir.exists():
            # Only a run folder directly under output/ is deleted, so a stale config
            # can't delete anything else.
            if _train_dir.parent != Path("output").resolve():
                raise RuntimeError(
                    f"Refusing to delete {_train_dir}: it is not a run folder directly under output/"
                )
            _last_write = max(
                (path.stat().st_mtime for path in _train_dir.rglob("*")),
                default=_train_dir.stat().st_mtime,
            )
            _minutes = (time.time() - _last_write) / 60
            if _minutes < 15:
                raise RuntimeError(
                    f"Refusing to delete {_train_dir}: it was written to {_minutes:.0f} minutes ago, so a "
                    "training run may still be using it. Use a new RUN_NAME, or wait until it's finished."
                )
            print(f"Deleting {_train_dir} (an earlier run with this RUN_NAME)")
            shutil.rmtree(_train_dir)

    if TRAIN:
        run(
            [
                "uv",
                "run",
                "python",
                "tools/fast_train.py",
                "--training_config=training_parameters.yaml",
                "--restore_checkpoint",
                str(int(RESTORE_CHECKPOINT)),
                "--workers",
                str(WORKERS),
                "--class_weights",
                CLASS_WEIGHTS,
                *MODEL_ARGS,
            ]
        )

    # Converts and tests the best weights. "--train 0" tells model_train_eval to skip its own
    # training, which fast_train.py above replaces, so TRAIN and RESTORE_CHECKPOINT don't apply
    # to this step. The quantized streaming model goes to
    # output/<RUN_NAME>/tflite_stream_state_internal_quant/.
    run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "microwakeword.model_train_eval",
            "--training_config=training_parameters.yaml",
            "--train",
            "0",
            "--test_tf_nonstreaming",
            "0",
            "--test_tflite_nonstreaming",
            "0",
            "--test_tflite_nonstreaming_quantized",
            "0",
            "--test_tflite_streaming",
            "0",
            "--test_tflite_streaming_quantized",
            "1",
            "--use_weights",
            "best_weights",
            *MODEL_ARGS,
        ]
    )

    # For docker workflow:
    # docker run --gpus all \
    #   --rm -i --device=/dev/dxg \
    #   -v /usr/lib/wsl:/usr/lib/wsl:ro \
    #   -v "$(pwd)":"$(pwd)" \
    #   -w "$(pwd)" \
    #   tensorflow/tensorflow:2.16.2-gpu \
    #   python train.py
    return


if __name__ == "__main__":
    app.run()
