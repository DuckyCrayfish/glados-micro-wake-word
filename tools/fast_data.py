"""Training and validation batches for tools/fast_train.py.

microwakeword.data builds each training batch in the training loop's own thread, reading
and copying one spectrogram at a time, and reads the whole validation set from disk again
at every evaluation. Here, worker processes build the training batches ahead of the
training loop, straight into shared memory, and the validation sets are read once and
kept in memory.

Batches are drawn the same way as microwakeword.data: for each sample, a feature set is
picked by its sampling_weight, then one of its spectrograms uniformly, which is cut to the
model's input length by the set's truncation_strategy.

Only numpy, mmap_ninja and the standard library are imported, so the worker processes
start quickly and never load TensorFlow.
"""

# Keeping this in line with microWakeWord
# ---------------------------------------
# This follows microwakeword/data.py as of the commit named in tools/fast_train.py, which
# also says how to find what changed upstream. Where upstream's code went:
#
#   FeatureHandler.__init__, reading the "features" config  -> FeatureSet.__init__
#   MmapFeatureGenerator.__init__, finding the *_mmap folders -> FeatureSet.__init__
#   MmapFeatureGenerator's durations (get_mode_duration)      -> AmbientSet.duration_s
#   get_data("training"), choosing a feature set per sample   -> TrainingSampler
#   MmapFeatureGenerator.get_random_spectrogram               -> TrainingSampler.fill
#   fixed_length_spectrogram, and the uint16 scaling          -> cut_into, UINT16_SCALE
#   get_data("validation"), truncation "truncate_start"       -> load_spectrograms
#   get_data("validation_ambient"), truncation "split"        -> AmbientSet
#   spec_augment                                              -> spec_augment
#   ClipsHandlerWrapperGenerator (feature type "clips")       -> not supported, raises
#
# The code is rewritten rather than copied, so port upstream changes by what they do. What
# has to stay the same as upstream:
#   - cut_into: exactly the array fixed_length_spectrogram returns, scaled the same way.
#   - load_spectrograms and AmbientSet: the same spectrograms and labels as get_data, in
#     any order, and the same total duration.
#   - TrainingSampler and spec_augment: the same probabilities. The random numbers differ.
#
# Deliberate differences:
#   - Random numbers come from a numpy Generator in each worker process instead of Python's
#     random and np.random. The batches drawn differ from upstream's, but are drawn the same
#     way. spec_augment draws the masks for a whole batch at once, in a different order.
#   - The validation spectrograms aren't shuffled. The metrics don't depend on their order.
#   - Settings upstream supports and this doesn't raise an error instead of silently doing
#     something else: feature type "clips", fixed_right_cutoffs other than [0], and the
#     truncation strategies "none" and "fixed_right_cutoff".
#
# Not microWakeWord, but depended on: FeatureFile reads mmap_ninja's files directly
# (data.ninja, the starts array and the memmap's offset), as laid out by mmap_ninja 0.8.0.
# After upgrading mmap_ninja, check that cut_into still matches fixed_length_spectrogram
# on your feature files.

import ctypes
import multiprocessing as mp
import os
import queue
import signal
import traceback
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from mmap_ninja.ragged import RaggedMmap

# microwakeword.data multiplies spectrograms stored as uint16 by this to get the features.
UINT16_SCALE = np.float32(0.0390625)


class FeatureFile:
    """One mmap_ninja RaggedMmap folder (*_mmap/). Each read is a single pread of only the
    frames that are used, instead of page faults on a memory map of the whole file."""

    def __init__(self, path):
        ragged = RaggedMmap(path)
        self.path = Path(path)
        self.size = len(ragged)
        if not self.size:
            return
        self.dtype = ragged.memmap.dtype
        self.n_features = int(ragged.shapes[0][1])
        self._data_offset = ragged.memmap.offset
        # Stays memory mapped, so the worker processes share one copy in the page cache.
        self._starts = ragged.starts
        sizes = np.asarray(ragged.ends) - np.asarray(ragged.starts)
        if np.any(sizes % self.n_features):
            raise ValueError(f"{path}: the spectrograms don't all have {self.n_features} features")
        self.frames = (sizes // self.n_features).astype(np.int32)
        self._fd = os.open(self.path / "data.ninja", os.O_RDONLY)
        weakref.finalize(self, os.close, self._fd)
        # The reads jump around the file, so reading ahead only wastes disk time and memory.
        os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_RANDOM)

    def read(self, index, first, count):
        """Frames first to first + count of spectrogram `index`, as stored (not scaled)."""
        row_bytes = self.n_features * self.dtype.itemsize
        offset = self._data_offset + int(self._starts[index]) * self.dtype.itemsize + first * row_bytes
        data = os.pread(self._fd, count * row_bytes, offset)
        if len(data) != count * row_bytes:
            raise OSError(f"{self.path}: short read of spectrogram {index}")
        return np.frombuffer(data, self.dtype).reshape(count, self.n_features)


def cut_into(out, file, index, strategy, rng=None):
    """Upstream: microwakeword.data.fixed_length_spectrogram, then the uint16 scaling.

    Writes spectrogram `index` of `file` into `out` (float32, frames x features). A longer
    spectrogram is cut by `strategy`, a shorter one is padded with zeros at the start. Each
    strategy branch matches the upstream branch of the same name; a new upstream strategy
    needs a new branch here."""
    length = int(file.frames[index])
    wanted = len(out)
    if length > wanted:
        if strategy == "random":
            # Same range as np.random.randint(0, length - wanted) upstream.
            first = int(rng.integers(0, length - wanted))
        elif strategy == "truncate_start":
            first = length - wanted
        elif strategy == "truncate_end":
            first = 0
        else:
            raise ValueError(f"truncation_strategy {strategy!r} isn't supported for training or validation")
        out[:] = file.read(index, first, wanted)
    else:
        out[: wanted - length] = 0
        out[wanted - length :] = file.read(index, 0, length)
    if file.dtype == np.uint16:
        out *= UINT16_SCALE


class FeatureSet:
    """The spectrograms of one entry of the training config's "features" list, in one
    mode (training, validation, validation_ambient, ...).

    Upstream: one MmapFeatureGenerator, made from the entry by FeatureHandler.__init__. The
    entry's keys are read the same way; a new key upstream reads may be needed here too."""

    def __init__(self, entry, mode):
        if entry["type"] != "mmap":
            raise NotImplementedError(f"feature sets of type {entry['type']!r} aren't supported, only 'mmap'")
        if list(entry.get("fixed_right_cutoffs", [0])) != [0]:
            raise NotImplementedError("fixed_right_cutoffs isn't supported")
        self.label = float(entry["truth"])
        self.sampling_weight = float(entry["sampling_weight"])
        self.penalty_weight = float(entry["penalty_weight"])
        self.truncation_strategy = entry["truncation_strategy"]
        # Same search as microwakeword.data.MmapFeatureGenerator.
        directory = Path(os.path.abspath(os.path.join(entry["features_dir"], mode)))
        files = [FeatureFile(path) for path in sorted(directory.glob("**/*_mmap/"))]
        self.files = [file for file in files if file.size]
        self._ends = np.cumsum([file.size for file in self.files])
        self.size = int(self._ends[-1]) if self.files else 0

    def locate(self, i):
        """The file and index within it of spectrogram i of the whole set."""
        k = int(np.searchsorted(self._ends, i, side="right"))
        return self.files[k], i - (int(self._ends[k - 1]) if k else 0)


def feature_sets(features_config, mode):
    """The feature sets that have spectrograms in `mode`, in config order."""
    sets = [FeatureSet(entry, mode) for entry in features_config]
    sets = [s for s in sets if s.size]
    n_features = {file.n_features for s in sets for file in s.files}
    if len(n_features) > 1:
        raise ValueError(f"{mode} spectrograms have different numbers of features: {sorted(n_features)}")
    return sets


class TrainingSampler:
    """Upstream: FeatureHandler.get_data("training") and get_random_spectrogram.

    For each sample, upstream picks a feature set with random.choices weighted by
    sampling_weight (among the sets with training spectrograms), then one of the set's
    spectrograms with random.choice, and cuts it with the set's truncation_strategy. The
    label and penalty_weight come from the set."""

    def __init__(self, features_config):
        self.sets = feature_sets(features_config, "training")
        if not self.sets:
            raise ValueError("no training spectrograms found in any feature set")
        weights = np.array([s.sampling_weight for s in self.sets])
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError(f"sampling weights must be >= 0 with a positive total: {weights}")
        self.probabilities = weights / weights.sum()
        self.labels = np.array([s.label for s in self.sets])
        self.penalty_weights = np.array([s.penalty_weight for s in self.sets])
        self.n_features = self.sets[0].files[0].n_features

    def fill(self, spectrograms, set_indices, rng):
        """Fills spectrograms (batch x frames x features) and the feature set of each."""
        set_indices[:] = rng.choice(len(self.sets), size=len(spectrograms), p=self.probabilities)
        for out, k in zip(spectrograms, set_indices):
            feature_set = self.sets[k]
            file, index = feature_set.locate(int(rng.integers(feature_set.size)))
            cut_into(out, file, index, feature_set.truncation_strategy, rng)


def spec_augment(spectrograms, time_mask_max_size, time_mask_count, freq_mask_max_size, freq_mask_count, rng):
    """Upstream: microwakeword.data.spec_augment, which get_data applies to each spectrogram.

    This does a whole batch at once, in place. Each spectrogram gets the same number of
    masks, with sizes and positions drawn from the same ranges as upstream."""
    batch_size, n_frames, n_features = spectrograms.shape
    for count, max_size, length, view in (
        (time_mask_count, time_mask_max_size, n_frames, spectrograms),
        (freq_mask_count, freq_mask_max_size, n_features, spectrograms.transpose(0, 2, 1)),
    ):
        positions = np.arange(length)
        for _ in range(count):
            # Same as int(np.random.uniform(0, max_size)) and random.randint(0, length - size).
            size = rng.uniform(0, max_size, size=batch_size).astype(np.int64)
            start = rng.integers(0, length - size + 1)
            masked = (positions >= start[:, None]) & (positions < (start + size)[:, None])
            view[masked] = 0


class BatchLoader:
    """Worker processes that fill training batches ahead of the training loop.

    next() returns (spectrograms, labels, penalty_weights) for the next batch. The
    spectrograms are a view into shared memory, valid until the following next() call.
    Use it in a with statement, so the workers are stopped at the end."""

    def __init__(self, features_config, batch_size, features_length, workers, slots=None):
        self.sampler = TrainingSampler(features_config)
        slots = slots or max(8, 3 * workers)
        shape = (slots, batch_size, features_length, self.sampler.n_features)
        ctx = mp.get_context("spawn")
        shared_spectrograms = ctx.RawArray(ctypes.c_float, int(np.prod(shape)))
        shared_set_indices = ctx.RawArray(ctypes.c_int32, slots * batch_size)
        self._spectrograms = np.frombuffer(shared_spectrograms, np.float32).reshape(shape)
        self._set_indices = np.frombuffer(shared_set_indices, np.int32).reshape(slots, batch_size)
        # Slots go round: free -> a worker fills it -> ready -> the training loop uses it -> free.
        self._free = ctx.Queue()
        self._ready = ctx.Queue()
        for slot in range(slots):
            self._free.put(slot)
        self._held = None
        self._workers = [
            ctx.Process(
                target=_worker,
                args=(features_config, shape, shared_spectrograms, shared_set_indices,
                      self._free, self._ready, os.getpid()),
                name=f"batch-worker-{i}",
                daemon=True,
            )
            for i in range(workers)
        ]
        for worker in self._workers:
            worker.start()

    def next(self):
        if self._held is not None:
            self._free.put(self._held)
            self._held = None
        while True:
            try:
                slot = self._ready.get(timeout=5)
                break
            except queue.Empty:
                for worker in self._workers:
                    if not worker.is_alive():
                        raise RuntimeError(f"{worker.name} exited with code {worker.exitcode}") from None
        if isinstance(slot, str):
            raise RuntimeError(f"a batch worker failed:\n{slot}")
        self._held = slot
        set_indices = self._set_indices[slot]
        return (
            self._spectrograms[slot],
            self.sampler.labels[set_indices],
            self.sampler.penalty_weights[set_indices],
        )

    def close(self):
        for _ in self._workers:
            self._free.put(None)
        for worker in self._workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _worker(features_config, shape, shared_spectrograms, shared_set_indices, free, ready, parent_pid):
    # Ctrl+C reaches the whole process group. The training loop handles it and stops the workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        sampler = TrainingSampler(features_config)
        spectrograms = np.frombuffer(shared_spectrograms, np.float32).reshape(shape)
        set_indices = np.frombuffer(shared_set_indices, np.int32).reshape(shape[:2])
        rng = np.random.default_rng()
        while True:
            try:
                slot = free.get(timeout=5)
            except queue.Empty:
                if os.getppid() != parent_pid:
                    return  # the training process is gone
                continue
            if slot is None:
                return
            sampler.fill(spectrograms[slot], set_indices[slot], rng)
            ready.put(slot)
    except Exception:
        ready.put(traceback.format_exc())


def load_spectrograms(features_config, mode, features_length, threads=16):
    """Upstream: FeatureHandler.get_data(mode, truncation_strategy="truncate_start"), as
    validate_nonstreaming calls it.

    Every spectrogram of `mode` (validation or testing) cut to its last features_length
    frames, in one float32 array, with their labels. Not shuffled, unlike upstream."""
    sets = feature_sets(features_config, mode)
    if not sets:
        raise ValueError(f"no {mode} spectrograms found in any feature set")
    total = sum(s.size for s in sets)
    spectrograms = np.empty((total, features_length, sets[0].files[0].n_features), np.float32)
    labels = np.empty(total)
    chunks = []
    row = 0
    for s in sets:
        labels[row : row + s.size] = s.label
        for file in s.files:
            for first in range(0, file.size, 1024):
                chunks.append((file, first, min(first + 1024, file.size), row + first))
            row += file.size

    def load(chunk):
        file, first, end, row = chunk
        for index in range(first, end):
            cut_into(spectrograms[row + index - first], file, index, "truncate_start")

    # pread releases the GIL, so threads keep several reads in flight.
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(load, chunks))
    return spectrograms, labels


class AmbientSet:
    """Upstream: FeatureHandler.get_data(mode, truncation_strategy="split") and
    get_mode_duration(mode), as validate_nonstreaming calls them.

    The long background recordings of `mode` (validation_ambient), cut into overlapping
    windows like upstream's "split" in MmapFeatureGenerator.get_feature_generator. The
    recordings are kept whole in memory and the windows are cut as each batch is made."""

    def __init__(self, features_config, mode, features_length, step_s, stride):
        hop = int(1000 * step_s * stride)  # same expression as microwakeword.data
        self.features_length = features_length
        self.recordings = []
        self.windows = []
        labels = []
        # Hours of audio, for false positives per hour. Summed like
        # microwakeword.data.MmapFeatureGenerator.
        self.duration_s = 0.0
        for s in feature_sets(features_config, mode):
            for file in s.files:
                for index in range(file.size):
                    length = int(file.frames[index])
                    recording = np.empty((length, file.n_features), np.float32)
                    cut_into(recording, file, index, "truncate_start")
                    self.duration_s += step_s * length
                    for start in range(0, length - features_length, hop):
                        self.windows.append((len(self.recordings), start))
                        labels.append(s.label)
                    self.recordings.append(recording)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.windows)

    def batches(self, batch_size):
        for first in range(0, len(self.windows), batch_size):
            windows = self.windows[first : first + batch_size]
            spectrograms = np.stack(
                [self.recordings[k][start : start + self.features_length] for k, start in windows]
            )
            yield spectrograms, self.labels[first : first + batch_size]
