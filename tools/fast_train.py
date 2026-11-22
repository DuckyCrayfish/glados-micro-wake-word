"""Trains a microWakeWord model like `python -m microwakeword.model_train_eval --train 1`,
without its CPU bottleneck.

The model, data, loss, metrics, schedule and choice of best weights are the same, and so is
the run folder (weights, checkpoints, TensorBoard logs, training_config.yaml). What changes is
how the data gets to the GPU (see tools/fast_data.py):
  - Worker processes build the training batches ahead of the training loop, so the GPU no
    longer waits for one Python thread to read and copy every spectrogram.
  - The validation sets are read from disk once and kept in memory, instead of being read
    again and copied into new arrays at every evaluation.

It only trains. Convert and test the model afterwards with
`python -m microwakeword.model_train_eval --train 0`, with the same config and model arguments.

    python tools/fast_train.py --training_config=training_parameters.yaml mixednet --stride 3 ...
"""

# Keeping this in line with microWakeWord
# ---------------------------------------
# This file and tools/fast_data.py follow microWakeWord (libs/microWakeWord) as of commit
# 4665173cd35f1cff9a61e06fc427f124766c488e ("Pin dependencies (#98)"). After updating the
# submodule, list what changed in the code they follow:
#
#     git -C libs/microWakeWord diff 4665173 HEAD -- microwakeword/train.py \
#         microwakeword/data.py microwakeword/model_train_eval.py
#
# Port each change to where that code went (below), then put the new commit above.
#
# Where upstream's code went:
#   microwakeword/train.py
#     train()                  -> train(). Same order of steps; comments starting with
#                                 "Upstream" mark each difference, the rest is copied.
#     validate_nonstreaming()  -> validate(). The metric calculations are copied verbatim.
#     swap_attribute()         -> not needed (see validate()).
#   microwakeword/data.py      -> tools/fast_data.py, mapped in its header.
#   microwakeword/model_train_eval.py
#     __main__ flags           -> parse_flags(), the training flags only.
#     train_model()            -> main().
#     load_config(), the models and their flags (mixednet, inception), utils
#                              -> imported, so these follow upstream without changes here.
#     evaluate_model(), conversion to TFLite and testing
#                              -> not used here. The notebook runs upstream's
#                                 model_train_eval with --train 0 after training.
#
# Upstream changes that need more than copying:
#   - A new setting that changes how batches are made (like a new augmentation): batches
#     are built in worker processes, so it goes in fast_data.TrainingSampler or
#     fast_data.spec_augment, and train() has to pass it on.
#   - Changes to how validate_nonstreaming() calls model.evaluate: see validate().
#   - A fix to the class weights broadcasting to a batch x batch matrix: make the
#     "upstream" branch of --class_weights in train() match it.
#
# What must stay true after porting: validate() gives the same metrics as upstream's
# validate_nonstreaming() for the same weights, and fast_data.cut_into() gives exactly the
# spectrograms upstream's fixed_length_spectrogram() does. Both were checked identical at
# the commit above, by running each against upstream on the same data.

import argparse
import os
import time

import numpy as np

# Only numpy and the standard library are imported at the top of this file: the batch
# workers are started with "spawn", which imports this file again in every worker.
# TensorFlow and microwakeword are imported in main().
import fast_data

VALIDATION_BATCH_SIZE = 1024  # same as microwakeword.train
PROGRESS_EVERY = 100  # steps between progress lines


def parse_flags(inception, mixednet):
    """Upstream: the training flags of microwakeword.model_train_eval's __main__ block, plus
    --workers and --class_weights. The model flags come from upstream's model_parameters()."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--training_config", required=True, help="Path to the training parameters yaml.")
    parser.add_argument(
        "--restore_checkpoint", type=int, default=0,
        help="1 to restore the checkpoint in the run folder and train from it. The step count starts over, as upstream.",
    )
    parser.add_argument(
        "--workers", type=int, default=min(16, max(1, (os.cpu_count() or 2) // 2)),
        help="Processes that build training batches. Use more if the progress line shows the "
        "training loop waiting for batches much of the time.",
    )
    parser.add_argument(
        "--class_weights", choices=["upstream", "per_sample"], default="upstream",
        help="'upstream' applies positive_class_weight and negative_class_weight exactly as "
        "microwakeword.train does. Its weights broadcast to a batch x batch matrix, so the loss is "
        "scaled by the batch's average class weight instead of weighting each sample, which Adam "
        "almost cancels out. 'per_sample' weights each sample's loss by its class weight.",
    )
    parser.add_argument("--verbosity", default="INFO", choices=["DEBUG", "INFO", "WARN", "ERROR", "FATAL"])
    subparsers = parser.add_subparsers(dest="model_name", required=True, help="NN model name")
    inception.model_parameters(subparsers.add_parser("inception"))
    mixednet.model_parameters(subparsers.add_parser("mixednet"))
    flags, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(f"Unknown argument: {unparsed}")
    return flags


class ValidationData:
    """The validation sets, read once."""

    def __init__(self, config):
        started = time.perf_counter()
        self.spectrograms, self.labels = fast_data.load_spectrograms(
            config["features"], "validation", config["spectrogram_length"]
        )
        self.ambient = fast_data.AmbientSet(
            config["features"], "validation_ambient", config["spectrogram_length"],
            step_s=config["window_step_ms"] / 1000.0, stride=config["stride"],
        )
        self.load_seconds = time.perf_counter() - started
        self.gigabytes = (
            self.spectrograms.nbytes + sum(r.nbytes for r in self.ambient.recordings)
        ) / 1e9

    def batches(self, batch_size):
        for first in range(0, len(self.labels), batch_size):
            yield self.spectrograms[first : first + batch_size], self.labels[first : first + batch_size]


def validate(model, validation):
    """Upstream: microwakeword.train.validate_nonstreaming, on the in-memory validation sets.

    Only the evaluation calls differ. Upstream calls model.evaluate on the validation set,
    then again on the ambient set with model.reset_metrics swapped out, so the ambient
    results add onto the validation set's. test_on_batch never resets the metrics, so calling
    it on every batch of both sets in turn accumulates the same totals (true as of Keras 3.12;
    after a Keras upgrade, check that this still gives upstream's metrics for the same weights).

    From `metrics = {}` on, the code is upstream's. Copy changes to it line for line.

    One small difference: the ambient set is skipped when it has no windows. Upstream checks
    for ambient recordings instead, so recordings all shorter than the model input would
    give it an empty evaluation."""
    model.reset_metrics()
    for spectrograms, labels in validation.batches(VALIDATION_BATCH_SIZE):
        result = model.test_on_batch(spectrograms, labels.reshape(-1, 1), return_dict=True)

    metrics = {}
    metrics["accuracy"] = result["accuracy"]
    metrics["recall"] = result["recall"]
    metrics["precision"] = result["precision"]

    metrics["auc"] = result["auc"]
    metrics["loss"] = result["loss"]
    metrics["recall_at_no_faph"] = 0
    metrics["cutoff_for_no_faph"] = 0
    metrics["ambient_false_positives"] = 0
    metrics["ambient_false_positives_per_hour"] = 0
    metrics["average_viable_recall"] = 0

    test_set_fp = np.asarray(result["fp"])

    if len(validation.ambient) > 0:
        for spectrograms, labels in validation.ambient.batches(VALIDATION_BATCH_SIZE):
            ambient_predictions = model.test_on_batch(spectrograms, labels.reshape(-1, 1), return_dict=True)

        duration_of_ambient_set = validation.ambient.duration_s / 3600.0

        # Other than the false positive rate, all other metrics are accumulated across
        # both test sets
        all_true_positives = np.asarray(ambient_predictions["tp"])
        ambient_false_positives = np.asarray(ambient_predictions["fp"]) - test_set_fp
        all_false_negatives = np.asarray(ambient_predictions["fn"])

        metrics["auc"] = ambient_predictions["auc"]
        metrics["loss"] = ambient_predictions["loss"]

        recall_at_cutoffs = all_true_positives / (all_true_positives + all_false_negatives)
        faph_at_cutoffs = ambient_false_positives / duration_of_ambient_set

        target_faph_cutoff_probability = 1.0
        for index, cutoff in enumerate(np.linspace(0.0, 1.0, 101)):
            if faph_at_cutoffs[index] == 0:
                target_faph_cutoff_probability = cutoff
                recall_at_no_faph = recall_at_cutoffs[index]
                break

        if faph_at_cutoffs[0] > 2:
            # Use linear interpolation to estimate recall at 2 faph

            # Increase index until we find a faph less than 2
            index_of_first_viable = 1
            while faph_at_cutoffs[index_of_first_viable] > 2:
                index_of_first_viable += 1

            x0 = faph_at_cutoffs[index_of_first_viable - 1]
            y0 = recall_at_cutoffs[index_of_first_viable - 1]
            x1 = faph_at_cutoffs[index_of_first_viable]
            y1 = recall_at_cutoffs[index_of_first_viable]

            recall_at_2faph = (y0 * (x1 - 2.0) + y1 * (2.0 - x0)) / (x1 - x0)
        else:
            # Lowest faph is already under 2, assume the recall is constant before this
            index_of_first_viable = 0
            recall_at_2faph = recall_at_cutoffs[0]

        x_coordinates = [2.0]
        y_coordinates = [recall_at_2faph]

        for index in range(index_of_first_viable, len(recall_at_cutoffs)):
            if faph_at_cutoffs[index] != x_coordinates[-1]:
                # Only add a point if it is a new faph
                # This ensures if a faph rate is repeated, we use the highest recall
                x_coordinates.append(faph_at_cutoffs[index])
                y_coordinates.append(recall_at_cutoffs[index])

        # Use trapezoid rule to estimate the area under the curve, then divide by 2.0 to get the average recall
        average_viable_recall = np.trapezoid(np.flip(y_coordinates), np.flip(x_coordinates)) / 2.0

        metrics["recall_at_no_faph"] = recall_at_no_faph
        metrics["cutoff_for_no_faph"] = target_faph_cutoff_probability
        metrics["ambient_false_positives"] = ambient_false_positives[50]
        metrics["ambient_false_positives_per_hour"] = faph_at_cutoffs[50]
        metrics["average_viable_recall"] = average_viable_recall

    return metrics


def train(model, config, loader, validation, class_weight_mode):
    """Upstream: microwakeword.train.train, with batches from `loader` and validation from memory.

    The steps are in upstream's order, so the two can be read side by side. The comments
    starting with "Upstream" mark every place this differs. Everything else (settings and
    their defaults, loss, optimizer, metrics, checkpoints, logging, TensorBoard tags, saved
    weights and the best-weights rule) is copied, so copy upstream changes to it as they are."""
    import tensorflow as tf
    from absl import logging

    # Upstream also reads mix_up_augmentation_prob and freq_mix_augmentation_prob, but
    # microwakeword.data never applies them. Add them here if upstream starts to.
    # Assign default training settings if not set in the configuration yaml
    if not (training_steps_list := config.get("training_steps")):
        training_steps_list = [20000]
    if not (learning_rates_list := config.get("learning_rates")):
        learning_rates_list = [0.001]
    if not (time_mask_max_size_list := config.get("time_mask_max_size")):
        time_mask_max_size_list = [5]
    if not (time_mask_count_list := config.get("time_mask_count")):
        time_mask_count_list = [2]
    if not (freq_mask_max_size_list := config.get("freq_mask_max_size")):
        freq_mask_max_size_list = [5]
    if not (freq_mask_count_list := config.get("freq_mask_count")):
        freq_mask_count_list = [2]
    if not (positive_class_weight_list := config.get("positive_class_weight")):
        positive_class_weight_list = [1.0]
    if not (negative_class_weight_list := config.get("negative_class_weight")):
        negative_class_weight_list = [1.0]

    # Ensure all training setting lists are as long as the training step iterations
    def pad_list_with_last_entry(list_to_pad, desired_length):
        while len(list_to_pad) < desired_length:
            list_to_pad.append(list_to_pad[-1])

    training_step_iterations = len(training_steps_list)
    for settings in (
        learning_rates_list, time_mask_max_size_list, time_mask_count_list, freq_mask_max_size_list,
        freq_mask_count_list, positive_class_weight_list, negative_class_weight_list,
    ):
        pad_list_with_last_entry(settings, training_step_iterations)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam()

    cutoffs = np.linspace(0.0, 1.0, 101).tolist()

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.TruePositives(name="tp", thresholds=cutoffs),
        tf.keras.metrics.FalsePositives(name="fp", thresholds=cutoffs),
        tf.keras.metrics.TrueNegatives(name="tn", thresholds=cutoffs),
        tf.keras.metrics.FalseNegatives(name="fn", thresholds=cutoffs),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryCrossentropy(name="loss"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Upstream unwraps model.train_function with tf_decorator.unwrap here. With Keras 3,
    # train_function is a plain function that calls a tf.function, so unwrapping it does
    # nothing, and it's left out.

    # Configure checkpointer and restore if available
    checkpoint_directory = os.path.join(config["train_dir"], "restore/")
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    # Configure TensorBoard summaries
    train_writer = tf.summary.create_file_writer(os.path.join(config["summaries_dir"], "train"))
    validation_writer = tf.summary.create_file_writer(os.path.join(config["summaries_dir"], "validation"))

    stage_ends = np.cumsum(training_steps_list)
    training_steps_max = int(stage_ends[-1])

    best_minimization_quantity = 10000
    best_maximization_quantity = 0.0
    best_no_faph_cutoff = 1.0

    rng = np.random.default_rng()
    learning_rate = None
    progress_started = time.perf_counter()
    progress_steps = 0
    waited = 0.0

    for training_step in range(1, training_steps_max + 1):
        # Upstream finds the stage with a loop over training_steps_list and assigns the
        # learning rate every step. This picks the same stage (the first one ending at or after
        # this step) and assigns the learning rate only when it changes.
        stage = int(np.searchsorted(stage_ends, training_step))
        if learning_rates_list[stage] != learning_rate:
            learning_rate = learning_rates_list[stage]
            model.optimizer.learning_rate.assign(learning_rate)

        # Upstream: data_processor.get_data("training", ...), which also applies SpecAugment.
        # The batches are drawn the same way, in worker processes (see tools/fast_data.py).
        wait_started = time.perf_counter()
        train_fingerprints, train_labels, train_sample_weights = loader.next()
        waited += time.perf_counter() - wait_started

        if time_mask_count_list[stage] or freq_mask_count_list[stage]:
            fast_data.spec_augment(
                train_fingerprints,
                time_mask_max_size_list[stage], time_mask_count_list[stage],
                freq_mask_max_size_list[stage], freq_mask_count_list[stage],
                rng,
            )

        # Upstream: class_weights is a dict looked up with np.vectorize. np.where gives the same
        # (batch, 1) array. If upstream fixes the batch x batch broadcast below, make the
        # "upstream" branch match its fix.
        train_ground_truth = train_labels.reshape(-1, 1)
        class_weights = np.where(
            train_ground_truth == 1, positive_class_weight_list[stage], negative_class_weight_list[stage]
        )
        if class_weight_mode == "upstream":
            # (batch,) * (batch, 1) gives a batch x batch matrix, as in microwakeword.train.
            combined_weights = train_sample_weights * class_weights
        else:
            combined_weights = train_sample_weights * class_weights[:, 0]

        # Upstream: the same call, with results as a list instead of a dict.
        # Blocks until the step has run, so the batch's shared memory can be refilled after this.
        result = model.train_on_batch(
            train_fingerprints, train_ground_truth, sample_weight=combined_weights, return_dict=True
        )

        # Upstream prints a different progress line here, without speed or waiting time.
        progress_steps += 1
        is_last_step = training_step == training_steps_max
        is_eval_step = (training_step % config["eval_step_interval"]) == 0 or is_last_step
        if training_step % PROGRESS_EVERY == 0 or is_eval_step:
            elapsed = time.perf_counter() - progress_started
            print(
                f"Step {training_step}/{training_steps_max}: {progress_steps / elapsed:.0f} steps/s, "
                f"waiting for batches {100 * waited / elapsed:.0f}% of the time; "
                f"accuracy = {float(result['accuracy']):.3f}; recall = {float(result['recall']):.3f}; "
                f"precision = {float(result['precision']):.3f}; loss = {float(result['loss']):.4f}",
                end="\r",
                flush=True,
            )

        if is_eval_step:
            print()
            logging.info(
                "Step #%d: rate %f, accuracy %.2f%%, recall %.2f%%, precision %.2f%%, cross entropy %f",
                training_step, learning_rate, result["accuracy"] * 100, result["recall"] * 100,
                result["precision"] * 100, result["loss"],
            )

            with train_writer.as_default():
                tf.summary.scalar("loss", result["loss"], step=training_step)
                tf.summary.scalar("accuracy", result["accuracy"], step=training_step)
                tf.summary.scalar("recall", result["recall"], step=training_step)
                tf.summary.scalar("precision", result["precision"], step=training_step)
                tf.summary.scalar("auc", result["auc"], step=training_step)
                train_writer.flush()

            model.save_weights(os.path.join(config["train_dir"], "last_weights.weights.h5"))

            # Upstream: validate_nonstreaming(), reading the validation sets from disk.
            validation_started = time.perf_counter()
            nonstreaming_metrics = validate(model, validation)
            model.reset_metrics()  # reset metrics for next validation epoch of training
            # Upstream's message, plus how long validation took.
            logging.info(
                "Step %d (nonstreaming): Validation (%.1f s): recall at no faph = %.3f with cutoff %.2f, accuracy = %.2f%%, recall = %.2f%%, precision = %.2f%%, ambient false positives = %d, estimated false positives per hour = %.5f, loss = %.5f, auc = %.5f, average viable recall = %.9f",
                training_step,
                time.perf_counter() - validation_started,
                nonstreaming_metrics["recall_at_no_faph"] * 100,
                nonstreaming_metrics["cutoff_for_no_faph"],
                nonstreaming_metrics["accuracy"] * 100,
                nonstreaming_metrics["recall"] * 100,
                nonstreaming_metrics["precision"] * 100,
                nonstreaming_metrics["ambient_false_positives"],
                nonstreaming_metrics["ambient_false_positives_per_hour"],
                nonstreaming_metrics["loss"],
                nonstreaming_metrics["auc"],
                nonstreaming_metrics["average_viable_recall"],
            )

            with validation_writer.as_default():
                for name in ("loss", "accuracy", "recall", "precision", "recall_at_no_faph", "auc", "average_viable_recall"):
                    tf.summary.scalar(name, nonstreaming_metrics[name], step=training_step)
                validation_writer.flush()

            os.makedirs(os.path.join(config["train_dir"], "train"), exist_ok=True)

            model.save_weights(
                os.path.join(
                    config["train_dir"],
                    "train",
                    f"{int(best_minimization_quantity * 10000)}_weights_{training_step}.weights.h5",
                )
            )

            current_minimization_quantity = 0.0
            if config["minimization_metric"] is not None:
                current_minimization_quantity = nonstreaming_metrics[config["minimization_metric"]]
            current_maximization_quantity = nonstreaming_metrics[config["maximization_metric"]]
            current_no_faph_cutoff = nonstreaming_metrics["cutoff_for_no_faph"]

            # Save model weights if this is a new best model (same rule as microwakeword.train)
            if (
                (
                    current_minimization_quantity <= config["target_minimization"]  # achieved target false positive rate
                    and (
                        current_maximization_quantity > best_maximization_quantity  # either accuracy improved
                        or best_minimization_quantity > config["target_minimization"]  # or this is the first time we met the target
                    )
                )
                or (
                    current_minimization_quantity > config["target_minimization"]  # we haven't achieved our target
                    and current_minimization_quantity < best_minimization_quantity  # but we have decreased since the previous best
                )
                or (
                    current_minimization_quantity == best_minimization_quantity  # we tied a previous best
                    and current_maximization_quantity > best_maximization_quantity  # and we increased our accuracy
                )
            ):
                best_minimization_quantity = current_minimization_quantity
                best_maximization_quantity = current_maximization_quantity
                best_no_faph_cutoff = current_no_faph_cutoff

                # overwrite the best model weights
                model.save_weights(os.path.join(config["train_dir"], "best_weights.weights.h5"))
                checkpoint.save(file_prefix=checkpoint_prefix)

            logging.info(
                "So far the best minimization quantity is %.3f with best maximization quantity of %.5f%%; no faph cutoff is %.2f",
                best_minimization_quantity,
                (best_maximization_quantity * 100),
                best_no_faph_cutoff,
            )

            # Upstream: no such counters. Progress speed counts training steps only, not validation.
            progress_started = time.perf_counter()
            progress_steps = 0
            waited = 0.0

    # Save checkpoint after training
    checkpoint.save(file_prefix=checkpoint_prefix)
    model.save_weights(os.path.join(config["train_dir"], "last_weights.weights.h5"))


def main():
    import yaml
    from absl import logging

    from microwakeword import inception, mixednet, utils
    from microwakeword.model_train_eval import load_config

    # Upstream: the __main__ block of microwakeword.model_train_eval, up to train_model().
    flags = parse_flags(inception, mixednet)
    logging.set_verbosity(getattr(logging, flags.verbosity))
    model_module = {"inception": inception, "mixednet": mixednet}[flags.model_name]
    config = load_config(flags, model_module)

    # Upstream: microwakeword.model_train_eval.train_model, copied up to its train.train() call.
    try:
        os.makedirs(config["train_dir"])
        os.mkdir(config["summaries_dir"])
    except OSError:
        if not flags.restore_checkpoint:
            raise ValueError(f"model already exists in folder {config['train_dir']}") from None
    with open(os.path.join(config["train_dir"], "training_config.yaml"), "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    model = model_module.model(flags, config["training_input_shape"], config["batch_size"])
    logging.info(model.summary())
    utils.save_model_summary(model, config["train_dir"])

    logging.info("Reading the validation sets into memory")
    validation = ValidationData(config)
    logging.info(
        "Read %d validation spectrograms and %d ambient windows (%.1f GB) in %.0f s",
        len(validation.labels), len(validation.ambient), validation.gigabytes, validation.load_seconds,
    )

    logging.info("Starting %d batch workers", flags.workers)
    with fast_data.BatchLoader(
        config["features"], config["batch_size"], config["spectrogram_length"], flags.workers
    ) as loader:
        train(model, config, loader, validation, flags.class_weights)


if __name__ == "__main__":
    main()
