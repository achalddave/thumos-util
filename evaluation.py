"""Helpers for evaluating detections on THUMOS.

This file calls THUMOS' MATLAB scripts for the actual evaluation."""

from __future__ import division
import collections
import subprocess
import os

import numpy as np

Detection = collections.namedtuple('Detection',
                                   ['filename', 'start_seconds', 'end_seconds',
                                    'category', 'score'])


# TODO(achald): Is this the best place for this function?
def binarized_predictions_to_detection_tuples(binary_predictions):
    """
    Args:
        binary_predictions ((num_frames, 1) array): Binarized predictions for
            each frame in a video.

    Returns:
        detections (list of (start, end) tuples): Note that this is an
            un-Pythonic range, in that the end is part of the detection.

    >>> predictions = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1])
    >>> predictions = predictions.reshape((len(predictions), 1))
    >>> binarized_predictions_to_detection_tuples(predictions)
    [(3, 5), (8, 8)]
    """
    # Pad with 0s on both sides to detect changes at the beginning and end
    padded_binary_predictions = np.vstack(([0], binary_predictions, [0]))
    # state_changes[i] = +1 => padded_binary_predictions[i] = 0,
    #                          padded_binary_predictions[i+1] = 1
    #                       => binary_predictions[i-1] = 0,
    #                          binary_predictions[i] = 1
    # state_changes[i] = -1 => padded_binary_predictions[i] = 1,
    #                          padded_binary_predictions[i+1] = 0
    #                       => binary_predictions[i-1] = 1
    #                          binary_predictions[i] = 0
    state_changes = np.diff(padded_binary_predictions, axis=0)
    starts = np.where(state_changes == 1)[0]
    ends = np.where(state_changes == -1)[0] - 1  # Include end in detection
    return [(start, end) for start, end in zip(starts, ends)]


def dump_detections(detections, output_path):
    """
    Args:
        detections (list of Detection objects)
    """
    with open(output_path, 'wb') as f:
        for detection in detections:
            f.write(('{filename} {start_seconds} {end_seconds} '
                     '{category} {score}\n').format(**detection._asdict()))


def evaluate_detections(detections,
                        detections_output_path,
                        test_annotations_dir,
                        subset='val',
                        intersection_over_union_threshold=0.1,
                        call_max_f=False,
                        single_confidence_hack=False):
    """
    Run THUMOS' MATLAB evaluation script on detections.

    This simply calls the MATLAB script, and does not return anything.

    Args:
        detections (list of Detections): List containing Detection objects.
        detections_output_path (str): Path where detections will be output.
            This is then passed to the MATLAB evaluation script.
        test_annotations_dir (str): Path to test annotations dir.
        subset (str): 'val' or 'test'
        intersection_over_union_threshold (float)
        call_max_f (bool): As in call_matlab_evaluate.
        single_confidence_hack (bool): As in call_matlab_evaluate.
    """
    dump_detections(detections, detections_output_path)
    call_matlab_evaluate(detections_output_path, test_annotations_dir, subset,
                         intersection_over_union_threshold, call_max_f,
                         single_confidence_hack)


def call_matlab_evaluate(detections_output_path,
                         test_annotations_dir,
                         subset,
                         intersection_over_union_threshold,
                         call_max_f=False,
                         single_confidence_hack=False):
    """
    TODO(achald): This 'cd's into util, which doesn't really make sense.

    Args:
        detections_output_path (str): Path where detections will be output.
            This is then passed to the MATLAB evaluation script.
        test_annotations_dir (str): Path to test annotations dir.
        subset (str): 'val' or 'test'
        intersection_over_union_threshold (float)
        call_max_f (bool): If true, calls pr_at_max_f.m instead of
            TH14EvalDet.m.
        single_confidence_hack (bool): Only valid if call_max_f is True. Passed
            to pr_at_max_f as single_confidence_hack parameter.
    """
    if single_confidence_hack and not call_max_f:
        raise ValueError('single_confidence_hack can only be used in'
                         ' conjunction with call_max_f')

    detections_output_path = os.path.abspath(detections_output_path)
    command = ['matlab', '-nodesktop', '-nojvm', '-nosplash', '-r']
    matlab_commands = "cd util/thumos-eval;"
    if call_max_f:
        # Matlab uses true/false, Python uses True/False.
        single_confidence_hack_string = str(single_confidence_hack).lower()
        matlab_commands += (
                "pr_at_max_f("
                    "'{detections_output_path}',"
                    "'{test_annotations_dir}',"
                    "'{subset}',"
                    "{intersection_over_union_threshold},"
                    "{single_confidence_hack_string}"
                ");").format(**locals())
    else:
        matlab_commands += (
                "TH14evalDet("
                    "'{detections_output_path}',"
                    "'{test_annotations_dir}',"
                    "'{subset}',"
                    "{intersection_over_union_threshold}"
                ");").format(**locals())

    command.append(matlab_commands)
    subprocess.call(command, stdin=open(os.devnull, 'r'))


def compute_average_precision(groundtruth, predictions):
    """
    Computes average precision for a binary problem. This is based off of the
    PASCAL VOC implementation.

    From:
        https://github.com/achalddave/average-precision/blob/adafd5d29c0932a0002e4cf7ddbaf7970aefff5c/python/ap.py

    Args:
        groundtruth (array-like): Binary vector indicating whether each sample
            is positive or negative.
        predictions (array-like): Contains scores for each sample.

    Returns:
        Average precision.

    """
    predictions = np.asarray(predictions)
    groundtruth = np.asarray(groundtruth, dtype=float)

    sorted_indices = np.argsort(predictions)[::-1]
    predictions = predictions[sorted_indices]
    groundtruth = groundtruth[sorted_indices]
    # The false positives are all the negative groundtruth instances, since we
    # assume all instances were 'retrieved'. Ideally, these will be low scoring
    # and therefore in the end of the vector.
    false_positives = 1 - groundtruth

    tp = np.cumsum(groundtruth)      # tp[i] = # of positive examples up to i
    fp = np.cumsum(false_positives)  # fp[i] = # of false positives up to i

    num_positives = tp[-1]

    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    recalls = tp / num_positives

    # Append end points of the precision recall curve.
    precisions = np.concatenate(([0.], precisions))
    recalls = np.concatenate(([0.], recalls))

    # Find points where prediction score changes.
    prediction_changes = set(
        np.where(predictions[1:] != predictions[:-1])[0] + 1)

    num_examples = predictions.shape[0]

    # Recall and scores always "change" at the first and last prediction.
    c = prediction_changes | set([0, num_examples])
    c = np.array(sorted(list(c)), dtype=np.int)

    precisions = precisions[c[1:]]

    # Set precisions[i] = max(precisions[j] for j >= i)
    # This is because (for j > i), recall[j] >= recall[i], so we can always use
    # a lower threshold to get the higher recall and higher precision at j.
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    ap = np.sum((recalls[c[1:]] - recalls[c[:-1]]) * precisions)

    return ap
