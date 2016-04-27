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
    >>> binarized_predictions_to_detections(predictions)
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
                        call_max_f=False):
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
    """
    dump_detections(detections, detections_output_path)
    call_matlab_evaluate(detections_output_path, test_annotations_dir, subset,
                         intersection_over_union_threshold, call_max_f)


def call_matlab_evaluate(detections_output_path, test_annotations_dir, subset,
                         intersection_over_union_threshold, call_max_f=True):
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
    """
    detections_output_path = os.path.abspath(detections_output_path)
    command = ['matlab', '-nodesktop', '-nojvm', '-nosplash', '-r']
    function_name = 'pr_at_max_f' if call_max_f else 'TH14evalDet'
    matlab_commands = ('cd util/thumos-eval; '
                       '{function_name}('
                         '\'{detections_output_path}\','
                         '\'{test_annotations_dir}\','
                         '\'{subset}\','
                         '{intersection_over_union_threshold}'
                       ');'
                       'exit;').format(**locals())

    command.append(matlab_commands)
    subprocess.call(command, stdin=open(os.devnull, 'r'))


def compute_average_precision(groundtruth, predictions):
    """
    Computes average precision for a binary binary problem.

    See:
    <https://en.m.wikipedia.org/wiki/Information_retrieval#Average_precision>

    This is what sklearn.metrics.average_precision_score should do, but it is
    broken:
    https://github.com/scikit-learn/scikit-learn/issues/5379
    https://github.com/scikit-learn/scikit-learn/issues/6377

    Args:
        groundtruth (array-like): Binary vector indicating whether each sample
            is positive or negative.
        predictions (array-like): Contains scores for each sample.

    Returns:
        Average precision.
    """
    sorted_indices = sorted(
        range(predictions.size),
        key=lambda x: predictions[x],
        reverse=True)

    average_precision = 0
    true_positives = 0
    for num_guesses, index in enumerate(sorted_indices):
        # The second clause here is crucial; otherwise, we give points to the
        # predictor for samples it did not retrieve. See
        # <http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html>
        if groundtruth[index] and predictions[index] > 0:
            true_positives += 1
            precision = true_positives / (num_guesses + 1)
            average_precision += precision
    if sum(predictions) == 0:
        print 'No predictions!'
    average_precision /= sum(groundtruth)
    return average_precision
