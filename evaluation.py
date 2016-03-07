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
                        intersection_over_union_threshold=0.1):
    """
    Run THUMOS' MATLAB evaluation script on detections.

    This simply calls the MATLAB script, and does not return anything.

    Args:
        detections (list of Detections): List containing Detection objects.
        detections_output_path (str): Path where detections will be output. This
            is then passed to the MATLAB evaluation script.
        test_annotations_dir (str): Path to test annotations dir.
        subset (str): 'val' or 'test'
        intersection_over_union_threshold (float)
    """
    dump_detections(detections, detections_output_path)
    call_matlab_evaluate(detections_output_path, test_annotations_dir, subset,
                         intersection_over_union_threshold)

def call_matlab_evaluate(detections_output_path, test_annotations_dir, subset,
                         intersection_over_union_threshold):
    """
    Args:
        detections_output_path (str): Path where detections will be output. This
            is then passed to the MATLAB evaluation script.
        test_annotations_dir (str): Path to test annotations dir.
        subset (str): 'val' or 'test'
        intersection_over_union_threshold (float)
    """
    detections_output_path = os.path.abspath(detections_output_path)
    command = ['matlab', '-nodesktop', '-nojvm', '-nosplash', '-r']
    matlab_commands = ('cd util/thumos-eval; '
                       'TH14evalDet('
                         '\'{detections_output_path}\','
                         '\'{test_annotations_dir}\','
                         '\'{subset}\','
                         '{intersection_over_union_threshold}'
                       ');'
                       'exit;').format(**locals())

    command.append(matlab_commands)
    subprocess.call(command, stdin=open(os.devnull, 'r'))
