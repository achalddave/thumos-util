import collections
import subprocess
import os

Detection = collections.namedtuple('Detection',
                                   ['filename', 'start_seconds', 'end_seconds',
                                    'category', 'score'])

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

    Returns: None
    """
    dump_detections(detections, detections_output_path)
    detections_output_path = os.path.abspath(detections_output_path)
    command = ['matlab', '-nodesktop', '-nosplash', '-r']
    matlab_commands = ('cd util/thumos-eval; '
                       'TH14evalDet('
                         '\'{detections_output_path}\','
                         '\'{test_annotations_dir}\','
                         '\'{subset}\','
                         '{intersection_over_union_threshold}'
                       ');'
                       'exit;').format(**locals())

    command.append(matlab_commands)
    subprocess.call(command, stdin=open(os.devnull, 'wb'))
