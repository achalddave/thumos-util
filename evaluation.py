# TODO(achald): It would be nice to be able to call the MATLAB functions from my
# code.

import collections

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

def evaluate(detections, subset='val', ):
    """
    Args:
        detections (list of dicts): Each detection should contain the fields
            'filename', 'start_sec', 'end_sec', 'category', 'score'
        subset (str): 'val' or 'test'
        intersection_over_union_threshold (float)

    Returns:
        precision_recalls (list of (prec
    """
