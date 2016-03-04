import collections
import json
import numpy as np

Annotation = collections.namedtuple(
    'Annotation', ['filename', 'start_frame', 'end_frame', 'start_seconds',
                   'end_seconds', 'frames_per_second', 'category'])

"""
7 BaseballPitch
9 BasketballDunk
12 Billiards
21 CleanAndJerk
22 CliffDiving
23 CricketBowling
24 CricketShot
26 Diving
31 FrisbeeCatch
33 GolfSwing
36 HammerThrow
40 HighJump
45 JavelinThrow
51 LongJump
68 PoleVault
79 Shotput
85 SoccerPenalty
92 TennisSwing
93 ThrowDiscus
97 VolleyballSpiking
"""
THUMOS_indices = [7, 9, 12, 21, 22, 23, 24, 26, 31, 33, 36, 40, 45, 51, 68, 79,
                  85, 92, 93, 97]


def load_annotations_json(annotations_json_path, category=None):
    """Load annotations into a dictionary mapping filenames to annotations."""
    with open(annotations_json_path) as f:
        annotations_list = json.load(f)
    annotations = collections.defaultdict(list)
    # Extract annotations for category
    for annotation in annotations_list:
        annotation = Annotation(**annotation)
        annotations[annotation.filename].append(annotation)
    if category is not None:
        annotations = filter_annotations_by_category(annotations, category)
    return annotations


def filter_annotations_by_category(annotations, category):
    """
    Return only annotations that belong to category.

    Args:
        annotations (dict): Maps filenames to list of Annotations.
    Returns:
        filtered_annotations (dict): Maps filenames to list of Annotations.

    >>> SimpleAnnotation = collections.namedtuple(
    ...         'SimpleAnnotation', ['category'])
    >>> annotations = {'file1': [SimpleAnnotation('class1'),
    ...                          SimpleAnnotation('class2')],
    ...                'file2': [SimpleAnnotation('class2')]}
    >>> filtered = filter_annotations_by_category(annotations, 'class1')
    >>> filtered.keys()
    ['file1']
    >>> len(filtered['file1'])
    1
    """
    filtered_annotations = {}
    for filename, annotations in annotations.items():
        filtered = [x for x in annotations if x.category == category]
        if filtered:
            filtered_annotations[filename] = filtered
    return filtered_annotations


def calculate_duration_mean_std(annotation_groundtruth):
    """
    Args:
        annotation_groundtruth (dict): Maps filenames to groundtruth
            annotations.

    Returns:
        mean, stderr of durations
    """
    durations = []
    for annotations in annotation_groundtruth.values():
        durations.extend([annotation.end_frame - annotation.start_frame + 1
                          for annotation in annotations])
    durations = np.array(durations)
    return durations.mean(), durations.std()
