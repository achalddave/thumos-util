import collections
import json

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
        if category is not None and annotation.category != category:
            continue
        annotations[annotation.filename].append(annotation)
    return annotations
