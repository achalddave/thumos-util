"""Tools for parsing THUMOS annotation files."""

import collections
import csv
import os
from os import path
from math import ceil, floor

from .evaluation import Detection
from .video_tools.util.annotation import Annotation


def load_class_mapping(class_list_path):
    """Load a class mapping from a file.

    Each line should be of the form "<class_index> <class_name>".

    Returns:
        mapping (OrderedDict): Maps category index to category name. The order
            of insertion is the same as the order of lines in the file.
    """
    mapping = collections.OrderedDict()
    with open(class_list_path) as f:
        for line in f:
            details = line.strip().split(' ')
            mapping[int(details[0])] = ' '.join(details[1:])
    return mapping


def parse_frame_info_file(video_frames_info_path):
    """Parse file containing FPS and num frames for each video.

    Each line should be of the form "<video_name>,<fps>,<num_frames>

    Returns:
        video_frame_info (dict): Maps filename to (fps, num_frames).
    """
    video_frame_info = dict()
    with open(video_frames_info_path) as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip headers
        for row in reader:
            video_frame_info[row[0]] = (float(row[1]), int(row[2]))
    return video_frame_info


def parse_video_fps_file(video_fps_file):
    """(DEPRECATED): Use parse_frame_info_file.

    Parse video frame info file.

    Each line should be of the form "<video_name>,<fps>[,<any_other_fields>]

    Returns:
        video_fps (dict): Maps filename to fps.
    """
    video_fps = dict()
    with open(video_fps_file) as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip headers
        for row in reader:
            video_fps[row[0]] = float(row[1])
    return video_fps



def parse_annotation_file(annotation_path, video_fps, category):
    """Parse THUMOS annotations.

    Args:
        annotation_path (str): Path to annotation file. Each line should be of
            the form: "<video_name> <start_time> <end_time>" or
            "<video_name>  <start_time> <end_time>" (notice one extra space).
        video_fps (dict): Maps video file names to frames per second.
        category (str): Category that these annotations belong to.

    Returns:
        annotations (list of Annotation)
    """
    annotations = []
    with open(annotation_path) as f:
        for line in f:
            # The THUMOS temporal labels have *two spaces* between the first two
            # fields (unfortunately), while the MultiTHUMOS labels have one
            # space.
            details = line.strip().split(' ')
            if details[1] == '':  # There were two spaces after the first field.
                details.pop(1)
            filename, start, end = details
            start, end = float(start), float(end)
            current_fps = video_fps[filename]
            start_frame = floor(start * current_fps)
            end_frame = ceil(end * current_fps)
            annotations.append(Annotation(**{'filename': filename,
                                             'start_seconds': start,
                                             'end_seconds': end,
                                             'start_frame': start_frame,
                                             'end_frame': end_frame,
                                             'frames_per_second': current_fps,
                                             'category': category}))
    return annotations


def load_thumos_annotations(annotations_dir, video_frames_info):
    annotation_paths = ["%s/%s" % (annotations_dir, x)
                        for x in os.listdir(annotations_dir)]

    # Maps video name to frames per second
    video_fps = parse_video_fps_file(video_frames_info)
    annotations = []
    for annotation_path in annotation_paths:
        category = path.splitext(path.basename(annotation_path))[0]
        if category.endswith('_val'):
            category = category[:-len('_val')]
        elif category.endswith('_test'):
            category = category[:-len('_test')]
        annotation_details = parse_annotation_file(annotation_path, video_fps,
                                                   category)
        annotations.extend(annotation_details)

    return annotations


def load_detections(detections_path):
    """Load detections from the format used by THUMOS '14's evaluation script.

    Args:
        detections_path (str): Path to a file with lines of the form
            <video_name> <start_second> <end_second> <category_index> <score>

    Returns:
        detections (list of Detection)
    """
    detections = []
    with open(detections_path) as f:
        for line in f:
            details = line.strip().split(' ')
            video_name = details[0]
            start_seconds = float(details[1])
            end_seconds = float(details[2])
            category_index = int(details[3])
            score = float(details[4])
            detections.append(Detection(video_name, start_seconds, end_seconds,
                                        category_index, score))
    return detections
