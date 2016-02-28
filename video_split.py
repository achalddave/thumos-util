class VideoSplitEnum:
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'


def get_video_split(video_name):
    """Returns one of VideoSplitEnum.{train,validation,test}"""
    if '_validation_' in video_name:
        return VideoSplitEnum.VALIDATION
    elif '_test_' in video_name:
        return VideoSplitEnum.TEST
    else:
        return VideoSplitEnum.TRAIN
