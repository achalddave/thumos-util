import yaml


def validate_config(provided_config_keys,
                    expected_config_keys,
                    example_config_file=None):
    if provided_config_keys != expected_config_keys:
        unknown_keys = provided_config_keys - expected_config_keys
        missing_keys = expected_config_keys - provided_config_keys
        error = 'Invalid config:'
        if unknown_keys:
            error += ' Config contains unknown keys: %s.' % unknown_keys
        if missing_keys:
            error += ' Config is missing keys: %s.' % missing_keys
        if example_config_file:
            error += ' See %s for an example config' % example_config_file
        raise ValueError(error)


def load_config(config_yaml_path, config_namedtuple, example_config_path=None):
    with open(config_yaml_path) as config_file:
        config_dict = yaml.load(config_file)
        validate_config(
            set(config_dict.keys()), set(config_namedtuple._fields),
            example_config_path)
        return config_namedtuple(**config_dict)
