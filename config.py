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
