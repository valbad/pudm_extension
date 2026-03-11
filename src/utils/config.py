"""JSON config reading utilities."""
import json
import copy


def replace_list_with_string_in_a_dict(dictionary):
    """Convert lists to their string representation (for printing)."""
    for key in dictionary.keys():
        if isinstance(dictionary[key], list):
            dictionary[key] = str(dictionary[key])
        if isinstance(dictionary[key], dict):
            dictionary[key] = replace_list_with_string_in_a_dict(dictionary[key])
    return dictionary


def restore_string_to_list_in_a_dict(dictionary):
    """Restore string representations of lists back to actual lists."""
    for key in dictionary.keys():
        if isinstance(dictionary[key], str):
            try:
                evaluated = eval(dictionary[key])
                if isinstance(evaluated, list):
                    dictionary[key] = evaluated
            except:
                pass
        if isinstance(dictionary[key], dict):
            dictionary[key] = restore_string_to_list_in_a_dict(dictionary[key])
    return dictionary


def load_config(config_path):
    """Load and parse a JSON config file, restoring string-encoded lists."""
    with open(config_path) as f:
        config = json.loads(f.read())
    config = restore_string_to_list_in_a_dict(config)
    return config


def print_config(config):
    """Pretty-print a config dict."""
    printable = replace_list_with_string_in_a_dict(copy.deepcopy(config))
    print(json.dumps(printable, indent=4))
