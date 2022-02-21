""" Utilities for parsing YAML configuration files. Supports variable substitution and includes. """
import yaml
import functools
import io
import string
import os
from braingeneers.utils import smart_open


def parse_yaml_cfg(yaml_file_or_str: (str, io.IOBase, list, tuple), is_file: bool = True,
                   working_dir: str = None, includes: (tuple, list, str) = (),
                   replace: bool = True):
    """
    Parse a yaml file with additional features:
        - Global variables will be replaced using bash style variable substitution when this file is parsed.
        - Variables used in from this file (use bracket notation for hierarchical variable names, ex: modelparams[sample_width]) or the users environment.
        - When there is a conflict the users environment takes precedence over yaml, and values passed to kwargs_overrides over environment.
        - Bash variable substitution will follow type casting rules of YAML.

    Overrides:
        Any number of 1-line YAML strings can be included and will take precedence over the configuration existing
        in the file. Use https://onlineyamltools.com/minify-yaml to minify YAML to a single line.

    :param yaml_file_or_str: the name of a yaml file (local or S3) or a yaml text string, or a list/tuple of such objects
    :param is_file: True if yaml_file_or_str is a file, false to pass a raw yaml string, applies to all values if
        yaml_file_or_str is a list or tuple.
    :param working_dir: Not normally set by the user. This is the base directory from which relative paths to
        include files will be resolved, it defaults to python working directory. Any INCLUDE statement in the YAML
        will be path-relative to the file containing the INCLUDE statement regardless of how this value is set.
    :param includes: a tuple, list, or single 1-line YAML string(s) to include.
        Use https://onlineyamltools.com/minify-yaml.
        Use this parameter to override a specific value in the yaml file, such as with a hyperparameter search.
    :param replace: Enables or disables {variable} replacement. Defaults to True.
    :return: object form of the yaml
    """
    includes_tup = includes if isinstance(includes, (tuple, list)) else (includes,) if isinstance(includes, str) else ()

    # load from a file, iostream, raw yaml string, or recurse into lists
    if isinstance(yaml_file_or_str, (list, tuple)):
        yaml_objs = [parse_yaml_cfg(y, is_file=is_file, working_dir=working_dir, replace=False) for y in yaml_file_or_str]
        yaml_obj = functools.reduce(merge, yaml_objs)
    elif is_file:
        if isinstance(yaml_file_or_str, io.IOBase):
            yaml_raw = yaml_file_or_str.read()
        else:
            base_path = working_dir if working_dir is not None and not yaml_file_or_str.startswith('s3://') else ''
            with smart_open.open(os.path.join(base_path, yaml_file_or_str), 'r') as f:
                yaml_raw = f.read()
        yaml_obj = yaml.safe_load(yaml_raw)
    else:
        yaml_raw = yaml_file_or_str
        yaml_obj = yaml.safe_load(yaml_raw)

    # perform variable replacement by iterating through the lists and dictionaries in the yaml
    # yaml_obj = recursive_str_format(yaml_obj, yaml_obj)

    # parse INCLUDE statements in the yaml file
    if yaml_obj is not None and 'INCLUDE' in yaml_obj:
        assert isinstance(yaml_obj['INCLUDE'], list)
        relative_path = os.path.dirname(yaml_file_or_str) if is_file else os.getcwd()
        parsed_includes = [parse_yaml_cfg(y, is_file=True, working_dir=relative_path, replace=False) for y in yaml_obj['INCLUDE']]
        del yaml_obj['INCLUDE']
        yaml_obj_merged = functools.reduce(merge, [yaml_obj] + parsed_includes)
    else:
        yaml_obj_merged = yaml_obj

    # parse optional include snippets into the yaml
    for include in includes_tup:
        if include == '':
            continue
        include_yaml = parse_yaml_cfg(include, is_file=False, working_dir=working_dir, replace=False)
        yaml_obj_merged = merge(yaml_obj_merged, include_yaml)

    # perform a variable replacement only after all loads have completed
    if replace:
        yaml_obj_merged = recursive_str_format(yaml_obj_merged, yaml_obj_merged)

    return yaml_obj_merged


def recursive_str_format(target, keys):
    """
    Recursively formats strings in nested dict & list structures (e.g. yaml objects) using EnhancedFormatter.
    This function enables variable substitution in yaml configuration files and is used by parse_yaml_cfg.

    :param target: A potentially nested data structure of dicts, lists, and scalars, e.g. a parsed yaml object, the
        function will iterate over all values and perform string.format() replacements using keys.
    :param keys: A potentially nested data structure of
    :return: a new (deep) copy of target, all dict and list objects are deep copied, other objects are assumed immutable
        (e.g. yaml scalars) and are not copied.
    """
    formatter = EnhancedFormatter()

    if isinstance(target, dict):
        result = dict([(k, recursive_str_format(v, keys)) for k, v in target.items()])
    elif isinstance(target, list):
        result = [recursive_str_format(v, keys) for v in target]
    elif isinstance(target, str):
        result = yaml.safe_load(formatter.format(target, **keys))
    else:
        result = target

    return result


class EnhancedFormatter(string.Formatter):
    """
    A custom formatter proving 2 additional conversion functions for lower and upper case.
    Usage: {!l} and {!u} or using field name or index: {field_name!l} or {0!u}
    For more formatting documentation follow conversion specification in docs:
        https://docs.python.org/3/library/string.html#format-string-syntax

    """
    def __init__(self, default='{{{0}}}'):
        self.default = default

    def get_value(self, key, args, kwds):
        if isinstance(key, str):
            return kwds.get(key, self.default.format(key))
        else:
            return string.Formatter.get_value(self, key, args, kwds)

    def convert_field(self, value, conversion):
        if conversion == 'l':
            return str(value).lower()
        elif conversion == 'u':
            return str(value).upper()
        else:
            return super(EnhancedFormatter, self).convert_field(value, conversion)


def merge(a, b, path=None):
    """
    merges b into a with b taking precedence
    Source: https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
    """
    if path is None:
        path = []

    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                # raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
                a[key] = b[key]  # b overrides a
        else:
            a[key] = b[key]

    return a
