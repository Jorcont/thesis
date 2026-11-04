import yaml
from pathlib import Path
from typing import List, Dict
from functools import reduce

ERROR_PREFIX: str = "There is differences in the current config and the loading config."

def check_other_config(current_config: str | Path, other_config: str | Path):
    if not Path(other_config).exists(): # Only compare if it actually exists
        return

    with open(current_config, 'r') as file:
        current = yaml.safe_load(file)

    with open(other_config, 'r') as file:
        other = yaml.safe_load(file)

    check_values(current, other, "algorithm")
    check_dict_equality(current, other, "model_config", allow_missing=["checkpoint", "local_checkpoint"])

def check_values(expected_data, actual_data, keys: str | List[str]):
    expected_val = nested_get(expected_data, keys)
    actual_val = nested_get(actual_data, keys)

    if expected_val != actual_val:
        raise ValueError(f"{ERROR_PREFIX} Expected {expected_val} (got {actual_val}) instead")

def check_dict_equality(current, other, keys: str | List[str], ignore_keys: List[str] = [], allow_missing: List[str] = []):
    errors = nested_dict_equal_all(nested_get(current, keys, default={}), nested_get(other, keys, default={}),
                                   ignore_keys=ignore_keys, allow_missing=allow_missing)

    if len(errors) > 0:
        raise ValueError(f"{ERROR_PREFIX} Differences found in current and loaded external config:\n" + "\n".join(str(e) for e in errors))

def nested_get(d: Dict, keys: str | List[str], default=None):
    if isinstance(keys, str):
        return d.get(keys, default)

    try:
        return reduce(lambda acc, key: acc[key], keys, d)
    except (KeyError, TypeError):
        return default

def nested_dict_equal_all(current, other, ignore_keys: List[str] = [], allow_missing: List[str] = []) -> List[str]:
    """Collect all differences, then raise."""
    errors = []

    keys = set(current.keys()) | set(other.keys())
    keys = [k for k in keys if k not in ignore_keys]

    for key in keys:
        if key not in current:
            if key not in allow_missing:
                errors.append(f"Key '{str(key)}' is not in the current config, but in the other loaded config")
            continue
        if key not in other:
            if key not in allow_missing:
                errors.append(f"Key '{str(key)}' is in the current config, but in the other loaded config")
            continue

        v1, v2 = current[key], other[key]

        if type(v1) != type(v2):
            errors.append(f"Type mismatch at '{str(key)}': {type(v1).__name__} vs {type(v2).__name__}")
        elif isinstance(v1, dict):
            errors.extend(nested_dict_equal_all(v1, v2))
        elif v1 != v2:
            errors.append(f"Value mismatch at '{str(key)}': {v1!r} != {v2!r}")
    return errors