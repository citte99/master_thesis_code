import torch


def recursive_to_tensor(value, device, datatype=torch.float32):
    # If value is a dictionary, process each key/value pair recursively.
    if isinstance(value, dict):
        return {k: recursive_to_tensor(v, device) for k, v in value.items()}

    # If value is a list:
    elif isinstance(value, list):
        # If the list is non-empty and every element is a number, convert the entire list into a tensor.
        if value and all(isinstance(item, (int, float)) for item in value):
            return torch.as_tensor(value, device=device, dtype=datatype)
        else:
            # Otherwise, process each element recursively.
            return [recursive_to_tensor(item, device) for item in value]

    # If value is a tuple, process each element recursively and return a tuple.
    elif isinstance(value, tuple):
        return tuple(recursive_to_tensor(item, device) for item in value)

    # Try to convert the value to a tensor if possible.
    else:
        try:
            return torch.as_tensor(value, device=device, dtype=datatype)
        except Exception:
            return value