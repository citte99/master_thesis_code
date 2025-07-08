import torch

"""
This function was moved to shared utils.
"""
# def recursive_to_tensor(value, device):
#     # If value is a dictionary, process each key/value pair recursively.
#     if isinstance(value, dict):
#         return {k: recursive_to_tensor(v, device) for k, v in value.items()}
    
#     # If value is a list, process each element recursively.
#     elif isinstance(value, list):
#         return [recursive_to_tensor(item, device) for item in value]
    
#     # If value is a tuple, process each element recursively and return a tuple.
#     elif isinstance(value, tuple):
#         return tuple(recursive_to_tensor(item, device) for item in value)
    
#     # Try to convert numerical values (or lists/arrays of numbers) to a tensor.
#     else:
#         try:
#             # torch.as_tensor will convert if possible.
#             return torch.as_tensor(value, device=device)
#         except Exception:
#             # If conversion fails (e.g., for strings), return the original value.
#             return value
        




