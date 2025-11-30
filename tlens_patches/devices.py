"""Devices.

Utilities to get the correct device, and assist in distributing model layers across multiple
devices.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

import transformer_lens

AvailableDeviceMemory = list[tuple[int, int]]
"""
This type is passed around between different CUDA memory operations.
The first entry of each tuple will be the device index.
The second entry will be how much memory is currently available.
"""


def calculate_available_device_cuda_memory(i: int) -> int:
    """Calculates how much memory is available at this moment for the device at the indicated index

    Args:
        i (int): The index we are looking at

    Returns:
        int: How memory is available
    """
    total = torch.cuda.get_device_properties(i).total_memory
    allocated = torch.cuda.memory_allocated(i)
    return total - allocated


def determine_available_memory_for_available_devices(max_devices: int) -> AvailableDeviceMemory:
    """Gets all available CUDA devices with their current memory calculated

    Returns:
        AvailableDeviceMemory: The list of all available devices with memory precalculated
    """
    devices = []
    for i in range(max_devices):
        devices.append((i, calculate_available_device_cuda_memory(i)))

    return devices


def sort_devices_based_on_available_memory(devices: AvailableDeviceMemory) -> AvailableDeviceMemory:
    """Sorts all available devices with devices with the most available memory returned first

    Args:
        devices (AvailableDeviceMemory): All available devices with memory calculated

    Returns:
        AvailableDeviceMemory: The same list of passed through devices sorted with devices with most
        available memory first
    """
    return sorted(devices, key=lambda x: x[1], reverse=True)


def get_best_available_cuda_device(max_devices: Optional[int] = None) -> torch.device:
    """Gets whichever cuda device has the most available amount of memory for use

    Raises:
        EnvironmentError: If there are no available devices, this will error out

    Returns:
        torch.device: The specific device that should be used
    """
    max_devices = max_devices if max_devices is not None else torch.cuda.device_count()
    devices = determine_available_memory_for_available_devices(max_devices)

    if len(devices) <= 0:
        raise EnvironmentError(
            "TransformerLens has been configured to use CUDA, but no available devices are present"
        )

    sorted_devices = sort_devices_based_on_available_memory(devices=devices)

    return torch.device("cuda", sorted_devices[0][0])


def get_best_available_device(cfg: "transformer_lens.HookedTransformerConfig") -> torch.device:
    """Gets the best available device to be used based on the passed in arguments"""
    assert cfg.device is not None
    # Check for explicit GPU assignment
    if hasattr(cfg, "gpu_indices") and cfg.gpu_indices is not None and len(cfg.gpu_indices) > 0:
        return torch.device(f"cuda:{cfg.gpu_indices[0]}")
    # Original logic
    device = torch.device(cfg.device)
    if device.type == "cuda" and cfg.n_devices > 1:
        return get_best_available_cuda_device(cfg.n_devices)
    else:
        return device


def get_device_for_block_index(
    index: int,
    cfg: "transformer_lens.HookedTransformerConfig",
    device: Optional[Union[torch.device, str]] = None,
):
    """Determine the device for a given layer index based on the model configuration."""
    # Check for explicit GPU assignment
    if hasattr(cfg, "gpu_indices") and cfg.gpu_indices is not None and len(cfg.gpu_indices) > 0:
        gpu_list = cfg.gpu_indices[:cfg.n_devices]
        if len(gpu_list) == 1:
            return torch.device(f"cuda:{gpu_list[0]}")
        layers_per_device = cfg.n_layers // len(gpu_list)
        remainder = cfg.n_layers % len(gpu_list)
        device_idx = 0
        layer_count = 0
        for i in range(len(gpu_list)):
            device_layers = layers_per_device + (1 if i < remainder else 0)
            if index < layer_count + device_layers:
                device_idx = i
                break
            layer_count += device_layers
        return torch.device(f"cuda:{gpu_list[device_idx]}")
    # Original logic
    assert cfg.device is not None
    layers_per_device = cfg.n_layers // cfg.n_devices
    if device is None:
        device = cfg.device
    device = torch.device(device)
    if device.type == "cpu":
        return device
    device_index = (device.index or 0) + (index // layers_per_device)
    return torch.device(device.type, device_index)

def move_to_and_update_config(
    model: Union[
        "transformer_lens.HookedTransformer",
        "transformer_lens.HookedEncoder",
        "transformer_lens.HookedEncoderDecoder",
    ],
    device_or_dtype: Union[torch.device, str, torch.dtype],
    print_details=True,
):
    """
    Wrapper around `to` that also updates `model.cfg`.
    """
    if isinstance(device_or_dtype, torch.device):
        model.cfg.device = device_or_dtype.type
        if print_details:
            print("Moving model to device: ", model.cfg.device)
    elif isinstance(device_or_dtype, str):
        model.cfg.device = device_or_dtype
        if print_details:
            print("Moving model to device: ", model.cfg.device)
    elif isinstance(device_or_dtype, torch.dtype):
        model.cfg.dtype = device_or_dtype
        if print_details:
            print("Changing model dtype to", device_or_dtype)
        # change state_dict dtypes
        for k, v in model.state_dict().items():
            model.state_dict()[k] = v.to(device_or_dtype)
    return nn.Module.to(model, device_or_dtype)
