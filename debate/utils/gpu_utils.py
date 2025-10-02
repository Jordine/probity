import torch
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """Information about available GPUs"""
    num_gpus: int
    gpu_names: List[str]
    gpu_memory: List[float]  # In GB
    recommended_assignment: Optional[Tuple[int, int]]  # (honest_gpu, dishonest_gpu)


def detect_gpu_configuration() -> GPUInfo:
    """Detect available GPUs and recommend model assignment"""
    
    if not torch.cuda.is_available():
        return GPUInfo(
            num_gpus=0,
            gpu_names=[],
            gpu_memory=[],
            recommended_assignment=None
        )
    
    num_gpus = torch.cuda.device_count()
    gpu_names = []
    gpu_memory = []
    
    for i in range(num_gpus):
        gpu_names.append(torch.cuda.get_device_name(i))
        # Get memory in GB
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        gpu_memory.append(total_memory)
    
    # Recommend GPU assignment
    if num_gpus == 0:
        recommended = None
    elif num_gpus == 1:
        recommended = (0, 0)  # Both models on same GPU
    elif num_gpus >= 2:
        # Assign to two different GPUs with most memory
        sorted_gpus = sorted(enumerate(gpu_memory), key=lambda x: x[1], reverse=True)
        recommended = (sorted_gpus[0][0], sorted_gpus[1][0])
    
    return GPUInfo(
        num_gpus=num_gpus,
        gpu_names=gpu_names,
        gpu_memory=gpu_memory,
        recommended_assignment=recommended
    )


def estimate_model_memory(model_name: str) -> float:
    """Estimate memory requirements for a model in GB"""
    
    model_name_lower = model_name.lower()
    
    # Rough estimates for common model sizes
    if "70b" in model_name_lower or "65b" in model_name_lower:
        return 140.0  # ~140GB for 70B model in bfloat16
    elif "34b" in model_name_lower or "30b" in model_name_lower:
        return 68.0   # ~68GB for 34B model
    elif "13b" in model_name_lower:
        return 26.0   # ~26GB for 13B model
    elif "8b" in model_name_lower or "7b" in model_name_lower:
        return 16.0   # ~16GB for 7-8B model
    elif "3b" in model_name_lower:
        return 6.0    # ~6GB for 3B model
    else:
        return 20.0   # Default estimate


def can_fit_models(gpu_info: GPUInfo, model1_name: str, model2_name: str) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """
    Check if models can fit on available GPUs and return assignment.
    Returns (can_fit, (gpu1, gpu2)) where gpu1 and gpu2 are device indices.
    """
    
    if gpu_info.num_gpus == 0:
        return False, None
    
    mem_required1 = estimate_model_memory(model1_name)
    mem_required2 = estimate_model_memory(model2_name)
    
    # Add overhead for activations and generation
    overhead = 4.0  # 4GB overhead per model
    mem_required1 += overhead
    mem_required2 += overhead
    
    if gpu_info.num_gpus == 1:
        # Check if both fit on single GPU
        if gpu_info.gpu_memory[0] >= (mem_required1 + mem_required2):
            return True, (0, 0)
        else:
            return False, None
    
    # Multiple GPUs - try to distribute
    for i in range(gpu_info.num_gpus):
        for j in range(gpu_info.num_gpus):
            if i == j:
                # Same GPU - check combined memory
                if gpu_info.gpu_memory[i] >= (mem_required1 + mem_required2):
                    return True, (i, j)
            else:
                # Different GPUs - check individual memory
                if gpu_info.gpu_memory[i] >= mem_required1 and gpu_info.gpu_memory[j] >= mem_required2:
                    return True, (i, j)
    
    return False, None