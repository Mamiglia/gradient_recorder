import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call
from typing import Any, Dict, List, Optional, Tuple, Collection
from pathlib import Path
from functools import cache
from tqdm import tqdm

# ---------------------------------------
# Core Computation Functions
# ---------------------------------------
# This is the most important function in this script. It computes the gradients, loss, and logits for a given batch of data.
# The function is designed to work with PyTorch's autograd system and uses the `vmap` function from the `torch.func` module.
# The `vmap` function is used to vectorize the computation over the batch dimension, which allows us to compute gradients for each sample in the batch.

# The function returns sample-level gradients, loss, logits, and parameter-level gradients.
# The sample-level gradients are similar to Grad-CAM, which can be used to visualize the importance of different parts of the input for the model's prediction.
# The parameter-level gradients are the gradients of the loss with respect to the model's parameters.

# If you have any questions ask Tao!
def compute_loss_and_gradients(
    x: torch.Tensor, y: torch.Tensor, model: nn.Module, 
    params: Dict[str, torch.Tensor] = None, buffers: Dict[str, torch.Tensor] = None, 
    chunk_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute gradients, loss, and logits for a given batch of data using a specified model.

    Parameters:
    - x (torch.Tensor): The input tensor for the batch.
    - y (torch.Tensor): The target tensor for the batch.
    - model (nn.Module): The neural network model.
    - params (Dict[str, torch.Tensor], optional): A dictionary of model parameters. If not provided, the function will extract parameters from the model.
    - buffers (Dict[str, torch.Tensor], optional): A dictionary of model buffers. If not provided, the function will extract buffers from the model.
    - chunk_size (int, optional): The size of chunks for batched gradient computation. Default is 32.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]: A tuple containing:
      - grads_sample (torch.Tensor): Sample-level gradients.
      - loss (torch.Tensor): Loss for the batch.
      - logits (torch.Tensor): Logits for the batch.
      - grads_params (Dict[str, torch.Tensor]): Parameter-level gradients.
    """
    if params is None or buffers is None:
        params = {k: v.detach() for k, v in model.named_parameters()}
        buffers = {k: v.detach() for k, v in model.named_buffers()}
    
    compute_loss_grad_fn = construct_batched_gradient_computation(chunk_size)
    (grads_sample, grads_params), (loss, logits) = compute_loss_grad_fn(x, y, params, buffers, model)
    return grads_sample.cpu(), loss.cpu(), logits.cpu(), [{k: v[i].cpu() for k, v in grads_params.items()} for i in range(x.size(0))]

def baseline_loss_fn(x: torch.Tensor, y: torch.Tensor, params: Collection[torch.Tensor], 
                     buffer: Collection[torch.Tensor], model: nn.Module) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    This is a simple loss function that computes the cross-entropy loss and the logits for a given batch of data.
    For more complex loss functions, you can define your own loss function and use it in the `compute_loss_and_gradients` function.
    """
    logits = functional_call(model, (params, buffer), (x.unsqueeze(0),))
    loss = F.cross_entropy(logits, y.unsqueeze(0))
    return loss, (loss, logits.squeeze())

@cache
def construct_batched_gradient_computation(chunk_size: int, loss_fn=None):
    """
    Construct a function that computes gradients, loss, and logits for a batch of data using a specified loss function.
    Usage:
    def loss_fn(x, y, params, buffer, model):
        # Define your loss function here
        pass
        
    compute_loss_grad_fn = construct_batched_gradient_computation(chunk_size, loss_fn)
    (grads_sample, grads_params), (loss, logits) = compute_loss_grad_fn(x, y, params, buffer, model)
    
    Parameters:
    - chunk_size (int): The size of chunks for batched gradient computation.
    - loss_fn (Callable, optional): The loss function to use for computing the loss and logits. Default is `baseline_loss_fn`.
    Returns:
    - Callable: A function that computes gradients, loss, and logits for a batch of data.
        - The function takes the following arguments:
            - x (torch.Tensor): The input tensor for the batch.
            - y (torch.Tensor): The target tensor for the batch.
            - params (Collection[torch.Tensor]): A collection of model parameters.
            - buffer (Collection[torch.Tensor]): A collection of model buffers.
            - model (nn.Module): The neural network model.
        - The function returns a tuple containing:
            - grads_sample (torch.Tensor): Sample-level gradients.
            - grads_params (Dict[str, torch.Tensor]): Parameter-level gradients.
            - loss (torch.Tensor): Loss for the batch.
            - logits (torch.Tensor): Logits for the batch.    
    """
    loss_fn = loss_fn or baseline_loss_fn
    return vmap(              # vmap over the batch dimension
            grad(                               # gradient over the model parameters and the input
                loss_fn, 
                argnums=(0, 2), has_aux=True    # argnums=(0, 2) means that we want to differentiate with respect to the first and third argument
                                                # has_aux=True means that the function returns also the loss and the logits
            ), in_dims=(0, 0, None, None, None),# in_dims specifies which is the batch dimension of x, y
            chunk_size=chunk_size)

# ---------------------------------------
# Saving Functions
# ---------------------------------------

def save_grads_sample(path: Path, grads_sample: torch.Tensor, batch_idx: int = 0):
    """Save sample-level gradients to disk."""
    for i, grad in enumerate(grads_sample):
        torch.save(grad, path / f'grad_sample/{batch_idx}_{i}.pt', pickle_protocol=4)

def save_grads_params(path: Path, grads_params: List[Dict[str, torch.Tensor]], batch_idx: int = 0):
    """Save parameter-level gradients to disk."""
    for i, grads in enumerate(grads_params):
        torch.save(grads, path / f'grad_params/{batch_idx}_{i}.pt', pickle_protocol=4)

def save_metadata(path: Path, loss: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, batch_idx: int = 0):
    """Save metadata to disk."""
    for i in range(len(loss)):
        torch.save({
            'batch_index': batch_idx,
            'sample_index': i,
            'loss': loss[i].item(),
            'logits': logits[i],
            'label': labels[i].item(),
        }, path / f'loss_logits/{batch_idx}_{i}.pt', pickle_protocol=4)

# ---------------------------------------
# Main Functions
# ---------------------------------------

def compute_grad_loss_batch(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, params: Dict[str, torch.Tensor], 
    buffers: Dict[str, torch.Tensor], device: torch.device, chunk_size: int = 32, 
    path_to_disk: Optional[Path] = None, batch_idx: int = 0
) -> Optional[List[Dict[str, Any]]]:
    """Compute gradients, loss, and logits for a batch."""
    x, y = x.to(device), y.to(device)
    model = model.to(device)
    batch_size = x.size(0)
    grads_sample, loss, logits, grads_params = compute_loss_and_gradients(x, y, model, params, buffers, chunk_size)

    if path_to_disk:
        save_grads_sample(path_to_disk, grads_sample, batch_idx=batch_idx)
        save_grads_params(path_to_disk, grads_params, batch_idx=batch_idx)
        save_metadata(path_to_disk, loss, logits, y, batch_idx=batch_idx)
        return None

    return [{'grads_sample': grads_sample[i], 'grads_params': grads_params[i], 
             'loss': loss[i], 'logits': logits[i], 'label': y[i]} for i in range(batch_size)]

def record_model(model, loader, device='cuda', chunk_size=32, path_to_disk=None):
    """Record model behavior over a dataset."""
    model.eval().to(device)
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    if path_to_disk:
        path_to_disk = Path(path_to_disk)
        shutil.rmtree(path_to_disk, ignore_errors=True)
        (path_to_disk / 'grad_sample').mkdir(parents=True)
        (path_to_disk / 'grad_params').mkdir(parents=True)
        (path_to_disk / 'loss_logits').mkdir(parents=True)

    results = []
    for batch_idx, (x, y) in enumerate(tqdm(loader)):
        result = compute_grad_loss_batch(
            model, x, y, params, buffers, device, chunk_size, path_to_disk, batch_idx
        )
        if result:
            results.extend(result)

    return results if not path_to_disk else None


if __name__ == '__main__':
    # Example Usage    
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))
    
    # Generate some random data
    x = torch.randn(32, 10)  # Batch of 32 samples, each with 10 features
    y = torch.randint(0, 2, (32,))  # Batch of 32 target labels
    
    # Compute gradients, loss, and logits
    grads_sample, loss, logits, grads_params = compute_loss_and_gradients(x, y, model)
    
    print("Loss:", loss.shape)
    print("Logits:", logits.shape)
    print("Sample-level Gradients:", grads_sample.shape) # This is similar to grad-cam
    print("Parameter-level Gradients:", grads_params[0].keys()) 
    
    
    ## Record model behavior over a dataset
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.dataset import random_split
    print("Recording model behavior over a dataset...")
    dataset = TensorDataset(torch.randn(1500, 10), torch.randint(0, 2, (1500,)))
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    record_model(model, train_loader, device='cuda', chunk_size=32, path_to_disk='temp')