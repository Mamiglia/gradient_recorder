import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call
from typing import Collection, Tuple, Mapping, Optional
from torch import Tensor 
from tqdm import tqdm
import pandas as pd
import os
from pathlib import Path


def loss_fn(
    x: Tensor, y: Tensor, params: Collection[Tensor], buffer:Collection[Tensor], model: nn.Module) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    """Compute the loss and logits for a single batch of data.
    
    Args:
        x: input data
        y: target labels
        params: model parameters
        buffer: buffer parameters
        model: model to evaluate
    """
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    # we need to use functional call of the model as this function is used in the vmap
    # thus no stateful operations are allowed
    logits = functional_call(model, (params, buffer), (x,)) 
    loss = F.cross_entropy(logits, y)
    return loss, (loss, logits.squeeze())

def record_model(
    model, loader, device='cuda', chunk_size=-1, 
    path_to_disk=None, keep_memory=False
):
    """Record the behaviour of the model on the given loader."""
    assert not keep_memory or not path_to_disk, "Either keep_memory or path_to_disk must be True"
    model.eval().to(device)
    
    if path_to_disk:
        path_to_disk = Path(path_to_disk)
        shutil.rmtree(path_to_disk, ignore_errors=True)  # Remove the directory and all its contents
        path_to_disk.mkdir(parents=True, exist_ok=True)
        (path_to_disk / 'grad_sample').mkdir()
        (path_to_disk / 'grad_params').mkdir()
        (path_to_disk / 'loss_logits').mkdir()
    
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    chunk_size = chunk_size if chunk_size > 0 else loader.batch_size
    
    # define a function that records the behaviour of the model on a single sample of data
    # we do this through functional transforms to improve performance
    compute_gradients = vmap(              # vmap over the batch dimension
        grad(                              # gradient over the model parameters and the input
            loss_fn, 
            argnums=(0, 2), has_aux=True   # argnums=(0, 2) means that we want to differentiate with respect to the first and third argument
                                            # has_aux=True means that the function returns also the loss and the logits
        ), in_dims=(0, 0, None, None, None),# in_dims specifies which is the batch dimension of x, y
        chunk_size=chunk_size)
    
        
    results = [
        result
        for batch_idx, (x, y) in enumerate(tqdm(loader))
        for result in process_batch(
            model, x, y, params, buffers, compute_gradients, batch_idx, 
            device, path_to_disk, keep_memory
        )
    ]
    
    if path_to_disk:
        df = pd.DataFrame([r for r in results if isinstance(r, dict)])
        df.to_csv(path_to_disk / 'metadata.csv', index=False)
        return df
    
    if keep_memory:
        return results
    

def process_batch(
    model, x, y, params, buffers, compute_loss_and_gradients, batch_idx, 
    device, path_to_disk=None, keep_memory=False
):
    """Process a single batch of data."""
    x, y = x.to(device), y.to(device)
    (grads_sample, grads_params), (loss, logits) = compute_loss_and_gradients(x, y, params, buffers, model)
    
    # detach the tensors and move them to the cpugrad_sample
    grads_sample = grads_sample.detach().cpu()
    loss = loss.detach().cpu()
    logits = logits.detach().cpu()
    y = y.detach().cpu()
    grads_params = {k: v.detach().cpu() for k, v in grads_params.items()}
    
    if path_to_disk:
        save_batch(path_to_disk, batch_idx, grads_sample, grads_params, loss, logits, y)
    
    if keep_memory:
        return [
            {
                'grads_sample': grads_sample[i],
                'grads_params': {k: v[i] for k, v in grads_params.items()},
                'loss': loss[i],
                'logits': logits[i],
                'label': y[i],
                'batch_index': batch_idx,
                'sample_index': i,
                'grad_sample_path': path_to_disk / f'grad_sample/{batch_idx}_{i}.pt',
                'grad_params_path': path_to_disk / f'grad_params/{batch_idx}_{i}.pt',
                'loss_logits_path': path_to_disk / f'loss_logits/{batch_idx}_{i}.pt',
            }
            for i in range(grads_sample.shape[0])
        ]
    return []

def save_batch(path, batch_idx, grads_sample, grads_params, loss, logits, labels):
    """Saves batch results to a single file."""
    
    for i in range(grads_sample.shape[0]):
        torch.save(grads_sample[i], path  / f'grad_sample/{batch_idx}_{i}.pt', pickle_protocol=4)
        torch.save({
                'batch_index': batch_idx,   
                'sample_index': i,
                'loss': loss[i].item(),
                'logits': logits[i],
                'label': labels[i].item(),
                'grad_sample_path': path / f'grad_sample/{batch_idx}_{i}.pt',
                'grad_params_path': path / f'grad_params/{batch_idx}_{i}.pt',
            }, path / f'loss_logits/{batch_idx}_{i}.pt', pickle_protocol=4)
        
        params = {k: v[i] for k, v in grads_params.items()}
        torch.save(params, path / f'grad_params/{batch_idx}_{i}.pt', pickle_protocol=4)  
        