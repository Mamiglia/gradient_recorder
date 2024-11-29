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
    model: nn.Module, loader, 
    device='cuda', chunk_size=32, 
    path_to_disk=None,
    keep_memory=False
    ) -> Optional[Tuple[Mapping[str, Tensor], Tensor, Tensor, Tensor]]:
    """Record the behaviour of the model on the given loader.
    
    For each sample computes and stores the:
    - loss
    - logits
    - gradients of the loss with respect to the model parameters
    - gradients of the loss with respect to the input
    
    Args:
    model: model to record
    loader: data loader
    device: device to use
    chunk_size: size of the chunk for the vmap operation
    path_to_disk: path to save the recorded data, if None the data is not saved
    keep_memory: if True, the recorded data is kept in memory
    """
    assert not keep_memory or not path_to_disk, "Either keep_memory or path_to_disk must be True"
    model.eval()
    model = model.to(device)
    
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    
    # define the derivative of the loss with respect to the model parameters and the input
    loss_logits_grads = vmap(                       # vmap over the batch dimension
    grad(                                       # gradient over the model parameters and the input
        loss_fn, argnums=(0, 2), has_aux=True   # argnums=(0, 2) means that we want to differentiate with respect to the first and third argument
                            # has_aux=True means that the function returns also the loss and the logits
    ), in_dims=(0, 0, None, None, None),        # in_dims specifies which is the batch dimension of x, y
    chunk_size=chunk_size)
    
    if path_to_disk:
        os.makedirs(path_to_disk, exist_ok=True)
    
    # define a function that records the behaviour of the model on a single batch of data
    def record_fn(x, y, batch_idx) -> Tuple[Tensor, Mapping[str, Tensor], Tensor, Tensor]:
        """Record the behaviour of the model on a single batch of data."""
        x = x.to(device)
        y = y.to(device)
        
        samples = []
        
        (grads_sample, grads_params), (loss, logits) = loss_logits_grads(x, y, params, buffers, model)
        
        if path_to_disk:
            for i in range(x.shape[0]):
                path = Path(path_to_disk) / f'{batch_idx}_{i}'
                os.makedirs(path, exist_ok=True)
                
                torch.save(grads_sample[i], path / 'grads_sample.pt')
                torch.save(loss[i], path / 'loss.pt')
                torch.save(logits[i], path / 'logits.pt')
                torch.save({k:p[i] for k,p in grads_params.items()}, path / 'grads_params.pt')
                
                samples.append({
                    'path': path,
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'grads_sample': path / 'grads_sample.pt',
                    'grads_params': path / 'grads_params.pt',
                    'loss': path / 'loss.pt',
                    'logits': path / 'logits.pt',
                    'label': y[i].item()
                })
                
        if not keep_memory:
            return (samples,)
        
        grads_sample, loss, logits = grads_sample.detach().cpu(), loss.detach().cpu(), logits.detach().cpu()
        for k in grads_params:
            grads_params[k] = grads_params[k].detach().cpu()
        
        return samples, grads_sample, grads_params, loss, logits
    
    # records is a list of tuples (grads_sample, grads_params, loss, logits)
    # we need to concatenate the tensors along the batch dimension
    records = [record_fn(x, y, idx) for idx, (x, y) in enumerate(tqdm(loader))]
    
    if path_to_disk:
        df = pd.DataFrame([s for r in records for s in r[0]])
        df.to_csv(Path(path_to_disk) / 'metadata.csv', index=False)
    
    if not keep_memory:
        return df
    
    grads_samples, grads_params, losses, logits = zip(*records)
    
    grads_samples = torch.cat(grads_samples, dim=0)
    losses = torch.cat(losses, dim=0)
    logits = torch.cat(logits, dim=0)
    grads_params = {k: torch.cat([g[k] for g in grads_params], dim=0) for k in grads_params[0]}
    
    return df, grads_params, grads_samples, logits, losses
    


    


        