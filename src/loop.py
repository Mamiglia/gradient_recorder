import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call
from typing import Collection, Tuple, Mapping
from torch import Tensor 
from tqdm import tqdm


def loss_fn(x: Tensor, y: Tensor, params: Collection[Tensor], buffer:Collection[Tensor], model: nn.Module) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
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
    logits = functional_call(model, (params, buffer), (x,))
    loss = F.cross_entropy(logits, y)
    return loss, (loss, logits.squeeze())



def record_model(model: nn.Module, loader, device='cuda', chunk_size=32) -> Tuple[Mapping[str, Tensor], Tensor, Tensor, Tensor]:
    """Record the behaviour of the model on the given loader.
    
    For each sample computes and stores the:
    - loss
    - logits
    - gradients of the loss with respect to the model parameters
    - gradients of the loss with respect to the input
    """
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
    
    # define a function that records the behaviour of the model on a single batch of data
    def record_fn(x, y) -> Tuple[Tensor, Mapping[str, Tensor], Tensor, Tensor]:
        """Record the behaviour of the model on a single batch of data."""
        x = x.to(device)
        y = y.to(device)
        
        (grads_sample, grads_params), (loss, logits) = loss_logits_grads(x, y, params, buffers, model)
        
        grads_sample, loss, logits = grads_sample.detach().cpu(), loss.detach().cpu(), logits.detach().cpu()
        for k in grads_params:
            grads_params[k] = grads_params[k].detach().cpu()
        return grads_sample, grads_params, loss, logits
    
    # records is a list of tuples (grads_sample, grads_params, loss, logits)
    # we need to concatenate the tensors along the batch dimension
    records = [ record_fn(x, y) for x, y in tqdm(loader) ]
    
    grads_samples, grads_params, losses, logits = zip(*records)
    
    grads_samples = torch.cat(grads_samples, dim=0)
    losses = torch.cat(losses, dim=0)
    logits = torch.cat(logits, dim=0)
    grads_params = {k: torch.cat([g[k] for g in grads_params], dim=0) for k in grads_params[0]}
    
    return grads_params, grads_samples, logits, losses
    


    


        