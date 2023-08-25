
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from typing import Tuple, Union, Optional

import json
import numpy as np
from abc import ABC


def estimate_fdr(labels, negative_labels, positive_labels) -> float:
    is_negative_reads = np.asarray([l in negative_labels for l in labels])
    num_positives = np.sum(~is_negative_reads)
    num_negatives = len(is_negative_reads) - num_positives
    fdr = num_negatives / (num_positives + num_negatives + 1e-8) * len(positive_labels) / len(negative_labels)
    return fdr





class RoundChannelProjector:
    def __init__(self, codebook, crosstalk, tensor_kwargs):
        self.codebook = torch.tensor(codebook.astype('float32'), **tensor_kwargs)
        self.codebook_binary = self.codebook.clone()

        if crosstalk is not None:
            self.crosstalk = torch.tensor(crosstalk.astype('float32'), **tensor_kwargs)
            for r,c in enumerate(self.crosstalk):
                self.codebook[:,r,:] = torch.einsum('ci,mi->mc', c, self.codebook[:,r,:])

        self.codebook = self.codebook.flatten(start_dim=1).T
        self.codebook_binary = self.codebook_binary.flatten(start_dim=1).T
        self.codebook = self.codebook / self.codebook.sum(axis=0, keepdim=True)            

    def forward(self, tensor):
        m, ny, nx = tensor.shape
        return (self.codebook @ tensor.view((m,ny*nx))).view((-1,ny,nx))

    def backward(self,tensor):
        m, ny, nx = tensor.shape
        return (self.codebook.t() @ tensor.view((m,ny*nx))).view((-1,ny,nx))
    



def load_gaussian(n:int, tensor_kwargs, sigma:float=1.0,s:float=1.0):
    if sigma == 0:
        return 1

    ns = int(np.ceil(n*s)); sigmas = sigma*s
    x = torch.linspace(0,ns-1,n); xs = torch.arange(ns)
    b = torch.exp(-0.5*(xs.reshape((-1,1)) - x)**2/sigmas**2).float()
    b = b / b.sum(axis=1,keepdim=True)

    bclosed = torch.zeros_like(b, **tensor_kwargs)
    f = 2*np.ceil(3*sigmas)+1
    w = int(f // 2)
    for i in range(x.shape[0]):
        xi = int(x[i])
        start = max(0,xi-w); stop = min(xs.shape[0],xi+w+1) 
        slize = slice(start,stop)
        bclosed[slize,i] = b[slize,i]
    bclosed = bclosed / b.sum(axis=0,keepdim=True)
    return bclosed



class SpatialProjector:
    def __init__(self,sigma: Tuple[float,float], shape: Tuple[int,int],  tensor_kwargs,  upscale: float = 1):
        self.ndim = np.sum(np.array(shape) > 1)
        if sigma[0] > 0:
            self.By = load_gaussian(shape[0], tensor_kwargs ,sigma=sigma[0], s=upscale).t()
        else:
            self.By = 1
        self.Bx = load_gaussian(shape[1], tensor_kwargs, sigma=sigma[1], s=upscale)

    def forward(self, tensor):
        if self.ndim > 1:
            return self.By @ tensor @ self.Bx
        return tensor @ self.Bx

    def backward(self, tensor):
        if self.ndim > 1:
            return self.By.t() @ tensor @ self.Bx.t()
        return tensor @ self.Bx.t()

    def down_pool(self, tensor):
        if self.ndim > 1:
            return (self.By != 0).float() @ tensor @ (self.Bx != 0).float()
        return tensor @ (self.Bx != 0).float()

    def up_pool(self, tensor):
        if self.ndim > 1:
            return (self.By.t() != 0).float() @ tensor @ (self.Bx.t() != 0).float()
        return tensor @ (self.Bx.t() != 0).float()



def find_local_maximas(tensor, window_size):

    padding = window_size // 2
    
    if tensor.dim() == 4:  # 3D tensor (m, z, y, x)
        tensor_padded = F.pad(tensor, (padding, padding, padding, padding, padding, padding), mode='constant', value=float('-inf'))
    elif tensor.dim() == 3:  # 2D tensor (m, y, x)
        tensor_padded = F.pad(tensor, (padding, padding, padding, padding), mode='constant', value=float('-inf'))
    else:
        raise ValueError("Tensor dimension must be 3 or 4")

    if tensor.dim() == 4:
        pool = F.max_pool3d
    else:
        pool = F.max_pool2d

    local_maximas = pool(tensor_padded, kernel_size=window_size, stride=1)
    local_maximas = torch.eq(tensor, local_maximas)

    return local_maximas


def _nonmaxsuppress(tensor, radius):
    padd = [radius, radius]
    kernel_sz = (2*radius+1, 2*radius+1)
    mask = torch.nn.functional.max_pool2d(tensor, kernel_sz, stride=1, padding=padd) == tensor
    return mask

def _compute_quality(tensor, Y, B, psf, codebook, psf_support_scaled):
    # Pool intensities spatially
    
    tensor_blurr = torch.nn.functional.avg_pool2d(tensor, \
        psf_support_scaled,\
        stride=1,\
        divisor_override=1,\
        padding=tuple(t//2 for t in psf_support_scaled))


    tensor_blurr2 = psf.up_pool(torch.nn.functional.relu(Y-B))

    # Compute quality feature
    Q = tensor_blurr / codebook.backward(tensor_blurr2)
    Q[torch.isnan(Q)] = 0
    return Q


def istdeco_decode(
    image_data:np.ndarray,
    codebook:np.ndarray,
    psf_sigma:Tuple[int,int]=(1.5,1.5),
    crosstalk:np.ndarray=None,
    min_integrated_intensity:float=500,
    niter:int=50,
    lowpass_sigma:Optional[Tuple[int,int]]=(7.0, 7.0),
    device='cpu',
    jitter: float = 0,
    l1_reg = 0,
    l2_reg = 0,
    radius=2,
    min_correct_spots=3.2
    ):
    """
    Decode barcodes from in situ sequencing imaging data.

    Args:
        image_data (np.ndarray): 4D array of shape (rounds, channels, height, width)
            containing the imaging data.
        codebook (np.ndarray): 3D array of shape (num_genes, rounds, channels) representing
            the gene codebook.
        psf_sigma (Tuple[int, int]): Tuple of integers representing the standard deviation
            of the point-spread function in both spatial dimensions. 
            Default is (1.5, 1.5).
        crosstalk (np.ndarray, optional): Crosstalk matrix of shape (rounds, channels, channels)
            representing the crosstalk between different channels. Defaults to None.
        min_integrated_intensity (float, optional): Minimum integrated intensity threshold
            for decoding spots. Defaults to 500.
        niter (int, optional): Number of iterations for optimization. Defaults to 50.
        lowpass_sigma (Tuple[int, int], optional): Tuple of integers representing the
            standard deviation of the low-pass filter in both spatial dimensions.
            Defaults is (7.0, 7.0).
        device (str, optional): Device to use for computations ('cpu' or 'cuda'). Defaults to 'cpu'.
        jitter (float, optional): Jitter parameter to model potential "jitter" in x and position of 
            spots in different rounds. Default to 0.
        l1_reg (float, optional): L1 regularization parameter. Defaults to 0.
        l2_reg (float, optional): L2 regularization parameter. Defaults to 0.
        min_correct_spots (float, optional): Minimum number of correct spots for decoding.
            Defaults to 3.2.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the decoded barcode information:
            'Target id', 'Y', 'X', 'Intensity', 'Num explained spots', 'Rounds', 'Channels'.

    Example:
        Example usage of istdeco_decode with image data and codebook:

        ```
        decoded_results = istdeco_decode(
            image_data, codebook, (2, 2), None, 1000, 50, None, 'cpu', 0, 0, 3.0
        )
        ```
    """
   

    _SMALL_CONSTANT = 1e-6

    # Extra arguments for creating PyTorch tensors
    tensor_args = {'device': device, 'dtype': torch.float32, 'requires_grad' : False}
    
    # Convert to tensor land
    Y = torch.tensor(image_data.astype('float32'), **tensor_args)
    # Y is of shape (rc, y, x)
    Y = Y.flatten(start_dim=0, end_dim=1)

    # Get number of spatial dimensions
    spatial_shape = Y.shape[1:]
    spatial_dims = len(spatial_shape)



    if jitter > 1.0:
        jitter_sigmas = tuple(jitter / 2.0 for _ in range(2))
        Y = SpatialProjector(
            jitter_sigmas, 
            spatial_shape,
            tensor_args
        ).forward(Y)

        psf_sigma = tuple(np.sqrt(s1**2 + s2**2) for s1,s2 in zip(jitter_sigmas, psf_sigma))



    # Create forward and back projector for 
    # the point-spread function
    psf = SpatialProjector(
        psf_sigma,
        spatial_shape,
        tensor_args
    )

    # Create forward and back projector for background
    if lowpass_sigma is not None:
        bg = SpatialProjector(
            lowpass_sigma,
            spatial_shape,
            tensor_args
        )
    else:
        bg = None

    num_codes = codebook.shape[0]

    # Create forward and back projector across rounds and channels
    cb = RoundChannelProjector(
        codebook,
        crosstalk,
        tensor_args
    ) 


    # Set up output
    output_shape = (num_codes, ) + spatial_shape

    # Initial guess
    X = torch.ones(output_shape, **tensor_args)

    # Estimate additative background
    if bg is not None:
        B = bg.forward(Y) * 1.0025 + _SMALL_CONSTANT
    else:
        B = _SMALL_CONSTANT
    # Create scaling
    scale = cb.backward(
        psf.backward(
            torch.ones_like(Y)
        )
    ) + l1_reg + 2 * l2_reg * X.sum(axis=0) + _SMALL_CONSTANT

    # Create cost function
    mse = torch.nn.MSELoss()

    # Loop over num iterations
    for i in range(niter):
        # Current fit
        Y_pred = psf.forward(cb.forward(X)) + B
        #  Root mean squared error
        loss = mse(Y_pred, Y)
        # Take a multiplicative step
        X = X * cb.backward(
            psf.backward(
                Y / Y_pred
            )
        ) / scale

    # Extract barcodes
    ## Integrate intensities in spatial neighborhoods
    bh = int(2*np.ceil(3*psf_sigma[0])+1) 
    bw =  int(2*np.ceil(3*psf_sigma[1])+1) 
    psf_support_scaled = (bh, bw)

    Q = _compute_quality(X, Y, B, psf, cb, psf_support_scaled)
    mask = _nonmaxsuppress(X, radius)
    X = X * mask
    Q = Q * mask
    
    #if (l1_reg > 0) and (l2_reg == 0):
    #    X = X * (1 + l1_reg / codebook.sum(axis=(1,2), keepdims=True))             
        
    # Pick out sparse data
    X = X.detach().cpu().numpy()
    Q = Q.detach().cpu().numpy()

    indices = np.where((X > min_integrated_intensity) & (Q > min_correct_spots))
    target_ids = indices[0]

    codebook_sparse = [np.transpose(np.where(code)) for code in codebook]
    rounds = [';'.join([str(item) for item in codebook_sparse[id][:,0]]) for id in  target_ids]
    channels = [';'.join([str(item) for item in codebook_sparse[id][:,1]]) for id in target_ids ]
    
    output = {
        'Target id' : target_ids,
        'Y' : indices[1],
        'X' : indices[2],
        'Intensity' : X[indices],
        'Num explained spots' : Q[indices],
        'Rounds' : rounds,
        'Channels' : channels
    }

    return output

if __name__ == '__main__':


    import pickle
    d = pickle.load(open('data.pkl','rb'))
    istdeco_decode(d['data'], d['cb'], (2.0, 2.0))
   