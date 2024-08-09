import torch
from psfmodels import make_psf as estimate_psf
from typing import Tuple,Dict,Any
import numpy as np
from tqdm.auto import tqdm

def __fftblur(signal:torch.Tensor,filter_f:torch.Tensor,filter_shape:Tuple):
    r = tuple(np.array(signal.shape) + np.array(filter_shape)-1)
    nd = signal.ndim
    ifun = lambda m, n: slice(int(np.ceil((n-1)/2)), int(m+np.ceil((n-1)/2)))
    slices = [ifun(signal.shape[d], filter_shape[d]) for d in range(nd)]
    af = torch.fft.fftn(signal, r)
    out = torch.real(torch.fft.ifftn(af * filter_f))
    out = out[slices]
    out[out < 1e-3] = 0
    return out



def __deconvolve(data:np.array, psf:np.array, niter:int, device:str = 'cpu', bg:float = 1e-3, pbar=None):
    # Constants
    psf = psf / psf.sum()
    pad = np.array([int(s//2+1) for s in psf.shape])
    # Slices for depading in the end
    depad = tuple([slice(p, -p) for p in pad])
    # Flip the order of the pads (PyTorch wants it this way)
    pad = np.flip(pad)
    # Duplicate all pad elements (we want to pad both sides)
    pad = np.array(np.kron(pad, np.ones(2)), dtype = 'int')
    # Create a function for bluring (using FFT)
    _blur = lambda x, filt_f: __fftblur(x, filt_f, psf.shape)
    # Input data type.
    inpt_dtype = data.dtype
    
    # Setup tensors
    inpt = torch.tensor(
        np.array(data, dtype='float32'), 
        dtype = torch.float32, 
        requires_grad = False, 
        device = device
    )

    psf = torch.tensor(
        psf, 
        dtype = torch.float32, 
        requires_grad = False, 
        device = device
    )

    # Pad input
    inpt = torch.nn.functional.pad(inpt, tuple(pad))

    # Precompute FFT for PSF
    r = tuple(np.array(inpt.shape) + np.array(psf.shape)-1)
    psf_f = torch.fft.fftn(psf, r)

    # Ugly masking to avoid ringing (boudnary crap)
    mask = inpt > 0
    alpha = _blur(torch.ones(inpt.shape, dtype = torch.float32, requires_grad = False, device = device) * mask, psf_f)
    w = torch.zeros(alpha.shape, dtype = torch.float32, requires_grad = False, device = device)
    w[alpha > 0] = 1 / alpha[alpha > 0]

    # Output tensor
    outpt = torch.ones(
        alpha.shape, 
        dtype = torch.float32, 
        requires_grad = False, 
        device = device
    )

    # Optimization loop
    for _ in range(niter):
        if pbar is not None:
            pbar.update(1)

        yhat = _blur(outpt, psf_f) + bg
        outpt = outpt * (_blur(inpt / yhat, psf_f) * w)

    # Convert back to numpy
    outpt = outpt.cpu().detach().numpy()
    outpt = outpt[depad]                        # Remove padding using list of slices
    outpt = np.array(outpt, dtype = inpt_dtype)    # Correct datatype
    return outpt

def __slice(shape,max_width):
    if max_width is None:
        return [tuple([slice(0,s) for s in shape])]
    slices = []
    for s in shape:
        slices.append([
            slice(max_width*i,np.minimum(max_width*(i+1),s)) 
            for i in range(s//max_width+1) if max_width*i != s
        ])

    all_slices = []
    if len(shape) == 3:
        zslice,yslice,xslice = tuple(slices)
        for z in zslice:
            for y in yslice:
                for x in xslice:
                    all_slices.append((z,y,x))
    else:
        yslice,xslice = tuple(slices)
        for y in yslice:
            for x in xslice:
                all_slices.append((y,x))
    return all_slices


def __make_pbar(niter, nmini_tiles):
    return tqdm(total=niter*nmini_tiles)

def lucy_richardson(
    image:np.ndarray,
    psf:np.ndarray=None,
    psf_params:Dict[str,Any]=None,
    niter:int=30, 
    device:str='cpu', 
    bg:float=1e-3, 
    tile_width:int=None) -> np.ndarray:
    
    
    # Check inputs
    if psf is None and psf_params is None:
        raise ValueError('Input parameter psf or psf_params must be provided.')
    if psf is None and (psf_params is not None):
        psf = estimate_psf(**psf_params)
    if psf_params is not None and psf is not None:
        raise Warning('Both psf and psf_params were given. Ignoring psf_params.')
    if image.ndim != psf.ndim:
        raise ValueError(f'Input image data has {image.ndim} dimensions, but provided PSF has {psf.ndim} number of dimensions.')
    if not isinstance(niter,int):
        raise ValueError(f'Number of iterations must be an integer.')
    if niter < 1:
        raise ValueError(f'Number of iterations must be a postive non-zero integer.')
    if tile_width is not None:
        if not isinstance(tile_width,int):
            raise ValueError(f'Tile width must be an integer.')
        if tile_width < 1:
            raise ValueError(f'Tile width must be a non-zero integer.')
        for i,s in enumerate(image.shape):
            if s < tile_width:
                raise ValueError(f'Tile width must be smaller than input image width. Input image has width {s} along dimension {i}, but tile width is set to {tile_width}.')


    if isinstance(image,np.ndarray):
        inptdata = image
    outptdata = np.zeros_like(inptdata,dtype=inptdata.dtype)
    slices = __slice(inptdata.shape,tile_width)
    pbar = __make_pbar(niter, len(slices))
    for slice in slices:
        outptdata[slice] = __deconvolve(inptdata[slice], psf, niter, device, bg, pbar)
    return outptdata

