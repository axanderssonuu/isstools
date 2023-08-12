
import itertools
import numpy as np
import torch
from typing import Any, Dict, Tuple, Union, List, Optional, Literal, Sequence, Callable
import psfmodels
from scipy.ndimage import convolve
from tqdm.auto import tqdm
from skimage.filters import gaussian as gaussian_filter
from skimage.morphology import disk, ball, local_maxima
from scipy.ndimage import convolve
import pandas as pd


import json
import numpy as np

class CodebookEntry:
    def __init__(self, gene_name, rounds_with_ones, channels_with_ones, **attributes):
        self.gene_name = gene_name
        self.rounds_with_ones = rounds_with_ones
        self.channels_with_ones = channels_with_ones
        self.attributes = attributes

class Codebook:
    def __init__(self, num_rounds, num_channels):
        self.num_rounds = num_rounds
        self.num_channels = num_channels
        self.entries = []

    def add_code(self, gene_name, round_indices, channel_indices, **attributes):
        entry = CodebookEntry(gene_name, round_indices, channel_indices, **attributes)
        self.entries.append(entry)

    def load_from_json(self, json_file_path):
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for entry_data in data:
                self.add_code(**entry_data)

    def get_label(self, gene_name):
        for entry in self.entries:
            if entry.gene_name == gene_name:
                combinatorial_label = self._generate_combinatorial_label(entry)
                return combinatorial_label
        return None

    def get_label_matrix(self):
        label_matrix = np.zeros((len(self.entries), self.num_rounds, self.num_channels))
        for idx, entry in enumerate(self.entries):
            combinatorial_label = self._generate_combinatorial_label(entry)
            label_matrix[idx] = np.array(combinatorial_label)
        return label_matrix

    def save_to_json(self, json_file_path):
        data = []
        for entry in self.entries:
            entry_data = {
                "gene_name": entry.gene_name,
                "rounds_with_ones": entry.rounds_with_ones,
                "channels_with_ones": entry.channels_with_ones,
                **entry.attributes
            }
            data.append(entry_data)

        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

    def _generate_combinatorial_label(self, entry):
        combinatorial_label = [[0] * self.num_channels for _ in range(self.num_rounds)]
        for round_idx, channel_idx in zip(entry.rounds_with_ones, entry.channels_with_ones):
            combinatorial_label[round_idx][channel_idx] = 1
        return combinatorial_label



def estimate_fdr(labels, negative_labels, positive_labels) -> float:
    is_negative_reads = np.asarray([l in negative_labels for l in labels])
    num_positives = np.sum(~is_negative_reads)
    num_negatives = len(is_negative_reads) - num_positives
    fdr = num_negatives / (num_positives + num_negatives + 1e-8) * len(positive_labels) / len(negative_labels)
    return fdr

        
class _Codebook:

    def __init__(self, codebook:np.ndarray, crosstalk=None, tensor_args={}):
        self.codebook_binary = codebook.copy()
        self.codebook_binary = torch.tensor(self.codebook_binary > 0, **tensor_args)

        if crosstalk is not None:
            codebook = np.einsum('ci,mri->mrc',crosstalk,codebook)
        self.codebook = torch.tensor(codebook, **tensor_args)
        self.codebook = self.codebook / self.codebook.sum(axis=(1,2),keepdim=True)

    def conv_T(self, x):
        return torch.einsum('rc...,mrc->m...',x,self.codebook)

    def conv(self, x):
        return torch.einsum('m...,mrc->rc...',x,self.codebook)

class PSF:
    def __init__(self, psf:Any, mode:str, data_shape, tensor_args={}):
        valid_modes = ['gaussian', 'psfmodels', 'ndarray']
        self.data_shape = data_shape
        self.conv = None
        self.conv_T = None
        self.tensor_args = tensor_args
        if mode == 'gaussian':
            self.setup_gaussian(psf)
        elif mode == 'psfmodels':
            self.setup_psfmodels(psf)
        elif mode == 'ndarray':
            self.setup_ndarray(psf)




    def setup_gaussian(self, psf):

        x = self.create_gaussiasn_toeplitz(self.data_shape[-1], psf[-1])
        y = self.create_gaussiasn_toeplitz(self.data_shape[-2], psf[-2])
        z = self.create_gaussiasn_toeplitz(self.data_shape[-3], psf[-3]) if len(self.data_shape) == 5 else None

        def conv(data:torch.Tensor,x:torch.Tensor,y:torch.Tensor,z:torch.Tensor,trsp):
            if not trsp:
                out = torch.einsum('...i,ij->...j',data,x)
                out = torch.einsum('...ix,ij->...jx',out,y)
                if z is not None:
                    out = torch.einsum('...iyx,ij->...jyx',out,z)
                return out
            else:
                out = torch.einsum('...i,ji->...j',data,x)
                out = torch.einsum('...ix,ji->...jx',out,y)
                if z is not None:
                    out = torch.einsum('...iyx,ji->...jyx',out,z)
                return out  
                
        self.conv = lambda data,axis=(0,1): conv(data,x,y,z,False)
        self.conv_T = lambda data,axis=(0,1): conv(data,x,y,z,True)


    def create_gaussiasn_toeplitz(self, width:int, sigma:float):
        base = np.eye(width)
        w = int(2*np.ceil(3*sigma)+1)
        gaussian = np.exp(-(np.arange(w)-w//2)**2 / (2*sigma**2)).reshape((-1,1))
        filt =np.array(gaussian)
        mat = convolve(base,filt)
        mat = mat / mat.sum(axis=0,keepdims=True)
        mat = torch.tensor(mat,**self.tensor_args)
        return mat

    def setup_ndarray(self, psf):
        psf = torch.tensor(psf, **self.tensor_args)
        filter_shape = psf.shape
        spatial_shape = (self.data_shape[2],self.data_shape[3]) if len(self.data_shape) == 4 else  (self.data_shape[2],self.data_shape[3],self.data_shape[4])
        psf_f = torch.fft.fftn(psf, tuple(np.array(spatial_shape) + np.array(filter_shape)-1))
        self.conv = PSF.__create_fft_conv(psf_f, filter_shape)
        self.conv_T = PSF.__create_fft_conv(psf_f, filter_shape)
        
    def setup_psfmodels(self,psf):
        psf = psfmodels.make_psf(**psf)
        self.setup_ndarray(psf)


    @staticmethod
    def __create_fft_conv(psf_f:torch.Tensor,filter_shape:Tuple):
        def conv(data:torch.Tensor,psf_f,filter_shape:Tuple,loopaxis=(0,1)):
            out = torch.zeros_like(data)
            loops = [range(n) if ax in loopaxis else [slice(None)] for ax,n in enumerate(data.shape)]
            for idx in itertools.product(*loops):
                out[idx] = PSF.__fftblur(data[idx],psf_f, filter_shape)
            out[out<0] = 0
            return out
        fun = lambda x,axis=(0,1): conv(x, psf_f, filter_shape, loopaxis=axis)
        return fun

    @staticmethod
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

class ISTDECO:
    """ISTDECO - In Situ Transcriptomics by Deconvolution

    ISTDECO (In Situ Transcriptomics Decoding by Deconvolution) is a simple tool
    for decoding in situ transcriptomics data. Im biref: In Situ Transcriptomcis data
    usually follow a combinatorial labelling strategy. Combination of fluorescent spots present
    and absent across imaging rounds and channels form a type code (often called a barcode).
    This barcode labels a particular type of RNA species. ISTDECO is tool for automatically
    locating barcodes using a simple "deconvolution" method.

    In essence, we assume that the observed image data, Y (shape: (r, c, y, x)), can be 
    explained as:

    Y ~ Poiss(DXG+B),

    where D (shape: (m, r, c)) is a codebook containing barcodes, X (shape: (m, y, x))
    is a "barcode magnitude" matrix that indicate the location of individual barcodes in
    a Cartesian space, G is a convolution with a "point-spread-function" (PSF), and
    B is some low-frequency background.

    The codebook and the PSF are known a priori. The background is easily estimated using 
    a standard low-pass filtering (see fit_intercept method), and X is found solving an
    inverse problem (fit method).




    """

    def __init__(self,
        data:np.ndarray,
        codebook:Codebook, 
        psf:Union[Tuple[float,float],Tuple[float,float,float],np.ndarray], 
        intercept:np.ndarray,
        crosstalk:np.ndarray=None,
        device:str='cpu'):
        """Initialize ISTDECO  model



        Parameters
        ----------
        data : np.ndarray
            The data that is to be deconvolved. Should be of shape `(r, c, y, x)`
            or `(r, c, z, y, x)`, where `r` is the number of rounds, `c` is the
            number of channels, `z`, `y` and `x` are the size of the 
            spatial dimensions. 

        codebook : Codebook
            Class containing information regarding targeted genes and their
            combinatorial labels. 

        psf : Union[Tuple[float,float],Tuple[float,float,float],np.ndarray]
            A point-spread-function modelling the shape of the spots.
            Can be an ndarray of shape `([z],y,x)` with a custom PSF model or a 
            Tuple containing sigmas (standard deviations) parameterizing
            the shape of a Gaussian PSF. Roughly 2*sigma should match the radius of a spot.

            If a Gaussian PSF is chosen, convolutions are done in a seperable manner,
            which dramatically speeds up deconvolution when running on the GPU.


        intercept : np.ndarray
            ndarray with shape matching data containing the background of image data.
            The `intercept` can be estimated by bluring `data` with a Gaussian filter
            using a sigma much larger than the width of the spots. 

            The intercept term can be estimated as follows:

            >>> from isttools.decoding import ISTDECO
            >>> intercept = ISTDECO.fit_intercept(sigma=(5.0,5.0))

        device : str, optional
            Which device to run the deconvolution on, by default 'cpu'.
            Common options are 'cpu' or 'cuda' (GPU). If GPU is available,
            we strongly recommend using it.

        Example
        -------

        Simple example (2D data) on how to deconvolve data of shape (r, c, y, x)
        using a codebook of shape (m, r, c) and a Gaussian psf.

        Output data will be a collection of m images. One image per targeted gene.

        >>> image_data = ... # Load image data (r, c, y, x)
        >>> codebook = ... # Load codebook (m, r, c)
        >>> from isttools.decoding import ISTDECO
        >>> intercept = ISTDECO.fit_intercept((5.0,5.0))
        >>> model = ISTDECO(data,codebook=codebook,psf=(2.0,2.0),intercept=intercept)
        >>> deconvolved_images = model.fit(niter=50) # (m, y, x)


        """        


        # Check inputs
        self.codebook_parent = codebook
        codebook = codebook.get_label_matrix()

        if data.ndim != 4 and data.ndim != 5:
            raise ValueError('Input `data` must have 4 (r, c, y, x) or 5 (r, c, z, y, x) dimensions.')
        if isinstance(psf,tuple):
            if data.ndim == 4 and len(psf) != 2:
                raise ValueError(f'PSF was passed as a tuple with {len(psf)} elemenets, but expected only 2 elements (sigmas in the y and x dimension.')
            if data.ndim == 5 and len(psf) != 3:
                raise ValueError(f'PSF was passed as a tuple with {len(psf)} elemenets, but expected only 3 elements (sigmas in the z, y, and x dimension.')
            for s in psf:
                if s <= 0:
                    raise ValueError(f'PSF was passed as the tuple {psf}, but expected only positive elements.')
        if intercept.ndim != data.ndim:
            raise ValueError(f'Number of dimensions of intercept {intercept.ndim} must match number of dimensions of input data {data.ndim}')
        
        for sz_in, sz_data in zip(*(intercept.shape,data.shape)):
            if sz_in != sz_data:
                raise ValueError(f'Size of `intercept` {intercept.shape} must match size of `data` {data.shape}')

        # Additional arguments for creating PyTorch tensor
        self.__tensor_args = {'device': device, 'dtype': torch.float32, 'requires_grad' : False}


        if isinstance(psf,Tuple):
            psf_mode = 'gaussian'
        elif type(psf) is np.ndarray:
            psf_mode = 'ndarray'


        # Create PSF
        self.__psf = PSF(
            psf, 
            psf_mode, 
            data.shape,
            self.__tensor_args
        )

        # Create codebook
        self.__codebook = _Codebook(codebook, crosstalk=crosstalk, tensor_args=self.__tensor_args)
        # Check input datatype
        self.__inpt_dtype = data.dtype


        # Check if input data is 3D or 2D (r,c,z,y,x) or (r,c,y,x)
        if len(data.shape) == 5:
            spatial_shape = (data.shape[2], data.shape[3],data.shape[4]) 
        else:
            spatial_shape = (data.shape[2], data.shape[3])

        # Shape of output data (m,z,y,x) or (m,y,x)
        # where m is the number of targeted barcodes
        self.__output_shape = (self.__codebook.codebook.shape[0],)+spatial_shape


        # Initial guess (all magnitudes. equal)
        self.barcode_magnitudes = torch.ones(self.__output_shape,**self.__tensor_args)
        # Observed data as a tensor
        self.observed_data = torch.tensor(data.astype('float32'), **self.__tensor_args)
        self.intercept = torch.tensor(intercept.astype('float32'), **self.__tensor_args)
        # Normalizer used in RL deconv.
        self._denominator = self.__codebook.conv_T(self.__psf.conv(torch.ones_like(self.observed_data)))




    def fit(self, 
        niter:int=50, 
        l2_reg:float=0, 
        l1_reg:float=0) -> np.ndarray:
        """Fit the point-spread-function and codebook to image data.

        Parameters
        ----------
        niter : int, optional
            Number of iterations, by default 50
        l2_reg : float, optional
            Non-negative float for scaling L2 regularization term, by default 0 (no regularization)
            A large l2_reg promotes smoothness in the deconvolved data.
        l1_reg : float, optional
             Non-negative float for scaling L1 regularization term, by default 0 (no regularization)
             A large l1_reg promotes sparsity in the deconvolved data.

        Returns
        -------
        np.ndarray
            Numpy array with decoded barcodes, i.e., an (m, [z], y, x)
            shaped array, where m is the number of codes in the codebook.
            Values in the array correspond to decoded intensities of different barcodes.

        """

        if not isinstance(niter,int) or niter < 1:
            raise ValueError('Number of iterations must be a positive integer.')        
        if l2_reg < 0:
            raise ValueError('`l2_reg` must be a non-negative number.')        
        if l1_reg < 0:
            raise ValueError('`l1_reg` must be a non-negative number.')        


                
        # Big loop
        with tqdm(np.arange(niter), unit="iterations") as tq:
            for _ in tq:
                # Current fit
                fit = self.__psf.conv(self.__codebook.conv(self.barcode_magnitudes)) + self.intercept
                #  Root mean squared error
                RMSE = np.sqrt(torch.mean((fit-self.observed_data)**2).item())
                # Update barcode magnitudes
                self.barcode_magnitudes = self.barcode_magnitudes * self.__codebook.conv_T(
                    self.__psf.conv_T( 
                        self.observed_data / fit
                    )
                ) / (self._denominator + l1_reg + 2*l2_reg*torch.sum(self.barcode_magnitudes,axis=0) + 1e-5)
                
                # Update progress bar
                tq.set_description(f'RMSE: {RMSE:02f}')
        # Convert back to numpy pumpy
        out = self.barcode_magnitudes.detach().cpu().numpy().copy()
        out = np.array(out)
        return out


    def get_reads(self, min_integrated_intensity:float=None,min_radius:float=2, min_fraction_explained:float=None, codebook_keys: Optional[List[str]] = None) -> Dict[str,Any]:
        """Extract position and type of decoded barcodes. 

        Performs a non-max supression to localize decoded barcode in the decoded image data.

        Parameters
        ----------
        min_integrated_intensity : float, optional
            Minimum intensity of a decoded barcode (integrated in disk with radius `min_radius`).
            If set to None, 99.95th percentile of decoded image data is used.
        min_radius : float, optional
            Radius of non-max suppression, by default 2
        min_fraction_explained : float, optional
            A quality score. If a decoded barcode explains a large portion of observed intensities, 
            then `min_fraction_explained` is close to 1, and 
            we can consider that barcode as a high-quality barcode. 
            
            If we expect one spot in each of the `nr` number of rounds, and if the maximum Hamming 
            distance between two codes in the codebook is `d_h`, then a good value for
            min_fraction_explained is `1-d_h/nr`.

            If set to None, an automatic value for min_fraction_explained is computed.
        codebook_keys : list, optional
            List of keys in the codebook that are to be included in the output.
            For example, if a codebook item contains the key `gene_name`,
            the output will have an additional dicionary entry called `gene_name`
            containing the name of each detected read.

        Returns
        -------
        Dictionary with following key-value pairs:
            'x' : np.ndarray
                x position of detected reads.
            'y' : np.ndarray
                y position of detected reads.
            'z' : np.ndarray
                z position of detected reads iff input image data has a 
                    z dimension.
            'target_id' : np.ndarray
                Array with categorical labels (integers) labelling the type of
                the detected read.
            'intensity' : np.ndarray
                Integrated intensities of the detected reads
            'fraction_overlap' : np.ndarray
                Fraction of explained (model fitted) intensities over
                observed intensities. Value close to 1 indicate a high-quality read.
            'round' : np.ndarray
                Array with strings, each string is comma separated list of integers
                indicating in which rounds the found reads is expected to show signal in.
                For instance, a read present in round 0-4 would have a value of "0;1;2;3;4".
                These are values are used by the TissUUmap viewer's In Situ Sequencing Spot_Inspector
                quality control plugin to draw traces across different rounds.
            'channel : np.ndarray  
                Array with strings, each string is comma separated list of integers
                indicating in which channel the found read is expected to show signal in.
                These are values are used by the TissUUmap viewer's In Situ Sequencing Spot_Inspector
                quality control plugin to draw traces across different rounds.


        Example
        -------
            >>> data = ... # Load image data (rounds,channels,y,x) ndarray
            >>> codebook = ... # Load codebook (m,rounds,channels) ndarray
            >>> psf = (2.0, 2.0) # Gaussian PSF with sigma = 2.0 in x amd y.
            >>> # Fit intercepts 
                intercept = ISTDECO.fit_intercept(data, sigma=(5.0, 5.0))
            >>> model = ISTDECO(data,codebook,psf,intercept)
            >>> model.fit(niter=50)
            >>> decoded_table = model.get_reads()

        Notes
        -----


        """        


        if min_integrated_intensity is not None:
            if min_integrated_intensity < 0:
                raise ValueError(f'min_ontegrated_intensity must be a positive float.')
        if min_radius is not None:
            if min_radius < 0:
                raise ValueError(f'min_radius must a non-negative number.')
        if min_fraction_explained is not None:
            if min_fraction_explained < 0 or min_fraction_explained > 1:
                raise ValueError('min_fraction_explained must be a non-negative number betweeon 0 and 1.')

        # We do this in numpy land. Mainly so we can use scikit image
        X_numpy = self.barcode_magnitudes.detach().cpu().numpy().copy()
        # Pool intensities & suppress non maximas
        nsd = int(self.observed_data.ndim - 2) # Number of spatial dimensions
        footprint = disk(min_radius) if nsd == 2 or nsd == 1 else ball(min_radius)
        for i in range(X_numpy.shape[0]):
            small_jitter = np.random.rand(*X_numpy[i].shape) * 1e-5
            mask = local_maxima(X_numpy[i] + small_jitter, connectivity=int(np.ceil(min_radius)))
            X_numpy[i] = convolve(X_numpy[i], footprint)
            X_numpy[i] = X_numpy[i] * mask
        if min_integrated_intensity is None:
            min_integrated_intensity = np.percentile(X_numpy,99.95)

        # Pool intensities spatially in observed data
        Y_numpy = self.observed_data.detach().cpu().numpy().copy()
        intercept_numpy = self.intercept.detach().cpu().numpy().copy()
        Y_numpy = Y_numpy - intercept_numpy
        Y_numpy[Y_numpy < 0] = 0
        footprint = disk(min_radius) if nsd == 2 or nsd == 1 else ball(min_radius)
        for r in range(Y_numpy.shape[0]):
            for c in range(Y_numpy.shape[1]):
                Y_numpy[r,c] = convolve(Y_numpy[r,c], footprint)

        # Fraction of overlap
        codebook_numpy = self.__codebook.codebook.cpu().detach().numpy()
        fraction_overlap = (X_numpy + 1e-5) / (np.einsum('mrc,rc...->m...' ,codebook_numpy, Y_numpy)+1e-5)

        #codebook_gram = np.einsum('mrc,nrc->mn',codebook_numpy,codebook_numpy)
        #foo2 = (X_numpy + 1e-5) / (np.einsum('mn,n...->m...' ,codebook_gram, X_numpy)+1e-5)

        # Find coords
        pks = np.where(X_numpy > min_integrated_intensity)

        if nsd == 3:
            target_id, z,y,x = pks
        elif nsd == 2:
            target_id,y,x = pks
            z = np.zeros(x.shape)

        intensity = np.array([X_numpy[pk] for pk in zip(*pks)])
        fraction_overlap = np.array([fraction_overlap[pk] for pk in zip(*pks)])
        
        codebook_numpy_binary = self.__codebook.codebook_binary.float().cpu().numpy()
        n_expected_spots_in_a_barcode = np.max(codebook_numpy_binary.sum(axis=(1,2)))
        barcode_overlaps = np.einsum('mij,nij->mn',codebook_numpy_binary,codebook_numpy_binary)
        np.fill_diagonal(barcode_overlaps, 0)
        barcode_overlaps = np.max(barcode_overlaps, axis=0)
        fraction_explained = fraction_overlap / n_expected_spots_in_a_barcode

        # For each detected barcode, compute two strings indicating the round & channel we expect to see
        # fluorescent spots in. The strings are semi-colon separated, for instance
        # round = '0;1;2;3;4', channel = '2;2;1;1;2'. 
        # These strings are used by the TissUUmaps viewer to draw traces along rounds and channels. 
        round,channel = zip(*[np.where(codebook_numpy_binary[i]) for i in range(codebook_numpy_binary.shape[0])])
        round = [';'.join(map(str,r)) for r in round]
        channel = [';'.join(map(str,c)) for c in channel]
        round = np.array([round[id] for id in target_id])
        channel = np.array([channel[id] for id in target_id])
        


        if min_fraction_explained is None:
            nrounds = Y_numpy.shape[0]
            min_fraction_explained = barcode_overlaps / nrounds
            min_fraction_explained = np.array([min_fraction_explained[id] for id in target_id])

        ind = fraction_explained > min_fraction_explained
        
        # Save everything in a dictionary
        out = {
            'x' : x[ind],
            'y' : y[ind],
            'z' : z[ind],
            'target_id' : target_id[ind],
            'intensity' : intensity[ind],
            'quality' : fraction_explained[ind],
            'rounds' : round[ind],
            'channels' : channel[ind],
        }
        
        # Add additional codebook items to output
        if codebook_keys is not None:
            if isinstance(codebook_keys, list):
                for item in codebook_keys:
                    val =  [self.codebook_parent.entries[item][id] for id in target_id[ind]]
                    out[item] = val                    
            elif isinstance(codebook_keys, str):
                    val =  [self.codebook_parent.entries[codebook_keys][id] for id in target_id[ind]]
                    out[codebook_keys] = val

        return out


    @staticmethod
    def fit_intercept(data:np.ndarray, sigma:Union[Tuple[float,float,float],Tuple[float,float]]):
        """Fit intercept the intercept term by bluring image data with Gaussian low-pass filter.

        Parameters
        ----------
        data : np.ndarray
            Input iamge data. Must be of shape (r,c,z,y,x)
            or (r,c,y,x).
        sigma : Union[Tuple[float,float,float],Tuple[float,float]]
            Sigma of the Gaussian filter along the y,x axes (or z,y,x axes).

        Returns
        -------
        Image data filtered.

        """

        if data.ndim != 4 and data.ndim != 5:
            raise ValueError('Expected data to have 4 (r,c,y,x) or 5 (r,c,z,y,x) number of dimensions.')
        if data.ndim == 4 and len(sigma) != 2:
            raise ValueError('Input data has 4 dimensions (r,c,y,x), expected sigma to a tuple of 2 elements, specifying width of a Gaussian filter along y and x axis.')
        if data.ndim == 5 and len(sigma) != 3:
            raise ValueError('Input data has 4 dimensions (r,c,z,y,x), expected sigma to a tuple of 2 elements, specifying width of a Gaussian filter along z, y and x axis.')


        intercept = np.zeros_like(data)
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                intercept[r,c] = gaussian_filter(data[r,c],sigma,preserve_range=True)
        intercept = intercept + 1e-5
        return intercept


def istdeco_decode(
    image_data:np.ndarray,
    codebook:Codebook,
    psf_sigma:Union[Tuple[float,float],Tuple[float,float,float]],
    crosstalk:np.ndarray=None,
    min_integration_radius:int=3,
    min_integrated_intensity:float=1000,
    niter:int=50,
    min_fraction_explained:float=None,
    lowpass_sigma:Union[Tuple[float,float],Tuple[float,float,float], None]=None,
    origin:Union[Tuple[float,float],Tuple[float,float,float],None] = None,
    codebook_keys: Optional[List[str]] = None,
    device='cpu'
    ):
    '''Decode data using ISTDECO

        Wrapper function for deconvolving and extracting barcodes from image data.
    
        Parameters
        ----------
        image_data : np.ndarray
            The data that is to be deconvolved. Should be of shape `(r, c, y, x)`
            or `(r, c, z, y, x)`, where `r` is the number of rounds, `c` is the
            number of channels, `z`, `y` and `x` are the size of the 
            spatial dimensions. 

        codebook : Codebook
            Class containing information regarding targeted genes and their
            combinatorial labels. 

        psf_sigma : Union[Tuple[float,float],Tuple[float,float,float]]
            A point-spread-function modelling the shape of the spots.
            Tuple containing sigmas (standard deviations) parameterizing
            the shape of a Gaussian PSF. Roughly 2*sigma should match the radius of a spot.

            If a Gaussian PSF is chosen, convolutions are done in a seperable manner,
            which dramatically speeds up deconvolution when running on the GPU.

        min_integration_radius : float, optional
            Radius of the non-max suppression windows used for detecting 
            deconvolved spot, by default 3

        min_integrated_intensity : float, optional
            Minimum intensity of a decoded barcode (integrated in disk with radius `min_integration_radius`).
            Default 1000. Should 

        min_fraction_explained : float, optional
            A quality score. If a decoded barcode explains a large portion of observed intensities, 
            then `min_fraction_explained` is close to 1, and 
            we can consider that barcode as a high-quality barcode. 
            
            If we expect one spot in each of the `nr` number of rounds, and if the maximum Hamming 
            distance between two codes in the codebook is `d_h`, then a good value for
            min_fraction_explained is `1-d_h/nr`.

            If set to None, an automatic value for min_fraction_explained is computed
            using the above formula.

        lowpass_sigma : Union[Tuple[float,float],Tuple[float,float,float]], optional
            Sigmas parameterizing Gaussian lowpass filters along [z], y and x dimension.
            Rule of thumb is to choose a sigma larger than the radius of a spot. 
            If set to None, `lowpass_sigma` will be set to `5*psf_sigma`.


        origin : Union[Tuple[float,float], None]
            Tuple containing the spatial origin (y,x) of the image data.
            Position of detected barcodes will be shifted according to this origin.
            If set to None (default), no shift will be performed.


        device : str, optional
            Which device to run the deconvolution on, by default 'cpu'.
            Common options are 'cpu' or 'cuda' (GPU). If GPU is available,
            we strongly recommend using it.


        Returns
        -------
        Dictionary with following key-value pairs:
            'x' : np.ndarray
                x position of detected barcodes.
            'y' : np.ndarray
                y position of detected barcodes.
            'z' : np.ndarray
                z position of detected barcodes iff input image data has a 
                    z dimension.
            'target_id' : np.ndarray
                Array with categorical labels (integers) labelling the type of
                the detected barcode.
            'intensity' : np.ndarray
                Integrated intensities of the detected barcodes
            'fraction_overlap' : np.ndarray
                Fraction of explained (model fitted) intensities over
                observed intensities. Value close to 1 indicate a high-quality barcode.
            'round' : np.ndarray
                Array with strings, each string is comma separated list of integers
                indicating in which rounds the found barcode is expected to show signal in.
                For instance, a barcode present in round 0-4 would have a value of "0;1;2;3;4".
                These are values are used by the TissUUmap viewer's In Situ Sequencing Spot_Inspector
                quality control plugin to draw traces across different rounds.
            'channel : np.ndarray  
                Array with strings, each string is comma separated list of integers
                indicating in which channel the found barcode is expected to show signal in.
                These are values are used by the TissUUmap viewer's In Situ Sequencing Spot_Inspector
                quality control plugin to draw traces across different rounds.


        Example
        -------
            >>> data = ... # Load image data (rounds,channels,y,x) ndarray
            >>> codebook = ... # Load codebook (m,rounds,channels) ndarray
            >>> psf = (2.0, 2.0) # Gaussian PSF with sigma = 2.0 in x amd y.
            >>> # Fit intercepts 
                intercept = ISTDECO.fit_intercept(data, sigma=(5.0, 5.0))
            >>> model = ISTDECO(data,codebook,psf,intercept)
            >>> model.fit(niter=50)
            >>> decoded_table = model.get_barcodes()
    
    
    '''

    if image_data.ndim != 5 and image_data.ndim != 4:
        raise ValueError('Input data must be of shape (rounds, channels, z, y, x) or (rounds, channels, y, x).')
    if image_data.ndim == 5 and len(psf_sigma) != 3:
        raise ValueError(f'Input data of shape (rounds, channels, z, y, x) has 3 spatial dimensions (z, y, x), but `psf_sigma` only contains {len(psf_sigma)} elements.')
    if image_data.ndim == 4 and len(psf_sigma) != 2:
        raise ValueError(f'Input data of shape (rounds, channels, y, x) has 2 spatial dimensions (y, x), but `psf_sigma` only contains {len(psf_sigma)} elements.')
    if image_data.ndim == 5 and lowpass_sigma is not None and len(lowpass_sigma) != 3:
        raise ValueError(f'Input data of shape (rounds, channels, z, y, x) has 3 spatial dimensions (z, y, x), but `lowpass_sigma` only contains {len(psf_sigma)} elements.')
    if image_data.ndim == 4 and lowpass_sigma is not None and len(lowpass_sigma) != 2:
        raise ValueError(f'Input data of shape (rounds, channels, y, x) has 2 spatial dimensions (y, x), but `lowpass_sigma` only contains {len(psf_sigma)} elements.')
    if min_integration_radius < 1:
        raise ValueError('`min_integrated_radius` must be atleast 1.')
    if min_integrated_intensity < 0:
        raise ValueError('`min_integrated_intensity` must be non-negative.')
    if min_fraction_explained is not None:
        if min_fraction_explained < 0 or min_fraction_explained > 1:
            raise ValueError('`min_fraction_explained` must be between 0 and 1.')
    if not isinstance(niter,int):
        raise ValueError('Number of iterations must be an integer.')
    if niter < 1:
        raise ValueError('Number of iterations must be >= 1.')
    if lowpass_sigma is None:
        lowpass_sigma = tuple([5*s for s in psf_sigma])

    if origin is not None and len(origin) != 2:
        raise ValueError(f'Origin must be a tuple of two elements (y, x)-location of the image data.')

    intercept = ISTDECO.fit_intercept(image_data, lowpass_sigma) * 1.025
    model = ISTDECO(image_data, codebook, psf_sigma, intercept, crosstalk=crosstalk, device=device)
    model.fit(niter=niter)

    decoded_table = model.get_reads(
        min_integrated_intensity=min_integrated_intensity,
        min_radius=min_integration_radius,
        min_fraction_explained=min_fraction_explained,
        codebook_keys=codebook_keys
    )

    if origin is not None:
        if len(decoded_table['x']) > 0 and len(decoded_table['y']) > 0:
            decoded_table['x'] += origin[1]
            decoded_table['y'] += origin[0]

    return pd.DataFrame(decoded_table)