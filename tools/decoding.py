import torch
import numpy as np
import pandas as pd
from typing import Sequence, List, Tuple, Any
from scipy.ndimage import convolve
from skimage.morphology import disk, local_maxima
from skimage.filters import gaussian as gaussian_filter


class Codebook:

    def __init__(self, nrounds: int, nchannels: int) -> None:
        self.nrounds = nrounds
        self.nchannels = nchannels
        self.code_ids = []
        self._code_arrays = []
        self.codebook_mat = None
        self.unexpected_codes = []

    def add_code(self, id: str,  codeword: np.ndarray, is_unexpected: bool = False):
        if codeword.shape != (self.nrounds, self.nchannels):
            raise ValueError(f'Codeword must a matrix of size num_rounds x num_channels.')
        if id in self.code_ids: 
            raise ValueError(f'Code with id: {id} already in codebook.')
        self.code_ids.append(id)
        self._code_arrays.append(codeword)
        self.codebook_mat = np.array(self._code_arrays)
        if is_unexpected:
            self.unexpected_codes.append(id)
    
    def remove_code(self, id: str):
        if id in self.code_ids:
            ind = self.code_ids.index(id)
            del self.code_ids[ind]
            del self._code_arrays[ind]
            self.codebook_mat = np.array(self._code_arrays)
            if id in self.unexpected_codes:
                ind = self.unexpected_codes.index(id)
                del self.unexpected_codes[ind]

    def get_unexpected(self) -> List:
        return self.unexpected_codes
    def get_expected(self) -> List:
        return [c for c in self.code_ids if c not in self.unexpected_codes]
    

def discretize(input_array: np.ndarray, nsteps: int = 100) -> np.ndarray:
    """Discretize an input array by sorting and dividing into nsteps bins."""
    sorted_array = np.sort(input_array)
    bin_indices = np.linspace(0, len(sorted_array) - 1, nsteps, dtype=int)
    return sorted_array[bin_indices]

def calculate_fdr(targets: Sequence, unexpected: Sequence) -> float:
    """Calculate the false discovery rate."""
    is_unexpected = np.array([t in unexpected for t in targets])
    expected = {t for t in targets if t not in unexpected}
    num_expected_observations = np.sum(~is_unexpected)
    num_unexpected_observations = np.sum(is_unexpected)
    fdr = (num_unexpected_observations / (num_expected_observations + num_unexpected_observations + 1e-8)) * (len(expected) / len(unexpected))
    return fdr

def filter_to_fdr(
    decoded_table: pd.DataFrame, codebook: Codebook, target_fdr: float = 0.01
) -> Tuple[pd.DataFrame, float, float]:
    """Filter decoded table to achieve a target false discovery rate (FDR)."""
   
    # Extracting data from the decoded_table
    fraction_explained = np.array(decoded_table['Score'])
    intensity = np.array(decoded_table['Intensity'])
    is_negative_reads = np.array(decoded_table['Is unexpected'])

    # Separating gene IDs into negative and positive IDs based on the codebook
    unique_negative_ids = codebook.get_unexpected()
    unique_positive_ids = codebook.get_expected()
    nsteps = 300

    # Discretize quality and intensity thresholds for grid search
    quality_thresholds = discretize(fraction_explained, nsteps)
    intensity_thresholds = discretize(intensity, nsteps)

    # Initialize grids for count and FDR
    count_grid = np.zeros((nsteps, nsteps))
    fdr_grid = np.zeros((nsteps, nsteps))

    # Grid search for optimal thresholds
    for i, quality_threshold in enumerate(quality_thresholds):
        # Apply quality threshold to identify target IDs with high enough quality
        ind_quality = fraction_explained >= quality_threshold
        intensity_quality = intensity[ind_quality]

        for j, intensity_threshold in enumerate(intensity_thresholds):
            # Apply intensity threshold to further filter target IDs
            ind_intensity = intensity_quality >= intensity_threshold
            reads_passing_thresholds = is_negative_reads[ind_quality][ind_intensity]

            # Calculate count and FDR for the current thresholds
            num_detected = len(reads_passing_thresholds)
            num_positives = np.sum(~reads_passing_thresholds)
            num_negatives = len(reads_passing_thresholds) - num_positives
            fdr = num_negatives / (num_positives + num_negatives + 1e-8) * len(unique_positive_ids) / len(unique_negative_ids)
            count_grid[i, j] = num_detected
            fdr_grid[i, j] = fdr

    # Find indices of thresholds that satisfy the target FDR constraint
    row_indices, col_indices = np.where(fdr_grid <= target_fdr)
    counts = count_grid[row_indices, col_indices]
    best_index = np.argmax(counts)

    # Extract optimal thresholds
    best_intensity_threshold = intensity_thresholds[col_indices[best_index]]
    best_quality_threshold = quality_thresholds[row_indices[best_index]]

    # Apply the optimal thresholds to select target IDs
    target_ids_selected = is_negative_reads[(fraction_explained >= best_quality_threshold) & (intensity >= best_intensity_threshold)]

    # Calculate FDR with the selected target IDs
    num_positives = np.sum(~target_ids_selected)
    num_negatives = len(target_ids_selected) - num_positives
    fdr = num_negatives / (num_positives + num_negatives + 1e-8) * len(unique_positive_ids) / len(unique_negative_ids)

    return decoded_table.query('Score >= @best_quality_threshold and Intensity >= @best_intensity_threshold').copy().reset_index(drop=True), best_quality_threshold, best_intensity_threshold






class ISTDecoModel:

    def __init__(self, data: np.ndarray, codebook_mat: np.ndarray, spot_sigma: float, l1_reg:float, l2_reg:float, tensor_args: Any) -> None:
        _, _, height, width = data.shape
        num_targets = codebook_mat.shape[0]
        self.l2_reg = l2_reg
        self.Y = torch.tensor(data, **tensor_args)
        self.X = torch.ones((num_targets, height, width), **tensor_args)
        self.D = torch.tensor(codebook_mat, **tensor_args)
        self.Gh = self._make_gaussian_blur(spot_sigma, height, tensor_args)
        self.Gw = self._make_gaussian_blur(spot_sigma, width, tensor_args)
        self.scale = torch.ones_like(self.Y)
        self.scale = self.back_project(self.scale) + l1_reg

    def back_project(self, x: torch.Tensor):
        x = self.backward_blur(x)
        x = torch.einsum('rc...,mrc->m...', x, self.D)
        return x
    

    def forward_blur(self, x: torch.Tensor):
        x = torch.einsum('...hi, iw->...hw', x, self.Gw)
        x = torch.einsum('...iw,ih->...hw', x, self.Gh)
        return x
    
    def backward_blur(self, x: torch.Tensor):
        x = torch.einsum('...hi,wi->...hw', x, self.Gw)
        x = torch.einsum('...iw,hi->...hw', x, self.Gh)
        return x
    def forward(self):
        x = torch.einsum('m...,mrc->rc...', self.X , self.D)
        x = self.forward_blur(x)
        x = x + 1e-6
        return x
    
    def update(self, fit: torch.Tensor):
        c = self.back_project(self.Y / fit) / (self.scale + self.l2_reg * self.X)
        self.X = c * self.X 

    def update_spatially(self):
        Y = self.X.clone()
        X = self.X.clone()
        for _ in range(50):
            c = self.backward_blur( Y / (self.forward_blur(X) + 1e-6))
            X = c * X
        self.X = X
    def to_numpy(self):
        return self.X.cpu().numpy()
    
    def _make_gaussian_blur(self, spot_sigma: float, size: int, tensor_args: Any):
        x = np.arange(size).reshape(-1, 1)
        x = np.exp(-0.5 * (x.T - x)**2 / spot_sigma**2)
        x = x / x.sum(axis=0, keepdims=True)
        return torch.tensor(x, **tensor_args)



    


def high_pass(data: np.ndarray, sigma: float):
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            data[r,c] = data[r,c] - gaussian_filter(data[r,c], sigma, preserve_range=True) - 1e-5
            data[r,c] = np.maximum(data[r,c], 0.0)
    return data


def band_pass(data: np.ndarray, sigma_low: float, sigma_high: float):
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            data[r,c] = gaussian_filter(data[r,c], sigma_low, preserve_range=True) - gaussian_filter(data[r,c], sigma_high, preserve_range=True) - 1e-5
            data[r,c] = np.maximum(data[r,c], 0.0)
    return data


def normalize(data: np.ndarray, prc: float):
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            data[r,c] = data[r,c]  / (np.percentile(data[r,c], prc, keepdims=True) + 1e-5)
    return data


def nonmaxsuppress(X: np.ndarray, radius: float):
    from scipy.ndimage import grey_dilation
    footprint = disk(radius)
    X_max = X.max(axis=0)
    X_max = grey_dilation(X_max, footprint=footprint)
    mask = X == np.expand_dims(X_max, 0)
    for i in range(X.shape[0]):
        X[i] = convolve(X[i], footprint)
    return X * mask

def quality(X: np.ndarray, Y: np.ndarray, codebook: np.ndarray, radius: np.ndarray):
    footprint = disk(radius)
    for r in range(Y.shape[0]):
        for c in range(Y.shape[1]):
            Y[r,c] = convolve(Y[r,c], footprint)
    Y_fit = np.einsum('mrc,m...->rc...' ,codebook, X)
    Y_fit = Y_fit / (np.linalg.norm(Y_fit,axis=(0,1), keepdims=True) + 1e-5)
    Y = Y / (np.linalg.norm(Y,axis=(0,1), keepdims=True) + 1e-5)
    cosine_similarity = (Y_fit * Y).sum(axis=(0,1))
    cosine_similarity = np.expand_dims(cosine_similarity, 0)
    cosine_similarity = cosine_similarity * (X > 0)
    return cosine_similarity


def istdeco(
    data: np.ndarray,
    codebook: Codebook,
    spot_sigma: float=1.5,
    highpass_sigma: float | None = None,
    min_score: float = 0.5,
    min_integrated_intensity: float | None = None,
    niter: int = 50,
    normalizing_percentile: float = 99.5,
    l1_reg: float = 0.05,
    l2_reg: float = 0.0,
    device: str='cpu',
    return_fit: bool = False
):
    

    output_columns = ['X', 'Y', 'Name', 'Is unexpected', 'Intensity', 'Score', 'Rounds', 'Channels']
    output_dict = {c : [] for c in output_columns}
    if data.sum() == 0:
        return output_dict

    if highpass_sigma is None:
        highpass_sigma = spot_sigma * 3.0


    num_targets = len(codebook.codebook_mat)

    # Pre-process
    data = data.astype('float32')
    #data = band_pass(data, spot_sigma / 2, spot_sigma * 3)
    #data = high_pass(data, highpass_sigma)
    data = normalize(data, normalizing_percentile)
    if return_fit:
        data_out = data.copy()

    if min_integrated_intensity is None:
        min_integrated_intensity = 0.5

    tensor_args = {'device': device, 'dtype': torch.float32, 'requires_grad' : False}
    mdl = ISTDecoModel(data, codebook.codebook_mat, spot_sigma, l1_reg, l2_reg, tensor_args)

    for _ in range(niter):
        fit = mdl.forward()
        mdl.update(fit)
    #mdl.update_spatially()

    # Pick out codes
    X = mdl.to_numpy()
    if return_fit:
        X_out = X.copy()
    Y = data

    # Run non max suppression
    spot_radius = 2*spot_sigma
    X = nonmaxsuppress(X, spot_radius)
    # Compute quality
    Q = quality(X, Y, codebook.codebook_mat, spot_radius)

    # Prepare output
    unexpected = set(codebook.get_unexpected())
    target_ind, y, x = np.where(X > min_integrated_intensity)
    
    
    output_dict['Name'] = [codebook.code_ids[id] for id in target_ind]
    output_dict['Is unexpected'] = [n in unexpected for n in output_dict['Name']]
    output_dict['Intensity'] = X[target_ind, y, x].tolist()
    output_dict['Score'] = Q[target_ind, y, x].tolist()
    output_dict['X'] = x.tolist()
    output_dict['Y'] = y.tolist()

    round, channel = zip(*[np.where(codebook.codebook_mat[i]) for i in range(num_targets)])
    round = [';'.join(map(str,r)) for r in round]
    channel = [';'.join(map(str,c)) for c in channel]
    output_dict['Rounds'] = [round[id] for id in target_ind]
    output_dict['Channels'] = [channel[id] for id in target_ind]
    output_dict = pd.DataFrame(output_dict)
    output_dict = output_dict.query('Score >= @min_score').reset_index(drop=True)
    if not return_fit:
        return output_dict
    else:
        return output_dict, X_out, data_out
 #   return output_dict if not return_fit else output_dict, X_out, data_out





