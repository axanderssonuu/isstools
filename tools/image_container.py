from __future__ import annotations

import numpy as np
import re
import os
import shutil
import tempfile
import pathlib
from PIL import Image
from typing import Tuple, Sequence, Dict, Generator, Optional, Callable, List
from os.path import abspath
Image.MAX_IMAGE_PIXELS = 100000000000000


class IncompleteDatasetError(Exception):
    pass



def mip(output_filepattern:str, dataset:ISSDataContainer) -> ISSDataContainer:
    """
    Compute Maximum Intensity Projection (MIP) across a dataset and save the result.

    Args:
        output_filepattern (str): File pattern for saving the MIP images.
        dataset (ISSDataContainer): The dataset containing images for MIP computation.

    Returns:
        ISSDataContainer: The data container containing the MIP images.

    Example:
        ```
        container = ISSDataContainer()
        container.add_images_from_filepattern('S{stage}_R{round}_C{channel}.tif')
        mip_container = mip('mip_round{round}_{channel}.tif', container)
        ```
    """
        

    # The iterate dataset allows us to iterate the dataset over stages, rounds and channels.
    for index, small_dataset in dataset.iterate_dataset(iter_stages=True, iter_rounds=True, iter_channels=True):
        # Load the small dataset
        small_dataset.load()
        # Get the image data
        data = small_dataset.data
        # MIP the data
        data = np.squeeze(data).max(axis=0)
        # Save the data
        imwrite(output_filepattern.format(**index), data)
        # Finally, we unload the images (otherwise we might run oom)
        small_dataset.unload()


    # Load the MIP:ed dataset
    container = ISSDataContainer()
    container.add_images_from_filepattern(output_filepattern)
    return container

def imwrite(filename:str,data:np.array):
    """
    Write a NumPy array as an image file.

    Args:
        filename (str): Path to the output image file.
        data (np.array): NumPy array representing the image data.

    """
    rawtiff=Image.fromarray(data)
    rawtiff.save(filename)

def imread(filename:str) -> np.ndarray:
    """
    Read an image file into a NumPy array.

    Args:
        filename (str): Path to the input image file.

    Returns:
        np.ndarray: NumPy array representing the image data.

    Example:
        ```
        image_data = imread('input_image.tif')
        ```
    """
    return np.asarray(Image.open(filename))
    
def project2d(issdataset: ISSDataContainer, project_fun: Callable[[np.ndarray], np.ndarray], output_filepattern: str) -> ISSDataContainer:
    """
    Project images using a given projection function and save the results.

    Args:
        issdataset (ISSDataContainer): The dataset containing images to project.
        project_fun (Callable[[np.ndarray], np.ndarray]): Function for image projection.
        output_filepattern (str): File pattern for saving the projected images.

    Returns:
        ISSDataContainer: The data container containing the projected images.

    Example:
        ```
        def my_projection(data):
            # Perform custom image projection here
            return data.mean(axis=0)  # Replace with your projection logic

        container = ISSDataContainer()
        container.add_images_from_filepattern('S{stage}_R{round}_C{channel}.tif')
        projected_container = project2d(container, my_projection, 'projected_round{round}_{channel}.tif')
        ```
    """
    num_stages, num_rounds, num_channels = issdataset.get_dataset_shape()
    for stg in range(num_stages):
        for rnd in range(num_rounds):
            for chn in range(num_channels):

                # Select data given the stage, round and channel
                data = issdataset.select(stage=stg, round=rnd, channel=chn)

                # Load into memory
                data.load()

                # Squeeze the dataset
                data = data.get_loaded_images_array().squeeze()

                # Send the data to the projection fun
                data = project_fun(data)

                # Make sure that output is 2D
                if data.ndim != 2:
                    raise ValueError('Projected data must have two dimensions.')
                
                # Write the image given the filepattern
                imwrite(output_filepattern.format(round=rnd, stage=stg, channel=chn), data)
    
    # Return the 2D projected images
    container = ISSDataContainer()
    container.add_images_from_filepattern(output_filepattern)
    return container

def stitch_ashlar(
        output_filepattern: str, 
        issdataset: ISSDataContainer, 
        stage_locations: Dict[int,Tuple[int,int]], 
        reference_channel: int, 
        maximum_shift:int=500, 
        filter_sigma:float=5.0, 
        verbose:bool=True, 
        overlap:float=0.1) -> ISSDataContainer:
    
    """
    Stitch images using the ASHLAR algorithm.

    Args:
        output_filepattern (str): File pattern for the output stitched images.
        issdataset (ISSDataContainer): The ISSDataContainer containing images to stitch.
        stage_locations (Dict[int, Tuple[int, int]]): Dictionary mapping stage indices to (y, x) stage locations.
        reference_channel (int): Index of the reference fluorescent channel.
        maximum_shift (int, optional): Maximum expected shift during alignment. Default is 500.
        filter_sigma (float, optional): Filter sigma for alignment. Default is 5.0.
        verbose (bool, optional): Whether to print verbose messages. Default is True.
        overlap (float, optional): Overlap between adjacent tiles. Default is 0.1.

    Returns:
        ISSDataContainer: The data container containing the stitched images.

    Example:
        The example below demonstrates how to stitch images using ASHLAR and
        save them with the specified output file pattern. The `output_filepattern`
        parameter should contain placeholders like '{round}' and '{cycle}'.

        ```
        container = ISSDataContainer()
        container = stitch_ashlar(
            output_filepattern='stitched_round{round}_{channel}.tif',
            issdataset=container,
            stage_locations={...},
            reference_channel=0,
            maximum_shift=500,
            filter_sigma=5.0,
            verbose=True,
            overlap=0.1
        )
        ```
    """

    import ashlar.scripts.ashlar as ashlar
    from os import mkdir, symlink
    from os.path import exists, join,dirname


        
    output_path = dirname(output_filepattern)
    if not exists(output_path):
        try:
            mkdir(output_path)
        except:
            pass
        #except:
        #   raise OSError(f'Could not create directory {output_path}.')

    if not '{round}' in output_filepattern:
        raise ValueError('Input filepattern must contain placeholder for round id, for instance my_file_{round}_{channel}.tif.')
    if not '{channel}' in output_filepattern:
        raise ValueError('Input filepattern must contain placeholder for channel id, for instance my_file_{round}_{channel}.tif.')
    if filter_sigma < 0:
        raise ValueError('Filter sigma must be a postivie number or None (no filter applied).')
    if maximum_shift < 0:
        raise ValueError('Maximum expected shift must be a positive number.')


    height, width = issdataset.get_common_image_shape()
    ystride = int((1.0-overlap)*height)
    xstride = int((1.0-overlap)*width)
    
    myp = min(item[0] for item in stage_locations.values())
    mxp = min(item[1] for item in stage_locations.values())

    # Map a stage to a grid coordinate
    stage2grid = {
        s : (int((yp-myp)//ystride), int((xp-mxp)//xstride))
            for  s, (yp, xp) in stage_locations.items()
    }

    # Map a stage to a pixel coordinate
    stage2pixel = {
        s : (int(yp-myp), int(xp-mxp))
            for  s, (yp, xp) in stage_locations.items()
    }


    ny = max(item[0] for item in stage2grid.values()) + 1
    nx = max(item[1] for item in stage2grid.values()) + 1

    gridsz = (ny, nx)
    _, rounds, channels = issdataset.get_dataset_shape()

    
    temp_path = join(output_path, '.ashlar_formatted')
    if not exists(temp_path):
        mkdir(temp_path)
    # Create temp data with symlinks
    grid = np.zeros(gridsz, dtype = 'bool')
    FILES = set()

    for image in issdataset.images:
        
        (i, j) = stage2grid[image['stage']]

        cycle_dir = f'round{image["round"]:1d}'
        base_path = join(temp_path, cycle_dir)

        # Create folder incase it does not exists
        if not exists(base_path):
            mkdir(base_path)

        # Path to symlink
        symlink_path = join(
            base_path, 
            'round{round:1}_r{row:1}_c{col:1}_ch{channel:1}.tif'.format(round=image['round'],row=i, col=j, channel=image['channel'])
        )

        # Pattern argument to ASHLAR
        pattern = 'round{round:1}'.format(round=image['round']) + '_r{row:1}_c{col:1}_ch{channel:1}.tif'
        FILES.add('filepattern|'+base_path+f'|pattern={pattern}|overlap={overlap}|pixel_size=1')

        # Create the symlink
        #if os.path.islink(symlink_path):
        #    os.unlink(symlink_path)
            
        
        
        shutil.copy(abspath(image['image_files'][0]), symlink_path)

        # Mark which images in the grid have data
        grid[i, j] = True


    _i = range(grid.shape[0])
    _j = range(grid.shape[1])
    
    # Create "black" images to force rectangular grid.
    if np.any(~grid):
        black_fov = np.zeros((height, width), dtype=issdataset.get_common_dtype())
        dummy_path = join(temp_path, 'dummy.tif')
        # Create the black fov
        imwrite(dummy_path, black_fov)
        for r in rounds:
            for c in channels:
                for i in _i:
                    for j in _j:
                        # If image does not exist, use black image
                        if not grid[i, j]:
                            cycle_dir = f'round{r:1d}'
                            base_path = join(temp_path, cycle_dir)
                            new_path = join(
                                base_path, 
                                'round{round:1}_r{row:1}_c{col:1}_ch{channel:1}.tif'.format(round=r,row=i,col=j,channel=c)
                            )
                            symlink(dummy_path, new_path)

    # Sort all files and call ASHlAR
    FILES = list(sorted(FILES))
    

    # Setup ASHLAR
    aligner_args = {}
    aligner_args['channel'] = reference_channel
    aligner_args['verbose'] = verbose
    aligner_args['max_shift'] = maximum_shift
    if filter_sigma is not None:
        aligner_args['filter_sigma'] = filter_sigma
    mosaic_path_format = str(pathlib.Path('.') / output_filepattern.replace('{round}', '{cycle}'))
    ashlar.process_single(
                FILES, mosaic_path_format, flip_x=False, flip_y=False, 
                ffp_paths=False, dfp_paths=False, aligner_args=aligner_args, mosaic_args={}, 
                pyramid=False, quiet=not verbose
    )

    # Remove all symlinks
    shutil.rmtree(temp_path)


    container = ISSDataContainer()
    container.add_images_from_filepattern(output_filepattern)

    return container


class TileIterator:
    def __init__(self, container: ISSDataContainer, tile_width, tile_height, squeeze, use_vips):
        self.image_data = container.loaded_images
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.image_shape = container.get_common_image_shape()
        self.stages, self.rounds, self.channels = container.get_dataset_indices()
        self.slice_indices = self.calculate_slice_indices()
        self.num_tiles = len(self.slice_indices)
        self.current_tile_index = 0
        self.squeeze = squeeze
        self.use_vips = use_vips
        self.dtype = container.get_common_dtype()


        self.stage2ind = { s : i for i,s in enumerate(self.stages)}
        self.round2ind = { s : i for i,s in enumerate(self.rounds)}
        self.channel2ind = { s : i for i,s in enumerate(self.channels)}

        self.ns, self.nr, self.nc = len(self.stages), len(self.rounds), len(self.channels)


        if use_vips:
            import pyvips

            self.nz = len(container.images[0]['image_files'])
            self.dtype = container.get_common_dtype()

            self.vips_dict = {
                r : {
                    c : []
                    for c in self.channels
                }
                for r in self.rounds
            }

            for image in container.images:
                for file in image['image_files']:

                    self.vips_dict[image['round']][image['channel']].append(
                        pyvips.Image.new_from_file(file)
                    )

     
    
    def calculate_slice_indices(self):
        indices = []
        for h_start in range(0, self.image_shape[-2], self.tile_height):
            h_end = min(h_start + self.tile_height, self.image_shape[-2])
            for w_start in range(0, self.image_shape[-1], self.tile_width):
                w_end = min(w_start + self.tile_width, self.image_shape[-1])
                indices.append((h_start, h_end, w_start, w_end))
        return indices
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.use_vips:
            if self.current_tile_index < self.num_tiles:
                slice_indices = self.slice_indices[self.current_tile_index]
                tile = self.image_data[..., slice_indices[0]:slice_indices[1], slice_indices[2]:slice_indices[3]]
                if self.squeeze:
                    tile = np.squeeze(tile)
                self.current_tile_index += 1
                return tile, (slice_indices[0], slice_indices[2])
            else:
                raise StopIteration
        else:
            if self.current_tile_index < self.num_tiles:
                slice_indices = self.slice_indices[self.current_tile_index]
                h, w = slice_indices[1] - slice_indices[0], slice_indices[3] - slice_indices[2] 
                tile = np.zeros((self.ns, self.nr, self.nc, self.nz, h, w), dtype=self.dtype)
                for ir, r in enumerate(self.rounds):
                    for ic, c in enumerate(self.channels):
                        for iz, vips_image in enumerate(self.vips_dict[r][c]): 
                            tile_slice = vips_image.crop(slice_indices[2], slice_indices[0], w, h)

                            tile_slice_np = np.ndarray(
                                buffer=tile_slice.write_to_memory(),
                                dtype=self.dtype,
                                shape=[h, w, tile_slice.bands]
                            )

                            tile[0, ir, ic, iz, :, :] = tile_slice_np.squeeze()
                if self.squeeze:
                    tile = np.squeeze(tile)
                self.current_tile_index += 1
                return tile, (slice_indices[0], slice_indices[2])
            else:
                raise StopIteration



class ISSDataContainer:

        
    def __init__(self):
        """
        Initialize an ISSDataContainer instance.
        """
        
        self.images = []
        self.image_shape = None  # Store the common image shape
        self.image_dtype = None
        self.loaded_images = None

    def add_image(self, image_files:List[str], stage:int, round:int, channel:int) -> ISSDataContainer:
        """
        Add an image to the data container.

        Args:
            image_files (List[str]): List of image file paths.
            stage (int): Index of the staging location.
            round (int): Index of the sequencing round.
            channel (int): Index of the fluorescent channel.

        Returns:
            ISSDataContainer: The updated data container instance.
        """
    
        if not isinstance(image_files, list) or not all(isinstance(file, str) for file in image_files):
            raise ValueError("image_files should be a list of strings.")
        if not isinstance(stage, int):
            raise ValueError("stage should be an integer.")
        if not isinstance(round, int):
            raise ValueError("round should be an integer.")
        if not isinstance(channel, int):
            raise ValueError("channel should be an integer.")
    
        # Assuming the image shape is determined from the first image in the list
        if self.image_shape is None and image_files:
            self.image_shape = self._get_image_shape(image_files[0])
        if self.image_dtype is None and image_files:
            self.image_dtype = self._get_image_dtype(image_files[0])

        self.images.append({
            'image_files': image_files,
            'stage': stage,
            'round': round,
            'channel': channel,
            'loaded': False
        })
    
        return self

    def load(self) -> ISSDataContainer:
        """
        Load image data into memory.

        Returns:
            ISSDataContainer: The updated data container instance.
        """

        for image in self.images:
            # Simulating loading the image into memory
            image['loaded'] = True
        self.loaded_images = self.get_loaded_images_array()
        return self

    def unload(self) -> ISSDataContainer:
        """
        Unload images from memory.

        Returns:
            ISSDataContainer: The updated data container instance.
        """
        
        for image in self.images:
            # Simulating loading the image into memory
            image['loaded'] = False
            self.loaded_images = None
        return self
    

    @property
    def data(self) -> np.ndarray:
        """
        Get the loaded image data.

        Returns:
            np.ndarray: Loaded image data.
        """
        return self.loaded_images

    def select(self, stage:Optional[int]=None, round:Optional[int]=None, channel:Optional[int]=None) -> ISSDataContainer:
        """
        Select images based on given criteria.

        Args:
            stage (Optional[int]): Index of the staging location.
            round (Optional[int]): Index of the sequencing round.
            channel (Optional[int]): Index of the fluorescent channel.

        Returns:
            ISSDataContainer: New data container with selected images.

        """
                
        selected_images = ISSDataContainer()

        if stage is not None and isinstance(stage, int):
            stage = [stage]

        if round is not None and isinstance(round, int):
            round = [round]

        if channel is not None and isinstance(channel, int):
            channel = [channel]

        if stage is not None and isinstance(stage, slice):
            stage = list(range(stage.start, stage.stop, stage.step if stage.step else 1))

        if round is not None and isinstance(round, slice):
            round = list(range(round.start, round.stop,  round.step if round.step else 1))

        if channel is not None and isinstance(channel, slice):
            channel = list(range(channel.start, channel.stop,  channel.step if channel.step else 1))

        
        for image in self.images:
            if (stage is None or image['stage'] in stage) and \
               (round is None or image['round'] in round) and \
               (channel is None or image['channel'] in channel):
                selected_images.add_image(image['image_files'], image['stage'], image['round'], image['channel'])
        
        return selected_images
    
    def get_loaded_images(self) -> np.ndarray:
        loaded_images = []
        
        for image in self.images:
            if image['loaded']:
                loaded_images.append(image)
        
        return loaded_images
    

    def get_dataset_indices(self) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
        sets = {}
        keys = ['stage','round','channel']
        for key in keys:
            sets[key] = set({})
        for key in keys:
            for image in self.images:
                sets[key].add(image[key])
        for key in keys:
            sets[key] = list(sorted(list(sets[key])))
        return tuple(sets[k] for k in keys)
        
    def get_loaded_images_array(self) -> np.ndarray:
        loaded_images = self.get_loaded_images()
        
        if not loaded_images:
            return None
        
        stages, rounds, channels = self.get_dataset_indices()
        
        stage2ind = { s : i for i,s in enumerate(stages)}
        round2ind = { s : i for i,s in enumerate(rounds)}
        channel2ind = { s : i for i,s in enumerate(channels)}
        

        num_stages, num_rounds, num_channels = self.get_dataset_shape()
        z_planes = max(len(image['image_files']) for image in loaded_images)
        y_size, x_size = None, None
        
        for image in loaded_images:
            y_size, x_size = np.asarray(Image.open(image['image_files'][0])).shape[0:2]
            break
        
        image_array = np.zeros((num_stages, num_rounds, num_channels, z_planes, y_size, x_size), dtype=self.get_common_dtype())
        
        for image in loaded_images:
            stage = image['stage']
            rnd = image['round']
            chn = image['channel']
            z_planes = len(image['image_files'])
            
            for z, image_file in enumerate(image['image_files']):
                image_data =  np.asarray(Image.open(image_file))
                image_array[stage2ind[stage], round2ind[rnd], channel2ind[chn], z, :, :] = image_data[:, :]
        
        return image_array
    
    def get_dataset_shape(self) -> Tuple[int, int, int]:        
        return tuple(len(s) for s in self.get_dataset_indices())
    
    def get_common_image_shape(self) -> Tuple[int, int]:
        return self.image_shape
    
    def _get_image_shape(self, image_file):
        from PIL import Image
        with Image.open(image_file) as img:
            x_size, y_size = img.size
        return y_size, x_size
    
    def _get_image_dtype(self, image_file):
        from PIL import Image
        # Mapping from image mode to data type
        mode_to_dtype = {
            '1': 'bool',
            'L': 'uint8',
            'LA': 'uint8',
            'P': 'uint8',
            'RGB': 'uint8',
            'RGBA': 'uint8',
            'CMYK': 'uint8',
            'YCbCr': 'uint8',
            'I': 'int32',
            'I;16': 'uint16',  # Added mapping for uint16
            'F': 'float32'
        }
        with Image.open(image_file) as img:
            dtype = mode_to_dtype[img.mode]
        return dtype
    

    @property    
    def shape(self) -> Tuple[int, int, int]:
        dataset_shape = self.get_dataset_shape()
        nz = len(self.images[0]['image_files'])
        spatial_shape = self.get_common_image_shape()
        return dataset_shape + (nz,) + spatial_shape
    
    def is_dataset_complete(self) -> bool:
        if not self.images:
            raise IncompleteDatasetError("Dataset is empty.")
        


        channels_per_stage = {}
        rounds_per_stage = {}
        z_planes_per_stage = {}
        stages = set({})
        
        for image in self.images:
            stage = image['stage']
            rnd = image['round']
            chn = image['channel']
    
            if stage not in channels_per_stage:
                channels_per_stage[stage] = [chn]
            else:
                channels_per_stage[stage].append(chn)

            if stage not in rounds_per_stage:
                rounds_per_stage[stage] = [rnd]
            else:
                rounds_per_stage[stage].append(rnd)

            if stage not in z_planes_per_stage:
                z_planes_per_stage[stage] = len(image['image_files'])

            stages.add(stage)

        num_channels_per_stage = {len(chn) for chn in channels_per_stage.values()}
        num_rounds_per_stage = {len(rnd) for rnd in rounds_per_stage.values()}
        num_zplanes_per_stage = {zplanes for zplanes in z_planes_per_stage.values()}

        if len(num_channels_per_stage) != 1:
            raise IncompleteDatasetError('Found different number of channels for a given stage location.')
        if len(num_rounds_per_stage) != 1:
            raise IncompleteDatasetError('Found different number of rounds for a given stage location.')
        if len(num_zplanes_per_stage) != 1:
            raise IncompleteDatasetError('Found different number of z-planes for a given stage location.')
       
        return True
    

    def iterate_tiles(self, tile_width:int, tile_height:int, squeeze: bool = False, use_vips: bool = False) -> TileIterator:
        """
        Iterate over tiles of loaded images.

        Args:
            tile_width (int): Width of each tile.
            tile_height (int): Height of each tile.
            squeeze  (bool)  : Wether the iterated tiles
                should be squeezed or not.
                If set to False, each tile has shape
                (stages, rounds, channels, z, tile_width, tile_height)
            use_vips (bool) : Wether to load the data using Vips.
                Vips allow us to read image data without explicitly
                loading all image files into memory at once.
                This dramatically reduces memory at the expense
                of longer loading time.

        Returns:
            TileIterator: Iterator for image tiles. Each iterable 
            is a tuple with (image_data , image_loc) where
            image_data is an np.ndarray with height and width set to 
            tile_width and tile_height and image_loc is a is a tuple
            with the y and x location of the tile. 
        """  

        if not isinstance(tile_width, int) or tile_width <= 0:
            raise ValueError("tile_width should be a positive integer.")
        if not isinstance(tile_height, int) or tile_height <= 0:
            raise ValueError("tile_height should be a positive integer.")
    
        num_staegs, _, _ = self.get_dataset_shape()
        if num_staegs != 1:
            raise ValueError("The number of stages should be 1 for iterating over tiles.")
        
        if not use_vips:
            if self.loaded_images is None:
                raise ValueError("Images are not loaded yet. Call the 'load' method first.")
            
        return TileIterator(self, tile_width, tile_height, squeeze, use_vips)
    
    

    def iterate_dataset(self, iter_stages:bool=False, iter_rounds:bool=False, iter_channels:bool=False) -> Generator[Dict, np.ndarray]:
        """
        Iterate over the dataset based on given parameters.

        Args:
            iter_stages (bool): Whether to iterate over stage locations.
            iter_rounds (bool): Whether to iterate over sequencing rounds.
            iter_channels (bool): Whether to iterate over fluorescent channels.

        Yields:
            Dict: Dictionary containing keys for iteration parameters.
            np.ndarray: Selected images.
        """
                
        import itertools
        if not isinstance(iter_stages, bool):
            raise ValueError("iter_stages should be a boolean.")
        if not isinstance(iter_rounds, bool):
            raise ValueError("iter_rounds should be a boolean.")
        if not isinstance(iter_channels, bool):
            raise ValueError("iter_channels should be a boolean.")
        
        num_stages, num_rounds, num_channels = self.get_dataset_shape()

        if iter_stages and iter_rounds and iter_channels:
            indices = itertools.product(range(num_stages), range(num_rounds), range(num_channels))
            keys = ['stage', 'round', 'channel']
        elif iter_stages and iter_rounds:
            indices = itertools.product(range(num_stages), range(num_rounds))
            keys = ['stage', 'round']

        elif iter_stages and iter_channels:
            indices = itertools.product(range(num_stages), range(num_channels))
            keys = ['stage', 'channel']

        elif iter_rounds and iter_channels:
            indices = itertools.product(range(num_rounds), range(num_channels))
            keys = ['round', 'channel']
        elif iter_stages:
            indices = itertools.product(range(num_stages))
            keys = ['stage']
        elif iter_rounds:
            indices = itertools.product(range(num_rounds))
            keys = ['round']
        elif iter_channels:
            indices = itertools.product(range(num_channels))
            keys = ['channel']

        for index in indices:
            key = { k : i for k,i in zip(*(keys,index))}
            yield key, self.select(**key)



    def get_common_dtype(self) -> str:
        """
        Get the common data type of images.

        Returns:
            str: Common data type.
        """
        return self.image_dtype
    
    def add_images_from_filepattern(self, filepattern: str, silent: bool = False) -> ISSDataContainer:
        """
        Add images to the data container based on a file pattern.

        Args:
            filepattern (str): File pattern for matching image files.
            silent (bool): Whether to print a message for each tile added.

        Returns:
            ISSDataContainer: The updated data container instance.

        Example:
            Example below shows how to add images on the form
            `S{stage}_R{round}_C{channel}.tif` where {stage},
            {round} and {channel} are placeholders for image files.

            `
                cnt = ISSDataContainer()
                cnt.add_images_from_filepattern('S{stage}_R{round}_C{channel}.tif')
    	    `
        """
        if not isinstance(filepattern, str):
            raise ValueError("filepattern should be a string.")
        
        search_dir = os.path.dirname(filepattern)
        filename_template = os.path.basename(filepattern)

        keys = ['stage', 'round', 'channel', 'z']
        sets = {
            key : set({})
            for key in keys
        }

        if not search_dir or ('/' not in search_dir and '\\' not in search_dir) and search_dir[0] != '.':
            search_dir = '.' + search_dir


        for filename in os.listdir(search_dir):
            match = re.match(filename_template.format(round=r'(?P<round>\d+)', channel='(?P<channel>\d+)', stage=r'(?P<stage>\d+)?', z=r'(?P<z>\d+)?'), filename)
            if match:
                matched_dict = match.groupdict()
                for key in keys:
                    if key in matched_dict:
                        sets[key].add(int(matched_dict[key]))
                        
        rounds = sorted(list(sets['round']))
        stages = sorted(list(sets['stage']))
        channels = sorted(list(sets['channel']))
        zs = sorted(list(sets['z']))

        if not zs:
            zs.append(0)
        if not stages:
            stages.append(0)
                    

        for rnd in rounds:
            for stg in stages:
                for chn in channels:
                    filenames = []
                    for z in zs:
                        filename = filepattern.format(round=rnd, stage=stg, channel=chn, z=z)
                        if os.path.exists(filename):
                            filenames.append(filename)                    
                    self.add_image(filenames, stg, rnd, chn)
                    if not silent:
                        print(f"Added {filename}. Stage: {stg}, Round: {rnd}, Channel: {chn}")


        return self

if __name__ == '__main__':
    # Load ISS data

    from decoding import istdeco_decode, Codebook
    import pandas as pd


    # Load data
    iss_data = ISSDataContainer()
    channels = ['Atto_425','Alexa_488','Alexa_568','Alexa_647']
    chid2str = ['425', '488', '568', '647']
    nrounds = 4
    nchannels = 4
    for r in range(4):
        for c in range(4):
            stain = chid2str[c]
            iss_data.add_image(
                [f'crop_r{r}_{stain}.tif'], 0, r, c
            )
    
    import json
    codebook = json.load(open('iss-mouse\\codebook.json','r'))
    ncodes = len(codebook['mappings'])

    unique_genes = []
    codebook_np = np.zeros((ncodes, nrounds, nchannels))
    gene_list = []

    for g,code_data in enumerate(codebook['mappings']):
        item = {}
        item['is_negative'] = False
        item['target'] = code_data['target']
        for r in range(4):
            for c in range(4):
                item[f'r{r}_c{c}'] = 0

        for items in code_data['codeword']:
            r, c, v = items['r'], items['c'], items['v']
            item[f'r{r}_c{c}'] = v
        gene_list.append(item)


    codebook = Codebook(gene_list,4,4)


    # Run the decoding
    results = []
    tile_idx = 1
    for tile, origin in iss_data.iterate_tiles(tile_height=512, tile_width=512, squeeze=True, use_vips=True):
        print(f'Decoding tile: {tile_idx}')
        tile_idx += 1
        # Decode the data using matrix factorization

        # Depending on your data, you might want to adjust the parameter min_integrated_intensity
        # or min_correct_spots
        # Usually a quality threshold between 0.5 and 0.85 works fine. 

        # This is really slow unless we can run on the GPU.

        import matplotlib.pyplot as plt
        #plt.figure(); plt.imshow(tile.sum(axis=(0,1))); plt.show()
        decoded_table = istdeco_decode(tile, codebook, psf_sigma=(2.0, 2.0), min_barcode_quality=0.1, min_barcode_intensity=1e-5)
        decoded_table.to_csv('results.csv')


    results.to_csv('results.csv', index=False)