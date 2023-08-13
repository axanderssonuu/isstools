from __future__ import annotations

import numpy as np
import re
import os
import shutil
import tempfile
import pathlib
from PIL import Image

from typing import Tuple, Sequence, Dict, Generator, Optional, Callable

Image.MAX_IMAGE_PIXELS = 100000000000000


class IncompleteDatasetError(Exception):
    pass



def mip(output_filepattern:str, dataset:ISSDataContainer) -> ISSDataContainer:
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
    rawtiff=Image.fromarray(data)
    rawtiff.save(filename)

def imread(filename:str) -> np.ndarray:
    return np.asarray(Image.open(filename))
    
def project2d(issdataset: ISSDataContainer, project_fun: Callable[[np.ndarray], np.ndarray], output_filepattern: str) -> ISSDataContainer:
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

def stitch_ashlar(output_filepattern: str, issdataset: ISSDataContainer, stage_locations: Dict[int,Tuple[int,int]], reference_channel: int, maximum_shift:int=500, filter_sigma:float=5.0, verbose:bool=True, overlap:float=0.1) -> ISSDataContainer:

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

    temp_path = tempfile.mkdtemp()
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
        symlink(image['image_files'][0], symlink_path)

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
    def __init__(self, image_data, tile_width, tile_height):
        self.image_data = image_data
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.image_shape = image_data.shape
        self.slice_indices = self.calculate_slice_indices()
        self.num_tiles = len(self.slice_indices)
        self.current_tile_index = 0
    
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
        if self.current_tile_index < self.num_tiles:
            slice_indices = self.slice_indices[self.current_tile_index]
            tile = self.image_data[..., slice_indices[0]:slice_indices[1], slice_indices[2]:slice_indices[3]]
            self.current_tile_index += 1
            return tile, slice_indices
        else:
            raise StopIteration


class ISSDataContainer:

        
    def __init__(self):



        self.images = []
        self.image_shape = None  # Store the common image shape
        self.image_dtype = None
        self.loaded_images = None

    def add_image(self, image_files:str, stage:int, round:int, channel:int) -> ISSDataContainer:
   
    
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

        for image in self.images:
            # Simulating loading the image into memory
            image['loaded'] = True
        self.loaded_images = self.get_loaded_images_array()
        return self

    def unload(self) -> ISSDataContainer:
        
        for image in self.images:
            # Simulating loading the image into memory
            image['loaded'] = False
            self.loaded_images = None
        return self
    @property
    def data(self) -> np.ndarray:
        return self.loaded_images

    def select(self, stage:Optional[int]=None, round:Optional[int]=None, channel:Optional[int]=None) -> ISSDataContainer:
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
    
    def get_common_image_shape(self) -> Tuple[int, int, int]:
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
    

    def is_dataset_complete(self) -> bool:
        if not self.images:
            raise IncompleteDatasetError("Dataset is empty.")
        
        stages_per_round = {}
        z_planes_per_stage_round_channel = {}
        
        for image in self.images:
            stage = image['stage']
            rnd = image['round']
            chn = image['channel']
            z_planes = len(image['image_files'])
            if stage not in stages_per_round:
                stages_per_round[stage] = {rnd: [chn]}
            else:
                if rnd not in stages_per_round[stage]:
                    stages_per_round[stage][rnd] = [chn]
                else:
                    stages_per_round[stage][rnd].append(chn)

        for image in self.images:
            stage = image['stage']
            rnd = image['round']
            chn = image['channel']
            z_planes = len(image['image_files'])
            if stage in stages_per_round:
                if rnd not in stages_per_round[stage] or chn not in stages_per_round[stage][rnd]:
                    raise IncompleteDatasetError("Dataset is incomplete.")
    
            if (stage, rnd, chn) in z_planes_per_stage_round_channel:
                if z_planes != z_planes_per_stage_round_channel[(stage, rnd, chn)]:
                    raise IncompleteDatasetError("Dataset is incomplete.")
       
        
        return True
    

    def iterate_tiles(self, tile_width:int, tile_height:int) -> TileIterator:
  
                
        if not isinstance(tile_width, int) or tile_width <= 0:
            raise ValueError("tile_width should be a positive integer.")
        if not isinstance(tile_height, int) or tile_height <= 0:
            raise ValueError("tile_height should be a positive integer.")
    
        num_staegs, _, _ = self.get_dataset_shape()
        if num_staegs != 1:
            raise ValueError("The number of stages should be 1 for iterating over tiles.")
        
        if self.loaded_images is None:
            raise ValueError("Images are not loaded yet. Call the 'load' method first.")
        
        return TileIterator(self.loaded_images, tile_width, tile_height)
    
    

    def iterate_dataset(self, iter_stages:bool=False, iter_rounds:bool=False, iter_channels:bool=False) -> Generator[Dict, np.ndarray]:

                
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
        return self.image_dtype
    
    def add_images_from_filepattern(self, filepattern: str) -> ISSDataContainer:
        if not isinstance(filepattern, str):
            raise ValueError("filepattern should be a string.")
        
        search_dir = os.path.dirname(filepattern)
        filename_template = os.path.basename(filepattern)

        keys = ['stage', 'round', 'channel', 'z']
        sets = {
            key : set({})
            for key in keys
        }

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
                    print(f"Added {filename}. Stage: {stg}, Round: {rnd}, Channel: {chn}")


        return self

if __name__ == '__main__':
    # Load ISS data
    container = ISSDataContainer()
    container.add_images_from_filepattern('C:\\Users\\Axel\\Documents\\ISTDECO\\downloads\\liver_2d\\liver_2d_locid{stage}_r{round}_c0{channel}.tif')
    container.load()

    for a,b in container.iterate_dataset(iter_rounds=True):
        print(a,b)

    small = container.select(stage=0)
    small.load()
    iterator = small.iterate_tiles(tile_width=512, tile_height=512)
    for a in iterator:
        print(a)


    stage_locations = {
        0: (0, 0), 
        1: (0, 1843), 
        2: (0, 3686), 
        3: (0, 5529), 
        4: (1843, 0), 
        5: (1843, 1843), 
        6: (1843, 3686), 
        7: (1843, 5529), 
        8: (3686, 0), 
        9: (3686, 1843), 
        10: (3686, 3686), 
        11: (3686, 5529), 
        12: (5529, 0), 
        13: (5529, 1843), 
        14: (5529, 3686), 
        15: (5529, 5529)
    }


    stitch_ashlar(
        'R{round}_C{channel}.tif',
        container,
        stage_locations,
        reference_channel=4
    )

