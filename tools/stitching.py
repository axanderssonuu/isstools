from typing import Tuple, Dict
import numpy as np
from .image_container import ISSDataContainer
import pathlib
from ashlar import reg

class _ASHLARMetadata(reg.Metadata):
    
    def __init__(self, container: ISSDataContainer, fov_locations, overlap):
        self.overlap = overlap
        self._pixel_size = 1
        self.dataset = container
        self.fov_locations = fov_locations
        self.nfovs, self.nrounds, self.nchannels, self.nz, self.tile_height, self.tile_width = self.dataset.shape
        self._enumerate_tiles()

    def _enumerate_tiles(self):
        rows = set()
        cols = set()
        channels = set()
        n = 0
        for (key, _) in self.dataset.iterate_dataset(iter_stages=True, iter_channels=True):
            fov, channel = key['stage'], key['channel']
            row, col = self.fov_locations[fov]
            rows.add(row)
            cols.add(col)
            channels.add(channel)
            n += 1

        if n != len(rows) * len(cols) * len(channels):
            raise Exception("Tiles do not form a full rectangular grid")

        self._actual_num_images = len(rows) * len(cols)
        self.channel_map = dict(enumerate(sorted(channels)))
        self.height = len(rows)
        self.width = len(cols)
        self.row_offset = min(rows)
        self.col_offset = min(cols)
        
        self._tile_size = np.array((self.tile_height, self.tile_width))
        self.multi_channel_tiles = False
        # Handle multi-channel tiles (pattern must not include channel).
        if self.nz > 1:
            self.channel_map = {c: None for c in range(self.nz)}
            self.multi_channel_tiles = True
        self._num_channels = len(self.channel_map)

    @property
    def _num_images(self):
        return self._actual_num_images

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def pixel_dtype(self):
        return np.uint16

    def tile_position(self, i):
        row, col = self.tile_rc(i)
        return [row, col] * self.tile_size(i) * (1 - self.overlap)

    def tile_size(self, i):
        return self._tile_size

    def tile_rc(self, i):
        row = i // self.width + self.row_offset
        col = i % self.width + self.col_offset
        return row, col



class _ASHLARReader(reg.Reader):

    def __init__(self, dataset: ISSDataContainer, round:int, fov_locations, overlap):
        # See FilePatternMetadata for an explanation of the pattern syntax.
        self.dataset = dataset.select(round=round)
        self.overlap = overlap
        self.metadata = _ASHLARMetadata(
            dataset, fov_locations, overlap
        )

    def read(self, series, c):
        img = self.dataset.select(stage=series, channel=c).load().data
        return img.squeeze()




def stitch(
        dataset:ISSDataContainer, 
        output_filepattern:str, 
        fov_locations: Dict[int,Tuple[int,int]], 
        reference_channel: int,
        overlap:float=0.2,
        max_shift:int=500,
        filter_sigma:float=10
    ):

    writer_args = {}
    mosaics = []
    mosaic_args = {}

    aligner_args = {}
    aligner_args['channel'] = reference_channel
    aligner_args['verbose'] = 1
    aligner_args['max_shift'] = max_shift
    aligner_args['filter_sigma'] = filter_sigma
    quiet = False
    output_format = str(pathlib.Path('.') / output_filepattern.replace('{round}', '{cycle}'))

    reader = _ASHLARReader(dataset, 0, fov_locations, overlap)
    ea_args = aligner_args.copy()
    edge_aligner = reg.EdgeAligner(reader, **ea_args)
    edge_aligner.run()
    mshape = edge_aligner.mosaic_shape
    mosaic_args_final = mosaic_args.copy()
    mosaics.append(reg.Mosaic(edge_aligner, mshape, **mosaic_args_final))
    stages, rounds, channels = dataset.get_dataset_indices()
    for round in rounds:
        if round > 0:
            reader = _ASHLARReader(dataset, round, fov_locations, overlap)
            layer_aligner = reg.LayerAligner(reader, edge_aligner, **aligner_args)
            layer_aligner.run()
            mosaic_args_final = mosaic_args.copy()
            mosaics.append(reg.Mosaic(layer_aligner, mshape, **mosaic_args_final))

    # Disable reader caching to save memory during mosaicing and writing.
    edge_aligner.reader = edge_aligner.reader.reader
    writer_class = reg.TiffListWriter
    writer = writer_class(
        mosaics, output_format, verbose=not quiet, **writer_args
    )
    writer.run()


if __name__ == '__main__':
    # Create the container
    issdata = ISSDataContainer()

    from os.path import join
    # First we load the miped data
    iss_data_miped = ISSDataContainer()
    iss_data_miped.add_images_from_filepattern(r'C:\Users\axela\Documents\GitHub\isstools\datasets\tutorial\mipped' + '\\S{stage}_R{round}_C{channel}.tif')

    fov_locations = {
    0: (0, 0), 
    1: (0, 1),
    }


    stitch(iss_data_miped, join('stitched','R{round}_C{channel}.tif'), fov_locations, reference_channel=4)