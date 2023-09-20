# The iterate dataset allows us to iterate the dataset over stages, rounds and channels.
import numpy as np
from imaging_utils import imwrite
from os.path import join

from imaging_utils import ISSDataContainer

# Create the container
issdata = ISSDataContainer()

# Add images
# join('downloads', 'stage{stage}_rounds{round}_z{z}_channel{channel}.tif')
pattern = 'decoding_tutorial\\S{stage}_R{round}_C{channel}_Z{z}.tif'
issdata.add_images_from_filepattern(pattern)



for index, small_dataset in issdata.iterate_dataset(iter_stages=True, iter_rounds=True, iter_channels=True):
    # Load the small dataset
    small_dataset.load()
    # Get the image data
    data = small_dataset.data
    # MIP the data
    data = np.squeeze(data).max(axis=0)
    # Save the data
    imwrite(join('MIP','S{stage}_R{round}_C{channel}.tif'.format(**index)), data)
    # Finally, we unload the images (otherwise we might run oom)
    small_dataset.unload()

# Or equivalently ...
# from ISSDataset import mip
# mip(join('mip','stage{stage}_round{round}_channel{channel}.tif'), issdata)
