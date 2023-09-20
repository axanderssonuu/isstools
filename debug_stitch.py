from imaging_utils import stitch_ashlar, ISSDataContainer
from os.path import join

# First we load the miped data
iss_data_miped = ISSDataContainer()
iss_data_miped.add_images_from_filepattern(join('MIP','S{stage}_R{round}_C{channel}.tif'))


from imaging_utils import stitch_ashlar
stage_locations = {
    0: (0, 0), 
    1: (0, 1843), 
}


# Stitch using ASHLAR
stitch_ashlar(join('stitched','R{round}_C{channel}.tif'), iss_data_miped, stage_locations, reference_channel=4)