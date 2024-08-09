import numpy as np
import pandas as pd
import json

from os.path import join
from tools.image_container import ISSDataContainer
from tools.decoding import istdeco, Codebook, calculate_fdr, filter_to_fdr


############################### STEP 0. ###################################
## PARAMETERS
base_path = join('datasets','iss-mouse-cropped')
stain2ch = {'425' : 0, '488' : 1, '568' : 2, '647' : 3}
codebook_path = join(base_path, 'codebook.json')
nrounds, nchannels = 4, 4
fake_codes = {'Fake1', 'Fake2', 'Fake3', 'Fake4', 'Fake5'}
psf_sigma = (2.0, 2.0)  # Assuming that the spots are Gaussian, this should, 
# roughly, correspond to their standard devation. 
# Score threshold (should be in the range 0, 1). A value above 0.7 is usually fine.
# We keep this low for now since we can always filter the signals in post-processing
min_score = 0.25   
min_intensity = 0.001

############################### STEP 1. ###################################
# Add data to image container 
iss_data = ISSDataContainer()
for r in range(nrounds):
    for stain, c in stain2ch.items():
        iss_data.add_image(
            [join(base_path, f'crop_r{r}_{stain}.tif')], 0, r, c
        )


############################### STEP 2. ###################################
# Create the codebook
codebook = Codebook(nrounds, nchannels)
codebook_json = json.load(open(codebook_path, 'r'))
for item in codebook_json['mappings']:
    gene = item['target']
    barcode = np.zeros((nrounds, nchannels))
    for barcode_dok in item['codeword']:
        barcode[barcode_dok['r'], barcode_dok['c']] = barcode_dok['v']
    is_fake = True if gene in fake_codes else False
    codebook.add_code(gene, barcode, is_fake)

for i in range(4):
    barcode = np.zeros((nrounds, nchannels))
    barcode[:,i] = 1.0
    codebook.add_code(f'AUTO-{i}', barcode)



############################### STEP 3. ###################################
# Decode the data.

result_df = []
# This will iterate the date in tiles of size 512 by 512
tile_iterator = iss_data.iterate_tiles(128, 128, squeeze=True, use_vips=True)

for tile, tile_pos in tile_iterator:
    results = istdeco(
        tile, 
        codebook, 
        spot_sigma=1.5,
        min_integrated_intensity=min_intensity,
        min_score=min_score,
        normalizing_percentile=50,
        l1_reg=1e-1,
        device='cuda'
    )
    results['X'] += tile_pos[1]
    results['Y'] += tile_pos[0]
    
    result_df.append(results)
results = pd.concat(result_df, ignore_index=True)

############################### STEP 4. ###################################
# Post-processing
bad_targets = [f'AUTO-{i}' for i in range(4)]
results = pd.concat(result_df, ignore_index=True).query('Score >= 0.65 and Name not in @bad_targets')
fdr = calculate_fdr(results, codebook.get_unexpected())
results, quality_threshold, intensity_threshold = filter_to_fdr(results, codebook, 0.01)
results['Decoder'] = 'ISTDECO'
other_data = pd.read_csv(join(base_path, 'methods.csv'))
all_data = pd.concat([other_data, results], ignore_index=True)
all_data.to_csv(join(base_path, 'results.csv'), index=False)
all_data.query('Decoder == "ISTDECO" and Score >= 0.65').to_csv(join(base_path, 'results_istdeco.csv'), index=False)


