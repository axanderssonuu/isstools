# ISS Decoding Pipeline


This repositor provides instructions for setting up and installing a simple decoding pipeline for in situ sequencing. This repository mainly contains tools used by Wahlby  group (Uppsala University) for decoding in situ sequencing data.

## Getting Started
Make sure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

### Setting Up Conda Environment

Run the following commands from a terminal (Linux/Mac) or command prompt (Windows):

Create conda environtment with Python 3.10

   ```bash
   conda create -y -n isstools python=3.10
   ```

Activate the conda environment:

   ```bash
   conda activate isstools
   ```

In the activated environment, install the dependencies:

    ```bash
    conda install -y -c conda-forge numpy scipy matplotlib networkx scikit-image=0.19 scikit-learn "tifffile>=2023.3.15" zarr pyjnius blessed
    ```

These dependencies are necessary for [Ashlar](https://github.com/labsyspharm/ashlar), a Python package for image stitching and registration.

Run the following command to install Ashlar:

    ```bash
    pip install ashlar
    ```

Finally, install additional dependencies:

    
    ```bash
        pip install pandas pillow torch psfmodels tqdm
    ```

We recommend using [TissUUmaps](https://github.com/TissUUmaps/TissUUmaps) for visualization and quality control.

### Usage
Now that you have set up the environment and installed the required packages, you're ready to use the pipeline. See the notebook
[tutorial.ipynb](tutorial.ipynb) for usage.


 