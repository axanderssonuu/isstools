import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="In Situ Transcriptomics Tools",
    version='1.0',
    author="Axel Andersson",
    author_email="axel.andersson@it.uu.se",
    description="Pipeline for decoding In Situ Transcriptomics (IST) data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD 3-Clause License",
    url="https://github.com/wahlby-lab/ISTDECO/tree/main",
    packages=["isttools"],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "scipy>=1.7.0",
        "scikit-image>=0.19.2",
        "scikit-learn>=1.1.0",
        "tqdm>=4.64.0",
        "numpy>=1.21.6",
        "matplotlib>=3.5.2",
        "h5py>=3.6.0",
        "pillow>=9.1.1",
        "pandas>=1.4.2",
        "ashlar>=1.16.0",
        "torch>=1.11.0",
        "tissuumaps>=3.0.9.5",
        "napari>=0.4.16",
        "psfmodels>=0.3.2",
        "wget>=3.2"
    ]
)