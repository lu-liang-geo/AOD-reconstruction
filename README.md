# MCD19A2 AOD Reconstruction
This code provides both the deep learning model files as well as the code for pre-processing and reconstruction the AOD data.

## Usage Notes
The `main.py` file contains all code required to take a series of raw MCD19A2 HDF-EOS files and save a reconstructed counterpart. This specific functionality provided can be modified using the flags at runtime, as follows:  
**-h**: Help. Shows basic information on the various flags.  
**-i** FPS: File pattern for MCD19A2 HDF files  
**-in** NLCD_FP: Filepath to the NLCD data for masking water bodies  
**-is** SHP_FP: Filepath to the shapefile for cropping  
**-od** OUTPUT_DIR: Output directory for the reconstructed data. Default: Current directory  
**-px** OVERLAP_PX: Number of pixels to overlap when tiling  
**-ln** LOCATION_NAME: Name of the location (for output file naming)  
**-c** OUT_CRS: The desired CRS for the output files  
**-r**: Apply random shift (6px to 15px) for overlap