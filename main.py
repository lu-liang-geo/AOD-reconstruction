"""
Takes sequential days of MCD19A2 AOD data in the raw HDF-EOS form
and reconstructs the corresponding AOD imagery.

There must be at least 4 days of imagery prior to the day for which
reconstruction is desired for the method to work. Batch processing
is available just by including a range of sequential days worth of
imagery.

Usage example:
python main.py \
    -i "path/to/hdf_files" \
    -in path/to/nlcd.tif \
    -is path/to/crop.shp \
    -od outputs \
    -ln geographic_name \
    -c EPSG:4326 \
    -r

To see the various flags and their uses, please use:
python main.py -h

"""

import glob, os
from rioxarray.merge import merge_arrays
import xarray as xr
import numpy as np
from numpy.random import randint
import rasterio
import datetime
from tensorflow.keras.models import load_model
import geopandas as gpd
import time
import argparse

# Custom packages
import geoext.MODIS as MODIS
from geoext.geo import patch_rxr

# Path to the reconstruction model
model_fp = 'models/model_mcd19a2_fill'
# Stack count (how many sequential days worth of data to obtain)
STACK_CT = 5

###############################################################################

def extract_info_from_fp(
    fp: str
):
    """ Extract information about the file from its filename """
    # Get the original file name, without the extension
    # and extract important information
    orig_fn = '.'.join(os.path.basename(fp).split('.')[:-1])
    product, date_j, tile, *_ = orig_fn.split('.')
    # The date is formatted as <A/T><julien_date>, where A/T
    # is whether the image is aqua or terra.
    # Since we don't care, we truncate that and leave julien date
    date_j = date_j[1:]
    # We then convert to 'standard' date format (i.e. the format
    # that I want)
    date = datetime.datetime.strptime(date_j, '%Y%j').date()
    # Date converted to string
    date = date.strftime("%Y-%m-%d")

    return product, date, tile

def input_handler():
    # If true, will use a random overlap between 6 and 15 when tiling
    # the whole image. This provides a smoother composite image when
    # aggregating data over multiple days as the tiles do not align.
    default_random_overlap = False
    # Assuming random overlap is not being performed, the number of 
    # pixels to overlap the tiles during tiling. a higher value helps 
    # reduce harsh edges, but takes longer to process.
    default_overlap_px = 8

    # Create parser object to get arguments from user
    parser = argparse.ArgumentParser()

    # Create the arguments that can be used as input
    parser.add_argument(
        "-i", dest="fps", type=str, required=True,
        help=f"File pattern for MCD19A2 HDF files"
    )
    # Used for masking water bodies
    parser.add_argument(
        "-in", dest="nlcd_fp", type=str, required=True,
        help=f"Filepath to the NLCD data"
    )
    parser.add_argument(
        "-is", dest="shp_fp", type=str,
        help=f"Filepath to the shapefile for cropping"
    )
    parser.add_argument(
        "-od", dest="output_dir", type=str, default= '.', required=True,
        help=(
            "Output directory for the reconstructed data. "
            "Default: Current directory"
        )
    )
    parser.add_argument(
        "-px", dest="overlap_px", default=default_overlap_px, type=int,
        help=f"Number of pixels to overlap when tiling"
    )
    parser.add_argument(
        "-ln", dest="location_name", type=str, default='None',
        help=f"Name of the location (for output file naming)"
    )
    parser.add_argument(
        "-c", dest="out_crs", type=str, default='EPSG:4326',
        help=f"The desired CRS for the output files."
    )
    parser.add_argument(
        "-r", dest="random_overlap", default=default_random_overlap, 
        action='store_true',
        help=f"Apply random shift (6px to 15px) for overlap"
    )

    # Parse the arguments from the user
    args = parser.parse_args()

    return args

def create_gradient_along_edge(
    shape, 
    edge_width
):
    """
    Creates a 2D array consisting of a gradient (range=(0, 1)) of rings
    along the edge of width=edge_width. The center values (e.g. for a
    10x10 image with edge_width=4, this is the center 2x2 pixel array)
    are ones. The outer-most ring is not 0, but instead the next value
    in the gradient.
    img_side_len must be even.

    Note:
    Although this could be implemented with the modulo operation, 
    if-statements are used for readability.
    """

    if not(shape[-1] / edge_width >= 2):
        raise Exception(f"shape[-1] must be at least double edge_width")
    if not(shape[-2] / edge_width >= 2):
        raise Exception(f"shape[-2] must be at least double edge_width")
    if not(shape[-1] % 2 == 0):
        raise Exception(f"shape[-1] must be even. Value received: {shape[-1]}")
    if not(shape[-2] % 2 == 0):
        raise Exception(f"shape[-2] must be even. Value received: {shape[-2]}")
    
    arr = np.empty(shape, dtype=np.float32)

    height = shape[-2]
    width = shape[-1]

    step_size = 1 / (edge_width + 1)
    for x in range(height):
        for y in range(width):
            x_val = min(1, (x + 1) * step_size, (height - x) * step_size)
            y_val = min(1, (y + 1) * step_size, (width - y) * step_size)

            arr[x, y] = min(min(x_val, y_val), 1)

    return arr

def reconstruct(
    time_series: xr.DataArray,
    overlap_px: int
):
    """
    Patches the time series data into 48x48 patches, reconstructs the
    missing pixels, and then merges the reconstructed patches.

    Parameters
    ----------
    time_series: xr.DataArray
        The time series of AOD data.

    overlap_px: int
        The number of pixels to overlap when patching. Helps reduce
        sharp edges when patched back together. Must be < 48.

    """
    side_len = 48

    def alpha_composite_merge(
        merged_data, new_data, merged_mask, new_mask,
        index=None,roff=None,coff=None
    ):
        """
        A merging method for use with rasterio.merge.merge (for 
        merging raster data when it overlaps) that performs alpha 
        compositing. REQUIRES that the last band be an alpha band with
        a gradient of alpha values along the edge for optimal blending
        performance. merged_data is the array to update with the new 
        data. For more information on the other inputs, see: 
        https://v.gd/JFArOL.
        """

        # Extract the true values
        src = np.copy(merged_data[:-1, ...])
        dst = np.copy(new_data[:-1, ...])

        ## Extract the alpha channels.
        # We must first invert the binary masks as valid data is 
        # represented as False (0) rather than True (1) then convert to 
        # int 
        src_mask_alpha = np.copy((~merged_mask[-1:, ...]).astype(int))
        dst_mask_alpha = np.copy((~new_mask[-1:, ...]).astype(int))

        srcA = 1 - create_gradient_along_edge(
            (side_len, side_len), overlap_px
        )
        srcA = np.where(src_mask_alpha == 0, src_mask_alpha, srcA)

        dstA = new_data[-1:, ...]
        dstA = np.where(dst_mask_alpha == 0, dst_mask_alpha, dstA)    

        # Work out resultant alpha channel
        outA = srcA + dstA*(1-srcA)

        # Work out the resultant true values
        new_vals = (src*srcA + dst*dstA*(1-srcA)) / outA
        
        np.stack((
            new_vals.squeeze(),
            new_data[-1:, ...].squeeze()
        ), out=merged_data)

    # Patch the merged rasters to the correct shape
    patches = patch_rxr(
        time_series, side_len, drop_ends=False, overlap_px=overlap_px
    )

    for patch_i, patch in enumerate(patches):
        # We add both a batch axis and a channel axis to the data
        # to make the data the required shape for reconstruction
        patch_to_be_filled = patch.to_numpy()[np.newaxis, ..., np.newaxis]

        # Set the no data value to 0
        patch_to_be_filled[patch_to_be_filled==patch.rio.nodata]=0

        # Perform gap filling
        filled_patch = model(
            patch_to_be_filled
        ).numpy()[0, ..., 0]

        # Replace the original valid pixels in the image
        filled_patch = np.where(
            patch_to_be_filled[0, -1, :, :, 0] == 0,
            filled_patch,
            patch_to_be_filled[0, -1, :, :, 0]
        )

        # Replace the filled patch values in the place of 
        # the old patch values
        patches[patch_i] = patch.isel(band=-1).copy(data=filled_patch)

        # Create alpha channel for the patch that can be used for 
        # blending
        patch_alpha = create_gradient_along_edge(
            (side_len, side_len), overlap_px
        )
        patches[patch_i] = xr.concat([
            patches[patch_i], 
            patch.isel(band=-1).copy(data=patch_alpha)
        ], dim='band')

    # Re-merge all now-reconstructed patches
    # We select band 0 because the second band is the alpha band
    # we added for merging
    full_reconstruction = merge_arrays(
        patches, method=alpha_composite_merge
    ).isel(band=0)

    return full_reconstruction

###############################################################################

args = input_handler()

fps = glob.glob(args.fps, recursive = True)
nlcd_fp = args.nlcd_fp
output_dir = args.output_dir
random_overlap = args.random_overlap
overlap_px = args.overlap_px
output_crs = rasterio.crs.CRS.from_string(args.out_crs)
location_name = args.location_name
shp_fp = args.shp_fp

date_sorted_fps = {}

# Group the filepaths by their date by assigning them to the 
# corresponding dictionary; each date will be the key, and 
# the values will be a list of the filepaths for that date
for fp in fps:
    date = extract_info_from_fp(fp)[1]
    if date in date_sorted_fps:
        date_sorted_fps[date].append(fp)
    else:
        date_sorted_fps[date] = [fp]
# Sort the dictionary of <date>:<[fps]> by date
date_sorted_fps = dict(sorted(
    date_sorted_fps.items(), key=lambda item: item[0]
))

# Extract the desired shapefile shape to be used in cropping the output
# raster
if shp_fp is not None: crop_shp = gpd.read_file(shp_fp).iloc[[0]]

# Will hold the rasters used for the current time series
# in the subsequent loop
time_series = []

# Load the prediction model
model = load_model(model_fp, compile=False)

previous_date = datetime.date(1900, 1, 1)
for date_i, date in enumerate(date_sorted_fps):
    reconstruction_out_fp = os.path.join(
        output_dir,
        f'MCD19A2.Reconstructed.{date}.{location_name}.tif'
    )
    # The start time for processing the current day
    start_time = time.time()
    # If the previous date is not the day prior to the current day,
    # this means there is a date of data missing and the raster
    # stack should be reset to avoid an incorrect time series (i.e.
    # a time series including dates prior to today-stack_size)
    date_obj = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    if date_obj != previous_date + datetime.timedelta(days=1):
        time_series = []

    # Get the raster filepaths for the current day
    date_fps = date_sorted_fps[date]
    
    # Process all rasters for the current day (from the
    # raster fps) and merge them together
    rasters_to_be_merged = []
    for fp in date_fps:
        rasters_to_be_merged.append(MODIS.MCD19A2(fp, nlcd_fp).xd)
    merged_rasters = merge_arrays(rasters_to_be_merged)

    # Append the merged raster to the time series stack
    time_series.append(merged_rasters)

    # If the length of the raster time series list is greater
    # than the desired stack size, raise an error as something
    # has gone wrong in implementation.
    # If it is less than the stack size, continue to the next
    # loop as there is not enough days in the time series for 
    # gap filling
    # If the length of the raster time series list is equal to
    # the desired stack size, perform gap filling
    if len(time_series) > STACK_CT:
        raise Exception('Error: Stack exceeds desired stack size.')
    elif len(time_series) < STACK_CT:
        print(f"Not enough days in stack for {date}")
    else:
        # Crop the merged & stacked rasters to the extent of 
        # shapefile. Convert to same CRS first.
        if shp_fp is not None:
            crop_shp_reprojected = crop_shp.to_crs(time_series[0].rio.crs)
            for raster_i, raster in enumerate(time_series):
                time_series[raster_i] = time_series[raster_i].rio.clip_box(
                    *crop_shp_reprojected.total_bounds
                )

        # Stack the merged rasters in the time series
        time_series_xr = xr.concat([
            raster for raster in time_series
        ], dim='band')

        overlap_px = randint(low=6, high=16) if random_overlap else overlap_px
        full_reconstruction = reconstruct(time_series_xr, overlap_px)
        
        # Reproject the merged raster to the desired output CRS
        full_reconstruction = full_reconstruction.rio.reproject(
            output_crs, resolution=1000,
            resampling=rasterio.warp.Resampling.average
        )
        # Clip/crop the raster to the extent of specified shapefile
        if shp_fp is not None:
            full_reconstruction_clip = full_reconstruction.rio.clip(
                [crop_shp.to_crs(output_crs).geometry.iloc[0]]
            )
        # Save the raster
        full_reconstruction_clip.rio.to_raster(
            reconstruction_out_fp
        )

        # Remove oldest day from time series
        time_series = time_series[1:]

        end_time = time.time()
        print(f"Process time for {date}: {end_time - start_time}")

    # Set the new previous date
    previous_date = date_obj