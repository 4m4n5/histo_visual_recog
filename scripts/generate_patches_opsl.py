# +
import warnings

# Essentials
import os
import openslide as opsl
import numpy as np

# Image functions
from PIL import Image
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk


# +
def optical_density(tile):
    tile = tile.astype(np.float64)

    od = -np.log((tile+1)/240)
    return od

def keep_tile(tile_tuple, tile_size, tissue_threshold):
    slide_num, tile = tile_tuple
    if tile.shape[0:2] == (tile_size, tile_size):
        tile_orig = tile
        tile = rgb2gray(tile)
        tile = 1 - tile
        tile = canny(tile)
        tile = binary_closing(tile, disk(10))
        tile = binary_dilation(tile, disk(10))
        tile = binary_fill_holes(tile)
        percentage = tile.mean()
        check1 = percentage >= tissue_threshold

        tile = optical_density(tile_orig)
        beta = 0.15
        tile = np.min(tile, axis=2) >= beta
        tile = binary_closing(tile, disk(2))
        tile = binary_dilation(tile, disk(2))
        tile = binary_fill_holes(tile)
        percentage = tile.mean()
        check2 = percentage >= tissue_threshold

        return check1 and check2
    else:
        return False


# -

# Parameters
PATH = '/project/DSone/as3ek/data/WSI/cinn_celiac_zif/'
patch_size = 1000
resize_to = 1000
target = '/project/DSone/as3ek/data/WSI_patched/cinn_celiac_zif/' # for WSI
target_path_unnorm = '/project/DSone/as3ek/data/patches/1000/un_normalized/cinn_celiac_zif/' # for unnormalized patches
thresh = 0.40
save_WSI = False
overlap = 0.5 # %-age area

# +
num_files = len(os.listdir(PATH))

for i, file in enumerate(list(reversed(os.listdir(PATH)))):
    image = opsl.OpenSlide(PATH + file)
    rescale = resize_to / patch_size
    new_dims = int(rescale * (image.dimensions[0] // 256) * 256) , int(rescale * (image.dimensions[1] // 256) * 256)
    
    # Initialize x and y coord
    x_cord = 0
    y_cord = 0
    
    if save_WSI:
        joined_image = Image.new('RGB', (new_dims))
    
    while x_cord + patch_size < image.dimensions[0] - 1000:
        while y_cord + patch_size < image.dimensions[1] - 1000:
            patch = image.read_region((x_cord, y_cord), 0, (patch_size, patch_size))
        
            patch = patch.convert('RGB')
            patch = patch.resize((resize_to, resize_to))
            patch = np.array(patch)
            
            # Check if we should keep patch
            if keep_tile((0, patch), resize_to, thresh) == False:
                y_cord = int(y_cord + (1 - overlap) * patch_size)
                continue
            
            patch = Image.fromarray(patch)
            
            # Save unnormalized patch
            target_folder = target_path_unnorm
            if not os.path.exists(target_folder):
                os.mkdir(target_folder)
            filename = target_folder + file.split('.')[0] + '__' + str(x_cord) + '_' + str(y_cord) + '.jpg'
            patch.save(filename)
            
            if save_WSI:
                joined_image.paste(patch, (int(x_cord*rescale), int(y_cord*rescale)))
            
            # Taking care of overlap
            y_cord = int(y_cord + (1 - overlap) * patch_size)
        
        # Taking care of overlap
        x_cord = int(x_cord + (1 - overlap) * patch_size)
        y_cord = 0
    
    print(str(i + 1) + '/' + str(num_files) + ' Complete!')
    if save_WSI:
        if not os.path.exists(target):
            os.makedirs(target)
        joined_image.save(target + file.split('.')[0] + '.png')
# -


