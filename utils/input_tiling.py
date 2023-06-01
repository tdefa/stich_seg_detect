


import tifffile
import numpy as np
from pathlib import Path
from tqdm import tqdm












if __name__ == '__main__':

    # Tilling an image

    nb_tile_x = 3
    nb_tile_y = 3

    image = tifffile.imread("/media/tom/T7/sp_data/In_situ_Sequencing_16/dapi/Base_1_stitched-1.tif")


    tile_x = image.shape[0]//nb_tile_x
    tile_y = image.shape[1]//nb_tile_y
    print(tile_y, tile_x)

    for x in tqdm(range(nb_tile_x)):
        for y in range(nb_tile_y):
            if x == nb_tile_x-1:
                tile = image[y * tile_y:(y + 1) * tile_y, x * tile_x:(x + 1) * tile_x]

            tile = image[y*tile_y:(y+1)*tile_y, x*tile_x:(x+1)*tile_x]
            tifffile.imwrite("/media/tom/T7/sp_data/In_situ_Sequencing_16/dapi_tile/tile_y_{}_x_{}.tif".format(y, x), tile)


