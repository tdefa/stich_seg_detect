











import argparse
from os import listdir
from os.path import isfile, join

import numpy as np
import tifffile
from cellpose import models
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils.segmentation_processing import stitch3D_z, erase_solitary, erase_small_nuclei

from pathlib import Path


def segment_nuclei(path_to_dapi_folder,
                   regex_dapi,
                   path_to_mask_dapi,
                   dico_param,
                   model,
                   save=True,
                   ):


    """
    segment dapi image and save  them in th path_to_mask_dapi folder
    Args:
        path_to_dapi (str):
        path_to_mask_dapi (str):
        dico_param (dict):
        model (cellpose modem):
        save (bool):
    Returns:
        None
    """

    if path_to_mask_dapi[-1] != "/":
        path_to_mask_dapi += "/"
    print(list(Path(path_to_dapi_folder).glob(f"*{regex_dapi}*")))
    print(f'dico_param{dico_param}')
    for path_dapi in tqdm(list(Path(path_to_dapi_folder).glob(f"*{regex_dapi}*"))):
        path_dapi = str(path_dapi)
        print(path_dapi)
        img = tifffile.imread(path_dapi)
        print(img.shape)
        if dico_param["mip"] is True and len(img.shape) == 3:
            img = np.amax(img, 0)
        else:
            if len(img.shape) == 3:
                img = img.reshape(img.shape[0], 1, img.shape[1], img.shape[2])
                print(f'image dapi shape after reshape {img.shape}')
                img = list(img)
        masks, flows, styles, diams = model.eval(img,
                                                 diameter=dico_param["diameter"],
                                                 channels=[0, 0],
                                                 flow_threshold=dico_param["flow_threshold"],
                                                 do_3D=dico_param["do_3D"],
                                                 stitch_threshold=0)

        masks = np.array(masks, dtype=np.int16)

        if len(masks.shape) == 3:
            masks = stitch3D_z(masks, dico_param["stitch_threshold"])
            masks = np.array(masks, dtype = np.int16)
            if len(masks.shape) and dico_param["erase_solitary"]:
                masks = erase_solitary(masks)
        if dico_param["erase_small_nuclei"] is not None:
            print(f'erase_small_nuclei threshold {dico_param["erase_small_nuclei"]}')
            masks = erase_small_nuclei(masks)
        if save:
            image_name = path_dapi.split('/')[-1].split(f'_{regex_dapi}')[0]
            tifffile.imwrite(path_to_mask_dapi + image_name +'.tif', data=masks, dtype=masks.dtype)
            np.save(path_to_mask_dapi + "dico_param.npy", dico_param)



#### generate centroid dico

def generate_centroid_dico(final_masks):
    from skimage.measure import label, regionprops
    centroid_dico = {}
    props = regionprops(final_masks)
    for p in props:
        centroid_dico[p.label] = [p.centroid]
    return centroid_dico

#######


if __name__ == "__main__":




    main_folder = "/cluster/CBIO/data1/data3/tdefard/T7/sp_data/In_situ_Sequencing_16/dapi_tile/"
    path_to_dapi_folder = main_folder + "dapi_tile/"
    path_to_mask_dapi = main_folder + "mask_tile_pcw/"
    #path_to_dapi_folder = main_folder + "dapi_pcw14/"
    #path_to_mask_dapi = main_folder + "mask_pcw14/"
    #path_to_dapi_folder = main_folder + "dapi_2/"
    #path_to_mask_dapi = main_folder + "mask_pcw6/"
    Path(path_to_mask_dapi).mkdir(parents=True, exist_ok=True)
    model_name = "cyto"
    model = models.Cellpose(gpu=True, model_type=model_name)
    dico_param = {}
    dico_param["diameter"] = 80
    dico_param["flow_threshold"] = 0.9
    dico_param["mask_threshold"] = 0
    dico_param["do_3D"] = False
    dico_param["mip"] = False
    dico_param["projected_focused"] = False
    dico_param["stitch_threshold"] = 0.3
    dico_param["erase_solitary"] = False
    dico_param["erase_small_nuclei"] = None

    for i in [50, 40, 30, 20, 15, 10, 5]:
        print(i)
        dico_param["diameter"] = i

        path_to_mask_dapi_loop = path_to_mask_dapi + str(i) + "_" + model_name + "/"
        Path(path_to_mask_dapi_loop).mkdir(parents=True, exist_ok=True)

        segment_nuclei(path_to_dapi_folder=path_to_dapi_folder,
                       regex_dapi="Ba*ti",
                       path_to_mask_dapi=path_to_mask_dapi_loop,
                       dico_param=dico_param,
                       model=model,
                       save=True,
                       )




