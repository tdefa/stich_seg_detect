

#%%
import time
from os import listdir
from os.path import isfile, join
import bigfish.detection as detection
import bigfish.stack as stack
import numpy as np
import tifffile
from numpy import argmax, nanmax, unravel_index
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import pdist, squareform
from scipy import ndimage
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity
from tqdm import tqdm
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import math
from pathlib import Path

import itertools

import pandas as pd
from utils.signal_quality import mean_cos_tetha




def remove_artifact(filtered_fish, spots, order = 3, min_cos_tetha = 0.75):

    gz, gy, gx = np.gradient(filtered_fish)
    real_spots = []
    for s in spots:
        if     mean_cos_tetha(gz, gy, gx,
                          z=s[0], yc=s[1],
                          xc=s[2], order=order) > min_cos_tetha:
            real_spots.append(s)
    return real_spots



def remove_double_detection(input_array,
            threshold = 0.3,
            scale_z_xy = np.array([0.300, 0.103, 0.103])):
    """

    Args:
        input_list (np.array):
        threshold (float): min distance between point in um
        scale_z_xy (np.array):voxel scale in um

    Returns: list of point without double detection

    """
    unique_tuple = [tuple(s) for s in input_array]
    unique_tuple = list(set((unique_tuple)))

    combos = itertools.combinations(unique_tuple, 2)
    points_to_remove = [list(point2)
                        for point1, point2 in combos
                        if np.linalg.norm(point1 * scale_z_xy  - point2 * scale_z_xy) < threshold]

    points_to_keep = [point for point in input_array if list(point) not in points_to_remove]
    return points_to_keep






def detection_with_segmentation(rna,
                                sigma,
                                min_distance = [3,3, 3],
                              segmentation_mask = None,
                              diam_um = 20,
                              scale_xy = 0.103,
                              scale_z = 0.300,
                              min_cos_tetha = 0.75,
                              order = 5,
                              test_mode = False,
                              threshold_merge_limit = 0.330,
                              x_translation_mask_to_rna = 0,
                              y_translation_mask_to_rna = 0):


    """
    Args:
        rna (np.array): fish image
        sigma (): sigma in gaussian filter
        min_distance (): big fish parameter
        segmentation_mask (np.array):  nuclei segmentation
        diam (): radias size in um around nuclei where to detect spots
        scale_xy (): pixel xy size in um

        scale_z (): pixel e size in um
        min_cos_tetha (float): value between 0 and 1, if 0 it does not remove anything, if 1 it accept only perfect radial symetric spots
        order (): radias size in pixel around spots where to check radail symetrie spts
        threshold_merge_limit (float): threshold below to detected point are considere the same
    Returns:
    """


    rna_log = stack.log_filter(rna, sigma)
    mask = detection.local_maximum_detection(rna_log, min_distance=min_distance)
    rna_gaus = ndimage.gaussian_filter(rna, sigma)

    list_of_nuc = np.unique(segmentation_mask)
    if 0 in list_of_nuc:
        list_of_nuc = list_of_nuc[1:]
    assert all(i >= 1 for i in list_of_nuc)



    all_spots = []
    pbar = tqdm(list_of_nuc)
    for mask_id in pbar:
        pbar.set_description(f"detecting rna around cell {mask_id}")
        [Zm,Ym, Xm] = ndimage.center_of_mass(segmentation_mask == mask_id)
        Xm -= x_translation_mask_to_rna
        Ym -= y_translation_mask_to_rna
        Y_min = np.max([0, Ym - diam_um / scale_xy]).astype(int)
        Y_max = np.min([segmentation_mask.shape[1], Ym + diam_um / scale_xy]).astype(int)
        X_min = np.max([0, Xm - diam_um / scale_xy]).astype(int)
        X_max = np.min([segmentation_mask.shape[2], Xm + diam_um / scale_xy]).astype(int)
        crop_mask = mask[:, Y_min:Y_max, X_min:X_max]
        threshold = detection.automated_threshold_setting(rna_log[:, Y_min:Y_max, X_min:X_max], crop_mask)

        spots, _ = detection.spots_thresholding(rna_log[:, Y_min:Y_max, X_min:X_max], crop_mask, threshold)

        if min_cos_tetha is not None:

            new_spots = remove_artifact(filtered_fish = rna_gaus[:,Y_min:Y_max, X_min:X_max],
                            spots = spots,
                            order=order,
                            min_cos_tetha=min_cos_tetha)
            """new_spots = []
            for s in spots:
                if mean_cos_tetha(filtered_crop_fish = rna_gaus[:,Y_min:Y_max, X_min:X_max],
                                  z=s[0], yc=s[1],
                                  xc=s[2], order=order) > min_cos_tetha:
                    new_spots.append(s)"""

        if test_mode: ## test mode
            input = np.amax(rna[:,Y_min:Y_max,  X_min:X_max], 0)
            pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
            rna_scale = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
            fig, ax = plt.subplots(2, 1, figsize=(15, 15))
            plt.title(f' X {str([X_min, X_max])} + Y {str([Y_min, Y_max])}', fontsize=20)

            ax[0].imshow(rna_scale)
            ax[1].imshow(rna_scale)
            for s in spots:
                ax[0].scatter(s[-1], s[-2], c='red', s=28)
            plt.show()

            fig, ax = plt.subplots(2, 1, figsize=(15, 15))
            plt.title(f'with remove artf  order{order}  min_tetha {min_cos_tetha}  X {str([X_min, X_max])} + Y {str([Y_min, Y_max])}', fontsize=20)
            ax[0].imshow(rna_scale)
            ax[1].imshow(rna_scale)
            for s in new_spots:
                ax[0].scatter(s[-1], s[-2], c='red', s=28)
            plt.show()
        spots = new_spots
        spots = np.array(spots)
        if len(spots) > 0:
            spots = spots + np.array([0, Y_min, X_min])
            all_spots += list(spots)


    all_spots = remove_double_detection(
                input_array = np.array(all_spots),
                threshold =threshold_merge_limit,
                scale_z_xy = np.array([scale_z, scale_xy, scale_xy]))



    if test_mode:
        input = np.amax(rna, 0)
        pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
        rna_scale = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
        fig, ax = plt.subplots(2, 1, figsize=(40, 40))
        ax[0].imshow(rna_scale)
        ax[1].imshow(rna_scale)
        for s in all_spots:
            ax[0].scatter(s[-1], s[-2], c='red', s=28)
        plt.show()

    return all_spots


def detection_without_segmentation(
                            rna,
                            sigma,
                            min_distance = [3,3, 3],
                            threshold = None):


    rna_log = stack.log_filter(rna, sigma)
    mask = detection.local_maximum_detection(rna_log, min_distance=min_distance)

    if threshold is None:
        threshold = detection.automated_threshold_setting(rna_log, mask)
    spots, _ = detection.spots_thresholding(rna_log, mask, threshold)
    return spots, threshold

#%%

def detection_folder_with_segmentation(
                     round_folder_path = "/media/tom/T7/Stitch/acquisition/",
                     round_name_regex = "r",
                     image_name_regex = "opool1_1_MMStack_3",
                     channel_name_regex = "ch1",
                     fixed_round_name = "r1_bc1",
                     path_output_segmentaton = "/media/tom/T7/stich0504/segmentation_mask/",
                      min_distance=(4, 4, 4),
                      scale_xy=0.103,
                      scale_z=0.300,
                      sigma = 1.35,
                        ### detection parameters with segmentation
                    dico_translation = None, # np.load("/media/tom/T7/Stitch/acquisition/dico_translation.npy", allow_pickle=True).item(),
                    diam_um=20,
                    local_detection = True,
                    min_cos_tetha=0.75,
                    order=5,
                    test_mode=False,
                    threshold_merge_limit= 0.330,
                    file_extension = ".ti",
                    threshold_input =  {},):

    """
    :param sigma:
    :param rna_path:
    :param path_output_segmentaton:
    :param threshold_input:
    :param output_file:
    :param min_distance:
    :param local_detection:
    :param diam:
    :param scale_xy:
    :param scale_z:
    :param min_cos_tetha:
    :param order:
    :param test_mode:
    :return:
    """

    dico_spots = {}
    dico_threshold = {}

    for path_round in tqdm(list(Path(round_folder_path).glob(f"{round_name_regex}*"))[4:] + list(Path(round_folder_path).glob(f"{round_name_regex}*"))):
        print()
        print(path_round.name)
        dico_spots[path_round.name] = {}
        dico_threshold[path_round.name] = {}
        for path_rna_img in tqdm(list(path_round.glob(f"*{channel_name_regex}*{file_extension}*"))):

            if image_name_regex not in path_rna_img.name:
                continue
            print(path_rna_img.name)

            rna_img = tifffile.imread(path_rna_img)
            if local_detection:

                mask_name = path_rna_img.name.replace('_' + channel_name_regex, "")
                segmentation_mask = tifffile.imread(path_output_segmentaton + mask_name)
                ### get the translation between the mask and the rna image
                if path_round.name == fixed_round_name:
                    x_translation_mask_to_rna = 0
                    y_translation_mask_to_rna = 0

                else :
                    if dico_translation[path_rna_img.name][fixed_round_name][path_round.name]['x_translation'] is None\
                            or dico_translation[path_rna_img.name][fixed_round_name][path_round.name]['y_translation'] is None:
                        print("no translation found for round ")
                        continue
                    x_translation_mask_to_rna = dico_translation[path_rna_img.name][fixed_round_name][path_round.name]['x_translation']
                    y_translation_mask_to_rna = dico_translation[path_rna_img.name][fixed_round_name][path_round.name]['y_translation']



                all_spots = detection_with_segmentation(rna = rna_img,
                                            sigma = sigma,
                                            min_distance=min_distance,
                                            segmentation_mask=segmentation_mask,
                                            diam_um=diam_um,
                                            scale_xy=scale_xy,
                                            scale_z=scale_z,
                                            min_cos_tetha=min_cos_tetha,
                                            order=order,
                                            test_mode=test_mode,
                                            threshold_merge_limit=threshold_merge_limit,
                                            x_translation_mask_to_rna=x_translation_mask_to_rna,
                                            y_translation_mask_to_rna=y_translation_mask_to_rna)

            else:
                if threshold_input is not None and path_round.name in threshold_input.keys() and path_rna_img.name in threshold_input[path_round.name].keys():
                    threshold = threshold_input[path_round.name][path_rna_img.name]
                else:
                    threshold = None

                all_spots, threshold = detection_without_segmentation(
                            rna=rna_img,
                            sigma=sigma,
                            min_distance=min_distance,
                            threshold = threshold
                )
            dico_spots[path_round.name][path_rna_img.name] = all_spots
            dico_threshold[path_round.name][path_rna_img.name] = threshold

    return dico_spots, dico_threshold




#%%


if __name__ == "__main__":


    round_folder_path  = "/home/tom/Bureau/phd/Batrin/Images_pour_FISH/Expérience2/round_folder/"

    dico_spots = detection_folder(
        round_folder_path=round_folder_path,
        round_name_regex="ch",
        image_name_regex="ti",
        channel_name_regex=".",
        min_distance=(3, 3),
        sigma=1,
        fixed_round_name=None,
        path_output_segmentaton=None,
        scale_xy=None,
        scale_z=None,
        ### detection parameters with segmentation
        dico_translation=None,
        diam_um=None,
        local_detection=None,
        min_cos_tetha=None,
        order=None,
        test_mode=None,
        threshold_merge_limit=None)


    #### dico spot to csv

    np.save(round_folder_path+"dico_spots.npy", dico_spots)
    Path(round_folder_path+"detection_csv/").mkdir(parents=True, exist_ok=True)
    for round_name in dico_spots.keys():
        for image_name in dico_spots[round_name].keys():
            df = pd.DataFrame(dico_spots[round_name][image_name])
            df = df.rename(columns={0: "x", 1: "y"})
            df.to_csv(f"{round_folder_path}detection_csv/{image_name}.csv",
                      index=False)

