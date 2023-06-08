

import pandas
import numpy as np



### create a dataframe with the information of the spots

dico_signal_quality = np.load("/media/tom/Transcend/lustr2023/images/23mai_dico_signal_quality0.npy",
                              allow_pickle=True).item()

all_mean_intensity = []
all_mean_snr = []
all_mean_symmetry_coef = []
all_mean_background = []

all_median_intensity = []
all_median_snr = []
all_median_symmetry_coef = []
all_median_background = []

for round_str in dico_signal_quality:
    mean_intensity = []
    mean_snr = []
    mean_symmetry_coef = []
    mean_background = []
    for image in dico_signal_quality[round_str]:
        mean_intensity += dico_signal_quality[round_str][image]["intensity"]
        mean_background += dico_signal_quality[round_str][image]["background"]
        mean_snr += dico_signal_quality[round_str][image]["snr"]
        mean_symmetry_coef += dico_signal_quality[round_str][image]["symmetry_coef"]

    print(f"round {round_str} mean intensity {np.mean(mean_intensity)}")
    print(f"round {round_str} median intensity {np.median(mean_intensity)}")
    print()
    print(f"round {round_str} mean background {np.mean(mean_background)}")
    print(f"round {round_str} median background {np.median(mean_background)}")
    print()

    print(f"round {round_str} mean snr {np.mean(mean_snr)}")
    print(f"round {round_str} median snr {np.median(mean_snr)}")
    print()

    print(f"round {round_str} mean symmetry coef {np.mean(mean_symmetry_coef)}")
    print(f"round {round_str} median symmetry coef {np.median(mean_symmetry_coef)}")

    print()
    print()
    print()

    all_mean_intensity.append(np.mean(mean_intensity))
    all_mean_snr.append(np.mean(mean_snr))
    all_mean_symmetry_coef.append(np.mean(mean_symmetry_coef))
    all_mean_background.append(np.mean(mean_background))

    all_median_intensity.append(np.median(mean_intensity))
    all_median_snr.append(np.median(mean_snr))
    all_median_symmetry_coef.append(np.median(mean_symmetry_coef))
    all_median_background.append(np.median(mean_background))

    index  = list(dico_signal_quality.keys())


df = pandas.DataFrame({"mean_intensity":all_mean_intensity,
                       "median_intensity":all_median_intensity,
                       "mean_snr":all_mean_snr,
                          "median_snr":all_median_snr,
                       "mean_symmetry_coef":all_mean_symmetry_coef,
                            "median_symmetry_coef":all_median_symmetry_coef,
                       "mean_background":all_mean_background,
                       "median_background":all_median_background
                       }, index=index)


dico_translation = np.load("/media/tom/Transcend/lustr2023/images/23mai_dico_translation_old.npy",
                            allow_pickle=True).item()
dico_translation_new = {}
for k in dico_translation:
    position = "pos" + k.split("pos")[1].split("_")[0]
    dico_translation_new[position] = dico_translation[k]

np.save("/media/tom/Transcend/lustr2023/images/23mai_dico_translation.npy",
        dico_translation_new)

##### fuse dico loacal detection


import numpy as np
import pandas
from pathlib import Path

path_folder_dico = Path("/media/tom/Transcend/lustr2023/images/folder_detection_each_round")

mai_dico_spots_local_detection1 = {}

for path_d in path_folder_dico.glob("*.npy"):

    dico = np.load(path_d, allow_pickle=True).item()

    for round in dico:
        print(round)

        mai_dico_spots_local_detection1[round] = dico[round]

np.save("/media/tom/Transcend/lustr2023/images/mai_dico_spots_local_detection1.npy",mai_dico_spots_local_detection1)


X = np.array(dico_spots['r1_Cy3']['r1_pos0_ch0.tif'])

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(points_to_keep)

distances, indices = nbrs.kneighbors(points_to_keep)

import itertools
threshold = 0.3
scale_z_xy = np.array([0.3, 0.1, 0.1])
input_array = X
unique_tuple = [tuple(s) for s in input_array]
unique_tuple = list(set((unique_tuple)))

combos = itertools.combinations(unique_tuple, 2)
points_to_remove = [list(point2)
                    for point1, point2 in combos
                    if np.linalg.norm(point1 * scale_z_xy - point2 * scale_z_xy) < threshold]

points_to_keep = [point for point in unique_tuple if list(point) not in points_to_remove]

