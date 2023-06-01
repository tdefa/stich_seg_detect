

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