#!/bin/bash

#SBATCH -N 1

#nombre threads max sur GPU 48

#SBATCH -n 1


#SBATCH -J loc_spots_detection

#SBATCH -J cyto17_loc_spots_detection


#SBATCH --output="seg_lustra.out"
#SBATCH --mem 65000    # Memory per node in MB (0 allocates all the memory)

#SBATCH --ntasks=1              # Number of processes to run (default is 1)
#SBATCH --cpus-per-task=12      # CPU cores per process (default 1)


#SBATCH -p cbio-gpu



cd /cluster/CBIO/data1/data3/tdefard/stich_seg_detect

python main.py \
--folder_of_rounds /cluster/CBIO/data1/data3/tdefard/lustr2023/images/ \
--path_to_dapi_folder /cluster/CBIO/data1/data3/tdefard/lustr2023/images/r1_Cy3/ \
--path_to_mask_dapi /cluster/CBIO/data1/data3/tdefard/lustr2023/images/segmentation_mask/ \
--regex_dapi ch1 \
--fixed_round_name r1_Cy3 \
--folder_regex_round r \
--chanel_regex ch0 \
--name_dico 23mai \
--segmentation 1 \
--registration 0 \
--spots_detection 0 \
--signal_quality 0 \
--stitch 0 \
--local_detection 0 \



