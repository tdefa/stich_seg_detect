# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse

import tifffile

from segmentation import segment_nuclei
from cellpose import models
import numpy as np
from spots_detection import detection_folder_with_segmentation  # , detection_folder_without_segmentation
from registration import folder_translation
# from stiching import stich_with_image_J, parse_txt_file
import datetime
from pathlib import Path

# Press the green button in the gutter to run the script.

dico_bc_gene0 = {
    'r1_bc1': "Rtkn2",
    'r3_bc4': "Pecam1",
    'r4_bc5': "Ptprb",
    'r5_bc6': "Pdgfra",
    'r6_bc7': "Chil3",
    'r7_bc3': "Lamp3"
}

dico_bc_gene1 = {
    'r1_Cy3': "Rtkn2",
    'r2': "Lamp3",
    'r3': "Pecam1",
    'r4': "Ptprb",
    'r5': "Pdgfra",
    'r6': "Chil3",
    'r7': "Apln",
    "r8": "Fibin",
    "r9": "Pon1",
    "r10": "Cyp2s1",
    "r11": "C3ar1",
    "r12": "Hhip",
    #"r13": "Hhip",
}


dico_bc_noise = {"r1_Cy5": "artefact"}


if __name__ == '__main__':

    ####### segment individual tile

    # input folder name + regex_file_name
    # result a new save folder with the same name + _segmented

    parser = argparse.ArgumentParser(description='test')

    parser.add_argument("--folder_of_rounds",
                        type=str,
                        default="/media/tom/Transcend/lustr2023/images/",
                        help='')

    parser.add_argument("--path_to_dapi_folder",
                        type=str,
                        default="/media/tom/Transcend/lustr2023/images/r1_Cy3/",
                        help='')

    parser.add_argument("--path_to_mask_dapi",
                        type=str,
                        default="/media/tom/Transcend/lustr2023/images/segmentation_mask/",
                        help='')
    parser.add_argument("--regex_dapi",
                        type=str,
                        default="ch1",
                        help='')

    parser.add_argument("--fixed_round_name",
                        type=str,
                        default="r1_Cy3",
                        help='')
    parser.add_argument("--folder_regex_round",
                        type=str,
                        default="r",
                        help='')
    parser.add_argument("--chanel_regex",
                        type=str,
                        default="ch0",
                        help='')
    parser.add_argument("--image_name_regex",
                        type=str,
                        default="",
                        help='')

    parser.add_argument("--name_dico",
                        type=str,
                        default="23mai",
                        help='')

    ### param for detection
    parser.add_argument("--local_detection",
                        type=int,
                        default=1,
                        help='')


    ######## parameters stiching
    parser.add_argument("--image_shape",
                        type=list,
                        default=[37, 2048, 2048],
                        help='')

    parser.add_argument("--nb_tiles",
                        type=int,
                        default=5,
                        help='')

    parser.add_argument("--regex_image_stiching",
                        type=str,
                        default="r1_pos{i}_ch0.tif",
                        )

    parser.add_argument("--remove_double_detection", type = int, default = 1)
    parser.add_argument("--stich_segmentation_mask", type = int, default = 1)

    ##### task to do

    parser.add_argument("--segmentation", default=0, type=int)
    parser.add_argument("--registration", default=0, type=int)

    parser.add_argument("--spots_detection", default=1, type=int)
    parser.add_argument("--signal_quality", default=1, type=int)

    parser.add_argument("--stitch", default=0, type=int)
    parser.add_argument("--stich_spots_detection", default=1, type=int)

    parser.add_argument("--port", default=39948)
    parser.add_argument("--mode", default='client')
    parser.add_argument("--host", default='127.0.0.2')

    args = parser.parse_args()

    print(args)


    args.image_path_stiching = args.folder_of_rounds + args.fixed_round_name
    args.output_path_stiching = args.folder_of_rounds + args.fixed_round_name + "output_s"

    e = datetime.datetime.now()
    date_str = f"{e.month}_{e.day}_{e.hour}_{e.minute}_{e.second}"

    ####################
    ####### segment individual tile
    ####################

    if args.segmentation == 1:
        print("segmentation")
        Path(args.path_to_mask_dapi).mkdir(parents=True, exist_ok=True)
        model = models.Cellpose(gpu=True, model_type="nuclei")
        dico_param = {}
        dico_param["diameter"] = 80
        dico_param["flow_threshold"] = 0.7
        dico_param["mask_threshold"] = 0
        dico_param["do_3D"] = False
        dico_param["mip"] = False
        dico_param["projected_focused"] = False
        dico_param["stitch_threshold"] = 0.3
        dico_param["erase_solitary"] = True
        dico_param["erase_small_nuclei"] = 300

        segment_nuclei(path_to_dapi_folder=args.path_to_dapi_folder,
                       regex_dapi=args.regex_dapi,
                       path_to_mask_dapi=args.path_to_mask_dapi,
                       dico_param=dico_param,
                       model=model,
                       save=True,
                       )

    ########
    # register each channel to the dapi round channel
    #######

    if args.registration == 1:
        dico_translation = folder_translation(folder_of_rounds=args.folder_of_rounds,  # works ok
                                              fixed_round_name=args.fixed_round_name,
                                              folder_regex=args.folder_regex_round,
                                              chanel_regex=args.chanel_regex,
                                              registration_repeat=5)

        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_translation.npy", dico_translation)

    ####################
    ## Do individual spots detection (optional : using mask segmentation)
    ####################
    if args.spots_detection == 1:
        print("spots detection")
        if args.local_detection:
            dico_translation = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_translation.npy",
                                       allow_pickle=True).item()
        else:
            dico_translation = None
        threshold_input = None
        dico_spots, dico_threshold = detection_folder_with_segmentation(
            round_folder_path=args.folder_of_rounds,
            round_name_regex=args.folder_regex_round,
            image_name_regex=args.image_name_regex,
            channel_name_regex=args.chanel_regex,
            fixed_round_name=args.fixed_round_name,
            path_output_segmentaton=args.path_to_mask_dapi,
            min_distance=(4, 4, 4),
            scale_xy=0.103,
            scale_z=0.300,
            sigma=1.35,
            ### detection parameters with segmentation
            dico_translation=dico_translation,
            diam_um=20,
            local_detection=args.local_detection,
            min_cos_tetha=0.70,
            order=5,
            test_mode=False,
            threshold_merge_limit=0.330,
            file_extension='tif',
            threshold_input=threshold_input)

        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_local_detection{args.local_detection}_{args.folder_regex_round}.npy",
                dico_spots)
        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_threshold{args.local_detection}_{args.round_name_regex}.npy", dico_spots)

    ########
    # Compute signal quality for each round
    #######

    if args.signal_quality == 1:

        from utils.signal_quality import compute_quality_all_rounds

        dico_spots = np.load \
            (f"{args.folder_of_rounds}{args.name_dico}_dico_spots_local_detection{args.local_detection}.npy",
             allow_pickle=True).item()

        dico_signal_quality = compute_quality_all_rounds(
            dico_spots = dico_spots,
            round_folder_path=args.folder_of_rounds,
            round_name_regex=args.folder_regex_round,
            image_name_regex=args.image_name_regex,
            channel_name_regex=args.chanel_regex,
            file_extension="tif",
            voxel_size=[300, 108, 108],
            spot_radius=300,
            sigma=1.3,
            order=5,
            compute_sym=True,
            return_list=True,
        )
        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_signal_quality{args.local_detection}.npy", dico_signal_quality)

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

    ########
    # Stitch only ref round images
    #######

    if args.stitch == 1:
        from stiching import stich_with_image_J, parse_txt_file

        stich_with_image_J(
            grid_size_x=args.nb_tiles,
            grid_size_y=args.nb_tiles,
            tile_overlap=10,
            image_name=args.regex_image_stiching,
            image_path=args.image_path_stiching,
            output_path=args.output_path_stiching ,
        )

        #### generate registration dico
        print(" you need to mannualy modify the path to the registered file in the txt file")
        dico_stitch = parse_txt_file \
            (path_txt= "/media/tom/Transcend/lustr2023/images/registration_ref/TileConfiguration0506.registered.txt",
             image_name_regex="_pos",)
        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_stitch.npy",
                dico_stitch)



    ########
    # Stitch detected spots
    #######


    if args.stich_spots_detection:  ## get a dataframe with the spots codinates in the ref round

        from stiching import stich_segmask,stich_dico_spots

        dico_stitch = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_stitch.npy",
            allow_pickle=True).item()


        dico_spots = np.load \
            (f"{args.folder_of_rounds}{args.name_dico}_dico_spots_local_detection{args.local_detection}.npy",
             allow_pickle=True).item()
        dico_translation = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_translation.npy",
                                   allow_pickle=True).item()



        df_coord, new_spot_list_dico, missing_data = stich_dico_spots(dico_spots = dico_spots,
                                                                      dico_translation = dico_translation,
                                                                      dico_stitch = dico_stitch,
                                                                      ref_round=args.fixed_round_name,
                                                                      dico_bc_gene=dico_bc_gene1,
                                                                      image_shape=args.image_shape,
                                                                      nb_tiles=args.nb_tiles,
                                                                      check_removal = True,
                                                                      threshold_merge_limit=0.330,
                                                                      scale_xy=0.103,
                                                                      scale_z=0.300,
                                                                      )


        if args.remove_double_detection:

            df_coord_noise, _, _ = stich_dico_spots(dico_spots = dico_spots,
                                                                          dico_translation = dico_translation,
                                                                          dico_stitch = dico_stitch,
                                                                          ref_round=args.fixed_round_name,
                                                                          dico_bc_gene=dico_bc_noise,
                                                                          image_shape=args.image_shape,
                                                                          nb_tiles=args.nb_tiles,
                                                                            check_removal = False,
                                                                        )

            from sklearn.neighbors import NearestNeighbors
            X_noise = np.array(list(zip(df_coord_noise.z, df_coord_noise.y, df_coord_noise.x)))
            X = np.array(list(zip(df_coord.z, df_coord.y, df_coord.x)))
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_noise)

            distances, indices = nbrs.kneighbors(X)

            distances = np.min(distances, axis = 1)

            df_coord = df_coord[distances > 7]

            df_coord.index = range(len(df_coord))

        ########
        # Stitch stich_segmentation_mask
        #######

        if args.stich_segmentation_mask:
            from stiching import  stich_segmask

            final_masks, dico_centroid = stich_segmask(dico_stitch,
                                        path_mask=args.path_to_mask_dapi,
                                        image_shape=args.image_shape,
                                        nb_tiles=args.nb_tiles,
                                        compute_dico_centroid = True)

            np.save(f"{args.folder_of_rounds}{args.name_dico}_final_masks.npy", final_masks)

            np.save(f"{args.folder_of_rounds}{args.name_dico}_final_masks_mip.npy", np.amax(final_masks, 0))
            np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_nuclei_centroid.npy", dico_centroid)


            from utils.segmentation_processing import compute_dico_centroid

            dico_nuclei_centroid = compute_dico_centroid(mask_nuclei = final_masks)

            np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_nuclei_centroid.npy", dico_nuclei_centroid)


            del final_masks

            from stiching import stich_from_dico

            final_dapi = stich_from_dico(dico_stitch,
                            path_mask="/media/tom/Transcend/lustr2023/images/r1_Cy3",
                            regex="*_ch1*tif*",
                            image_shape=[37, 2048, 2048],
                            nb_tiles=5)
            np.save(f"{args.folder_of_rounds}{args.name_dico}_final_dapi.npy", final_dapi)

            np.save(f"{args.folder_of_rounds}{args.name_dico}_final_dapi_mip.npy", np.amax(final_dapi, 0))
            tifffile.imsave(f"{args.folder_of_rounds}{args.name_dico}_final_dapi_mip.npy", np.amax(final_dapi, 0))



        ########
        # Genrate comseg imput
        #######

        final_masks = np.load(f"{args.folder_of_rounds}{args.name_dico}_final_masks.npy")
        x_list = list(df_coord.x)
        y_list = list(df_coord.y)
        z_list = list(df_coord.z)
        nuc_prior = []
        in_nuc = []
        not_in_nuc = []
        for ix in range(len(z_list)):
            nuc_index_prior = final_masks[int(z_list[ix]), int(y_list[ix]), int(x_list[ix])]
            nuc_prior.append(nuc_index_prior)
            if nuc_index_prior != 0:
                in_nuc.append([int(y_list[ix]), int(x_list[ix])])
            else:
                not_in_nuc.append([int(y_list[ix]), int(x_list[ix])])

        df_coord["in_nucleus"] = nuc_prior

        dico_dico_commu = {"stich0": {"df_spots_label": df_coord, }}
        #np.save(f"/media/tom/Transcend/lustr2023/dico_dico_commu_1juin_global_detect", dico_dico_commu)

        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_dico_commu_local_detect_cy5_removal.npy", dico_dico_commu)



        compute_dico_centroid(mask_nuclei, dico_simu=None)
        ###############" add nuclei prior to spots detection

    #color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #viewer.add_points(in_nuc, name=f"in_nuc", face_color=color, edge_color=color, size=6)

    #color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #viewer.add_points(not_in_nuc, name=f"not_in_nuc", face_color=color, edge_color=color, size=6)

    ################ compute registration accross round

    ### input folder of folder with the rounds, regex to fish images to take into account

    ### output dictionary with dico[image_name][moving_image][static_round_image] = translation s.th. moving_image + translation = static image
    ### static image is typically round one here

    ################### perform spots detection wit
