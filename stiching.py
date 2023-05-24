
import napari
import random
from tqdm import tqdm

import ast

import numpy as np
import tifffile
import pandas as pd
from pathlib import Path
if False:
    import imagej, scyjava
    scyjava.config.add_option('-Xmx40g')
    ij = imagej.init('sc.fiji:fiji')

#https://forum.image.sc/t/grid-collection-stitching-registered-coordinates-not-saving-correctly-with-pyimagej/22942
def stich_with_image_J(
    grid_size_x =5,
    grid_size_y =5,
    tile_overlap = 10,
    image_name = "r1_pos0_ch0.tif",
    image_path = "/media/tom/Transcend/lustr2023/images/r1_Cy3",
    output_path = "/media/tom/Transcend/lustr2023/images/r1_Cy3/output_s",):


    ##find the option to save it in a file txt
    STITCHING_MACRO = """ 
    #@ String grid_size_x
    #@ String grid_size_y
    #@ String tile_overlap
    #@ String image_path
    #@ String image_name
    #@ String output_path
     run("Grid/Collection stitching",
        "type=[Grid: row-by-row] order=[Right & Down]" +
        " grid_size_x=" + grid_size_x +
        " grid_size_y=" + grid_size_y +
        " tile_overlap=" + tile_overlap +
        " first_file_index_i=0"+
        " directory=" + image_path +
        " file_names=" + image_name + 
        " output_textfile_name=TileConfiguration.txt"+
        " fusion_method=[Linear Blending]"+
        " regression_threshold=0.30"+
        " max/avg_displacement_threshold=2.50"+
        " absolute_displacement_threshold=3.50"+
        " compute_overlap" +   
        " computation_parameters=[Save memory (but be slower)]"+
        " output_directory="+output_path);
    """
    #  " image_output=[Write to disk]"+
    ### add compute overlap key word
    args = {"grid_size_x": grid_size_x,
            "grid_size_y": grid_size_y,
            "tile_overlap": tile_overlap,
            "image_path": image_path,
            "image_name": image_name,
            "output_path": output_path
            }
    res  = ij.py.run_macro(STITCHING_MACRO, args)


###############"" parse the .TXT file generated by imageJ


def parse_txt_file(path_txt =  "/media/tom/T7/Stitch/acquisition/r1_bc1/TileConfiguration.registered_ch1.txt",
                   image_name_regex = "opool1_1_MMStack"):


    file1 = open(path_txt, "r")
    list_line = file1.readlines()

    dico_stitch = {}

    for line in list_line:
        if image_name_regex in line:
            dico_stitch[line.split('; ; ')[0]] = ast.literal_eval(line.split('; ; ')[1])


    ### negative coordinate not allowed

    x_min = 0
    y_min = 0
    z_min = 0
    for img_name in dico_stitch.keys():
        x_min = min(x_min, np.min(np.array(dico_stitch[img_name])[0]))
        y_min = min(y_min, np.min(np.array(dico_stitch[img_name])[1]))
        z_min = min(z_min, np.min(np.array(dico_stitch[img_name])[2]))
    for img_name in dico_stitch.keys():
        dico_stitch[img_name] = np.array(dico_stitch[img_name])
        dico_stitch[img_name] -= np.array([x_min, y_min, z_min])
    return dico_stitch


#### stich dico_spots

def stich_dico_spots(dico_spots,
                    dico_translation,
                     dico_stitch,
                     ref_round = "r1_bc1",

                     dico_bc_gene={
                         'r1_bc1': "Rtkn2",
                         'r3_bc4': "Pecam1",
                         'r4_bc5': "Ptprb",
                         'r5_bc6': "Pdgfra",
                         'r6_bc7': "Chil3",
                         'r7_bc3': "Lamp3"
                     },
                     image_shape=[55, 2048, 2048],
                    nb_tiles = 3
                     ):

    ### register each round to the ref round
    dico_spots_registered = {}
    missing_data = []
    image_list = list(dico_spots[ref_round].keys())
    for round_t in dico_spots.keys():
        dico_spots_registered[round_t] = {}
        for image_name in image_list:
            if image_name not in dico_spots[round_t].keys():
                dico_spots_registered[round_t][image_name] = []
                missing_data.append([round_t, image_name])
                continue

            if round_t not in dico_translation[image_name][ref_round].keys() and round_t != ref_round:
                missing_data.append([round_t, image_name])
                dico_spots_registered[round_t][image_name] = []
                print(f'missing data {round_t} {image_name}')
                continue

            if round_t == ref_round:
                x_translation = 0
                y_translation = 0
            else:
                x_translation = dico_translation[image_name][ref_round][round_t]['x_translation']
                y_translation = dico_translation[image_name][ref_round][round_t]['y_translation']
            dico_spots_registered[round_t][image_name] = dico_spots[round_t][image_name] - np.array([0, x_translation, y_translation])


    ###  create df coord in ref round + gene
    image_lx = image_shape[-2]
    image_ly = image_shape[-1]
    final_shape_xy = np.array([image_lx * nb_tiles + 100, image_ly * nb_tiles + 100]) #ten pixels margin
    final_masks = np.zeros([image_shape[0], int(final_shape_xy[0]), int(final_shape_xy[1])], dtype= np.uint16 )

    #### stich all the spots
    list_x = []
    list_y = []
    list_z = []
    list_round = []
    list_gene = []
    new_spot_list_dico = {}

    for image_name in dico_translation:

        cx, cy, cz = dico_stitch[image_name]

        cx = round(cx)
        cy  = round(cy)
        cz = round(cz)
        for round_t in dico_spots_registered:
            spot_list =  dico_spots_registered[round_t][image_name]
            if round_t not in new_spot_list_dico.keys():
                new_spot_list_dico[round_t] = []
            for spot in spot_list:

                if final_masks[int(spot[0] + cz), int(spot[1] + cy), int(spot[2] + cx)] == 0:

                    list_z.append(spot[0] + cz)
                    list_y.append(spot[1] + cy)
                    list_x.append(spot[2] + cx)
                    list_round.append(round_t)
                    list_gene.append(dico_bc_gene[round_t])

                    new_spot_list_dico[round_t].append([spot[0] + cz, spot[1] + cy, spot[2] + cx])

        final_masks[cz : cz +image_shape[0] , cy : cy + image_ly, cx:cx +  image_lx] = np.ones([image_shape[0],image_ly, image_lx ])



    df_coord = pd.DataFrame()
    df_coord['x'] = list_x
    df_coord['y'] = list_y
    df_coord['z'] = list_z
    df_coord['round'] = list_round
    df_coord['gene'] = list_gene

    return df_coord, new_spot_list_dico, missing_data

    #df_coord.to_csv(f"{args.folder_of_rounds}{args.name_dico}_df_coord.csv", index=False)


#### stich segmask

def stich_segmask(dico_stitch, # np.load(f"/media/tom/T7/Stitch/acquisition/2mai_dico_stitch.npy",allow_pickle=True).item()
                  path_mask = "/media/tom/T7/stich0504/segmentation_mask",
                  image_shape=[55, 2048, 2048],
                  nb_tiles = 3):

    image_lx = image_shape[-2]
    image_ly = image_shape[-1]
    final_shape_xy = np.array([image_lx * nb_tiles + 100, image_ly * nb_tiles + 100]) #ten pixels margin
    final_masks = np.zeros([image_shape[0], int(final_shape_xy[0]), int(final_shape_xy[1])], dtype= np.uint16 )
    for path_ind_mask in list(Path(path_mask).glob("*.tif"))[:]:
        print(path_ind_mask.name)
        ind_mask = tifffile.imread(path_ind_mask)
        x_or, y_or, z_or = dico_stitch[path_ind_mask.name[:-4] + "_ch1.tif"]
        x_or, y_or, z_or, = round(x_or), round(y_or), round(z_or)
        ind_mask = ind_mask.astype(np.uint16)
        max_ind_cell =  final_masks.max()
        ind_mask[ind_mask > 0] = ind_mask[ind_mask > 0] + max_ind_cell

        #compute iou between mask and final mask

        local_mask  = final_masks[:, y_or:y_or + image_ly, x_or:x_or + image_lx]

        present_cell = np.unique(final_masks[:,  y_or:y_or + image_ly, x_or:x_or + image_lx])
        print(f'present_cell {present_cell}')
        if 0 in present_cell:
            present_cell = present_cell[1:]

        for cell in tqdm(present_cell):
            unique_inter_cell = np.unique(ind_mask[local_mask == cell])
            print(f'unique_inter_cell {unique_inter_cell} , cell {cell}')
            if 0 in unique_inter_cell:
                unique_inter_cell = unique_inter_cell[1:]

            for inter_cell in np.unique(unique_inter_cell):
                iou = np.logical_and(ind_mask == inter_cell,local_mask == cell).sum()\
                      / np.logical_or(ind_mask == inter_cell, local_mask == cell).sum()
                print(f"iou {iou} cell {cell} inter_cell {inter_cell-max_ind_cell}")

                if iou > 0.25:
                    ind_mask[ind_mask == inter_cell] = cell
                    print("iou MATCH ", iou)


        final_masks[:, y_or:y_or + image_ly, x_or:x_or + image_lx] = ind_mask
        #final_masks[:, x_or:x_or + image_lx, y_or:y_or + image_ly]
    return final_masks

if __name__ == "__main__":


    dico_spots = np.load(
        f"{args.folder_of_rounds}{args.name_dico}_dico_spots_local_detection{args.local_detection}.npy",
        allow_pickle=True).item()
    dico_translation = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_translation.npy",
                               allow_pickle=True).item()
    dico_stitch = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_stitch.npy",
                                allow_pickle=True).item()


    dico_spots = np.load("/media/tom/T7/Stitch/acquisition/2mai_dico_spots_local_detection0.npy", allow_pickle=True).item()
    dico_translation = np.load("/media/tom/T7/Stitch/acquisition/2mai_dico_translation.npy", allow_pickle=True).item()
    dico_stitch = np.load("/media/tom/T7/Stitch/acquisition/2mai_dico_stitch.npy", allow_pickle=True).item()

    df_coord, new_spot_list_dico, missing_data = stich_dico_spots(dico_spots,
                     dico_translation,
                     dico_stitch,
                     ref_round="r1_bc1",

                     dico_bc_gene={
                         'r1_bc1': "Rtkn2",
                         'r3_bc4': "Pecam1",
                         'r4_bc5': "Ptprb",
                         'r5_bc6': "Pdgfra",
                         'r6_bc7': "Chil3",
                         'r7_bc3': "Lamp3"
                     },
                     image_shape=[55, 2048, 2048],
                     nb_tiles=3
                     )
    import napari
    import random
    mip_dapi = tifffile.imread(f"/media/tom/T7/Stitch/acquisition/r1_bc1/mip_dapi.tif")


    viewer = napari.Viewer()
    viewer.add_image(mip_dapi, name = 'dapi')


    for round_t in new_spot_list_dico:
        spots = np.array(new_spot_list_dico[round_t])[:, 1:]

        ### radom color string
        color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

        viewer.add_points(spots, name = f"round_{round_t}", face_color = color,edge_color=color, size = 6)

    final_masks = stich_segmask(dico_stitch,
                  # np.load(f"/media/tom/T7/Stitch/acquisition/2mai_dico_stitch.npy",allow_pickle=True).item()
                  path_mask="/media/tom/T7/stich0504/segmentation_mask",
                  image_shape=[55, 2048, 2048],
                  nb_tiles=3)


    viewer.add_image(np.amax(final_masks, 0), name = 'final_masks')




























    ##############################
    #DRAFT
    ##############################

    dico_spots = np.load("/media/tom/T7/Stitch/acquisition/2mai_dico_spots_local_detection0.npy", allow_pickle=True).item()
    ref_round = "r1_bc1"
    image_shape = [55, 2048, 2048]
    ### register dico_spots

    image_lx = image_shape[-2]
    image_ly = image_shape[-1]
    final_shape_xy = np.array([image_lx * 3 + 100, image_ly * 3 + 100]) #ten pixels margin

    masks = np.zeros([image_shape[0], int(final_shape_xy[0]), int(final_shape_xy[1])] )


    dico_spots_registered = {}
    missing_data = []
    image_list = list(dico_spots[ref_round].keys())
    for round_t in dico_spots.keys():
        dico_spots_registered[round_t] = {}
        for image_name in image_list:
            if image_name not in dico_spots[round_t].keys():
                dico_spots_registered[round_t][image_name] = []
                missing_data.append([round_t, image_name])
                continue

            if round_t not in dico_translation[image_name][ref_round].keys() and round_t != ref_round:
                missing_data.append([round_t, image_name])
                dico_spots_registered[round_t][image_name] = []
                print(f'missing data {round_t} {image_name}')
                continue

            if round_t == ref_round:
                x_translation = 0
                y_translation = 0
            else:
                x_translation = dico_translation[image_name][ref_round][round_t]['x_translation']
                y_translation = dico_translation[image_name][ref_round][round_t]['y_translation']


            dico_spots_registered[round_t][image_name] = dico_spots[round_t][image_name] - np.array([0, x_translation, y_translation])


    ###  create df coord in ref round + gene

    dico_bc_gene = {
        'r1_bc1': "Rtkn2",
        'r3_bc4': "Pecam1",
        'r4_bc5': "Ptprb",
        'r5_bc6': "Pdgfra",
        'r6_bc7': "Chil3",
        'r7_bc3': "Lamp3"
    }

    list_x = []
    list_y = []
    list_z = []
    list_round = []
    list_gene = []
    new_spot_list_dico = {}

    for image_name in dico_translation:

        cx, cy, cz = dico_stitch[image_name]
        cx = round(cx)
        cy  = round(cy)

        for round_t in dico_spots_registered:
            spot_list =  dico_spots_registered[round_t][image_name]
            if round_t not in new_spot_list_dico.keys():
                new_spot_list_dico[round_t] = []
            for spot in spot_list:
                if
                list_z.append(spot[0] + cz)
                list_x.append(spot[1] + cy)
                list_y.append(spot[2] + cx)
                list_round.append(round_t)
                list_gene.append(dico_bc_gene[round_t])

                new_spot_list_dico[round_t].append([spot[0] + cz, spot[1] + cy, spot[2] + cx])


    df_coord = pd.DataFrame()
    df_coord['x'] = list_x
    df_coord['y'] = list_y
    df_coord['z'] = list_z
    df_coord['round'] = list_round
    df_coord['gene'] = list_gene

    df_coord.to_csv(f"{args.folder_of_rounds}{args.name_dico}_df_coord.csv", index=False)

    """import tifffile
    img_dapi = tifffile.imread(f"/media/tom/T7/Stitch/acquisition/r1_bc1/Fused_r1_bc1_ch3.tif")

    mip_dapi = np.amax(img_dapi, 0)
    tifffile.imsave(f"/media/tom/T7/Stitch/acquisition/r1_bc1/mip_dapi.tif", mip_dapi)

    import napari"""
    import napari
    mip_dapi = tifffile.imread(f"/media/tom/T7/Stitch/acquisition/r1_bc1/mip_dapi.tif")


    viewer = napari.Viewer()
    viewer.add_image(mip_dapi, name = 'dapi')


    for round_t in new_spot_list_dico:
        spots = np.array(new_spot_list_dico[round_t])[:, 1:]

        ### radom color string
        color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

        viewer.add_points(spots, name = f"round_{round_t}", face_color = color,edge_color=color, size = 6)


    ### stich mask

    import numpy as np
    import napari
    import tifffile
    from pathlib import Path
    from tqdm import tqdm

    dico_stitch = np.load(f"/media/tom/T7/Stitch/acquisition/2mai_dico_stitch.npy",
                          allow_pickle=True).item()
    path_mask = "/media/tom/T7/stich0504/segmentation_mask"
    image_shape = [55, 2048, 2048]
    image_lx = image_shape[-2]
    image_ly = image_shape[-1]
    final_shape_xy = np.array([image_lx * 3 + 100, image_ly * 3 + 100]) #ten pixels margin

    final_masks = np.zeros([image_shape[0], int(final_shape_xy[0]), int(final_shape_xy[1])], dtype= np.uint16 )
    for path_ind_mask in list(Path(path_mask).glob("*.tif"))[:]:
        print(path_ind_mask.name)
        ind_mask = tifffile.imread(path_ind_mask)
        x_or, y_or, z_or = dico_stitch[path_ind_mask.name[:-4] + "_ch1.tif"]
        x_or, y_or, z_or, = round(x_or), round(y_or), round(z_or)
        ind_mask = ind_mask.astype(np.uint16)
        max_ind_cell =  final_masks.max()
        ind_mask[ind_mask > 0] = ind_mask[ind_mask > 0] + max_ind_cell

        #compute iou between mask and final mask
        local_mask  = final_masks[:, y_or:y_or + image_ly, x_or:x_or + image_lx]
        #local_mask  = final_masks[:, x_or:x_or + image_lx,  y_or:y_or + image_ly]

        present_cell = np.unique(local_mask)
        print(f'present_cell {present_cell}')
        if 0 in present_cell:
            present_cell = present_cell[1:]

        for cell in tqdm(present_cell):
            unique_inter_cell = np.unique(ind_mask[local_mask == cell])
            print(f'unique_inter_cell {unique_inter_cell} , cell {cell}')
            if 0 in unique_inter_cell:
                unique_inter_cell = unique_inter_cell[1:]

            for inter_cell in np.unique(unique_inter_cell):
                iou = np.logical_and(ind_mask == inter_cell,local_mask == cell).sum()\
                      / np.logical_or(ind_mask == inter_cell, local_mask == cell).sum()
                print(f"iou {iou} cell {cell} inter_cell {inter_cell-max_ind_cell}")

                if iou > 0.25:
                    ind_mask[ind_mask == inter_cell] = cell
                    print("iou MATCH ", iou)


        final_masks[:, y_or:y_or + image_ly, x_or:x_or + image_lx] = ind_mask
        #final_masks[:, x_or:x_or + image_lx, y_or:y_or + image_ly]



    viewer = napari.Viewer()

    viewer.add_image(final_masks, name = 'final_masks')
