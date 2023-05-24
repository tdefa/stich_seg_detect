

import numpy as np
import cellpose
from tqdm import tqdm



def stitch3D_z(masks, stitch_threshold=0.25):
    """ stitch 2D masks into 3D volume with stitch_threshold on IOU """
    mmax = masks[0].max()
    for i in range(len(masks)-1):
        try:
            iou = cellpose.metrics._intersection_over_union(masks[i+1], masks[i])[1:,1:]
            iou[iou < stitch_threshold] = 0.0
            iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou.argmax(axis=1) + 1
            ino = np.nonzero(iou.max(axis=1)==0.0)[0]
            istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)
            mmax += len(ino)
            istitch = np.append(np.array(0), istitch)
            masks[i+1] = istitch[masks[i+1]]
        except Exception as e:
            print(e)
            print("in stich")
            continue
    return masks



def erase_solitary(mask): #mask en 3D
    """
    Erase nuclei  that are present in only one Z-slice
    Args:
        mask ():

    Returns:

    """
    mask_bis = np.zeros(mask.shape)
    current_nuclei = set(np.unique(mask[0]))
    post_nuclei = set(np.unique(mask[1]))
    nuclei_to_remove =  current_nuclei - post_nuclei
    nuclei_to_keep = current_nuclei - nuclei_to_remove # reminder: set operation are different from arithemtic operation
    for nuc in nuclei_to_keep:
        mask_bis[0] += (mask[0] == nuc) * mask[0]

    for i in range(1, len(mask)-1):
        pre_nuclei = set(np.unique(mask[i-1]))
        current_nuclei = set(np.unique(mask[i]))
        post_nuclei = set(np.unique(mask[i+1]))
        nuclei_to_remove =  current_nuclei - pre_nuclei - post_nuclei
        nuclei_to_keep = current_nuclei - nuclei_to_remove # reminder: set operation are different from arithemtic operation
        for nuc in nuclei_to_keep:
            mask_bis[i] += (mask[i] == nuc) *  mask[i]
    ##traiter le cas ou n = -1
    current_nuclei = set(np.unique(mask[-1]))
    pre_nuclei = set(np.unique(mask[-2]))
    nuclei_to_remove =  current_nuclei - pre_nuclei
    nuclei_to_keep = current_nuclei - nuclei_to_remove # reminder: set operation are different from arithemtic operation
    for nuc in nuclei_to_keep:
        mask_bis[-1] += (mask[-1] == nuc) * mask[-1]
    return mask_bis




def erase_small_nuclei(mask, min_size = 340):
    for nuc in tqdm(np.unique(mask)[1:]): ## remove zero
        sum_size = np.sum((mask == nuc).astype(int))
        print(sum_size)
        if sum_size < min_size:
                mask[mask == nuc] = 0
    return mask
