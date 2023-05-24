


import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
import tifffile
from tqdm import tqdm
import scipy
from pathlib import Path

import numpy as np

import pandas as pd
import tifffile

import SimpleITK as sitk




def compute_the_translation_fourier_shift(static_image, moving_image, # does not work
                            upsample_factor=100,
                            return_translated_image = False):

    shift, error, diffphase = phase_cross_correlation(static_image, moving_image,  upsample_factor=upsample_factor)
    # moving_image + shift = static_image

    if return_translated_image:
        offset_image = fourier_shift(np.fft.fftn(moving_image), shift)
        offset_image = np.fft.ifftn(offset_image).real
        return shift, error, diffphase, offset_image
    return shift, error, diffphase


def compute_euler_transform(fixed_image,
                            moving_image, #works ok
                             ndim = 2):

    """
    :param fixed_image: image to register to
    :param moving_image: image to register
    :param ndim: 2 or 3
    :return: thetha, x_translation, y_translation
    fixed_image + (x_translation, y_translation) = moving_image
    """

    assert ndim in [2, 3]
    fixed_image = sitk.GetImageFromArray(fixed_image)
    moving_image = sitk.GetImageFromArray(moving_image)

    if ndim == 3:
        raise NotImplementedError("3d registration not implemented yet")
        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                              moving_image,
                                                              sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

    else:

        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                              moving_image,
                                                              sitk.Euler2DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
    learningRate=1,
    numberOfIterations=100,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(fixed_image, moving_image)
    print(final_transform.GetParameters())

    print("Final metric value: {0}".format(registration_method.GetMetricValue()))
    print(
    "Optimizer's stopping condition, {0}".format(
        registration_method.GetOptimizerStopConditionDescription()
    )
    )
    final_metric_value = registration_method.GetMetricValue()
    thetha = final_transform.GetParameters()[0]
    y_translation = final_transform.GetParameters()[1]
    x_translation = final_transform.GetParameters()[2]
    assert thetha < 0.17, "angle is too big more than one degree"
    return final_metric_value, thetha, x_translation, y_translation


from skimage.transform import AffineTransform, warp
def shift(image, translation):
    transform = AffineTransform(translation=translation)
    shifted = warp(image, transform, mode='constant', preserve_range=True)

    shifted = shifted.astype(image.dtype)
    return shifted



def folder_translation(folder_of_rounds = "/media/tom/T7/Stitch/acquisition/", #works ok
                        fixed_round_name = "r1_bc1",
                       folder_regex = 'r',
                       chanel_regex = 'ch1',
                       registration_repeat = 5,
                       position_naming = True):
    """
    dico_translation[fixed_round_name][path_round.name]['x_translation']
    moving_image + shift = static_image
    """


    #### get image names form static folder

    dico_translation = {}
    list_path_img_static = list(Path(folder_of_rounds).joinpath(fixed_round_name).glob(f'*{chanel_regex}*tif*'))
    for path_img_static in tqdm(list_path_img_static):
        print()
        print(path_img_static.name)
        dico_translation[path_img_static.name] = {}
        dico_translation[path_img_static.name][fixed_round_name] = {}
        fixed_image = np.amax(tifffile.imread(path_img_static), 0).astype(float)
        for path_round in Path(folder_of_rounds).glob(f'{folder_regex}*'):
            if path_round.name != fixed_round_name:
                print(path_round.name)

                if position_naming:
                    moving_image_name = path_round.name + '_p' + path_img_static.name.split('_p')[1]
                else:
                    moving_image_name = path_img_static.name
                moving_image = np.amax(tifffile.imread(path_round.joinpath(moving_image_name)), 0).astype(float)
                assert fixed_image.shape == moving_image.shape
                assert fixed_image.ndim == 2
                rep_list = []
                try:
                    for rep in range(registration_repeat):

                        final_metric_value, thetha, x_translation, y_translation = compute_euler_transform(
                                            fixed_image = fixed_image,
                                                moving_image = moving_image,  # works ok
                                                ndim = fixed_image.ndim)
                        rep_list.append([final_metric_value, thetha, x_translation, y_translation])
                except RuntimeError:
                    print(f"registration failed {(path_img_static.name, fixed_round_name)}")
                    continue


                min_index = np.argmin(np.array(rep_list)[:, 0])
                thetha, x_translation, y_translation = rep_list[min_index][1:]


                dico_translation[path_img_static.name][fixed_round_name][path_round.name] = {'thetha': thetha,
                                                                                              'x_translation': x_translation,
                                                                                                'y_translation': y_translation}
    return dico_translation



if __name__ == "__main__":



    image1 =tifffile.imread("/media/tom/T7/Stitch/acquisition/r1_bc1/opool1_1_MMStack_3-Pos_3_ch1.tif").astype(float)
    image2 = tifffile.imread("/media/tom/T7/Stitch/acquisition/r7_bc3/opool1_1_MMStack_3-Pos_3_ch1.tif").astype(float)

    image1 = np.amax(tifffile.imread("/media/tom/T7/Stitch/acquisition/r1_bc1/opool1_1_MMStack_3-Pos_3_ch1.tif"),
                     0).astype(float)
    image2 = np.amax(tifffile.imread("/media/tom/T7/Stitch/acquisition/r7_bc3/opool1_1_MMStack_3-Pos_3_ch1.tif"),
                     0).astype(float)




    static_image = scipy.ndimage.gaussian_filter(image1, 10)
    moving_image = scipy.ndimage.gaussian_filter(image2, 10)



    shift, error, diffphase = compute_the_translation(
                            static_image = image1,
                            moving_image = image2,
                            upsample_factor=10,
                            return_translated_image=False)

    print(shift)
    print(error)

    tifffile.imread()



    image1 =tifffile.imread("/media/tom/T7/Stitch/acquisition/r1_bc1/opool1_1_MMStack_3-Pos_3_ch1.tif").astype(float)
    image2 = tifffile.imread("/media/tom/T7/Stitch/acquisition/r7_bc3/opool1_1_MMStack_3-Pos_3_ch1.tif").astype(float)

    image1 = np.amax(tifffile.imread("/media/tom/T7/Stitch/acquisition/r1_bc1/opool1_1_MMStack_3-Pos_3_ch1.tif"),
                     0).astype(float)
    image2 = np.amax(tifffile.imread("/media/tom/T7/Stitch/acquisition/r7_bc3/opool1_1_MMStack_3-Pos_3_ch1.tif"),
                     0).astype(float)

    fixed_image_r = np.amax(tifffile.imread("/media/tom/T7/Stitch/acquisition/r1_bc1/opool1_1_MMStack_3-Pos_3_ch1.tif"),
                     0).astype(float)
    moving_image_r = np.amax(tifffile.imread("/media/tom/T7/Stitch/acquisition/r7_bc3/opool1_1_MMStack_3-Pos_3_ch1.tif"),
                     0).astype(float)
    fixed_image = sitk.GetImageFromArray(image1)
    moving_image = sitk.GetImageFromArray(image2)




    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                      moving_image,
                                                      sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
    learningRate=1,
    numberOfIterations=100,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(fixed_image, moving_image)
    print(final_transform.GetParameters())

    print("Final metric value: {0}".format(registration_method.GetMetricValue()))
    print(
    "Optimizer's stopping condition, {0}".format(
        registration_method.GetOptimizerStopConditionDescription()
    )
    )
    y_translation = final_transform.GetParameters()[1]
    x_translation = final_transform.GetParameters()[2]



    shifted_image1 = shift(fixed_image_r, translation=(x_translation,y_translation))
    shifted_image2 = shift(moving_image_r, translation=(-x_translation,-y_translation))

    import napari
    viewer = napari.view_image(fixed_image_r, name="image1")
    viewer.add_image(moving_image_r, name="image2")
    viewer.add_image(shifted_image1, name="shifted_image1")

    viewer.add_image(shifted_image2, name="shifted_image22")






