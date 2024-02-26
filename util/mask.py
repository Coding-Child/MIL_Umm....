import cv2
from skimage import morphology as sk_morphology
import numpy as np


def mask_percent(np_img):
    """
    Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).
    Args:
        np_img: Image as a NumPy array.
    Returns:
        The percentage of the NumPy array that is masked.
    """

    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np.sum(np_img, axis=2)
        mask_percentage = 0 if np_sum.size == 0 else 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 0 if np_img.size == 0 else 100 - np.count_nonzero(np_img) / np_img.size * 100

    return mask_percentage


def tissue_mask(np_img):
    # To prevent selecting background patches, slides are converted to HSV, blurred,
    # and patches filtered out if maximum pixel saturation lies below 0.07
    # (which was validated to not throw out tumor data in the training set).

    np_tissue_mask = filter_purple_pink(np_img)
    np_tissue_mask = fill_small_holes(np_tissue_mask, area_threshold=3000 if np_img.shape[0] > 500 else 30)
    np_tissue_mask = remove_small_objects(np_tissue_mask, min_size=3000 if np_img.shape[0] > 500 else 30)
    return np_tissue_mask


def basic_threshold(np_img, threshold=0.0, output_type="bool"):
    """
    Return mask where a pixel has a value if it exceeds the threshold value.
    Args:
        np_img: Binary image as a NumPy array.
        threshold: The threshold value to exceed.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array pixel exceeds the threshold value.
    """

    result = (np_img > threshold)
    return parse_output_type(result, output_type)


def filter_purple_pink(np_img, output_type="bool"):
    """
    Create a mask to filter out pixels where the values are similar to purple and pink.
    Args:
        np_img: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing a mask where pixels with purple/pink values have been masked out.
    """

    np_img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(np_img_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (0, 50, 50), (179, 255, 255))
    mask = basic_threshold(mask, threshold=0, output_type="bool")

    return parse_output_type(mask, output_type)


def parse_output_type(np_img, output_type="bool"):
    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    return np_img


def remove_small_objects(np_img, min_size=3000, output_type="bool"):
    """
    Filter image to remove small objects (connected components) less than a particular minimum size.
    Args:
        np_img: Image as a NumPy array of type bool.
        min_size: Minimum size of small object to remove.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8).
    """

    result = np_img.astype(bool)  # make sure mask is boolean
    result = sk_morphology.remove_small_objects(result, min_size=min_size)
    return parse_output_type(result, output_type)


def fill_small_holes(np_img, area_threshold=3000, output_type="bool"):
    """
    Filter image to remove small holes less than a particular size.
    Args:
        np_img: Image as a NumPy array of type bool.
        area_threshold: Remove small holes below this area.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8).
    """

    result = sk_morphology.remove_small_holes(np_img, area_threshold=area_threshold)
    return parse_output_type(result, output_type)


def tissue_percent(np_img):
    """
    Determine the percentage of a NumPy array that is tissue (not masked).
    Args:
        np_img: Image as a NumPy array.
    Returns:
        The percentage of the NumPy array that is tissue.
    """

    return 100 - mask_percent(np_img)
