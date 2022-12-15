import cv2
import numpy as np
import os
import opencv_transforms.functional as FF
import torchvision.transforms.functional as TF


def color_cluster(img, nclusters=9):
    """
    Apply K-means clustering to the input image

    Args:
        img: Numpy array which has shape of (H, W, C)
        nclusters: # of clusters (default = 9)

    Returns:
        color_palette: list of 3D numpy arrays which have same shape of that of input image
        e.g. If input image has shape of (256, 256, 3) and nclusters is 4, the return color_palette is [color1, color2, color3, color4]
            and each component is (256, 256, 3) numpy array.

    Note:
        K-means clustering algorithm is quite computaionally intensive.
        Thus, before extracting dominant colors, the input images are resized to x0.25 size.
    """
    img_size = img.shape
    small_img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    sample = small_img.reshape((-1, 3))
    sample = np.float32(sample)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS

    _, _, centers = cv2.kmeans(sample, nclusters, None, criteria, 10, flags)
    centers = np.uint8(centers)
    color_palette = []

    for i in range(0, nclusters):
        dominant_color = np.zeros(img_size, dtype="uint8")
        dominant_color[:, :, :] = centers[i]
        color_palette.append(dominant_color)

    return color_palette


def make_tensor(image):
    image = FF.to_tensor(image)
    image = FF.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return image


def color_to_hex(color):
    """
    Convert color tensor to hex representation

    Args:
        color: color tensor with same shape of input image

    Returns:
        hex represntation of this color tensor
    """

    # only need one rgb representation
    color = color[0][0]
    color = color.add(1).div(2).mul(255)
    return "#" + f"{int(color[0]):02x}".upper() * 3


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f"{path} already exists")


def tensor_to_pillow_image(tensor):
    # denormalize to [0, 1]
    tensor = tensor.add(1).div(2)
    return TF.to_pil_image(tensor)
