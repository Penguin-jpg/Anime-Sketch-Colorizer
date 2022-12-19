import numpy as np
import torch
import opencv_transforms.functional as FF
from torchvision import datasets

from colorizer import utils


class PairImageFolder(datasets.ImageFolder):
    """
    A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    This class works properly for paired image in form of [sketch, color_image]

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        sketch_net: The network to convert color image to sketch image
        ncluster: Number of clusters when extracting color palette.
        slice_image: Whether slice the image or not when calling __getitem__

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples

     Getitem:
        img_edge: Edge image
        img: Color Image
        color_palette: Extracted color paltette
    """

    def __init__(self, root, transform, sketch_net, ncluster, slice_image=True):
        super(PairImageFolder, self).__init__(root, transform)
        self.ncluster = ncluster
        self.sketch_net = sketch_net
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.slice_image = slice_image

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(path)
        img = np.asarray(img)
        if self.slice_image:
            img = img[:, 0:512, :]
        img = self.transform(img)
        color_palette = utils.color_cluster(img, nclusters=self.ncluster)
        img = utils.make_tensor(img)

        with torch.no_grad():
            img_edge = (
                self.sketch_net(img.unsqueeze(0).to(self.device)).squeeze().permute(1, 2, 0).cpu().numpy()
            )
            img_edge = FF.to_grayscale(img_edge, num_output_channels=3)
            img_edge = FF.to_tensor(img_edge)

        for i in range(0, len(color_palette)):
            color = color_palette[i]
            color_palette[i] = utils.make_tensor(color)

        return img_edge, img, color_palette


class GetImageFolder(datasets.ImageFolder):
    """
    A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        sketch_net: The network to convert color image to sketch image
        ncluster: Number of clusters when extracting color palette.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples

     Getitem:
        img_edge: Edge image
        img: Color Image
        color_palette: Extracted color paltette
    """

    def __init__(self, root, transform, sketch_net, ncluster):
        super(GetImageFolder, self).__init__(root, transform)
        self.ncluster = ncluster
        self.sketch_net = sketch_net
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(path)
        img = np.asarray(img)
        img = self.transform(img)
        color_palette = utils.color_cluster(img, nclusters=self.ncluster)
        img = utils.make_tensor(img)

        with torch.no_grad():
            img_edge = (
                self.sketch_net(img.unsqueeze(0).to(self.device)).squeeze().permute(1, 2, 0).cpu().numpy()
            )
            img_edge = FF.to_grayscale(img_edge, num_output_channels=3)
            img_edge = FF.to_tensor(img_edge)

        for i in range(0, len(color_palette)):
            color = color_palette[i]
            color_palette[i] = utils.make_tensor(color)

        return img_edge, img, color_palette


class AnvilImageFolder(datasets.ImageFolder):
    """
    A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        sketch_net: The network to convert color image to sketch image
        ncluster: Number of clusters when extracting color palette.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples

     Getitem:
        img_edge: Edge image
        img: Color Image
    """

    def __init__(self, root, transform, sketch_net):
        super(AnvilImageFolder, self).__init__(root, transform)
        self.sketch_net = sketch_net
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)
        img = np.asarray(img)
        img = self.transform(img)
        img = utils.make_tensor(img)

        with torch.no_grad():
            img_edge = (
                self.sketch_net(img.unsqueeze(0).to(self.device)).squeeze().permute(1, 2, 0).cpu().numpy()
            )
            img_edge = FF.to_grayscale(img_edge, num_output_channels=3)
            img_edge = FF.to_tensor(img_edge)

        return img_edge, img
