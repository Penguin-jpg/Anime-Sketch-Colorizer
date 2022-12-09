import argparse
import torch
import models
import utils
import opencv_transforms.transforms as TF
import dataloader
import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# To ignore warning
import warnings

warnings.simplefilter("ignore", UserWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--color_model_path", type=str, help="Checkpoint path of trained Sketch2Color model")
    parser.add_argument("--sketch_model_path", type=str, help="Checkpoint path of trained Color2Sketch model")
    parser.add_argument("--data_path", type=str, help="Folder path of (un)colored image")
    parser.add_argument("--reference_path", type=str, help="Folder path of reference image")
    parser.add_argument(
        "--num_clusters", type=int, default=9, help="Cluster of K-means to find color palette"
    )
    parser.add_argument("--result_path", type=str, default="./results", help="Path to save result image")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    utils.make_dir(os.path.join(args.data_path, os.path.basename(args.data_path)))
    utils.make_dir(os.path.join(args.reference_path, os.path.basename(args.reference_path)))
    utils.make_dir(args.result_path)

    with torch.no_grad():
        netC2S = models.Color2Sketch(pretrained=True, checkpoint_path=args.sketch_model_path).to(device)
        netC2S.eval()

    channels = 3 * (args.num_clusters + 1)
    netG = models.Sketch2Color(nc=channels, pretrained=True, checkpoint_path=args.color_model_path).to(device)
    netD = models.Discriminator(nc=channels + 3).to(device)
    netG.eval()
    netD.eval()
    torch.backends.cudnn.benchmark = True

    transforms = TF.Compose([TF.Resize(512)])
    test_image_folder = dataloader.GetImageFolder(args.data_path, transforms, netC2S, args.num_clusters)
    test_data_loader = torch.utils.data.DataLoader(test_image_folder, batch_size=1, shuffle=False)
    reference_image_folder = dataloader.GetImageFolder(
        args.reference_path, transforms, netC2S, args.num_clusters
    )
    reference_data_loader = torch.utils.data.DataLoader(reference_image_folder, batch_size=1, shuffle=False)

    test_batch = next(iter(test_data_loader))
    reference_batch = next(iter(reference_data_loader))

    with torch.no_grad():
        edge = test_batch[0]
        reference = reference_batch[1].to(device)
        color_palette = reference_batch[2]
        # color_hexes = [utils.color_to_hex(color) for color in color_palette]
        input_tensor = torch.cat([edge] + color_palette, dim=1).to(device)
        fake = netG(input_tensor)
        result = torch.cat([reference, edge, fake], dim=-1).cpu()
        output = vutils.make_grid(result, nrow=1, padding=5, normalize=True).cpu().permute(1, 2, 0).numpy()
        plt.imsave(arr=output, fname=os.path.join(args.result_path, "compare.png"))
        image = utils.tensor_to_pillow_image(fake.squeeze(0))
        image.save(os.path.join(args.result_path, "result.png"))
    print("Colorization Done!")
