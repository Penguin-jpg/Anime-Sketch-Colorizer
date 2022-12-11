import argparse
from trainer import Trainer

# To ignore warning
import warnings

warnings.simplefilter("ignore", UserWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Folder path of color images")
    parser.add_argument(
        "--num_clusters", type=int, default=9, help="Cluster of K-means to find color palette"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--save_interval", type=int, default=1000, help="Interval of saving model")
    parser.add_argument(
        "--save_path",
        type=str,
        default="./checkpoint/sketch2color/sketch2color.pth",
        help="Path to save checkpoint",
    )
    parser.add_argument(
        "--sketch_model_path", type=str, hhelp="Checkpoint path of trained Color2Sketch model"
    )
    parser.add_argument("--slice_image", action="store_true", help="Whether slice the training image or not")
    parser.add_argument(
        "--only_store_generator", action="store_true", help="Whether only store generator model weight"
    )
    args = parser.parse_args()

    trainer = Trainer(
        args.data_path,
        args.slice_image,
        args.num_clusters,
        args.batch_size,
        args.num_epochs,
        args.save_interval,
        args.save_path,
        args.only_store_generator,
        args.sketch_model_path,
    )

    trainer.train()
