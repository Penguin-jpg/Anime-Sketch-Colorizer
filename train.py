import argparse
from trainer import Trainer

# To ignore warning
import warnings

warnings.simplefilter("ignore", UserWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path of color images")
    parser.add_argument(
        "--num_clusters", type=int, default=9, help="Cluster of K-means to find color palette"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--save_interval", type=int, default=1000, help="Interval of saving model")
    parser.add_argument("--slice_image", action="store_true", type="Whether slice the training image or not")
    args = parser.parse_args()

    trainer = Trainer(
        args.data_path,
        args.slice_image,
        args.num_clusters,
        args.batch_size,
        args.num_epochs,
        args.save_interval,
    )

    trainer.train()
