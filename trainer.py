import torch
from torch import nn
import torchvision
import opencv_transforms.transforms as TF

import models
import utils
import dataloader
import time
import os


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class Trainer:
    def __init__(
        self,
        data_path,
        slice_image=True,
        num_clusters=9,
        batch_size=2,
        num_epochs=25,
        save_interval=1000,
        save_path="./checkpoint/sketch2color/sketch2color.pth",
        only_store_generator=False,
        sketch_model_path="./ckeckpoint/color2sketch/color2sketch.pth",
        lr=2e-4,
        lambda1=100,
        lambda2=1e-4,
        lambda3=1e-2,
        beta1=0.5,
        beta2=0.999,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Training data...")
        train_transforms = TF.Compose(
            [
                TF.RandomResizedCrop(256),
                TF.RandomHorizontalFlip(),
            ]
        )
        with torch.no_grad():
            self.netC2S = models.Color2Sketch(pretrained=True, checkpoint_path=sketch_model_path).to(
                self.device
            )
            self.netC2S.eval()

        self.train_imagefolder = dataloader.PairImageFolder(
            data_path, train_transforms, self.netC2S, num_clusters, slice_image
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_imagefolder, batch_size=batch_size, shuffle=True
        )
        print("Done!")
        print(f"Training data size : {len(self.train_imagefolder)}")

        # A : Edge, B : Color
        channels = 3 * (num_clusters + 1)
        self.netG = models.Sketch2Color(nc=channels).to(self.device)
        self.netD = models.Discriminator(nc=channels + 3).to(self.device)
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_interval = save_interval
        self.save_path = save_path
        self.only_store_generator = only_store_generator
        self.lr = lr
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        # Loss functions
        self.criterion_GAN = nn.MSELoss()  # LSGAN
        self.criterion_L1 = nn.L1Loss()  # L1 Loss
        self.criterion_L2 = nn.MSELoss()  # L2 Loss
        self.criterion_TV = TVLoss()  # Total variance Loss

        # Setup Adam optimizers for both G and D
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta2, beta2))
        self.netEx = torchvision.models.vgg16(True).features[0:4].to(self.device)
        torch.backends.cudnn.benchmark = True

        utils.make_dir(save_path)
        num_params = sum(p.numel() for p in self.netG.parameters() if p.requires_grad) + sum(
            p.numel() for p in self.netD.parameters() if p.requires_grad
        )
        print(f"Number of parameters: {num_params}")

    def save(self, loss_list_D, loss_list_G, current_epoch):
        print(f"Saving epoch {current_epoch}!")
        if not self.only_store_generator:
            state = {
                "epoch": current_epoch,
                "netG": self.netG.state_dict(),
                "netD": self.netD.state_dict(),
                "loss_list_D": loss_list_D,
                "loss_list_G": loss_list_G,
                "optimizer_G": self.optimizer_G.state_dict(),
                "optimizer_D": self.optimizer_D.state_dict(),
            }
        else:
            state = {
                "netD": self.netD.state_dict(),
            }
        torch.save(state, os.path.join(os.path.dirname(self.save_path), f"sketch2color_{current_epoch}.pth"))
        print(f"Epoch {current_epoch} saved!")

    def train(self):
        self.netG.train()
        self.netD.train()
        print("Starting Training Loop...")

        # Lists to keep track of progress
        current_epoch = 0
        loss_list_D = []
        loss_list_G = []

        # For each epoch
        last_epoch = self.num_epochs + current_epoch - 1
        for epoch in range(current_epoch, self.num_epochs + current_epoch):
            current_epoch += 1

            start_time = time.time()
            total_time = 0

            print(f"Epoch [{epoch}/{last_epoch}]")
            for i, data in enumerate(self.train_loader, 0):

                # Set model input
                edge = data[0]
                color = data[1].to(self.device)
                color_list = data[2]
                input_tensor = torch.cat([edge] + color_list, dim=1).to(self.device)
                b_size = edge.size(0)

                # Real & Fake Lebel
                real_label = torch.autograd.Variable(
                    torch.cuda.FloatTensor(b_size).fill_(0.9), requires_grad=False
                )
                fake_label = torch.autograd.Variable(
                    torch.cuda.FloatTensor(b_size).fill_(0.0), requires_grad=False
                )

                ###### Outputs ######
                fake = self.netG(input_tensor)
                pred_fake = self.netD(input_tensor, fake)
                pred_real = self.netD(input_tensor, color)
                ##################################

                ###### Discriminator ######
                self.optimizer_D.zero_grad()

                # GAN loss
                loss_D_GAN = self.criterion_GAN(pred_fake, fake_label) + self.criterion_GAN(
                    pred_real, real_label
                )

                # Total loss
                loss_D = 1.0 * loss_D_GAN
                loss_D.backward()

                # Update
                self.optimizer_D.step()
                ###################################

                ###### Outputs ######
                fake = self.netG(input_tensor)
                pred_fake = self.netD(input_tensor, fake)
                pred_real = self.netD(input_tensor, color)
                fake_feature = self.netEx(fake)
                real_feature = self.netEx(color)
                ##################################

                ###### Generators ######
                self.optimizer_G.zero_grad()

                # GAN loss
                loss_G_GAN = self.criterion_GAN(pred_fake, real_label)

                # L1 Loss
                loss_L1 = self.criterion_L1(fake, color)

                # Variance Loss
                loss_TV = self.criterion_TV(fake)

                # Feature Loss
                loss_Feature = self.criterion_L2(fake_feature, real_feature)

                # Total loss
                loss_G = (
                    1.0 * loss_G_GAN
                    + self.lambda1 * loss_L1
                    + self.lambda2 * loss_TV
                    + self.lambda3 * loss_Feature
                )
                loss_G.backward()

                # Update
                self.optimizer_G.step()
                ###################################

                if i % 5 == 0:
                    # Time Info.
                    end_time = time.time()
                    taken_time = end_time - start_time
                    total_time += taken_time
                    average_time = total_time / (i + 1)

                    # Output training stats
                    print(
                        "\r[%d/%d] Loss D: %.2f / Loss_G: %.2f / Loss_G_GAN: %.2f / Loss_L1: %.2f /"
                        " Loss_TV: %.2f / Loss_Feature: %.2f / Time : %.2f (%.2f)"
                        % (
                            i + 1,
                            len(self.train_loader),
                            loss_D.item(),
                            loss_G.item(),
                            loss_G_GAN.item(),
                            loss_L1.item(),
                            loss_TV,
                            loss_Feature,
                            taken_time,
                            average_time,
                        ),
                        end="     ",
                    )
                    start_time = end_time

                if i % self.save_interval == 0:
                    print()
                    self.save(loss_list_D, loss_list_G, current_epoch)

                # Record loss
                loss_list_D.append(loss_D.cpu().item())
                loss_list_G.append(loss_G.cpu().item())
            print()

        print("Done")
