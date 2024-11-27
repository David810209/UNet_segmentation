import argparse
import logging
import math
import os
import yaml
import torchvision
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

from dataset import (
    Flare7kpp_Pair_Loader,
    Image_Pair_Loader,
    Light_Source_Extract_Loader,
)
from models.unet import U_Net
from models.uformer import Uformer
from models.encoder_decoder import EncoderDecoder


class Trainer:
    def __init__(self, opt):
        self.opt = opt

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = EncoderDecoder(input_ch=3, output_ch=1).to(self.device)
        self.model = U_Net(img_ch=3, output_ch=1).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr)
        # self.criterion = torch.nn.MSELoss()

        ## Use Weighted BCE Loss
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=opt.epochs
        )

        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer, step_size=opt.step_size, gamma=opt.gamma
        # )

        with open(opt.config, "r") as stream:
            config = yaml.safe_load(stream)

        self.train_dataset = Light_Source_Extract_Loader(
            config["datasets"],
        )
        self.test_dataset = Image_Pair_Loader(
            config["testing_dataset"],
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        self.start_epoch = 0

    def train(self):
        torch.backends.cudnn.benchmark = True

        self.model.zero_grad()  # zero the parameter gradients
        scaler = GradScaler()

        for epoch in range(self.start_epoch, self.opt.epochs):
            self.model.train()
            train_losses = []
            with tqdm(total=len(self.train_loader), unit="batch") as pbar:
                pbar.set_description(f"Epoch {epoch}")
                for i, batch in enumerate(self.train_loader):
                    # img = img.to(self.device)
                    # mask = mask.to(self.device)
                    img = batch["input"].to(self.device)
                    mask = batch["light_source_mask"].to(self.device)

                    with autocast():
                        output = self.model(img)
                        loss = self.criterion(output, mask)

                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()


                    # self.optimizer.zero_grad()
                    # loss = self.criterion(output, mask)
                    # loss.backward()
                    # self.optimizer.step()

                    train_losses.append(loss.item())
                    pbar.set_postfix(loss=f"{sum(train_losses) / len(train_losses):.4f}")
                    pbar.update(1)

                    # if i % 100 == 0:
                    #     logging.info(
                    #         f"Epoch [{epoch}/{self.opt.epochs}], Step [{i}/{len(self.train_loader)}], Loss: {loss.item():.4f}"
                    #     )

                self.lr_scheduler.step()

                pbar.set_postfix(loss=f"{sum(train_losses) / len(train_losses):.4f}")
                pbar.close()

                if (epoch + 1) % 2 == 0:
                    self.save_checkpoint(self.model, self.optimizer, epoch)
                    # self.validate()

        # done
        self.save_checkpoint(self.model, self.optimizer, self.opt.epochs)
        # self.validate()

    def validate(self):
        pass

    def save_checkpoint(self, model, optimizer, epoch):
        # remove previous checkpoints if ckpt more than 5
        if len(os.listdir(self.opt.save_ckpt_dir)) >= 3:
            ckpts = os.listdir(self.opt.save_ckpt_dir)
            ckpts = [int(ckpt.split("_")[1].split(".")[0]) for ckpt in ckpts]
            ckpts.sort()
            os.remove(os.path.join(self.opt.save_ckpt_dir, f"model_{ckpts[0]}.pth"))

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            os.path.join(self.opt.save_ckpt_dir, f"model_{epoch+1}.pth"),
        )

    def load_checkpoint(self, model, optimizer):
        checkpoint = torch.load(self.opt.checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        return model, optimizer, epoch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        help="Model to use for training",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/flare7kpp_dataset.yml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--save_ckpt_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save_res_dir",
        type=str,
        default="res",
        help="Directory to save results",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.exists(args.save_ckpt_dir):
        os.makedirs(args.save_ckpt_dir)

    if not os.path.exists(args.save_res_dir):
        os.makedirs(args.save_res_dir)

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()