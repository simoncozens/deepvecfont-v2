from collections import defaultdict
import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torchvision.utils import save_image
from rich.table import Table
from rich.console import Group
from rich.progress import (
    Progress,
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeRemainingColumn,
)

from deepvecfont.dataloader import get_loader
from deepvecfont.models.model_main import ModelMain
from deepvecfont.options import get_parser_main_model

from deepvecfont.models.util_funcs import device


class CustomProgress(Progress):
    def __init__(self, *args, **kwargs) -> None:
        self.losses = {}
        self.update_table({})
        super().__init__(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )

    def update_table(self, losses):
        table = Table(show_header=False)
        for loss, value in losses.items():
            table.add_row(loss, str(value))
        self.table = table

    def get_renderable(self):
        renderable = Group(self.table, *self.get_renderables())
        return renderable


class Trainer:
    def __init__(self, opts):
        self.opts = opts
        self.init_epoch = 0
        self.dir_exp = os.path.join("./experiments", opts.name_exp)
        self.dir_sample = os.path.join(self.dir_exp, "samples")
        self.dir_ckpt = os.path.join(self.dir_exp, "checkpoints")
        self.dir_log = os.path.join(self.dir_exp, "logs")
        self.logfile_train = open(os.path.join(self.dir_log, "train_loss_log.txt"), "w")
        self.logfile_val = open(os.path.join(self.dir_log, "val_loss_log.txt"), "w")

        self.train_loader = get_loader(opts, opts.batch_size)
        self.val_loader = get_loader(opts, opts.batch_size_val, mode="test")

        self.model_main = ModelMain(opts)

        if torch.cuda.is_available() and opts.multi_gpu:
            self.model_main = torch.nn.DataParallel(self.model_main)

        self.model_main.to(device)

        parameters_all = [
            {"params": self.model_main.img_encoder.parameters()},
            {"params": self.model_main.img_decoder.parameters()},
            {"params": self.model_main.modality_fusion.parameters()},
            {"params": self.model_main.transformer_main.parameters()},
            {"params": self.model_main.transformer_seqdec.parameters()},
        ]

        self.optimizer = Adam(
            parameters_all,
            lr=opts.lr,
            betas=(opts.beta1, opts.beta2),
            eps=opts.eps,
            weight_decay=opts.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.997
        )

        if opts.tboard:
            self.writer = SummaryWriter(self.dir_log)

    def train(self):
        if self.opts.restart:
            self.load_checkpoint()

        self.setup_seed(1111)
        with CustomProgress() as pb:
            epoch_pb = pb.add_task("Epoch", total=self.opts.n_epochs - self.init_epoch)
            batch_pb = pb.add_task("Batch", total=len(self.train_loader))

            for epoch in range(self.init_epoch, self.opts.n_epochs):
                for idx, data in enumerate(self.train_loader):
                    for key in data:
                        data[key] = data[key].to(device)
                    ret_dict, loss_dict, loss = self.train_step(data)

                    batches_done = epoch * len(self.train_loader) + idx + 1
                    if batches_done % self.opts.freq_log == 0:
                        self.log_message(pb, epoch, idx, loss_dict, loss, batches_done)
                        if self.opts.tboard:
                            self.write_tboard(ret_dict, loss_dict, loss, batches_done)

                    if (
                        self.opts.freq_sample > 0
                        and batches_done % self.opts.freq_sample == 0
                    ):
                        self.do_sample(epoch, ret_dict, batches_done)

                    if (
                        self.opts.freq_val > 0
                        and batches_done % self.opts.freq_val == 0
                    ):
                        self.val_step(pb, epoch, idx, batches_done)
                    pb.update(task_id=batch_pb, completed=idx + 1)
                pb.update(task_id=epoch_pb, completed=epoch + 1)

                self.scheduler.step()
                if epoch % self.opts.freq_ckpt == 0:
                    self.save_checkpoint(epoch, batches_done)
        self.logfile_train.close()
        self.logfile_val.close()

    def train_step(self, data):
        ret_dict, loss_dict = self.model_main(data)

        loss = (
            self.opts.loss_w_l1 * loss_dict["img_l1"]
            + self.opts.loss_w_pt_c * loss_dict["img_vgg_perceptual"]
            + self.opts.kl_beta * loss_dict["kl"]
            + loss_dict["svg_total"]
            + loss_dict["svg_parallel_total"]
        )

        # perform optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return ret_dict, loss_dict, loss

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def val_step(self, pb, epoch, idx, batches_done):
        with torch.no_grad():
            self.model_main.eval()
            val_loss = defaultdict(float)

            for val_data in self.val_loader:
                for key in val_data:
                    val_data[key] = val_data[key].to(device)
                _, loss_dict_val = self.model_main(val_data, mode="val")
                for item, loss in loss_dict_val.items():
                    val_loss[item] += loss

        # Average losses over the validation set
        for key in val_loss.keys():
            val_loss[key] /= len(self.val_loader)

        if self.opts.tboard:
            for key, loss in val_loss.items():
                self.writer.add_scalar(f"VAL/{key}", loss, batches_done)

        self.log_message_val(pb, epoch, idx, val_loss)

    def save_checkpoint(self, epoch, batches_done):
        torch.save(
            {
                "model": (
                    self.model_main.module.state_dict()
                    if self.opts.multi_gpu
                    else self.model_main.state_dict()
                ),
                "opt": self.optimizer.state_dict(),
                "n_epoch": epoch,
                "n_iter": batches_done,
            },
            f"{self.dir_ckpt}/{epoch}_{batches_done}.ckpt",
        )

    def load_checkpoint(self):
        checkpoint = torch.load(
            os.path.join(self.dir_ckpt, self.opts.name_ckpt + ".ckpt")
        )
        self.model_main.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["opt"])
        self.init_epoch = checkpoint["n_epoch"]
        print(f"Checkpoint loaded from {self.opts.name_ckpt}.ckpt")

    def do_sample(self, epoch, ret_dict, batches_done):
        img_sample = torch.cat(
            (ret_dict["target_image"].data, ret_dict["generated_image"].data), -2
        )
        save_file = os.path.join(
            self.dir_sample, f"train_epoch_{epoch}_batch_{batches_done}.png"
        )
        save_image(img_sample, save_file, nrow=8, normalize=True)

    def log_message_val(self, pb, epoch, idx, loss_val):
        val_msg = (
            f"Epoch: {epoch}/{self.opts.n_epochs}, Batch: {idx}/{len(self.train_loader)}, "
            f"Val loss img l1: {loss_val['img_l1']: .6f}, "
            f"Val loss img pt: {loss_val['img_vgg_perceptual']: .6f}, "
            f"Val loss total: {loss_val['svg_total']: .6f}, "
            f"Val loss cmd: {loss_val['svg_cmd']: .6f}, "
            f"Val loss args: {loss_val['svg_args']: .6f}, "
        )

        self.logfile_val.write(val_msg + "\n")
        pb.console.print(val_msg)

    def write_tboard(self, ret_dict, loss_dict, loss, batches_done):
        self.writer.add_scalar("Loss/loss", loss.item(), batches_done)
        for item, loss in loss_dict.items():
            self.writer.add_scalar(
                f"Loss/{item}",
                loss.item(),
                batches_done,
            )
        self.writer.add_image(
            "Images/target_image", ret_dict["target_image"][0], batches_done
        )
        self.writer.add_image(
            "Images/generated_output", ret_dict["generated_image"][0], batches_done
        )

    def log_message(self, pb, epoch, idx, loss_dict, loss, batches_done):
        message = {
            "Loss": f"{loss.item():.6f}",
            "img_l1_loss": f"{self.opts.loss_w_l1 * loss_dict['img_l1'].item():.6f}",
            "img_perceptual_loss": f"{self.opts.loss_w_pt_c * loss_dict['img_vgg_perceptual']:.6f}",
            "svg_total_loss": f"{loss_dict['svg_total'].item():.6f}",
            "svg_cmd_loss": f"{self.opts.loss_w_cmd * loss_dict['svg_cmd'].item():.6f}",
            "svg_args_loss": f"{self.opts.loss_w_args * loss_dict['svg_args'].item():.6f}",
            "svg_smooth_loss": f"{self.opts.loss_w_smt * loss_dict['svg_smt'].item():.6f}",
            "svg_aux_loss": f"{self.opts.loss_w_aux * loss_dict['svg_aux'].item():.6f}",
            "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
        }
        pb.update_table(message)
        message = (
            f"Epoch: {epoch}/{self.opts.n_epochs}, Batch: {idx}/{len(self.train_loader)}, "
            + ", ".join([f"{k}: {v}" for k, v in message.items()])
        )
        self.logfile_train.write(message + "\n")


def train(opts):
    Trainer(opts).train()


def main():

    opts = get_parser_main_model().parse_args()
    opts.name_exp = opts.name_exp + "_main_model"
    os.makedirs("./experiments", exist_ok=True)
    debug = True
    # Create directories
    experiment_dir = os.path.join("./experiments", opts.name_exp)
    os.makedirs(
        experiment_dir, exist_ok=debug
    )  # False to prevent multiple train run by mistake
    os.makedirs(os.path.join(experiment_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    print(f"Training on experiment {opts.name_exp}...")
    # Dump options
    with open(os.path.join(experiment_dir, "opts.txt"), "w") as f:
        for key, value in vars(opts).items():
            f.write(str(key) + ": " + str(value) + "\n")
    train(opts)


if __name__ == "__main__":
    main()
