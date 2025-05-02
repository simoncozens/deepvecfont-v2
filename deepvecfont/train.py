import os
import random
import shutil

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torchvision.utils import save_image

from deepvecfont.dataloader import get_loader
from deepvecfont.models.model_main import ModelMain
from deepvecfont.options import get_parser_main_model

from deepvecfont.models.util_funcs import device


class Trainer:
    def __init__(self, opts):
        self.opts = opts
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
        for epoch in range(self.opts.init_epoch, self.opts.n_epochs):
            for idx, data in enumerate(self.train_loader):
                for key in data:
                    data[key] = data[key].to(device)
                ret_dict, loss_dict, loss = self.train_step(data)

                batches_done = epoch * len(self.train_loader) + idx + 1
                if batches_done % self.opts.freq_log == 0:
                    self.log_message(epoch, idx, loss_dict, loss, batches_done)
                    if self.opts.tboard:
                        self.write_tboard(ret_dict, loss_dict, loss, batches_done)

                if (
                    self.opts.freq_sample > 0
                    and batches_done % self.opts.freq_sample == 0
                ):
                    self.do_sample(epoch, ret_dict, batches_done)

                if self.opts.freq_val > 0 and batches_done % self.opts.freq_val == 0:
                    self.val_step(epoch, idx, batches_done)

            self.scheduler.step()
            if epoch % self.opts.freq_ckpt == 0:
                self.save_checkpoint(epoch, batches_done)
        self.logfile_train.close()
        self.logfile_val.close()

    def train_step(self, data):
        ret_dict, loss_dict = self.model_main(data)

        loss = (
            self.opts.loss_w_l1 * loss_dict["img"]["l1"]
            + self.opts.loss_w_pt_c * loss_dict["img"]["vggpt"]
            + self.opts.kl_beta * loss_dict["kl"]
            + loss_dict["svg"]["total"]
            + loss_dict["svg_para"]["total"]
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

    def val_step(self, epoch, idx, batches_done):
        with torch.no_grad():
            self.model_main.eval()
            loss_val = {
                "img": {"l1": 0.0, "vggpt": 0.0},
                "svg": {"total": 0.0, "cmd": 0.0, "args": 0.0, "aux": 0.0},
                "svg_para": {"total": 0.0, "cmd": 0.0, "args": 0.0, "aux": 0.0},
            }

            for val_data in self.val_loader:
                for key in val_data:
                    val_data[key] = val_data[key].to(device)
                _, loss_dict_val = self.model_main(val_data, mode="val")
                for loss_cat in ["img", "svg"]:
                    for key, _ in loss_val[loss_cat].items():
                        loss_val[loss_cat][key] += loss_dict_val[loss_cat][key]

            for loss_cat in ["img", "svg"]:
                for key, _ in loss_val[loss_cat].items():
                    loss_val[loss_cat][key] /= len(self.val_loader)

            if self.opts.tboard:
                for loss_cat in ["img", "svg"]:
                    for key, _ in loss_val[loss_cat].items():
                        self.writer.add_scalar(
                            f"VAL/loss_{loss_cat}_{key}",
                            loss_val[loss_cat][key],
                            batches_done,
                        )

            self.log_message_val(epoch, idx, loss_val)

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
        self.opts.init_epoch = checkpoint["n_epoch"]
        print(f"Checkpoint loaded from {self.opts.name_ckpt}.ckpt")

    def do_sample(self, epoch, ret_dict, batches_done):
        img_sample = torch.cat(
            (ret_dict["img"]["trg"].data, ret_dict["img"]["out"].data), -2
        )
        save_file = os.path.join(
            self.dir_sample, f"train_epoch_{epoch}_batch_{batches_done}.png"
        )
        save_image(img_sample, save_file, nrow=8, normalize=True)

    def log_message_val(self, epoch, idx, loss_val):
        val_msg = (
            f"Epoch: {epoch}/{self.opts.n_epochs}, Batch: {idx}/{len(self.train_loader)}, "
            f"Val loss img l1: {loss_val['img']['l1']: .6f}, "
            f"Val loss img pt: {loss_val['img']['vggpt']: .6f}, "
            f"Val loss total: {loss_val['svg']['total']: .6f}, "
            f"Val loss cmd: {loss_val['svg']['cmd']: .6f}, "
            f"Val loss args: {loss_val['svg']['args']: .6f}, "
        )

        self.logfile_val.write(val_msg + "\n")
        print(val_msg)

    def write_tboard(self, ret_dict, loss_dict, loss, batches_done):
        self.writer.add_scalar("Loss/loss", loss.item(), batches_done)
        loss_img_items = ["l1", "vggpt"]
        loss_svg_items = ["total", "cmd", "args", "aux", "smt"]
        for item in loss_img_items:
            self.writer.add_scalar(
                f"Loss/img_{item}",
                loss_dict["img"][item].item(),
                batches_done,
            )
        for item in loss_svg_items:
            self.writer.add_scalar(
                f"Loss/svg_{item}",
                loss_dict["svg"][item].item(),
                batches_done,
            )
        for item in loss_svg_items:
            self.writer.add_scalar(
                f"Loss/svg_para_{item}",
                loss_dict["svg_para"][item].item(),
                batches_done,
            )
        self.writer.add_scalar(
            "Loss/img_kl_loss",
            self.opts.kl_beta * loss_dict["kl"].item(),
            batches_done,
        )
        self.writer.add_image("Images/trg_img", ret_dict["img"]["trg"][0], batches_done)
        self.writer.add_image(
            "Images/img_output", ret_dict["img"]["out"][0], batches_done
        )

    def log_message(self, epoch, idx, loss_dict, loss, batches_done):
        message = (
            f"Epoch: {epoch}/{self.opts.n_epochs}, Batch: {idx}/{len(self.train_loader)}, "
            f"Loss: {loss.item():.6f}, "
            f"img_l1_loss: {self.opts.loss_w_l1 * loss_dict['img']['l1'].item():.6f}, "
            f"img_pt_c_loss: {self.opts.loss_w_pt_c * loss_dict['img']['vggpt']:.6f}, "
            f"svg_total_loss: {loss_dict['svg']['total'].item():.6f}, "
            f"svg_cmd_loss: {self.opts.loss_w_cmd * loss_dict['svg']['cmd'].item():.6f}, "
            f"svg_args_loss: {self.opts.loss_w_args * loss_dict['svg']['args'].item():.6f}, "
            f"svg_smooth_loss: {self.opts.loss_w_smt * loss_dict['svg']['smt'].item():.6f}, "
            f"svg_aux_loss: {self.opts.loss_w_aux * loss_dict['svg']['aux'].item():.6f}, "
            f"lr: {self.optimizer.param_groups[0]['lr']:.6f}, "
            f"Step: {batches_done}"
        )
        self.logfile_train.write(message + "\n")
        print(message)


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
