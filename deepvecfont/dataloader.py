# data loader for training main model
import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T

from deepvecfont.data_utils.svg_utils import MAX_SEQ_LEN
from deepvecfont.options import get_charset


class SVGDataset(data.Dataset):
    def __init__(self, opts, mode="train"):
        super().__init__()

        root_path = opts.data_root
        lang = opts.language

        self.mode = mode
        self.img_size = opts.img_size
        self.glyphset_size = len(get_charset(opts))
        self.dim_seq = opts.dim_seq

        SetRange = T.Lambda(lambda X: 1.0 - X)  # convert [0, 1] -> [0, 1]
        self.trans = T.Compose([SetRange])
        self.font_paths = []
        self.dir_path = os.path.join(root_path, lang, self.mode)
        for root, dirs, _files in os.walk(self.dir_path):
            depth = root.count("/") - self.dir_path.count("/")
            if depth == 0:
                for dir_name in dirs:
                    self.font_paths.append(os.path.join(self.dir_path, dir_name))
        self.font_paths.sort()
        print(f"Finished loading {mode} paths, number: {str(len(self.font_paths))}")

    def __getitem__(self, index):
        item = {}
        font_path = self.font_paths[index]
        item = {}
        item["class"] = torch.LongTensor(np.load(os.path.join(font_path, "class.npy")))
        item["seq_len"] = torch.LongTensor(
            np.load(os.path.join(font_path, "seq_len.npy"))
        )
        item["sequence"] = torch.FloatTensor(
            np.load(os.path.join(font_path, "sequence_relaxed.npy"))
        ).view(self.glyphset_size, MAX_SEQ_LEN, self.dim_seq)
        item["pts_aux"] = torch.FloatTensor(
            np.load(os.path.join(font_path, "pts_aux.npy"))
        )
        item["rendered"] = (
            torch.FloatTensor(
                np.load(
                    os.path.join(font_path, "rendered_" + str(self.img_size) + ".npy")
                )
            ).view(self.glyphset_size, self.img_size, self.img_size)
            / 255.0
        )
        item["rendered"] = self.trans(item["rendered"])
        item["font_id"] = torch.FloatTensor(
            np.load(os.path.join(font_path, "font_id.npy")).astype(np.float32)
        )
        return item

    def __len__(self):
        return len(self.font_paths)


def get_loader(opts, batch_size, mode="train"):
    dataset = SVGDataset(opts, mode)
    assert len(dataset) > 0, (
        "No data found in the " + str(opts.data_root) + "/" + mode + " directory"
    )
    dataloader = data.DataLoader(dataset, batch_size, shuffle=(mode == "train"))
    return dataloader


# if __name__ == "__main__":
#     root_path = "data/new_data"
#     max_seq_len = 51
#     dim_seq = 10
#     batch_size = 1
#     glyphset_size = 52

#     loader = get_loader(root_path, glyphset_size, max_seq_len, dim_seq, batch_size, "train")
#     fout = open("train_id_record_old.txt", "w")
#     for idx, batch in enumerate(loader):
#         binary_fp = batch["font_id"].numpy()[0][0]
#         fout.write("%05d" % int(binary_fp) + "\n")
