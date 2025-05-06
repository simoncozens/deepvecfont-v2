import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
import tqdm

from deepvecfont.data_utils.extract_path import make_hb_font
from deepvecfont.data_utils.make_dataset import load_font_glyphs, render_glyph
from deepvecfont.data_utils.relax_rep import cal_aux_bezier_pts, relax_a_character
from deepvecfont.data_utils.svg_utils import MAX_SEQ_LEN, render
from deepvecfont.models.model_main import ModelMain
from deepvecfont.models.transformers import denumericalize
from deepvecfont.models.util_funcs import cal_iou, svg2img, device
from deepvecfont.options import get_charset, get_parser_main_model


def make_test_font(opts):
    charset = get_charset(opts)
    font, upem = make_hb_font(opts.font_path)
    cur_font_glyphs = load_font_glyphs(
        charset, opts.font_path, font, upem, missing_ok=True
    )
    sequence = []
    seq_len = []
    rendered = []
    relaxed = []
    for charid, char in enumerate(charset):
        example = cur_font_glyphs[charid]
        sequence.append(example["sequence"])
        assert example["seq_len"][0] <= MAX_SEQ_LEN
        seq_len.append(example["seq_len"])
        if example["seq_len"][0] > 0:
            rendering = render_glyph(font, upem, char, opts.img_size)
        else:
            rendering = np.zeros((opts.img_size, opts.img_size), dtype=np.uint8)
        rendered.append(rendering)

        this_sequence = np.array(example["sequence"]).reshape((MAX_SEQ_LEN + 1), -1)
        cmd = this_sequence[:, :4]
        args = this_sequence[:, 4:]
        relaxed.append(relax_a_character(example["seq_len"][0], cmd, args))
    item = {}
    item["seq_len"] = torch.LongTensor(np.array(seq_len))
    item["sequence"] = (
        torch.FloatTensor(np.array(relaxed))
        .reshape(len(charset), -1)
        .view(1, len(charset), MAX_SEQ_LEN + 1, opts.dim_seq)
    )
    pts_aux = cal_aux_bezier_pts(relaxed, len(charset))
    item["pts_aux"] = torch.FloatTensor(pts_aux)
    item["rendered"] = (
        torch.FloatTensor(
            np.array(rendered).reshape((len(charset), opts.img_size, opts.img_size))
        )
        / 255.0
    )
    item["rendered"] = T.Lambda(lambda X: 1.0 - X)(item["rendered"])
    return item


def test_main_model(opts, data):
    opts.mode = "test"
    opts.batch_size = 1

    model_main = ModelMain(opts)
    path_ckpt = os.path.join(
        "experiments", opts.name_exp, "checkpoints", opts.name_ckpt
    )
    model_main.load_state_dict(torch.load(path_ckpt, map_location=device)["model"])
    model_main.to(device)
    model_main.eval()

    dir_res = os.path.join("./experiments/", opts.name_exp, "results")
    charset = get_charset(opts)
    glyphset_size = len(charset)
    test_idx = len(list(Path(dir_res).glob("*")))
    dir_save = Path(dir_res) / f"{test_idx:04d}"
    print("Writing result to", dir_save)
    svg_merge_dir = dir_save / "svgs_merge"
    img_dir = dir_save / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    (dir_save / "svgs_single").mkdir(parents=True, exist_ok=True)
    svg_merge_dir.mkdir(parents=True, exist_ok=True)

    iou_max = np.zeros(glyphset_size)
    idx_best_sample = np.zeros(glyphset_size)

    syn_svg_merge_f = (
        svg_merge_dir / f"{opts.name_ckpt}_syn_merge_{test_idx}.html"
    ).open("w")

    result_idxs = [
        charset.index(ref_id) for ref_id in opts.ref_chars + opts.target_glyphs
    ]
    for sample_idx in tqdm.tqdm(range(opts.n_samples)):
        ret_dict_test = model_main(data, mode="test")[0]

        svg_sampled = ret_dict_test["sampled_svg_1"]
        sampled_svg_2 = ret_dict_test["sampled_svg_2"]
        trg_seq_gt = ret_dict_test["target_svg"]

        if sample_idx == 0:

            target_image = ret_dict_test["target_image"]
            generated_image = ret_dict_test["generated_image"]

            target_image = target_image[result_idxs, ...]
            generated_image = generated_image[result_idxs, ...]
            # Zero out the reference characters from the generated image
            generated_image[0 : len(opts.ref_chars), ...] = 0.0

            img_sample_merge = torch.cat((target_image.data, generated_image.data), -2)
            save_file_merge = os.path.join(dir_save, "imgs", f"merge_{sample_idx}.png")
            save_image(img_sample_merge, save_file_merge, nrow=8, normalize=True)

            for row_idx, char_idx in enumerate(result_idxs):
                img_target = (1.0 - target_image[row_idx, ...]).data
                save_file_target = img_dir / f"{char_idx:02d}_target.png"
                save_image(img_target, save_file_target, normalize=True)

                img_sample = (1.0 - generated_image[row_idx, ...]).data
                save_file = img_dir / f"{char_idx:02d}_{opts.img_size}.png"
                save_image(img_sample, save_file, normalize=True)

        # write results w/o parallel refinement
        svg_dec_out = svg_sampled.clone().detach()
        for i, one_seq in enumerate(svg_dec_out):
            syn_svg_outfile = os.path.join(
                os.path.join(dir_save, "svgs_single"),
                f"syn_{i:02d}_{sample_idx}_wo_refine.svg",
            )

            syn_svg_f_ = open(syn_svg_outfile, "w")
            svg = render(one_seq.cpu().numpy())
            syn_svg_f_.write(svg)
            # syn_svg_merge_f.write(svg)
            if i > 0 and i % 13 == 12:
                syn_svg_f_.write("<br>")
                # syn_svg_merge_f.write('<br>')

            syn_svg_f_.close()

            # write results w/ parallel refinement
        svg_dec_out = sampled_svg_2.clone().detach()
        for i, one_seq in enumerate(svg_dec_out):
            if i not in result_idxs:
                continue
            syn_svg_outfile = os.path.join(
                os.path.join(dir_save, "svgs_single"),
                f"syn_{i:02d}_{sample_idx}_refined.svg",
            )

            syn_svg_f = open(syn_svg_outfile, "w")
            svg = render(one_seq.cpu().numpy())
            syn_svg_f.write(svg)
            syn_svg_f.close()
            syn_img_outfile = syn_svg_outfile.replace(".svg", ".png")
            svg2img(syn_svg_outfile, syn_img_outfile, img_size=opts.img_size)
            iou_tmp, _l1_tmp = cal_iou(
                syn_img_outfile,
                os.path.join(dir_save, "imgs", f"{i:02d}_{opts.img_size}.png"),
            )
            if iou_tmp > iou_max[i]:
                iou_max[i] = iou_tmp
                idx_best_sample[i] = sample_idx

    for i in range(glyphset_size):
        # print(idx_best_sample[i])
        syn_svg_outfile_best = os.path.join(
            os.path.join(dir_save, "svgs_single"),
            f"syn_{i:02d}_{int(idx_best_sample[i])}_refined.svg",
        )
        syn_svg_merge_f.write(open(syn_svg_outfile_best, "r").read())
        if i > 0 and i % 13 == 12:
            syn_svg_merge_f.write("<br>")

    svg_target = trg_seq_gt.clone().detach()
    tgt_commands_onehot = F.one_hot(svg_target[:, :, :1].long(), 4).squeeze()
    tgt_args_denum = denumericalize(svg_target[:, :, 1:])
    svg_target = torch.cat([tgt_commands_onehot, tgt_args_denum], dim=-1)

    syn_svg_merge_f.write("\n<h2>Ground Truth</h2>\n")
    for i, one_gt_seq in enumerate(svg_target):
        gt_svg = render(one_gt_seq.cpu().numpy())
        syn_svg_merge_f.write(gt_svg)
        if i > 0 and i % 13 == 12:
            syn_svg_merge_f.write("<br>")

    syn_svg_merge_f.close()


def main():
    parser = get_parser_main_model()

    parser.add_argument("font_path", type=str, help="Path to the font file")
    parser.add_argument("target_glyphs", type=str, help="Target glyphs")
    opts = parser.parse_args()
    opts.name_exp = opts.name_exp + "_main_model"
    data = make_test_font(opts)
    print(f"Testing on experiment {opts.name_exp}...")
    # Dump options
    test_main_model(opts, data)


if __name__ == "__main__":
    main()
