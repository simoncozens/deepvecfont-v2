import argparse
from pathlib import Path


def add_language_arg(parser):
    charset_files = Path("data/char_set").glob("*.txt")
    charset_choices = sorted([f.stem for f in charset_files])
    if not charset_choices:
        raise ValueError("No charset files found in data/char_set")

    parser.add_argument("--language", type=str, default="eng", choices=charset_choices)


def get_charset(opts):
    if hasattr(opts, "data_root"):
        root = opts.data_root
    elif hasattr(opts, "output_path"):
        root = Path(opts.output_path)
    else:
        raise ValueError("No valid root path found in options")
    return (root.parent / "char_set" / f"{opts.language}.txt").read_text().strip()


def get_parser_main_model():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group = parser.add_argument_group("basic parameters training related")
    add_language_arg(group)
    group.add_argument(
        "--bottleneck_bits",
        type=int,
        default=512,
        help="latent code number of bottleneck bits",
    )
    group.add_argument("--ref_nshot", type=int, default=4, help="reference number")
    group.add_argument("--batch_size", type=int, default=64, help="batch size")
    group.add_argument(
        "--batch_size_val", type=int, default=8, help="batch size when do validation"
    )
    group.add_argument("--img_size", type=int, default=64, help="image size")
    group.add_argument(
        "--dim_seq",
        type=int,
        default=12,
        help="the dim of each stroke in a sequence, 4 + 8, 4 is cmd, and 8 is args",
    )
    group.add_argument(
        "--dim_seq_short",
        type=int,
        default=9,
        help="the short dim of each stroke in a sequence, 1 + 8, 1 is cmd class num, and 8 is args",
    )
    group.add_argument("--hidden_size", type=int, default=512, help="hidden_size")
    group.add_argument(
        "--dim_seq_latent", type=int, default=512, help="sequence encoder latent dim"
    )
    group.add_argument(
        "--ngf",
        type=int,
        default=16,
        help="the basic num of channel in image encoder and decoder",
    )
    group.add_argument(
        "--n_aux_pts",
        type=int,
        default=6,
        help="the number of aux pts in bezier curves for additional supervison",
    )

    group = parser.add_argument_group("experiment related")
    group.add_argument("--random_index", type=str, default="00")
    group.add_argument("--name_ckpt", type=str, default="600_192921")
    group.add_argument(
        "--restart", action="store_true", help="restart training", default=False
    )
    group.add_argument("--n_epochs", type=int, default=800, help="number of epochs")
    group.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="the number of samples for each glyph when testing",
    )
    group.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    group.add_argument(
        "--ref_char_ids", type=str, default="0,1,26,27", help="default is A, B, a, b"
    )

    group.add_argument(
        "--mode", type=str, default="train", choices=["train", "val", "test"]
    )
    group.add_argument("--multi_gpu", action="store_true", default=False)
    group.add_argument("--name_exp", type=str, default="dvf")
    group.add_argument("--data_root", type=Path, default="./data/vecfont_dataset/")
    group.add_argument(
        "--freq_ckpt", type=int, default=50, help="save checkpoint frequency of epoch"
    )
    group.add_argument(
        "--freq_sample", type=int, default=500, help="sample train output of steps"
    )
    group.add_argument("--freq_log", type=int, default=50, help="freq of showing logs")
    group.add_argument(
        "--freq_val", type=int, default=500, help="sample validate output of steps"
    )
    group.add_argument(
        "--beta1", type=float, default=0.9, help="beta1 of Adam optimizer"
    )
    group.add_argument(
        "--beta2", type=float, default=0.999, help="beta2 of Adam optimizer"
    )
    group.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    group.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    group.add_argument(
        "--tboard",
        action="store_true",
        default=True,
        help="whether use tensorboard to visulize loss",
    )

    group = parser.add_argument_group("loss weights")
    group.add_argument(
        "--kl_beta", type=float, default=0.01, help="latent code kl loss beta"
    )
    group.add_argument(
        "--loss_w_pt_c",
        type=float,
        default=0.001 * 10,
        help="the weight of perceptual content loss",
    )
    group.add_argument(
        "--loss_w_l1",
        type=float,
        default=1.0 * 10,
        help="the weight of image reconstruction l1 loss",
    )
    group.add_argument(
        "--loss_w_cmd", type=float, default=1.0, help="the weight of cmd loss"
    )
    group.add_argument(
        "--loss_w_args", type=float, default=1.0, help="the weight of args loss"
    )
    group.add_argument(
        "--loss_w_aux", type=float, default=0.01, help="the weight of pts aux loss"
    )
    group.add_argument(
        "--loss_w_smt", type=float, default=10.0, help="the weight of smooth loss"
    )

    return parser
