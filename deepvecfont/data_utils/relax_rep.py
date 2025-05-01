import numpy as np

from deepvecfont.data_utils.svg_utils import MAX_SEQ_LEN

# Denumericalize works fine with numpy arrays and torch tensors, numericalize only works with torch...
from deepvecfont.models.transformers import denumericalize


# ...so we need our own numericalize function
def numericalize(cmd, n=128):
    """NOTE: shall only be called after normalization"""
    # assert np.max(cmd.origin) <= 1.0 and np.min(cmd.origin) >= -1.0
    cmd = (cmd / 30 * n).round().clip(min=0, max=n - 1).astype(int)
    return cmd


def cal_aux_bezier_pts(font_seq, n_chars):
    """
    calculate aux pts along bezier curves
    """
    pts_aux_all = []

    for j in range(n_chars):
        char_seq = font_seq[j]  # shape: MAX_SEQ_LEN ,12
        pts_aux_char = []
        for k in range(MAX_SEQ_LEN):
            stroke_seq = char_seq[k]
            stroke_cmd = np.argmax(stroke_seq[:4], -1)
            stroke_seq[4:] = denumericalize(numericalize(stroke_seq[4:], n=64), n=64)
            p0, p1, p2, p3 = (
                stroke_seq[4:6],
                stroke_seq[6:8],
                stroke_seq[8:10],
                stroke_seq[10:12],
            )
            pts_aux_stroke = []
            if stroke_cmd == 0:
                for t in range(6):
                    pts_aux_stroke.append(0)
            elif stroke_cmd == 1:  # move
                for t in [0.25, 0.5, 0.75]:
                    coord_t = p0 + t * (p3 - p0)
                    pts_aux_stroke.append(coord_t[0])
                    pts_aux_stroke.append(coord_t[1])
            elif stroke_cmd == 2:  # line
                for t in [0.25, 0.5, 0.75]:
                    coord_t = p0 + t * (p3 - p0)
                    pts_aux_stroke.append(coord_t[0])
                    pts_aux_stroke.append(coord_t[1])
            elif stroke_cmd == 3:  # curve
                for t in [0.25, 0.5, 0.75]:
                    coord_t = (
                        (1 - t) * (1 - t) * (1 - t) * p0
                        + 3 * t * (1 - t) * (1 - t) * p1
                        + 3 * t * t * (1 - t) * p2
                        + t * t * t * p3
                    )
                    pts_aux_stroke.append(coord_t[0])
                    pts_aux_stroke.append(coord_t[1])

            pts_aux_stroke = np.array(pts_aux_stroke)
            pts_aux_char.append(pts_aux_stroke)

        pts_aux_char = np.array(pts_aux_char)
        pts_aux_all.append(pts_aux_char)

    pts_aux_all = np.array(pts_aux_all)

    return pts_aux_all


def relax_a_character(char_len, char_cmds, char_args):
    new_args = []
    pre_arg = None
    for k in range(char_len):
        cur_cls = np.argmax(char_cmds[k], -1)
        cur_arg = char_args[k]
        if k > 0:
            pre_arg = char_args[k - 1]
        if cur_cls == 1:  # when k == 0, cur_cls == 1
            cur_arg = np.concatenate(
                (np.array([cur_arg[-2], cur_arg[-1]]), cur_arg), -1
            )
        else:
            cur_arg = np.concatenate(
                (np.array([pre_arg[-2], pre_arg[-1]]), cur_arg), -1
            )
        new_args.append(cur_arg)

    while (len(new_args)) < MAX_SEQ_LEN + 1:
        new_args.append(np.array([0, 0, 0, 0, 0, 0, 0, 0]))

    new_args = np.array(new_args)
    new_seq = np.concatenate((char_cmds, new_args), -1)
    return new_seq


# No need for this step any more, it's part of write_data_to_dirs
