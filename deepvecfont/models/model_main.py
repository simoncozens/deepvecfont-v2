import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from deepvecfont.data_utils.svg_utils import MAX_SEQ_LEN
from deepvecfont.models import util_funcs
from deepvecfont.options import get_charset

from .image_decoder import ImageDecoder
from .image_encoder import ImageEncoder
from .modality_fusion import ModalityFusion
from .transformers import (
    Transformer,
    Transformer_decoder,
    denumericalize,
    numericalize,
    subsequent_mask,
)
from .util_funcs import device
from .vgg_perceptual_loss import VGGPerceptualLoss


class ModelMain(nn.Module):

    def __init__(self, opts, mode="train"):
        super().__init__()
        self.opts = opts
        self.glyphset_size = len(get_charset(opts))
        self.img_encoder = ImageEncoder(
            img_size=opts.img_size,
            input_nc=opts.ref_nshot,
            ngf=opts.ngf,
            norm_layer=nn.LayerNorm,
        )
        self.img_decoder = ImageDecoder(
            img_size=opts.img_size,
            input_nc=opts.bottleneck_bits + self.glyphset_size,
            output_nc=1,
            ngf=opts.ngf,
            norm_layer=nn.LayerNorm,
        )
        self.vggptlossfunc = VGGPerceptualLoss()
        self.modality_fusion = ModalityFusion(opts)
        self.transformer_main = Transformer(
            opts,
            input_channels=1,
            input_axis=2,  # number of axis for input data (2 for images, 3 for video)
            num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
            max_freq=10.0,  # maximum frequency, hyperparameter depending on how fine the data is
            depth=6,  # depth of net. The shape of the final attention mechanism will be:
            # depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents=256,  # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=opts.dim_seq_latent,  # latent dimension
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=64,  # number of dimensions per cross attention head
            latent_dim_head=64,  # number of dimensions per latent self attention head
            num_classes=1000,  # output number of classes
            attn_dropout=0.0,
            ff_dropout=0.0,
            weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data=True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn=2,  # number of self attention blocks per cross attention
        )

        self.transformer_seqdec = Transformer_decoder(opts)

    def forward(self, data, mode="train"):

        data = self.fetch_data(data, mode)
        reference_image = data["reference_image"]
        target_image = data["target_image"]
        ref_pad_mask = data["reference_sequence_padding_mask"]
        target_class = data["target_class"]
        ref_seq_cat = data["reference_sequence_cat"]
        target_sequence_gt = data["target_sequence_gt"]
        target_sequence = data["target_sequence"]
        target_sequencelen = data["target_sequencelen"]
        target_sequence_shifted = util_funcs.shift_right(target_sequence)
        target_auxiliary_points = data["target_auxiliary_points"]

        # image encoding
        img_feat = self.img_encoder(reference_image)  # shape = [batch_size, ngf * 2**6]

        # seq encoding
        reference_image_ = reference_image.view(
            reference_image.size(0) * reference_image.size(1),
            reference_image.size(2),
            reference_image.size(3),
        ).unsqueeze(
            -1
        )  # [max_seq_len, n_bs * n_ref, 9]
        seq_feat, _ = self.transformer_main(
            reference_image_, ref_seq_cat, mask=ref_pad_mask
        )  # [n_bs * n_ref, max_seq_len + 1, 9]

        # modality fusion
        mf_output, latent_feat_seq = self.modality_fusion(
            seq_feat, img_feat, ref_pad_mask=ref_pad_mask
        )
        latent_feat_seq = self.transformer_main.att_residual(
            latent_feat_seq
        )  # [n_bs, max_seq_len + 1, bottleneck_bits]
        z = mf_output["latent"]
        kl_loss = mf_output["kl_loss"]

        # image decoding
        trg_char_onehot = util_funcs.target_class_to_onehot(
            target_class, self.glyphset_size
        )
        img_decoder_out = self.img_decoder(z, trg_char_onehot, target_image)

        loss_dict = {}

        ret_dict = {
            "generated_image": img_decoder_out["gen_imgs"],
            "reference_image": reference_image,
            "target_image": target_image,
        }

        if mode in {"train", "val"}:
            # seq decoding (training or val mode)
            tgt_mask = (
                Variable(subsequent_mask(MAX_SEQ_LEN).type_as(ref_pad_mask.data))
                .unsqueeze(0)
                .expand(z.size(0), -1, -1, -1)
                .to(device)
                .float()
            )
            command_logits, args_logits, _attn = self.transformer_seqdec(
                x=target_sequence_shifted,
                memory=latent_feat_seq,
                trg_char=target_class,
                tgt_mask=tgt_mask,
            )
            command_logits_2, args_logits_2 = self.transformer_seqdec.parallel_decoder(
                command_logits,
                args_logits,
                memory=latent_feat_seq.detach(),
                trg_char=target_class,
            )

            total_loss = self.transformer_main.loss(
                command_logits,
                args_logits,
                target_sequence,
                target_sequencelen,
                target_auxiliary_points,
            )
            total_loss_parallel = self.transformer_main.loss(
                command_logits_2,
                args_logits_2,
                target_sequence,
                target_sequencelen,
                target_auxiliary_points,
            )
            vggpt_loss = self.vggptlossfunc(img_decoder_out["gen_imgs"], target_image)
            # loss and output
            loss_svg_items = ["total", "cmd", "args", "smt", "aux"]
            # for image
            loss_dict["img"] = {}
            loss_dict["img"]["l1"] = img_decoder_out["img_l1loss"]
            loss_dict["img"]["vggpt"] = vggpt_loss["pt_c_loss"]
            # for latent
            loss_dict["kl"] = kl_loss
            # for svg
            loss_dict["svg"] = {}
            loss_dict["svg_para"] = {}
            for item in loss_svg_items:
                loss_dict["svg"][item] = total_loss[f"loss_{item}"]
                loss_dict["svg_para"][item] = total_loss_parallel[f"loss_{item}"]

        else:  # testing (inference)

            trg_len = target_sequence_shifted.size(0)
            sampled_svg = torch.zeros(
                1, target_sequence.size(1), self.opts.dim_seq_short
            ).to(device)

            for _ in range(0, trg_len):
                tgt_mask = (
                    Variable(
                        subsequent_mask(sampled_svg.size(0)).type_as(ref_seq_cat.data)
                    )
                    .unsqueeze(0)
                    .expand(sampled_svg.size(1), -1, -1, -1)
                    .to(device)
                    .float()
                )
                command_logits, args_logits, _attn = self.transformer_seqdec(
                    x=sampled_svg,
                    memory=latent_feat_seq,
                    trg_char=target_class,
                    tgt_mask=tgt_mask,
                )
                prob_comand = F.softmax(command_logits[:, -1, :], -1)
                prob_args = F.softmax(args_logits[:, -1, :], -1)
                next_command = torch.argmax(prob_comand, -1).unsqueeze(-1)
                next_args = torch.argmax(prob_args, -1)
                predict_tmp = (
                    torch.cat((next_command, next_args), -1)
                    .unsqueeze(1)
                    .transpose(0, 1)
                )
                sampled_svg = torch.cat((sampled_svg, predict_tmp), dim=0)

            sampled_svg = sampled_svg[1:]
            cmd2 = sampled_svg[:, :, 0].unsqueeze(-1)
            arg2 = sampled_svg[:, :, 1:]
            command_logits_2, args_logits_2 = self.transformer_seqdec.parallel_decoder(
                cmd_logits=cmd2,
                args_logits=arg2,
                memory=latent_feat_seq,
                trg_char=target_class,
            )
            prob_comand = F.softmax(command_logits_2, -1)
            prob_args = F.softmax(args_logits_2, -1)
            update_command = torch.argmax(prob_comand, -1).unsqueeze(-1)
            update_args = torch.argmax(prob_args, -1)

            sampled_svg_parralel = torch.cat(
                (update_command, update_args), -1
            ).transpose(0, 1)

            commands1 = (
                F.one_hot(sampled_svg[:, :, :1].long(), 4).squeeze().transpose(0, 1)
            )
            args1 = denumericalize(sampled_svg[:, :, 1:]).transpose(0, 1)
            sampled_svg_1 = torch.cat(
                [commands1.cpu().detach(), args1[:, :, 2:].cpu().detach()], dim=-1
            )

            commands2 = (
                F.one_hot(sampled_svg_parralel[:, :, :1].long(), 4)
                .squeeze()
                .transpose(0, 1)
            )
            args2 = denumericalize(sampled_svg_parralel[:, :, 1:]).transpose(0, 1)
            sampled_svg_2 = torch.cat(
                [commands2.cpu().detach(), args2[:, :, 2:].cpu().detach()], dim=-1
            )

            ret_dict["sampled_svg_1"] = sampled_svg_1
            ret_dict["sampled_svg_2"] = sampled_svg_2
            ret_dict["target_svg"] = target_sequence_gt

        return ret_dict, loss_dict

    def fetch_data(self, data, mode):

        input_image = data[
            "rendered"
        ]  # [bs, self.glyphset_size, opts.img_size, opts.img_size]
        input_sequence = data["sequence"]  #  [bs, self.glyphset_size, opts.max_seq_len]
        input_sequence_length = data["seq_len"]
        input_sequence_length = input_sequence_length + 1
        input_pts_aux = data["pts_aux"]
        arg_quant = numericalize(input_sequence[:, :, :, 4:])
        cmd_cls = torch.argmax(input_sequence[:, :, :, :4], dim=-1).unsqueeze(-1)
        input_sequence = torch.cat([cmd_cls, arg_quant], dim=-1)  # 1 + 8 = 9 dimension

        # choose reference classes and target classes
        batch_size = input_image.size(0)
        reference_class = self.generate_reference_classes(mode, batch_size)

        if mode in {"train", "val"}:
            target_class = torch.randint(
                0, self.glyphset_size, (input_image.size(0), 1)
            ).to(device)
        else:
            target_class = torch.arange(0, self.glyphset_size).to(device)
            target_class = target_class.view(self.glyphset_size, 1)
            input_image = input_image.expand(self.glyphset_size, -1, -1, -1)
            input_sequence = input_sequence.expand(self.glyphset_size, -1, -1, -1)
            input_pts_aux = input_pts_aux.expand(self.glyphset_size, -1, -1, -1)
            input_sequence_length = input_sequence_length.expand(
                self.glyphset_size, -1, -1
            )

        reference_image = util_funcs.select_imgs(
            input_image, reference_class, self.opts
        )
        # select a target glyph image
        target_image = util_funcs.select_imgs(input_image, target_class, self.opts)
        # randomly select ref vector glyphs
        ref_seq = util_funcs.select_seqs(
            input_sequence, reference_class, self.opts.dim_seq_short
        )  # [opts.batch_size, opts.ref_nshot, opts.max_seq_len, opts.dim_seq_nmr]
        # randomly select a target vector glyph
        target_sequence = util_funcs.select_seqs(
            input_sequence, target_class, self.opts.dim_seq_short
        )
        target_sequence = target_sequence.squeeze(1)
        target_auxiliary_points = util_funcs.select_seqs(
            input_pts_aux, target_class, self.opts.n_aux_pts
        )
        target_auxiliary_points = target_auxiliary_points.squeeze(1)

        # shift target sequence
        target_sequence_gt = target_sequence.clone().detach()
        target_sequence_gt = torch.cat(
            (target_sequence_gt[:, :, :1], target_sequence_gt[:, :, 3:]), -1
        )
        target_sequence = target_sequence.transpose(0, 1)

        ref_seq_cat = ref_seq.view(
            ref_seq.size(0) * ref_seq.size(1), ref_seq.size(2), ref_seq.size(3)
        )
        ref_seq_cat = ref_seq_cat.transpose(0, 1)
        ref_seqlen = util_funcs.select_seqlens(input_sequence_length, reference_class)
        ref_seqlen_cat = ref_seqlen.view(
            ref_seqlen.size(0) * ref_seqlen.size(1), ref_seqlen.size(2)
        )
        ref_pad_mask = torch.zeros(
            ref_seqlen_cat.size(0), MAX_SEQ_LEN
        )  # value = 1 means pos to be masked
        for i in range(ref_seqlen_cat.size(0)):
            ref_pad_mask[i, : ref_seqlen_cat[i]] = 1
        ref_pad_mask = ref_pad_mask.to(device).float().unsqueeze(1)

        target_sequencelen = util_funcs.select_seqlens(
            input_sequence_length, target_class
        )
        target_sequencelen = target_sequencelen.squeeze()

        return {
            "reference_image": reference_image,
            "reference_sequence_cat": ref_seq_cat,
            "reference_sequence_padding_mask": ref_pad_mask,
            "target_image": target_image,
            "target_sequence": target_sequence,
            "target_sequence_gt": target_sequence_gt,
            "target_auxiliary_points": target_auxiliary_points,
            "target_class": target_class,
            "target_sequencelen": target_sequencelen,
        }

    def generate_reference_classes(self, mode, batch_size):
        if mode == "train":
            # Choose batch_size x ref_nshot random classes
            return torch.randint(
                0, self.glyphset_size, (batch_size, self.opts.ref_nshot)
            ).to(device)
        elif mode == "val":
            # Choose first ref_nshot classes
            return (
                torch.arange(0, self.opts.ref_nshot, 1)
                .to(device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        else:
            # Take from ref_char_ids
            ref_ids = self.opts.ref_char_ids.split(",")
            ref_ids = list(map(int, ref_ids))
            assert len(ref_ids) == self.opts.ref_nshot
            return (
                torch.tensor(ref_ids)
                .to(device)
                .unsqueeze(0)
                .expand(self.glyphset_size, -1)
            )
