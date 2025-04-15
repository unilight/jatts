#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VITS with mel-spectrogram output (i.e., without GAN)"""

import logging
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from jatts.modules.alignments import (AlignmentModule, average_by_duration,
                                      viterbi_decode)
from jatts.modules.conformer.encoder import Encoder as ConformerEncoder
from jatts.modules.duration_predictor import DurationPredictor
from jatts.modules.initialize import initialize
from jatts.modules.length_regulator import GaussianUpsampling, LengthRegulator
from jatts.modules.matchatts.flow_matching import CFM
from jatts.modules.positional_encoding import (PositionalEncoding,
                                               ScaledPositionalEncoding)
from jatts.modules.transformer.subsampling import Conv2dSubsampling
from jatts.modules.utils import make_non_pad_mask, make_pad_mask
from jatts.modules.variance_predictor import VariancePredictor
from jatts.modules.vits.posterior_encoder import PosteriorEncoder
from jatts.modules.vits.residual_coupling import ResidualAffineCouplingBlock
from jatts.modules.vits.text_encoder import TextEncoder
from typeguard import typechecked

# from espnet2.tts.gst.style_encoder import StyleEncoder # in the future


class VITS(torch.nn.Module):
    """VITS module

    This is a module of Matcha-TTS described in `Matcha-TTS: A fast TTS
    architecture with conditional flow matching`_,
    with the monotonic alignment search.

    .. _`Matcha-TTS: A fast TTS architecture with conditional flow matching`:
        https://arxiv.org/abs/2309.03199

    """

    @typechecked
    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        adim: int = 384,
        aheads: int = 4,
        reduction_factor: int = 1,
        # text encoder
        text_encoder_attention_heads: int = 2,
        text_encoder_ffn_expand: int = 4,
        text_encoder_blocks: int = 6,
        text_encoder_positionwise_layer_type: str = "conv1d",
        text_encoder_positionwise_conv_kernel_size: int = 1,
        text_encoder_positional_encoding_layer_type: str = "rel_pos",
        text_encoder_self_attention_layer_type: str = "rel_selfattn",
        text_encoder_activation_type: str = "swish",
        text_encoder_normalize_before: bool = True,
        text_encoder_dropout_rate: float = 0.1,
        text_encoder_positional_dropout_rate: float = 0.0,
        text_encoder_attention_dropout_rate: float = 0.0,
        text_encoder_conformer_kernel_size: int = 7,
        use_macaron_style_in_text_encoder: bool = True,
        use_conformer_conv_in_text_encoder: bool = True,
        # conformer decoder related
        dlayers: int = 6,
        dunits: int = 1536,
        decoder_positionwise_layer_type: str = "conv1d",
        decoder_positionwise_conv_kernel_size: int = 1,
        decoder_normalize_before: bool = True,
        decoder_concat_after: bool = False,
        transformer_dec_dropout_rate: float = 0.1,
        transformer_dec_positional_dropout_rate: float = 0.1,
        transformer_dec_attn_dropout_rate: float = 0.1,
        conformer_rel_pos_type: str = "legacy",
        conformer_pos_enc_layer_type: str = "rel_pos",
        conformer_self_attn_layer_type: str = "rel_selfattn",
        conformer_activation_type: str = "swish",
        use_macaron_style_in_conformer: bool = True,
        use_cnn_in_conformer: bool = True,
        conformer_dec_kernel_size: int = 31,
        # duration predictor
        duration_predictor_type: str = "deterministic",
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 384,
        duration_predictor_kernel_size: int = 3,
        duration_predictor_dropout_rate: float = 0.1,
        # posterior encoder related
        posterior_encoder_kernel_size: int = 5,
        posterior_encoder_layers: int = 16,
        posterior_encoder_stacks: int = 1,
        posterior_encoder_base_dilation: int = 1,
        posterior_encoder_dropout_rate: float = 0.0,
        use_weight_norm_in_posterior_encoder: bool = True,
        # flow related
        flow_flows: int = 4,
        flow_kernel_size: int = 5,
        flow_base_dilation: int = 1,
        flow_layers: int = 4,
        flow_dropout_rate: float = 0.0,
        use_weight_norm_in_flow: bool = True,
        use_only_mean_in_flow: bool = True,
        # extra embedding related
        spks: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "add",
        use_gst: bool = False,
        gst_tokens: int = 10,
        gst_heads: int = 4,
        gst_conv_layers: int = 6,
        gst_conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        gst_conv_kernel_size: int = 3,
        gst_conv_stride: int = 2,
        gst_gru_layers: int = 1,
        gst_gru_units: int = 128,
        # training related
        init_type: str = "xavier_uniform",
        init_enc_alpha: float = 1.0,
        use_masking: bool = False,
        use_weighted_masking: bool = False,
    ):
        """Initialize VITS module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            elayers (int): Number of encoder layers.
            eunits (int): Number of encoder hidden units.
            use_scaled_pos_enc (bool): Whether to use trainable scaled pos encoding.
            use_batch_norm (bool): Whether to use batch normalization in encoder prenet.
            encoder_normalize_before (bool): Whether to apply layernorm layer before
                encoder block.
            encoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in encoder.
            reduction_factor (int): Reduction factor.
            encoder_type (str): Encoder type ("transformer" or "conformer").
            transformer_enc_dropout_rate (float): Dropout rate in encoder except
                attention and positional encoding.
            transformer_enc_positional_dropout_rate (float): Dropout rate after encoder
                positional encoding.
            transformer_enc_attn_dropout_rate (float): Dropout rate in encoder
                self-attention module.
            conformer_rel_pos_type (str): Relative pos encoding type in conformer.
            conformer_pos_enc_layer_type (str): Pos encoding layer type in conformer.
            conformer_self_attn_layer_type (str): Self-attention layer type in conformer
            conformer_activation_type (str): Activation function type in conformer.
            use_macaron_style_in_conformer: Whether to use macaron style FFN.
            use_cnn_in_conformer: Whether to use CNN in conformer.
            zero_triu: Whether to use zero triu in relative self-attention module.
            conformer_enc_kernel_size: Kernel size of encoder conformer.
            duration_predictor_layers (int): Number of duration predictor layers.
            duration_predictor_chans (int): Number of duration predictor channels.
            duration_predictor_kernel_size (int): Kernel size of duration predictor.
            duration_predictor_dropout_rate (float): Dropout rate in duration predictor.
            spks (Optional[int]): Number of speakers. If set to > 1, assume that the
                sids will be provided as the input and use sid embedding layer.
            spk_embed_dim (Optional[int]): Speaker embedding dimension. If set to > 0,
                assume that spembs will be provided as the input.
            spk_embed_integration_type: How to integrate speaker embedding.
            use_gst (str): Whether to use global style token.
            gst_tokens (int): The number of GST embeddings.
            gst_heads (int): The number of heads in GST multihead attention.
            gst_conv_layers (int): The number of conv layers in GST.
            gst_conv_chans_list: (Sequence[int]):
                List of the number of channels of conv layers in GST.
            gst_conv_kernel_size (int): Kernel size of conv layers in GST.
            gst_conv_stride (int): Stride size of conv layers in GST.
            gst_gru_layers (int): The number of GRU layers in GST.
            gst_gru_units (int): The number of GRU units in GST.
            init_type (str): How to initialize transformer parameters.
            init_enc_alpha (float): Initial value of alpha in scaled pos encoding of the
                encoder.
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to apply weighted masking in loss
                calculation.

        """
        # initialize base classes
        torch.nn.Module.__init__(self)

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.reduction_factor = reduction_factor
        self.duration_predictor_type = duration_predictor_type
        self.use_gst = use_gst

        # use idx 0 as padding idx
        self.padding_idx = 0

        # MAS related
        self.viterbi_func = viterbi_decode

        # define encoder
        self.text_encoder = TextEncoder(
            vocabs=idim,
            attention_dim=adim,
            attention_heads=text_encoder_attention_heads,
            linear_units=adim * text_encoder_ffn_expand,
            blocks=text_encoder_blocks,
            positionwise_layer_type=text_encoder_positionwise_layer_type,
            positionwise_conv_kernel_size=text_encoder_positionwise_conv_kernel_size,
            positional_encoding_layer_type=text_encoder_positional_encoding_layer_type,
            self_attention_layer_type=text_encoder_self_attention_layer_type,
            activation_type=text_encoder_activation_type,
            normalize_before=text_encoder_normalize_before,
            dropout_rate=text_encoder_dropout_rate,
            positional_dropout_rate=text_encoder_positional_dropout_rate,
            attention_dropout_rate=text_encoder_attention_dropout_rate,
            conformer_kernel_size=text_encoder_conformer_kernel_size,
            use_macaron_style=use_macaron_style_in_text_encoder,
            use_conformer_conv=use_conformer_conv_in_text_encoder,
        )

        # define GST
        if self.use_gst:
            self.gst = StyleEncoder(
                idim=odim,  # the input is mel-spectrogram
                gst_tokens=gst_tokens,
                gst_token_dim=adim,
                gst_heads=gst_heads,
                conv_layers=gst_conv_layers,
                conv_chans_list=gst_conv_chans_list,
                conv_kernel_size=gst_conv_kernel_size,
                conv_stride=gst_conv_stride,
                gru_layers=gst_gru_layers,
                gru_units=gst_gru_units,
            )

        # define spk embedding
        self.spks = None
        if spks is not None and spks > 1:
            self.spks = spks
            self.sid_emb = torch.nn.Embedding(spks, adim)

        # define additional projection for speaker embedding
        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            self.spk_embed_dim = spk_embed_dim
            self.spk_embed_integration_type = spk_embed_integration_type
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)

        self.posterior_encoder = PosteriorEncoder(
            in_channels=odim,
            out_channels=adim,
            hidden_channels=adim,
            kernel_size=posterior_encoder_kernel_size,
            layers=posterior_encoder_layers,
            stacks=posterior_encoder_stacks,
            base_dilation=posterior_encoder_base_dilation,
            global_channels=spk_embed_dim,  # NOTE(unilight) 20250408: do we want to use spk_embed_dim as global_channels?
            dropout_rate=posterior_encoder_dropout_rate,
            use_weight_norm=use_weight_norm_in_posterior_encoder,
        )
        self.flow = ResidualAffineCouplingBlock(
            in_channels=adim,
            hidden_channels=adim,
            flows=flow_flows,
            kernel_size=flow_kernel_size,
            base_dilation=flow_base_dilation,
            layers=flow_layers,
            global_channels=spk_embed_dim,  # NOTE(unilight) 20250408: do we want to use spk_embed_dim as global_channels?
            dropout_rate=flow_dropout_rate,
            use_weight_norm=use_weight_norm_in_flow,
            use_only_mean=use_only_mean_in_flow,
        )

        # define duration predictor
        if duration_predictor_type == "deterministic":
            self.duration_predictor = DurationPredictor(
                idim=adim,
                n_layers=duration_predictor_layers,
                n_chans=duration_predictor_chans,
                kernel_size=duration_predictor_kernel_size,
                dropout_rate=duration_predictor_dropout_rate,
            )
        elif duration_predictor_type == "stochastic":
            self.duration_predictor = StochasticDurationPredictor(
                channels=adim,
                kernel_size=stochastic_duration_predictor_kernel_size,
                dropout_rate=stochastic_duration_predictor_dropout_rate,
                flows=stochastic_duration_predictor_flows,
                dds_conv_layers=stochastic_duration_predictor_dds_conv_layers,
                global_channels=-1,  # not used for now
            )
        else:
            raise ValueError(
                f"Duration predictor type: {duration_predictor_type} is not supported."
            )

        # define AlignmentModule
        self.alignment_module = AlignmentModule(adim, odim)

        # define length regulator
        # self.length_regulator = LengthRegulator()
        self.length_regulator = GaussianUpsampling()

        # define decoder
        self.decoder = ConformerEncoder(
            idim=0,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=dunits,
            num_blocks=dlayers,
            input_layer=None,
            dropout_rate=transformer_dec_dropout_rate,
            positional_dropout_rate=transformer_dec_positional_dropout_rate,
            attention_dropout_rate=transformer_dec_attn_dropout_rate,
            normalize_before=decoder_normalize_before,
            concat_after=decoder_concat_after,
            positionwise_layer_type=decoder_positionwise_layer_type,
            positionwise_conv_kernel_size=decoder_positionwise_conv_kernel_size,
            macaron_style=use_macaron_style_in_conformer,
            pos_enc_layer_type=conformer_pos_enc_layer_type,
            selfattention_layer_type=conformer_self_attn_layer_type,
            activation_type=conformer_activation_type,
            use_cnn_module=use_cnn_in_conformer,
            cnn_module_kernel=conformer_dec_kernel_size,
        )

        # define final projection
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
            init_enc_alpha=init_enc_alpha,
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        joint_training: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded token ids (B, T_text).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            feats (Tensor): Batch of padded target features (B, T_feats, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            durations (LongTensor): Batch of padded durations (B, T_text + 1).
            durations_lengths (LongTensor): Batch of duration lengths (B, T_text + 1).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            joint_training (bool): Whether to perform joint training with vocoder.

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.

        """
        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel

        batch_size = text.size(0)

        # Add eos at the last of sequence
        # xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        # for i, l in enumerate(text_lengths):
        # xs[i, l] = self.eos
        # ilens = text_lengths + 1
        xs = text
        ilens = text_lengths

        ys = feats
        olens = feats_lengths

        # forward propagation
        ret = self._forward(
            xs,
            ilens,
            ys,
            olens,
            spembs=spembs,
            sids=sids,
            lids=lids,
            is_inference=False,
        )

        # NOTE(unilight) 20250206: carefully fix this part in the future
        # modify mod part of groundtruth
        # if self.reduction_factor > 1:
        #     olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
        #     max_olen = max(olens)
        #     ys = ys[:, :max_olen]
        #     ret["ys"] = ys
        #     ret["olens"] = olens

        return ret

    def _forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        ys: Optional[torch.Tensor] = None,
        olens: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        n_timesteps: int = None,
        temperature: float = None,
        noise_scale: float = 0.667,
        is_inference: bool = False,
    ) -> Sequence[torch.Tensor]:

        ret = {}

        # forward encoder
        # hs, m_p, logs_p have shape [B, dim, T]
        hs, m_p, logs_p, x_mask = self.text_encoder(xs, ilens)

        # integrate with GST
        # NOTE(unilight) 20250408: think about how to integrate with GST in the future
        # if self.use_gst:
        #     style_embs = self.gst(ys)
        #     hs = hs + style_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        # alignment search, VAE, flow
        d_masks = make_pad_mask(ilens).to(xs.device)
        if is_inference:
            if ys is None:
                log_p_attn = None
                ds = None
                bin_loss = 0.0
            else:
                log_p_attn = self.alignment_module(hs.transpose(1, 2), ys, d_masks)
                ds, bin_loss = self.viterbi_func(log_p_attn, ilens, olens)

            # forward duration predictor
            if self.duration_predictor_type == "deterministic":
                d_outs = self.duration_predictor.inference(hs.transpose(1, 2), None)
            elif self.duration_predictor_type == "stochastic":
                _h_masks = make_non_pad_mask(ilens).to(hs.device)
                d_outs = self.duration_predictor(
                    hs.transpose(1, 2),
                    _h_masks.unsqueeze(1),
                    inverse=True,
                    noise_scale=self.stochastic_duration_predictor_noise_scale,
                ).squeeze(1)

            # upsampling
            d_masks = make_non_pad_mask(ilens).to(d_outs.device)
            m_p = self.length_regulator(
                m_p.transpose(1, 2), d_outs, None, d_masks
            ).transpose(1, 2)
            logs_p = self.length_regulator(
                logs_p.transpose(1, 2), d_outs, None, d_masks
            ).transpose(1, 2)

            # decoder
            z_p = (
                m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
            )  # [B, dim, T]
            y_lengths = torch.clamp_min(torch.sum(d_outs), 1).long().unsqueeze(0)
            y_mask = make_non_pad_mask(y_lengths).unsqueeze(1).to(z_p.device)
            z = self.flow(z_p, y_mask, g=spembs, inverse=True)
        else:
            # forward posterior encoder
            z, m_q, logs_q, y_mask = self.posterior_encoder(
                ys.transpose(1, 2), olens, g=spembs
            )  # NOTE(unilight) 20250408: right now the only condition is spembs

            # forward flow
            z_p = self.flow(z, y_mask, g=spembs)  # (B, H, T_feats)

            # forward alignment module and obtain duration
            log_p_attn = self.alignment_module(hs.transpose(1, 2), ys, d_masks)
            ds, bin_loss = self.viterbi_func(log_p_attn, ilens, olens)

            # forward duration predictor
            h_masks = make_non_pad_mask(ilens).to(hs.device)
            if self.duration_predictor_type == "deterministic":
                d_outs = self.duration_predictor(hs.transpose(1, 2), h_masks)
            elif self.duration_predictor_type == "stochastic":
                dur_nll = self.duration_predictor(
                    hs.transpose(1, 2),  # (B, T, C)
                    h_masks.unsqueeze(1),
                    w=ds.unsqueeze(1),  # (B, 1, T_text)
                )
                dur_nll = dur_nll / torch.sum(h_masks)
                ret["dur_nll"] = dur_nll

            m_p = self.length_regulator(
                m_p.transpose(1, 2),
                ds,
                make_non_pad_mask(olens).to(hs.device),
                make_non_pad_mask(ilens).to(ds.device),
            ).transpose(1, 2)
            logs_p = self.length_regulator(
                logs_p.transpose(1, 2),
                ds,
                make_non_pad_mask(olens).to(hs.device),
                make_non_pad_mask(ilens).to(ds.device),
            ).transpose(1, 2)

            ret["m_q"] = m_q  # from posterior encoder
            ret["logs_q"] = logs_q  # from posterior encoder

        # forward decoder
        if olens is not None and not is_inference:
            if self.reduction_factor > 1:
                olens_in = olens.new(
                    [
                        torch.div(olen, self.reduction_factor, rounding_mode="trunc")
                        for olen in olens
                    ]
                )
            else:
                olens_in = olens
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None
            olens_in = olens

        # decoder forward
        zs, _ = self.decoder(z.transpose(1, 2), h_masks)  # (B, T_feats, adim)
        outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)  # (B, T_feats, odim)

        # decoder reconstruction (during inference)
        if is_inference and ys is not None:
            z_bar, _, _, _ = self.posterior_encoder(
                ys.transpose(1, 2), olens, g=spembs
            )  # NOTE(unilight) 20250408: right now the only condition is spembs
            zs_bar, _ = self.decoder(
                z_bar.transpose(1, 2), h_masks
            )  # (B, T_feats, adim)
            outs_bar = self.feat_out(zs_bar).view(
                zs_bar.size(0), -1, self.odim
            )  # (B, T_feats, odim)
            ret["outs_bar"] = outs_bar

        # return ys, hs and h_masks for prior loss calculation
        ret = ret | {
            "outs": outs,
            "d_outs": d_outs,
            "ys": ys,
            "hs": hs,
            "olens_in": olens_in,
            "bin_loss": bin_loss,
            "log_p_attn": log_p_attn,
            "ds": ds,
            # kl loss related
            "m_p": m_p,  # from text encoder
            "logs_p": logs_p,  # from text encoder
            "z": z,  # from posterior encoder
            "y_mask": y_mask,  # from posterior encoder
            "z_p": z_p,  # from flow
        }

        return ret

    def inference(
        self,
        text: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        spembs: torch.Tensor = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        n_timesteps: int = None,
        temperature: float = None,
        noise_scale: float = 0.667,
        use_teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T_text,).
            feats (Optional[Tensor): Feature sequence to extract style (N, idim).
            durations (Optional[Tensor): Groundtruth of duration (T_text + 1,).
            spembs (Optional[Tensor): Speaker embedding vector (spk_embed_dim,).
            sids (Optional[Tensor]): Speaker ID (1,).
            alpha (float): Alpha to control the speed.
            use_teacher_forcing (bool): Whether to use teacher forcing.
                If true, groundtruth of duration, pitch and energy will be used.

        Returns:
            Dict[str, Tensor]: Output dict including the following items:
                * feat_gen (Tensor): Output sequence of features (T_feats, odim).
                * duration (Tensor): Duration sequence (T_text + 1,).

        """
        x, y = text, feats
        spemb, d = spembs, durations

        # add eos at the last of sequence
        # x = F.pad(x, [0, 1], "constant", self.eos)

        # setup batch axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs, ys = x.unsqueeze(0), None
        if y is not None:
            ys = y.unsqueeze(0)
            olens = torch.tensor([y.shape[0]], dtype=torch.long, device=y.device)
        else:
            ys = None
            olens = None
        if spemb is not None:
            spembs = spemb.unsqueeze(0)

        if use_teacher_forcing:
            # use groundtruth of duration, pitch, and energy
            ds, ps, es = d.unsqueeze(0), p.unsqueeze(0), e.unsqueeze(0)
            _ret = self._forward(
                xs,
                ilens,
                ys,
                ds=ds,
                ps=ps,
                es=es,
                olens=olens,
                spembs=spembs,
                sids=sids,
                lids=lids,
                n_timesteps=n_timesteps,
                temperature=temperature,
            )  # (1, T_feats, odim)
        else:
            _ret = self._forward(
                xs,
                ilens,
                ys,
                spembs=spembs,
                olens=olens,
                sids=sids,
                lids=lids,
                is_inference=True,
                n_timesteps=n_timesteps,
                temperature=temperature,
            )  # (1, T_feats, odim)

        ret = {
            # "feat_gen": _ret["feat_gen"][0].permute(1, 0),
            "feat_gen": _ret["outs"][0],
            "duration": _ret["d_outs"][0],
        }

        if _ret["log_p_attn"] is not None:
            ret["log_p_attn"] = _ret["log_p_attn"][0]
        else:
            ret["log_p_attn"] = None
        if _ret["ds"] is not None:
            ret["ds"] = _ret["ds"][0]
        else:
            ret["ds"] = None

        if "outs_bar" in _ret:
            ret["outs_bar"] = _ret["outs_bar"][0]

        return ret

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, T_text, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, T_text, adim).

        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.projection(torch.cat([hs, spembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

    def _reset_parameters(self, init_type: str, init_enc_alpha: float):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)
