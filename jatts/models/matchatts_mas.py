#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Matcha-TTS with Monotonic Alignment Search (MAS)."""

import logging
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from jatts.modules.conformer.encoder import Encoder as ConformerEncoder
from jatts.modules.duration_predictor import DurationPredictor
from jatts.modules.initialize import initialize
from jatts.modules.length_regulator import LengthRegulator, GaussianUpsampling
from jatts.modules.matchatts.flow_matching import CFM
from jatts.modules.positional_encoding import (
    PositionalEncoding,
    ScaledPositionalEncoding,
)
from jatts.modules.utils import make_non_pad_mask, make_pad_mask
from jatts.modules.variance_predictor import VariancePredictor
from jatts.modules.transformer.subsampling import Conv2dSubsampling
from jatts.modules.alignments import (
    AlignmentModule,
    average_by_duration,
    viterbi_decode,
)

from typeguard import typechecked

# from espnet2.tts.gst.style_encoder import StyleEncoder # in the future

MAX_DP_OUTPUT = 10


class MatchaTTS_MAS(torch.nn.Module):
    """Matcha-TTS module with monotonic alignment search (MAS).

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
        elayers: int = 6,
        eunits: int = 1536,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        use_scaled_pos_enc: bool = True,
        use_batch_norm: bool = True,
        encoder_normalize_before: bool = True,
        encoder_concat_after: bool = False,
        reduction_factor: int = 1,
        encoder_type: str = "transformer",
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        # only for conformer
        conformer_rel_pos_type: str = "legacy",
        conformer_pos_enc_layer_type: str = "rel_pos",
        conformer_self_attn_layer_type: str = "rel_selfattn",
        conformer_activation_type: str = "swish",
        use_macaron_style_in_conformer: bool = True,
        use_cnn_in_conformer: bool = True,
        zero_triu: bool = False,
        conformer_enc_kernel_size: int = 7,
        conformer_dec_kernel_size: int = 31,
        # decoder related
        decoder_channels=[256, 256],
        decoder_dropout: float = 0.05,
        decoder_attention_head_dim: int = 64,
        decoder_n_blocks: int = 1,
        decoder_num_mid_blocks: int = 2,
        decoder_num_heads: int = 2,
        decoder_act_fn: str = "snakebeta",
        # duration predictor
        duration_predictor_type: str = "deterministic",
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 384,
        duration_predictor_kernel_size: int = 3,
        duration_predictor_dropout_rate: float = 0.1,
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
        """Initialize FastSpeech2 module.

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
        self.encoder_type = encoder_type
        self.duration_predictor_type = duration_predictor_type
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.use_gst = use_gst

        # use idx 0 as padding idx
        self.padding_idx = 0

        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )

        # MAS related
        self.viterbi_func = viterbi_decode

        # check relative positional encoding compatibility
        if "conformer" in [encoder_type]:
            if conformer_rel_pos_type == "legacy":
                if conformer_pos_enc_layer_type == "rel_pos":
                    conformer_pos_enc_layer_type = "legacy_rel_pos"
                    logging.warning(
                        "Fallback to conformer_pos_enc_layer_type = 'legacy_rel_pos' "
                        "due to the compatibility. If you want to use the new one, "
                        "please use conformer_pos_enc_layer_type = 'latest'."
                    )
                if conformer_self_attn_layer_type == "rel_selfattn":
                    conformer_self_attn_layer_type = "legacy_rel_selfattn"
                    logging.warning(
                        "Fallback to "
                        "conformer_self_attn_layer_type = 'legacy_rel_selfattn' "
                        "due to the compatibility. If you want to use the new one, "
                        "please use conformer_pos_enc_layer_type = 'latest'."
                    )
            elif conformer_rel_pos_type == "latest":
                assert conformer_pos_enc_layer_type != "legacy_rel_pos"
                assert conformer_self_attn_layer_type != "legacy_rel_selfattn"
            else:
                raise ValueError(f"Unknown rel_pos_type: {conformer_rel_pos_type}")

        # define encoder
        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=adim, padding_idx=self.padding_idx
        )
        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                idim=idim,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=encoder_input_layer,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
        elif encoder_type == "conformer":
            self.encoder = ConformerEncoder(
                idim=idim,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=encoder_input_layer,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                macaron_style=use_macaron_style_in_conformer,
                pos_enc_layer_type=conformer_pos_enc_layer_type,
                selfattention_layer_type=conformer_self_attn_layer_type,
                activation_type=conformer_activation_type,
                use_cnn_module=use_cnn_in_conformer,
                cnn_module_kernel=conformer_enc_kernel_size,
                zero_triu=zero_triu,
            )
        else:
            raise ValueError(f"{encoder_type} is not supported.")

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

        self.encoder_proj = torch.nn.Linear(adim, odim * reduction_factor)

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
        self.decoder = CFM(
            in_channels=2 * odim * reduction_factor,  # because we concat x and mu
            out_channel=odim * reduction_factor,
            channels=decoder_channels,
            dropout=decoder_dropout,
            attention_head_dim=decoder_attention_head_dim,
            n_blocks=decoder_n_blocks,
            num_mid_blocks=decoder_num_mid_blocks,
            num_heads=decoder_num_heads,
            act_fn=decoder_act_fn,
        )

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
        durations: torch.Tensor = None, # dummy
        durations_lengths: torch.Tensor = None, # dummy
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
        is_inference: bool = False,
    ) -> Sequence[torch.Tensor]:

        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)  # (B, T_text, adim)

        # integrate with GST
        if self.use_gst:
            style_embs = self.gst(ys)
            hs = hs + style_embs.unsqueeze(1)

        # integrate with SID and LID embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        # forward duration predictor and variance predictors
        d_masks = make_pad_mask(ilens).to(xs.device)
        if is_inference:
            if ys is None:
                log_p_attn = None
                ds = None
                bin_loss = 0.0
            else:
                log_p_attn = self.alignment_module(hs, ys, d_masks)
                ds, bin_loss = self.viterbi_func(log_p_attn, ilens, olens)

            # forward duration predictor
            if self.duration_predictor_type == "deterministic":
                d_outs = self.duration_predictor.inference(hs, None)
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
            hs = self.length_regulator(hs, d_outs, None, d_masks)  # (B, T_feats, adim)
        else:
            # forward alignment module and obtain duration
            log_p_attn = self.alignment_module(hs, ys, d_masks)
            ds, bin_loss = self.viterbi_func(log_p_attn, ilens, olens)

            # forward duration predictor
            h_masks = make_non_pad_mask(ilens).to(hs.device)
            if self.duration_predictor_type == "deterministic":
                d_outs = self.duration_predictor(hs, h_masks)
            elif self.duration_predictor_type == "stochastic":
                dur_nll = self.duration_predictor(
                    hs.transpose(1, 2),  # (B, T, C)
                    h_masks.unsqueeze(1),
                    w=ds.unsqueeze(1),  # (B, 1, T_text)
                )
                dur_nll = dur_nll / torch.sum(h_masks)
                ret["dur_nll"] = dur_nll

            # upsampling (expand)
            hs = self.length_regulator(
                hs,
                ds,
                make_non_pad_mask(olens).to(hs.device),
                make_non_pad_mask(ilens).to(ds.device),
            )  # (B, T_feats, adim)

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
            # Matcha-TTS requires mask during inference too.
            # So we build a all-one mask using predicted duration
            olens_in = torch.clamp_min(d_outs.sum().unsqueeze(0), 1).long()
            h_masks = self._source_mask(olens_in)

        # project to odim
        hs = self.encoder_proj(hs)

        # since there is 2x upsampling in the decoder, truncate length to multiply of 2
        olens_in = olens_in.new([olen - olen % 2 for olen in olens_in])
        max_olen_in = max(olens_in)
        h_masks = self._source_mask(olens_in)
        hs = hs[:, :max_olen_in]
        if ys is not None:
            ys = ys[:, :max_olen_in]

        # return ys, hs and h_masks for prior loss calculation
        ret = {
            "d_outs": d_outs,
            "ys": ys,
            "hs": hs,
            "olens_in": olens_in,
            "bin_loss": bin_loss,
            "log_p_attn": log_p_attn,
            "ds": ds,
        }

        # decoder forward. Note that the input to the decoder should be [B, feat_dim, time]
        if is_inference:
            ret["feat_gen"] = self.decoder.inference(
                hs.permute(0, 2, 1), h_masks, n_timesteps, temperature
            )
        else:
            cfm_loss, _ = self.decoder.compute_loss(
                x1=ys.permute(0, 2, 1), mask=h_masks, mu=hs.permute(0, 2, 1)
            )
            ret["cfm_loss"] = cfm_loss

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
            "feat_gen": _ret["feat_gen"][0].permute(1, 0),
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

        # initialize alpha in scaled positional encoding
        if self.encoder_type == "transformer" and self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
