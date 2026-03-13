#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This file is adapted from: https://github.com/open-mmlab/Amphion
# Copyright 2023 Amphion team, licensed under MIT

from jatts.modules.valle_amphion.llama import LlamaConfig, LlamaForCausalLM, LlamaModel
import torch
import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn


class VALLE_AMPHION_AR(nn.Module):
    def __init__(
        self,
        idim: int,
        target_vocab_size=1024,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=12,
        num_attention_heads=16,
        pad_token_id=1281,
        bos_target_id=1282,
        eos_target_id=1283,
        bos_phone_id=1284,
        eos_phone_id=1285,
        use_input_embeds=False,
        emb_dim=256,
        **kwargs,
    ):
        super(VALLE_AMPHION_AR, self).__init__()
        self.pad_token_id = idim + target_vocab_size + 1
        self.bos_target_id = idim + target_vocab_size + 2
        self.eos_target_id = idim + target_vocab_size + 3
        self.bos_phone_id = idim + target_vocab_size + 4
        self.eos_phone_id = idim + target_vocab_size + 5
        self.config = LlamaConfig(
            vocab_size=idim + target_vocab_size + 10,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_target_id,
            eos_token_id=self.eos_target_id,
        )
        self.idim = idim
        self.target_vocab_size = target_vocab_size

        self.model = LlamaForCausalLM(self.config)

        self.use_input_embeds = use_input_embeds

        # no input embedding is used to provide speaker information
        if self.use_input_embeds:
            self.emb_linear = nn.Linear(emb_dim, hidden_size)
            self.emb_linear.weight.data.normal_(mean=0.0, std=0.01)
            self.emb_linear.bias.data.zero_()

    def forward(
        self, phone_ids, phone_mask, target_ids, target_mask, input_embeds=None
    ):
        if input_embeds is not None:
            input_embeds = self.emb_linear(input_embeds)
        phone_ids, phone_mask, phone_label = self.add_phone_eos_bos_label(
            phone_ids,
            phone_mask,
            self.eos_phone_id,
            self.bos_phone_id,
            self.pad_token_id,
        )
        target_ids, target_mask, target_label = self.add_target_eos_bos_label(
            target_ids,
            target_mask,
            self.eos_target_id,
            self.bos_target_id,
            self.pad_token_id,
        )
        input_token_ids = torch.cat([phone_ids, target_ids], dim=-1)
        attention_mask = torch.cat([phone_mask, target_mask], dim=-1)
        if input_embeds is not None:
            raise NotImplementedError
            attention_mask = torch.cat(
                [
                    torch.ones(
                        (input_embeds.shape[0], input_embeds.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                    attention_mask,
                ],
                dim=-1,
            )
        labels = torch.cat([phone_label, target_label], dim=-1)
        if input_embeds is not None:
            raise NotImplementedError
            labels = torch.cat(
                [
                    -100
                    * torch.ones(
                        (input_embeds.shape[0], input_embeds.shape[1]),
                        dtype=labels.dtype,
                        device=labels.device,
                    ),
                    labels,
                ],
                dim=-1,
            )

        if input_embeds is not None:
            raise NotImplementedError
            inputs_embeds = torch.cat(
                [input_embeds, self.model.model.embed_tokens(input_token_ids)], dim=1
            )
            out = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            return out

        # print("input_token_ids", input_token_ids[0])
        # print("attention_mask", attention_mask[0])
        # print("labels", labels[0])

        out = self.model(
            input_token_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        # calcualte top1, top5, top10 accuracy
        logits = out.logits
        logits = logits[:, -target_ids.shape[1] :]
        top1_acc = logits.argmax(-1)[..., :-1] == target_ids[:, 1:]
        top1_acc = (top1_acc * target_mask[..., :-1]).sum() / target_mask.sum()

        top5_acc = torch.topk(logits[..., :-1, :], 5, dim=-1)[1]
        top5_acc = top5_acc == target_ids[:, 1:].unsqueeze(-1)
        top5_acc = (
            top5_acc * target_mask[..., :-1].unsqueeze(-1)
        ).sum() / target_mask.sum()

        top10_acc = torch.topk(logits[..., :-1, :], 10, dim=-1)[1]
        top10_acc = top10_acc == target_ids[:, 1:].unsqueeze(-1)
        top10_acc = (
            top10_acc * target_mask[..., :-1].unsqueeze(-1)
        ).sum() / target_mask.sum()

        out.top1_acc = top1_acc
        out.top5_acc = top5_acc
        out.top10_acc = top10_acc

        return out

    def add_phone_eos_bos_label(
        self, phone_ids, phone_mask, phone_eos_id, phone_bos_id, pad_token_id
    ):
        # phone_ids: [B, T]
        # phone_mask: [B, T]

        # codec ids: 1 ~ self.target_vocab_size
        # phone ids: self.target_vocab_size + 1 ~ self.target_vocab_size + self.n_tokens
        # so we add self.target_vocab_size to phone ids
        phone_ids = phone_ids + self.target_vocab_size * phone_mask
        # I think this is not needed, but just in case
        phone_ids = phone_ids * phone_mask

        # make pad token eos token, add eos token at the end
        phone_ids = F.pad(phone_ids, (0, 1), value=0) + phone_eos_id * F.pad(
            1 - phone_mask, (0, 1), value=1
        )
        # add eos mask
        phone_mask = F.pad(phone_mask, (1, 0), value=1)  
        
        # restore pad token ids
        phone_ids = phone_ids * phone_mask + pad_token_id * (
            1 - phone_mask
        )
        
        phone_ids = F.pad(phone_ids, (1, 0), value=phone_bos_id)  # add bos token
        phone_mask = F.pad(phone_mask, (1, 0), value=1)  # add bos mask

        phone_label = -100 * torch.ones_like(
            phone_ids
        )  # loss for entire phone is not computed (passed to llama)

        return phone_ids, phone_mask, phone_label

    def add_target_eos_bos_label(
        self, target_ids, target_mask, target_eos_id, target_bos_id, pad_token_id
    ):
        # target_ids: [B, T]
        # target_mask: [B, T]

        # I think this is not needed, but just in case
        target_ids = target_ids * target_mask

        # add eos token at the end
        target_ids = F.pad(target_ids, (0, 1), value=0) + target_eos_id * F.pad(
            1 - target_mask, (0, 1), value=1
        )
        # add eos mask
        target_mask = F.pad(target_mask, (1, 0), value=1)

        # restore pad token ids
        target_ids = target_ids * target_mask + pad_token_id * (1 - target_mask)
        target_ids = F.pad(target_ids, (1, 0), value=target_bos_id)
        target_mask = F.pad(target_mask, (1, 0), value=1)

        target_label = target_ids * target_mask + (-100) * (
            1 - target_mask
        )  # loss for target is computed on unmasked tokens

        return target_ids, target_mask, target_label

    def sample_hf(
        self,
        phone_ids,  # the phones of prompt and target should be concatenated together
        prompt_ids,
        inputs_embeds=None,
        max_length=2000,
        temperature=1.0,
        top_k=100,
        top_p=0.9,
        repeat_penalty=1.0,
        num_beams=1,
    ):
        """
        Generation mode.
        
        Args:
            phone_ids: [B, T]
            prompt_ids: [B, T]
            inputs_embeds: [B, T, D]
            max_length: int
            temperature: float
            top_k: int
            top_p: float
            repeat_penalty: float
            num_beams: int

        """

        if inputs_embeds is not None:
            inputs_embeds = self.emb_linear(inputs_embeds)
        phone_mask = torch.ones_like(phone_ids)
        prompt_mask = torch.ones_like(prompt_ids)
        phone_ids, _, _ = self.add_phone_eos_bos_label(
            phone_ids,
            phone_mask,
            self.eos_phone_id,
            self.bos_phone_id,
            self.pad_token_id,
        )
        prompt_ids, _, _ = self.add_target_eos_bos_label(
            prompt_ids,
            prompt_mask,
            self.eos_target_id,
            self.bos_target_id,
            self.pad_token_id,
        )
        prompt_ids = prompt_ids[:, :-1]  # remove end token. Make it continue mode

        input_token_ids = torch.cat([phone_ids, prompt_ids], dim=-1)

        if inputs_embeds is not None:
            raise NotImplementedError
            inputs_embeds = torch.cat(
                [inputs_embeds, self.model.model.embed_tokens(input_token_ids)], dim=1
            )
            generated_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                do_sample=True,
                max_length=max_length,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_target_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repeat_penalty,
            )
            gen_tokens = generated_ids[:, :-1]
            return gen_tokens

        input_length = input_token_ids.shape[1]
        generated_ids = self.model.generate(
            input_token_ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_target_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repeat_penalty,
            num_beams=num_beams,
        )

        gen_tokens = generated_ids[:, input_length:-1]

        return gen_tokens