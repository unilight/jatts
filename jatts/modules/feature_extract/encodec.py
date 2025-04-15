from encodec import EncodecModel
from encodec.utils import convert_audio
from einops import rearrange
from pathlib import Path

import torch
import soundfile as sf


class EnCodec:
    def __init__(self, device="cuda"):
        # Instantiate a pretrained EnCodec model
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(6.0)
        self.model.to(device)

    @torch.no_grad()
    def encode(self, wav, sr, device="cuda"):
        """
        Args:
            wav: (t)
            sr: int
        """
        wav = torch.tensor(wav).unsqueeze(0)
        wav = convert_audio(wav, sr, self.model.sample_rate, self.model.channels)
        wav = wav.to(device)

        # Ensure wav is of shape (1, 1, T)
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)

        if wav.size(0) != 1 or wav.size(1) != 1:
            wav = wav.view(1, 1, -1)

        encoded_frames = self.model.encode(wav)
        qnt = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # (b q t)
        return qnt

    def unload_model(self):
        return self.model.cache_clear()

    @torch.no_grad()
    def decode(self, codes, device="cuda"):
        """
        Args:
            codes: (b q t)
        """
        assert codes.dim() == 3
        codes = codes.to(device)
        return self.model.decode([(codes, None)]), self.model.sample_rate

    def decode_to_file(self, resps, path: Path):
        assert resps.dim() == 2, f"Require shape (t q), but got {resps.shape}."
        resps = rearrange(resps, "t q -> 1 q t")
        wavs, sr = self.decode(resps)
        sf.write(str(path), wavs.cpu()[0, 0], sr)
