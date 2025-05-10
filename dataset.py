import torch, numpy as np, pandas as pd

class CryingDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, dia_config):
        self.df = pd.read_csv(manifest_path)
        self.cfg = dia_config
        self.text_pad = self.cfg.data.text_pad_value
        self.max_txt  = self.cfg.data.text_length
        self.audio_pad = self.cfg.data.audio_pad_value
        self.audio_bos = self.cfg.data.audio_bos_value
        self.max_aud   = self.cfg.data.audio_length

    def __len__(self):
        return len(self.df)

    def _encode_text(self, txt):
        ids = list(txt.encode("utf-8"))[:self.max_txt]
        arr = torch.full((self.max_txt,), self.text_pad, dtype=torch.long)
        arr[:len(ids)] = torch.tensor(ids)
        return arr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        txt_ids = self._encode_text(row.text)
        aud_ids = np.load(row.path.replace("wav_44k", "dac_tokens").replace(".wav", ".npy"))
        aud = torch.tensor(aud_ids, dtype=torch.long)

        audio_padded = torch.full((self.max_aud, self.cfg.data.channels),
                                  self.audio_pad, dtype=torch.long)
        audio_padded[0] = self.audio_bos

        T = min(aud.shape[0], self.max_aud - 1)
        audio_padded[1:T + 1] = aud[:T]

        return txt_ids, audio_padded, T + 1  # +1 because BOS