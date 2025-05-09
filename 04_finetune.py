import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from dia.model import Dia
from dataset import CryingDataset
from dia.state import EncoderInferenceState, DecoderInferenceState

device = "cuda"
dia = Dia.from_pretrained("nari-labs/Dia-1.6B",
                          compute_dtype="float16", device=device,
                          load_dac=False)

V = dia.config.model.tgt_vocab_size
C = dia.config.data.channels
BOS = dia.config.data.audio_bos_value        # 1026
PAD = dia.config.data.audio_pad_value        # 1025

# ---- freeze everything except decoder & its embeddings ----
for p in dia.model.parameters(): p.requires_grad_(False)
for p in dia.model.decoder.parameters(): p.requires_grad_(True)

ds = CryingDataset("crying/manifest.csv", dia.config)
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

criterion = nn.CrossEntropyLoss(ignore_index=dia.config.data.audio_pad_value)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, dia.model.parameters()),
                        lr=3e-5, weight_decay=1e-2)

EPOCHS = 5
for epoch in range(EPOCHS):
    dia.model.train()
    for txt, aud, T in loader:               # txt:[B,Ttxt] aud:[B,Taud,C]
        txt, aud = txt.to(device), aud.to(device)

        # ---- encoder ----
        src = txt.unsqueeze(1)               # [B,1,Ttxt]  matches EncoderInferenceState.new()
        enc_state = EncoderInferenceState.new(dia.config, src)
        enc_out   = dia.model.encoder(txt, enc_state)        # txt:[B,Ttxt]

        # ---- cross-attn cache for every decoder layer ----
        cross_cache = dia.model.decoder.precompute_cross_attn_cache(
            enc_out, enc_state.positions, enc_state.padding_mask
        )
        dec_state = DecoderInferenceState.new(
            dia.config, enc_state, enc_out, cross_cache, dia.compute_dtype
        )

        # ---- teacher forcing ----
        tgt_in  = aud[:, :-1, :]             # drop last frame
        tgt_out = aud[:, 1:, :]              # next-frame target
        logits  = dia.model.decoder(tgt_in, dec_state)       # [B,T-1,C,V]

        # reshape for CE loss
        logits  = logits.reshape(-1, V)                      # [(B*T-1*C),V]
        tgt_out = tgt_out.reshape(-1)                       # [(B*T-1*C)]
        loss    = criterion(logits, tgt_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch}: loss {loss.item():.4f}")

torch.save(dia.model.state_dict(), "dia_crying_finetuned.pt")