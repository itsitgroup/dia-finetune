#!/usr/bin/env python
# 04_finetune.py  – fine-tune Dia-1.6B decoder on (crying) clips
# --------------------------------------------------------------

import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

# optional – improves allocator fragmentation behaviour
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from dia.model import Dia
from dataset import CryingDataset               # make sure dataset.py is updated
from dia.state import EncoderInferenceState, DecoderInferenceState

device = "cuda"

# ------------------------------------------------------------------
# 1. Load Dia in full float-32 precision
# ------------------------------------------------------------------
print("[INFO] loading Dia-1.6B …")
torch.cuda.empty_cache()
print(f"[DEBUG]  GPU after empty-cache : {torch.cuda.memory_allocated()/1e9:.2f} GB")

dia = Dia.from_pretrained(
    "nari-labs/Dia-1.6B",
    compute_dtype="float32",         # 100 % FP32
    device=device,
    load_dac=False                   # no need for DAC during training
)

print(f"[DEBUG]  GPU after model load : {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ------------------------------------------------------------------
# 2. Shrink decoder sequence length to save VRAM
#    We never need the default 3072 frames for short crying clips
# ------------------------------------------------------------------
MAX_FRAMES = 256                     # <= 3 s @ 86 fps
dia.config.data.audio_length = MAX_FRAMES

# ------------------------------------------------------------------
# 3. Freeze everything except decoder + its embeddings
# ------------------------------------------------------------------
for p in dia.model.parameters():
    p.requires_grad_(False)
for p in dia.model.decoder.parameters():
    p.requires_grad_(True)

# ------------------------------------------------------------------
# 4. Dataset & DataLoader
# ------------------------------------------------------------------
ds = CryingDataset("crying/manifest.csv", dia.config, max_frames=MAX_FRAMES)
loader = DataLoader(ds,
                    batch_size=4,              # fits easily now
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True)

# ------------------------------------------------------------------
# 5. Optim / criterion
# ------------------------------------------------------------------
V   = dia.config.model.tgt_vocab_size
PAD = dia.config.data.audio_pad_value

criterion = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, dia.model.parameters()),
    lr=2e-5,                            # conservative LR
    weight_decay=1e-2
)
CLIP_NORM = 1.0

# ------------------------------------------------------------------
# 6. Training loop
# ------------------------------------------------------------------
EPOCHS = 3
for epoch in range(EPOCHS):
    dia.model.train()
    total_loss = 0.0

    for txt, aud, valid_len in loader:   # txt:[B,Ttxt]  aud:[B,MAX_FRAMES,C]
        txt = txt.to(device, dtype=torch.long)
        aud = aud.to(device, dtype=torch.long)

        # ------------ Encoder pass ------------
        src = txt.unsqueeze(1)                           # [B,1,Ttxt]
        enc_state = EncoderInferenceState.new(dia.config, src)

        B = txt.size(0)
        enc_state.padding_mask = enc_state.padding_mask[:B]
        enc_state.attn_mask    = enc_state.attn_mask[:B]

        enc_out = dia.model.encoder(txt, enc_state)

        # ------------ Decoder prep ------------
        cross_cache = dia.model.decoder.precompute_cross_attn_cache(
            enc_out, enc_state.positions, enc_state.padding_mask
        )
        dec_state = DecoderInferenceState.new(
            dia.config, enc_state, enc_out, cross_cache, dia.compute_dtype
        )

        seq = valid_len.max().item()          # longest length in batch (<= MAX_FRAMES)

        tgt_in  = aud[:, :seq-1, :]
        tgt_out = aud[:, 1:seq, :]

        # ------------ Decoder forward ------------
        logits = dia.model.decoder(tgt_in, dec_state)    # [B,seq-1,C,V]
        logits = logits.reshape(-1, V)
        tgt_out = tgt_out.reshape(-1)

        loss = criterion(logits, tgt_out)

        # ------------ Back-prop ------------
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dia.model.decoder.parameters(), CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"[EPOCH {epoch}]  avg_loss = {avg_loss:.4f}")

# ------------------------------------------------------------------
# 7. Save checkpoint
# ------------------------------------------------------------------
torch.save(dia.model.state_dict(), "dia_crying_finetuned.pt")
print("[INFO] checkpoint written -> dia_crying_finetuned.pt")