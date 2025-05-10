import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from dia.model import Dia
from dataset import CryingDataset
from dia.state import EncoderInferenceState, DecoderInferenceState

torch.cuda.empty_cache()
print(f"Before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB used")

device = "cuda"
dia = Dia.from_pretrained(
    "nari-labs/Dia-1.6B",
    compute_dtype="float32",      # FULL float32 precision enforced
    device=device,
    load_dac=False
)

print(f"After loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB used")

V = dia.config.model.tgt_vocab_size
PAD = dia.config.data.audio_pad_value

# ---- freeze everything except decoder & its embeddings ----
for p in dia.model.parameters():
    p.requires_grad_(False)
for p in dia.model.decoder.parameters():
    p.requires_grad_(True)

ds = CryingDataset("crying/manifest.csv", dia.config)
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)  # batch 4 now fine

criterion = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, dia.model.parameters()),
                        lr=2e-5, weight_decay=1e-2)
clip_norm = 1.0

EPOCHS = 3
for epoch in range(EPOCHS):
    dia.model.train()
    total_loss = 0
    for txt, aud, _ in loader:
        txt, aud = txt.to(device, dtype=torch.long), aud.to(device, dtype=torch.long)

        # ---- encoder ----
        src = txt.unsqueeze(1)  # [B,1,Ttxt]
        enc_state = EncoderInferenceState.new(dia.config, src)

        B = txt.size(0)
        enc_state.padding_mask = enc_state.padding_mask[:B]
        enc_state.attn_mask = enc_state.attn_mask[:B]

        enc_out = dia.model.encoder(txt, enc_state)

        # ---- decoder ----
        cross_cache = dia.model.decoder.precompute_cross_attn_cache(
            enc_out, enc_state.positions, enc_state.padding_mask
        )
        dec_state = DecoderInferenceState.new(
            dia.config, enc_state, enc_out, cross_cache, dia.compute_dtype
        )

        tgt_in  = aud[:, :-1, :]
        tgt_out = aud[:, 1:, :]

        logits = dia.model.decoder(tgt_in, dec_state)

        logits = logits.reshape(-1, V)
        tgt_out = tgt_out.reshape(-1)
        loss = criterion(logits, tgt_out)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dia.model.decoder.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item()

    print(f"epoch {epoch}: avg_loss {total_loss / len(loader):.4f}")

torch.save(dia.model.state_dict(), "dia_crying_finetuned.pt")