import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from dia.model import Dia
from dataset import CryingDataset        # same dir

device = "cuda"
dia = Dia.from_pretrained("nari-labs/Dia-1.6B",
                          compute_dtype="float16", device=device,
                          load_dac=False)              # we train, DAC not needed

# ---- freeze everything except decoder & its embeddings ----
for p in dia.model.parameters(): p.requires_grad_(False)
for p in dia.model.decoder.parameters(): p.requires_grad_(True)

ds = CryingDataset("crying/manifest.csv", dia.config)
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

criterion = nn.CrossEntropyLoss(ignore_index=dia.config.data.audio_pad_value)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, dia.model.parameters()),
                        lr=3e-5, weight_decay=1e-2)

EPOCHS = 3
for epoch in range(EPOCHS):
    dia.model.train()
    for txt, aud, T in loader:
        txt, aud = txt.to(device), aud.to(device)
        # encoder forward
        enc_state = dia.model.encoder(txt, dia.model.encoder.layers[0].pre_sa_norm)  # quick stub
        # decoder teacher forcing: shift right (BOS already added)
        tgt_in  = aud[:,:-1]          # teacher input
        tgt_out = aud[:,1:]           # prediction target
        dec_state = dia.model.decoder.precompute_cross_attn_cache(
                        enc_state, torch.arange(txt.shape[-1], device=device), None)
        logits = dia.model.decoder(tgt_in, dec_state)  # (B, T, C*V)
        logits = logits.view(-1, dia.config.model.tgt_vocab_size)
        loss   = criterion(logits, tgt_out.view(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    print(f"epoch {epoch}: loss {loss.item():.4f}")
torch.save(dia.model.state_dict(), "dia_crying_finetuned.pt")