import numpy as np, pandas as pd, pathlib, tqdm, torchaudio
from dia.model import Dia

MANIFEST = pathlib.Path("crying/manifest.csv")
DST_DAC  = pathlib.Path("crying/dac_tokens");  DST_DAC.mkdir(exist_ok=True)

# load Dia only for its DAC
dia = Dia.from_pretrained(
        "nari-labs/Dia-1.6B",
        compute_dtype="float16",
        device="cuda",          # or "cpu" if no GPU
        load_dac=True)

df = pd.read_csv(MANIFEST)

for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    out_npy = DST_DAC / (pathlib.Path(row["path"]).stem + ".npy")
    if out_npy.exists():
        continue

    wav, sr = torchaudio.load(row["path"])   # wav: [1, T]
    wav = wav.to(dia.device)                 # keep channel dim!
    
    tokens = dia._encode(wav)                # [T, C] int16
    np.save(out_npy, tokens.cpu().numpy())
    df.loc[idx, "num_frames"] = tokens.shape[0]

df.to_csv(MANIFEST, index=False)
print("✅  DAC encoding finished.")