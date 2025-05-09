import numpy as np, pandas as pd, torch, pathlib, tqdm
import torchaudio
from dia.model import Dia

MANIFEST   = pathlib.Path("crying/manifest.csv")
DST_DAC    = pathlib.Path("crying/dac_tokens"); DST_DAC.mkdir(exist_ok=True)

dia = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16",
                          device="cuda", load_dac=True)   # we only need dia.dac_model
dac = dia.dac_model          # Descript Audio Codec

df = pd.read_csv(MANIFEST)
for idx,row in tqdm.tqdm(df.iterrows(), total=len(df)):
    out_npy = DST_DAC / (pathlib.Path(row["path"]).stem + ".npy")
    if out_npy.exists(): continue
    wav, _ = torchaudio.load(row["path"])
    wav = wav.to(dia.device)
    tokens = dia._encode(wav.squeeze(0))    # [T, C] tensor of int
    np.save(out_npy, tokens.cpu().numpy())
    df.loc[idx,"num_frames"] = tokens.shape[0]
df.to_csv(MANIFEST, index=False)