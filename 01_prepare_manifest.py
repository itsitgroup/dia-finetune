import torchaudio, soundfile as sf, pandas as pd, json, pathlib

SRC_ROOT   = pathlib.Path("data/NonverbalVocalization")
DST_WAV    = pathlib.Path("crying/wav_44k");  DST_WAV.mkdir(parents=True, exist_ok=True)
MANIFEST   = pathlib.Path("crying/manifest.csv")

with open(SRC_ROOT / "Nonverbal_Vocalization.json") as f:
    meta = json.load(f)

entries = []
for item in meta:
    if item["label"] != "crying":           # keep only crying class
        continue
    in_wav = SRC_ROOT / item["path"]        # e.g.  WAV/crying/000123.wav
    out_wav = DST_WAV / in_wav.name
    # ---- resample to 44 100 Hz mono ----
    wav, sr = torchaudio.load(in_wav)
    if sr != 44_100:
        wav = torchaudio.functional.resample(wav, sr, 44_100)
    if wav.shape[0] > 1:                    # stereo → mono
        wav = wav.mean(0, keepdim=True)
    sf.write(out_wav, wav.squeeze().numpy(), 44_100)
    # ---- transcript ----
    txt = "[S1] (crying)"
    entries.append({"path": str(out_wav), "text": txt})
pd.DataFrame(entries).to_csv(MANIFEST, index=False)
print(f"Manifest with {len(entries)} crying clips written to {MANIFEST}")