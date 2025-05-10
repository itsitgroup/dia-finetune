from dia.model import Dia
import torch

dia = Dia.from_pretrained(
    "nari-labs/Dia-1.6B",
    device="cuda",
    compute_dtype="float32",   # FULL float32 for stable decoding
    load_dac=True
)
# dia.model.load_state_dict(torch.load("dia_crying_finetuned.pt"), strict=False)

txt = "[S1] Hello how are you?"
wav = dia.generate(txt, use_torch_compile=True, verbose=True)
dia.save_audio("crying_demo.wav", wav)