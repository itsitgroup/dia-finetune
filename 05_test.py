from dia.model import Dia
import torch

dia = Dia.from_pretrained("nari-labs/Dia-1.6B", device="cuda",
                          compute_dtype="float16", load_dac=True)
dia.model.load_state_dict(torch.load("dia_crying_finetuned.pt"), strict=False)

txt = "[S1] I’m sorry… (crying) [S2] Please calm down."
wav = dia.generate(txt, temperature=1.2, top_p=0.95, verbose=True)
dia.save_audio("crying_demo.wav", wav)