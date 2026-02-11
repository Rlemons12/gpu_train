from __future__ import annotations

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_dir", required=True)
    p.add_argument("--prompt", default="Instruction: say hello\nResponse:")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    ).to(device)
    model.eval()

    enc = tok(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=40,
            do_sample=False,
        )

    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
