#!/usr/bin/env python3
"""Greedy chat with the numpy float32 reference (no quantization at all).

    python tools/ref_chat.py models/minimind2 "你好" [max_new]

This is the ground truth the .sllm exports are supposed to approximate: run the
same prompt through fp32, int8 and int4 and diff the text to see whether a bad
answer is a quantization artefact or the model itself.
"""
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from export_model import read_safetensors
from ref_forward import ref_logits

SYSTEM = "You are a helpful assistant"


def main():
    mdir = Path(sys.argv[1])
    prompt = sys.argv[2] if len(sys.argv) > 2 else "你好"
    max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    system = sys.argv[4] if len(sys.argv) > 4 else SYSTEM

    cfg = json.loads((mdir / "config.json").read_text())
    sd = read_safetensors(mdir / "model.safetensors")
    tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))

    text = (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n")
    ids = tok.encode(text).ids
    eos = {2, 0}

    out = []
    for step in range(max_new):
        logits = ref_logits(sd, cfg, ids)
        nxt = int(np.argmax(logits))
        if nxt in eos:
            print(f"[eos after {step}]")
            break
        out.append(nxt)
        ids.append(nxt)
        print(tok.decode([nxt]), end="", flush=True)
    print()
    print("ids:", out)


if __name__ == "__main__":
    main()
