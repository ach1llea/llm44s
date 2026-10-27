#!/usr/bin/env python3
"""numpy float32 reference forward pass, used to validate the int8 C engine."""
import json
import struct
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from export_model import read_safetensors

SYSTEM = "You are a helpful assistant"


def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def ref_logits(sd, cfg, ids):
    dim = cfg["hidden_size"]
    nh = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    L = cfg["num_hidden_layers"]
    eps = cfg["rms_norm_eps"]
    theta = cfg["rope_theta"]

    half = hd // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / hd))
    n = len(ids)
    pos = np.arange(n)[:, None]
    freqs = pos * inv[None, :]
    cos = np.cos(freqs).astype(np.float32)      # (n, half)
    sin = np.sin(freqs).astype(np.float32)

    def rope(v):                                 # v: (n, nh, hd)
        a = v[..., :half]
        b = v[..., half:]
        return np.concatenate([a * cos[:, None, :] - b * sin[:, None, :],
                               b * cos[:, None, :] + a * sin[:, None, :]], axis=-1)

    x = sd["model.embed_tokens.weight"][ids].astype(np.float32)   # (n, dim)
    for l in range(L):
        p = f"model.layers.{l}."
        h = rmsnorm(x, sd[p + "input_layernorm.weight"], eps)
        q = (h @ sd[p + "self_attn.q_proj.weight"].T).reshape(n, nh, hd)
        k = (h @ sd[p + "self_attn.k_proj.weight"].T).reshape(n, nkv, hd)
        v = (h @ sd[p + "self_attn.v_proj.weight"].T).reshape(n, nkv, hd)
        q, k = rope(q), rope(k)
        rep = nh // nkv
        kk = np.repeat(k, rep, axis=1)
        vv = np.repeat(v, rep, axis=1)
        att = np.zeros((n, nh, hd), dtype=np.float32)
        for i in range(n):
            s = (kk[: i + 1] * q[i:i + 1]).sum(-1) / np.sqrt(hd)   # (i+1, nh)
            s = np.exp(s - s.max(0, keepdims=True))
            s = s / s.sum(0, keepdims=True)
            att[i] = (s[..., None] * vv[: i + 1]).sum(0)
        x = x + att.reshape(n, dim) @ sd[p + "self_attn.o_proj.weight"].T
        h = rmsnorm(x, sd[p + "post_attention_layernorm.weight"], eps)
        g = h @ sd[p + "mlp.gate_proj.weight"].T
        u = h @ sd[p + "mlp.up_proj.weight"].T
        act = g / (1.0 + np.exp(-g))
        x = x + (act * u) @ sd[p + "mlp.down_proj.weight"].T
    x = rmsnorm(x, sd["model.norm.weight"], eps)
    return x[-1] @ sd["model.embed_tokens.weight"].T


def main():
    mdir = Path(sys.argv[1])
    cpath = sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else "你好"

    cfg = json.loads((mdir / "config.json").read_text())
    sd = read_safetensors(mdir / "model.safetensors")
    tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))

    text = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n")
    ids = tok.encode(text).ids
    print(f"prompt tokens: {len(ids)}")

    ref = ref_logits(sd, cfg, ids)
    got = np.fromfile(cpath, dtype="<f4")
    print(f"logits: ref {ref.shape} got {got.shape}")
    assert ref.shape == got.shape

    r_ref = np.argsort(-ref)[:5]
    r_got = np.argsort(-got)[:5]
    print("ref top5:", [(tok.decode([int(i)]), float(ref[i])) for i in r_ref])
    print("got top5:", [(tok.decode([int(i)]), float(got[i])) for i in r_got])

    num = (ref * got).sum()
    den = np.linalg.norm(ref) * np.linalg.norm(got)
    print(f"cosine similarity : {num/den:.5f}")
    print(f"max |diff|        : {np.abs(ref-got).max():.4f}")
    print(f"mean |diff|       : {np.abs(ref-got).mean():.4f}")
    same_top1 = int(r_ref[0]) == int(r_got[0])
    print(f"top-1 match       : {same_top1}")


if __name__ == "__main__":
    main()
