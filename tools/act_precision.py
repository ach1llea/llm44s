#!/usr/bin/env python3
"""How much accuracy does each activation-quantization scheme cost?

The C engine quantizes activations before every matvec.  With int8 per-vector
scaling (the current scheme) a single outlier in a 768-wide vector squashes
everything else into a couple of levels, which is fine for int8 weights but
apparently fatal for int4 ones.  This script replays the forward pass with
several schemes so we can pick one that is fast *and* accurate.

usage: act_precision.py <model_dir> <file.sllm> ["prompt"]
"""
import sys
from pathlib import Path

import numpy as np

from export_model import read_safetensors
from verify_sllm import parse, to_state_dict, sim
from tokenizers import Tokenizer
import json

SYSTEM = "You are a helpful assistant"


def q_per_vector(x, bits):
    """symmetric, one scale per lane (what the C code does today)"""
    hi = (1 << (bits - 1)) - 1
    amax = np.abs(x).max(-1, keepdims=True)
    sc = np.maximum(amax / hi, 1e-8)
    return np.clip(np.rint(x / sc), -hi, hi) * sc


def q_per_group(x, bits, g):
    """symmetric, one scale per `g` contiguous elements"""
    hi = (1 << (bits - 1)) - 1
    n = x.shape[-1]
    pad = (-n) % g
    xx = np.pad(x, ((0, 0), (0, pad))) if pad else x
    v = xx.reshape(xx.shape[0], -1, g)
    amax = np.abs(v).max(-1, keepdims=True)
    sc = np.maximum(amax / hi, 1e-8)
    q = np.clip(np.rint(v / sc), -hi, hi) * sc
    return q.reshape(xx.shape)[:, :n]


def fwd(sd, cfg, ids, act):
    dim, nh, nkv, hd = cfg["dim"], cfg["H"], cfg["KV"], cfg["HD"]
    eps, theta, L = cfg["eps"], cfg["theta"], cfg["L"]
    half = hd // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / hd))
    n = len(ids)
    freqs = np.arange(n)[:, None] * inv[None, :]
    cos, sin = np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)

    def rope(v):
        a, b = v[..., :half], v[..., half:]
        return np.concatenate([a * cos[:, None, :] - b * sin[:, None, :],
                               b * cos[:, None, :] + a * sin[:, None, :]], axis=-1)

    def mm(h, w):
        return act(h) @ w.T

    x = sd["model.embed_tokens.weight"][ids].astype(np.float32)
    for l in range(L):
        p = f"model.layers.{l}."
        h = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * sd[p + "input_layernorm.weight"]
        q = mm(h, sd[p + "self_attn.q_proj.weight"]).reshape(n, nh, hd)
        k = mm(h, sd[p + "self_attn.k_proj.weight"]).reshape(n, nkv, hd)
        v = mm(h, sd[p + "self_attn.v_proj.weight"]).reshape(n, nkv, hd)
        q, k = rope(q), rope(k)
        rep = nh // nkv
        kk, vv = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
        att = np.zeros((n, nh, hd), dtype=np.float32)
        for i in range(n):
            s = (kk[:i + 1] * q[i:i + 1]).sum(-1) / np.sqrt(hd)
            s = np.exp(s - s.max(0, keepdims=True))
            s = s / s.sum(0, keepdims=True)
            att[i] = (s[..., None] * vv[:i + 1]).sum(0)
        x = x + mm(att.reshape(n, dim), sd[p + "self_attn.o_proj.weight"])
        h = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * sd[p + "post_attention_layernorm.weight"]
        g = mm(h, sd[p + "mlp.gate_proj.weight"])
        u = mm(h, sd[p + "mlp.up_proj.weight"])
        x = x + mm(g / (1.0 + np.exp(-g)) * u, sd[p + "mlp.down_proj.weight"])
    x = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * sd["model.norm.weight"]
    return x[-1] @ sd["model.embed_tokens.weight"].T


SCHEMES = [
    ("fp32 activations        ", lambda h: h),
    ("int16 per-vector        ", lambda h: q_per_vector(h, 16)),
    ("int8  per-vector        ", lambda h: q_per_vector(h, 8)),
    ("int8  per-group(32)     ", lambda h: q_per_group(h, 8, 32)),
    ("int8  per-group(16)     ", lambda h: q_per_group(h, 8, 16)),
    ("int16 per-group(32)     ", lambda h: q_per_group(h, 16, 32)),
]


def main():
    mdir = Path(sys.argv[1])
    spath = sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else "你好"

    cfg, tensors = parse(spath)
    hf = json.loads((mdir / "config.json").read_text())
    tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))
    ids = tok.encode(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                     f"<|im_start|>user\n{prompt}<|im_end|>\n"
                     f"<|im_start|>assistant\n").ids

    sd_exact = read_safetensors(mdir / "model.safetensors")
    c = dict(dim=hf["hidden_size"], H=hf["num_attention_heads"],
             KV=hf["num_key_value_heads"], HD=hf["head_dim"],
             eps=hf["rms_norm_eps"], theta=hf["rope_theta"], L=hf["num_hidden_layers"])
    print(f"file: quant={cfg['quant']} group={cfg['group']}\n")
    base = fwd(sd_exact, c, ids, lambda h: h)
    btop = tok.decode([int(np.argmax(base))])
    print(f"{'scheme':26s} {'cosine':>8s}  {'top1':<10s}")
    print("-" * 52)
    print(f"{'fp32 (ground truth)':26s} {1.0:8.4f}  {btop}")
    for name, act in SCHEMES:
        lg = fwd(sd_exact, c, ids, act)
        t = tok.decode([int(np.argmax(lg))]).replace("\n", "\\n")
        print(f"{name:26s} {sim(base, lg):8.4f}  {t!r}")


if __name__ == "__main__":
    main()
