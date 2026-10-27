#!/usr/bin/env python3
"""Dequantize a .sllm file with numpy and run the reference forward pass.

This isolates two very different failure modes that look identical from the
outside:

  * the C kernel unpacks/quantizes wrongly  -> cosine vs .sllm < 0.999
  * int4 is simply lossy                    -> cosine vs fp16 < 1.0 but the
                                               .sllm reference matches

usage: verify_sllm.py <model_dir> <file.sllm> ["prompt"]
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from export_model import read_safetensors

T_TOK_EMB = 0
(L_ATTN_NORM, L_WQ, L_WK, L_WV, L_WO, L_FFN_NORM, L_W1, L_W3, L_W2) = range(9)
N_PER_LAYER = 9
SYSTEM = "You are a helpful assistant"


def parse(path):
    d = Path(path).read_bytes()
    assert d[:4] == b"SLLM", "bad magic"
    ver, dim, L, H, KV, HD, FFN, V, SEQ = struct.unpack("<9i", d[4:40])
    eps, theta = struct.unpack("<2f", d[40:48])
    nt, quant, group = struct.unpack("<III", d[48:60])
    cfg = dict(dim=dim, L=L, H=H, KV=KV, HD=HD, FFN=FFN, V=V, SEQ=SEQ,
               eps=eps, theta=theta, quant=quant, group=group)
    tensors = {}
    for i in range(nt):
        off = 68 + i * 36
        tid, ndim, s0, s1, s2, s3, nbytes, doff, soff = struct.unpack(
            "<ii4iIII", d[off:off + 36])
        tensors[tid] = dict(ndim=ndim, shape=(s0, s1, s2, s3)[:ndim],
                            nbytes=nbytes, off=doff, soff=soff, data=d)
    return cfg, tensors


def bf16_to_f32(h):
    return (h.astype(np.uint32) << 16).view(np.float32)


def dequant(cfg, t):
    d = t["data"]
    rows, cols = t["shape"][0], (t["shape"][1] if t["ndim"] > 1 else 1)
    if t["soff"] == 0:          # norms are stored as raw fp32 (no scale block)
        return np.frombuffer(d[t["off"]:t["off"] + t["nbytes"]],
                             dtype="<f4").reshape(rows, cols).copy()

    if cfg["quant"] == 2:
        g = cfg["group"]
        ng = cols // g
        sb = np.frombuffer(d[t["soff"]:t["soff"] + rows * ng * 2], dtype="<u2")
        sc = bf16_to_f32(sb).reshape(rows, ng)
        raw = np.frombuffer(d[t["off"]:t["off"] + rows * cols // 2], dtype=np.uint8)
        p = raw.reshape(rows, ng * (g // 32), 16)
        lo = p & 0x0F
        hi = p >> 4
        lo = np.where(lo >= 8, lo.astype(np.int16) - 16, lo).astype(np.int8)
        hi = np.where(hi >= 8, hi.astype(np.int16) - 16, hi).astype(np.int8)
        vals = np.concatenate([lo, hi], axis=-1).reshape(rows, ng, g)
        return (vals.astype(np.float32) * sc[:, :, None]).reshape(rows, cols)

    sf = np.frombuffer(d[t["soff"]:t["soff"] + rows * 4], dtype="<f4")
    w = np.frombuffer(d[t["off"]:t["off"] + rows * cols], dtype=np.int8)
    return (w.reshape(rows, cols).astype(np.float32) * sf[:, None])


def to_state_dict(cfg, tensors):
    sd = {}
    sd["model.embed_tokens.weight"] = dequant(cfg, tensors[T_TOK_EMB])
    for l in range(cfg["L"]):
        p = f"model.layers.{l}."
        b = 1 + l * N_PER_LAYER
        a = p + "self_attn."
        m = p + "mlp."
        sd[p + "input_layernorm.weight"] = dequant(cfg, tensors[b + L_ATTN_NORM]).ravel()
        sd[a + "q_proj.weight"] = dequant(cfg, tensors[b + L_WQ])
        sd[a + "k_proj.weight"] = dequant(cfg, tensors[b + L_WK])
        sd[a + "v_proj.weight"] = dequant(cfg, tensors[b + L_WV])
        sd[a + "o_proj.weight"] = dequant(cfg, tensors[b + L_WO])
        sd[p + "post_attention_layernorm.weight"] = dequant(cfg, tensors[b + L_FFN_NORM]).ravel()
        sd[m + "gate_proj.weight"] = dequant(cfg, tensors[b + L_W1])
        sd[m + "up_proj.weight"] = dequant(cfg, tensors[b + L_W3])
        sd[m + "down_proj.weight"] = dequant(cfg, tensors[b + L_W2])
    sd["model.norm.weight"] = dequant(cfg, tensors[1 + cfg["L"] * N_PER_LAYER]).ravel()
    return sd


def ref_logits(sd, cfg, ids):
    dim, nh, nkv, hd = cfg["dim"], cfg["H"], cfg["KV"], cfg["HD"]
    eps, theta, L = cfg["eps"], cfg["theta"], cfg["L"]
    half = hd // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / hd))
    n = len(ids)
    freqs = np.arange(n)[:, None] * inv[None, :]
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)

    def rope(v):
        a, b = v[..., :half], v[..., half:]
        return np.concatenate([a * cos[:, None, :] - b * sin[:, None, :],
                               b * cos[:, None, :] + a * sin[:, None, :]], axis=-1)

    x = sd["model.embed_tokens.weight"][ids].astype(np.float32)
    for l in range(L):
        p = f"model.layers.{l}."
        h = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * sd[p + "input_layernorm.weight"]
        q = (h @ sd[p + "self_attn.q_proj.weight"].T).reshape(n, nh, hd)
        k = (h @ sd[p + "self_attn.k_proj.weight"].T).reshape(n, nkv, hd)
        v = (h @ sd[p + "self_attn.v_proj.weight"].T).reshape(n, nkv, hd)
        q, k = rope(q), rope(k)
        rep = nh // nkv
        kk, vv = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
        att = np.zeros((n, nh, hd), dtype=np.float32)
        for i in range(n):
            s = (kk[:i + 1] * q[i:i + 1]).sum(-1) / np.sqrt(hd)
            s = np.exp(s - s.max(0, keepdims=True))
            s = s / s.sum(0, keepdims=True)
            att[i] = (s[..., None] * vv[:i + 1]).sum(0)
        x = x + att.reshape(n, dim) @ sd[p + "self_attn.o_proj.weight"].T
        h = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * sd[p + "post_attention_layernorm.weight"]
        g = h @ sd[p + "mlp.gate_proj.weight"].T
        u = h @ sd[p + "mlp.up_proj.weight"].T
        x = x + (g / (1.0 + np.exp(-g)) * u) @ sd[p + "mlp.down_proj.weight"].T
    x = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * sd["model.norm.weight"]
    return x[-1] @ sd["model.embed_tokens.weight"].T


def sim(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


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

    print(f"sllm: quant={cfg['quant']} group={cfg['group']} "
          f"dim={cfg['dim']} L={cfg['L']} hd={cfg['HD']} ctx={cfg['SEQ']}")
    q = to_state_dict(cfg, tensors)
    ref_fp32 = ref_logits(read_safetensors(mdir / "model.safetensors"),
                          dict(dim=hf["hidden_size"], H=hf["num_attention_heads"],
                               KV=hf["num_key_value_heads"], HD=hf["head_dim"],
                               eps=hf["rms_norm_eps"], theta=hf["rope_theta"],
                               L=hf["num_hidden_layers"]), ids)
    ref_q = ref_logits(q, cfg, ids)

    print(f"cosine  dequant-vs-fp32 : {sim(ref_q, ref_fp32):.5f}   "
          f"(pure quantization loss)")
    print(f"mean|d| dequant-vs-fp32 : {np.abs(ref_q - ref_fp32).mean():.4f}")

    print("top5 fp32 :", [tok.decode([int(i)]) for i in np.argsort(-ref_fp32)[:5]])
    print("top5 quant:", [tok.decode([int(i)]) for i in np.argsort(-ref_q)[:5]])


if __name__ == "__main__":
    main()
