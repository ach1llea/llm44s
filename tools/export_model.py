#!/usr/bin/env python3
"""Export a HF safetensors Llama-family checkpoint into the .sllm format.

No torch / safetensors dependency: the safetensors container is parsed by hand
(8-byte LE header length + JSON header + raw buffers) and numpy handles the math.

.sllm layout (all little-endian, every tensor 64-byte aligned):

  Header  : magic "SLLM" | version | dim | n_layers | n_heads | n_kv_heads |
            head_dim | hidden_dim | vocab_size | max_seq | rms_eps |
            rope_theta | n_tensors | quant | pad[3]
  Dir     : n_tensors × TEntry(36B): id, ndim, shape[4], nbytes, off, soff
  Blobs   : int8 tensor  = f32 scales[rows] then int8 data[rows*cols]
            int4 tensor  = bf16 scales[rows*(cols/group)] then packed nibbles
            f32 tensor   = f32 data[n]
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"SLLM"
VERSION = 1

# tensor ids ---------------------------------------------------------------
T_TOK_EMB = 0
L_ATTN_NORM, L_WQ, L_WK, L_WV, L_WO, L_FFN_NORM, L_W1, L_W3, L_W2 = range(9)
N_PER_LAYER = 9


def tensor_id_final_norm(n_layers):
    return 1 + n_layers * N_PER_LAYER


def read_safetensors(path):
    """Return {name: np.ndarray(float32)} without torch/safetensors."""
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        dt = info["dtype"]
        shape = info["shape"]
        np_dt = {"F16": np.float16, "F32": np.float32, "BF16": None}[dt]
        f.seek(base + start)
        raw = f.read(end - start)
        if dt == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            arr = u16.view(np.float32)
        else:
            arr = np.frombuffer(raw, dtype=np_dt).astype(np.float32)
        out[name] = arr.reshape(shape)
    f.close()
    return out


def quantize_rows(w):
    """int8 per-row (per output channel) symmetric quantization."""
    rows, cols = w.shape
    amax = np.abs(w).max(axis=1, keepdims=True)
    amax = np.where(amax < 1e-8, 1e-8, amax)
    scale = (amax / 127.0).astype(np.float32)          # (rows,1)
    q = np.rint(w / scale).clip(-127, 127).astype(np.int8)
    return q, scale.reshape(rows).astype(np.float32)


def f32_to_bf16(a):
    """round-to-nearest-even truncation of fp32 to the upper 16 bits."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return r.astype(np.uint16)


def quantize_q4(w, group=64):
    """int4 group-wise symmetric quantization.

    Each group of `group` values shares one bf16 scale.  Values are packed two
    per byte: within every 32-value sub-block, element i goes in the low nibble
    of byte i and element 16+i in the high nibble.  Nibbles are signed two's
    complement, so the range is -8..7.
    """
    rows, cols = w.shape
    if cols % group != 0:
        raise SystemExit(f"q4: cols={cols} is not a multiple of group={group}")
    ng = cols // group
    x = w.reshape(rows, ng, group)
    amax = np.abs(x).max(axis=2, keepdims=True)
    amax = np.where(amax < 1e-8, 1e-8, amax)
    scale = (amax / 7.0).astype(np.float32)            # (rows,ng,1)
    q = np.rint(x / scale).clip(-8, 7).astype(np.int8)

    sub = q.reshape(rows, ng * (group // 32), 32)
    lo = (sub[:, :, :16].astype(np.uint8) & 0x0F)
    hi = (sub[:, :, 16:].astype(np.uint8) & 0x0F) << 4
    packed = (lo | hi).reshape(rows, cols // 2)
    return packed, f32_to_bf16(scale.reshape(rows, ng))


def main():
    if len(sys.argv) < 3:
        print("usage: export_model.py <model_dir> <out.sllm> [--seq N] [--q4 [GROUP]]")
        sys.exit(1)
    mdir = Path(sys.argv[1])
    outp = Path(sys.argv[2])
    seq_cap = 256
    if "--seq" in sys.argv:
        seq_cap = int(sys.argv[sys.argv.index("--seq") + 1])

    q4_group = 0
    if "--q4" in sys.argv:
        i = sys.argv.index("--q4")
        q4_group = 64
        if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
            q4_group = int(sys.argv[i + 1])

    # Which parts stay at int8 when int4 is enabled.  Attention is only ~23% of
    # the parameters but carries ~46% of the accuracy loss, so the default mix
    # keeps it (and the tied embedding / output head) at 8 bits.
    mix_i8 = set()
    if "--mix" in sys.argv:
        i = sys.argv.index("--mix")
        val = "attn,emb"
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
            val = sys.argv[i + 1]
        mix_i8 = {x.strip() for x in val.split(",") if x.strip()}
    elif q4_group:
        mix_i8 = {"attn", "emb"}

    cfg = json.loads((mdir / "config.json").read_text())
    dim = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads") or n_heads
    head_dim = cfg.get("head_dim") or dim // n_heads
    hidden = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    eps = cfg.get("rms_norm_eps", 1e-5)
    theta = cfg.get("rope_theta", 1e6)
    max_seq = min(int(cfg.get("max_position_embeddings", 32768)), seq_cap)

    print(f"config: dim={dim} L={n_layers} H={n_heads} KV={n_kv} hd={head_dim} "
          f"ffn={hidden} vocab={vocab} eps={eps} theta={theta}")

    sd = read_safetensors(mdir / "model.safetensors")

    # tensors to emit: (id, name, quantized?)
    spec = []
    # `kind`: None = fp32, 'i8' = per-row int8, 'i4' = grouped int4.
    # The layout is decided per tensor, so attention can stay at int8 (it is
    # only ~23% of the parameters but ~46% of the accuracy) while the
    # parameter-heavy MLP drops to int4.
    def kind_for(part):
        if not q4_group:
            return True, "i8"
        return True, ("i8" if part in mix_i8 else "i4")

    spec.append((T_TOK_EMB, "model.embed_tokens.weight", kind_for("emb")[1]))
    for l in range(n_layers):
        p = f"model.layers.{l}."
        a = p + "self_attn."
        m = p + "mlp."
        # gate/up are one "part" because their quantisation error largely cancels
        # through silu(g)*u; down_proj writes straight back into the residual
        # stream, so it gets its own switch and is usually worth keeping at int8.
        spec += [
            (1 + l * N_PER_LAYER + L_ATTN_NORM, p + "input_layernorm.weight", None),
            (1 + l * N_PER_LAYER + L_WQ, a + "q_proj.weight", kind_for("attn")[1]),
            (1 + l * N_PER_LAYER + L_WK, a + "k_proj.weight", kind_for("attn")[1]),
            (1 + l * N_PER_LAYER + L_WV, a + "v_proj.weight", kind_for("attn")[1]),
            (1 + l * N_PER_LAYER + L_WO, a + "o_proj.weight", kind_for("attn")[1]),
            (1 + l * N_PER_LAYER + L_FFN_NORM, p + "post_attention_layernorm.weight", None),
            (1 + l * N_PER_LAYER + L_W1, m + "gate_proj.weight", kind_for("up")[1]),
            (1 + l * N_PER_LAYER + L_W3, m + "up_proj.weight", kind_for("up")[1]),
            (1 + l * N_PER_LAYER + L_W2, m + "down_proj.weight",
             kind_for("down" if "down" in mix_i8 else "mlp")[1]),
        ]
    spec.append((tensor_id_final_norm(n_layers), "model.norm.weight", None))

    # lm_head is tied to the embedding in this checkpoint: verify, don't duplicate
    if "lm_head.weight" in sd:
        same = np.array_equal(sd["lm_head.weight"], sd["model.embed_tokens.weight"])
        print(f"lm_head present, tied={same}")
        if not same:
            print("WARNING: untied lm_head detected; .sllm v1 assumes tied embeddings")

    blobs = []          # (id, ndim, shape, nbytes, data_bytes, scale_bytes)
    n_i8 = n_i4 = 0
    for tid, name, kind in spec:
        w = sd[name].astype(np.float32)
        if kind == "i4":
            rows, cols = w.shape
            q, sc = quantize_q4(w, q4_group)
            blobs.append((tid, 2, (rows, cols), q.nbytes, q.tobytes(), sc.tobytes()))
            n_i4 += 1
        elif kind == "i8":
            rows, cols = w.shape
            q, sc = quantize_rows(w)
            blobs.append((tid, 2, (rows, cols), q.nbytes, q.tobytes(), sc.tobytes()))
            n_i8 += 1
        else:
            blobs.append((tid, 1, (w.size,), w.nbytes, w.tobytes(), b""))
    print(f"quant layout: {n_i8} int8 tensors, {n_i4} int4 tensors")

    # ---- lay out file: two passes so offsets and padding agree exactly ----
    def align64(x):
        return (x + 63) // 64 * 64

    hdr_len = 68 + len(blobs) * 36        # fixed header + tensor directory
    entries, placed, off = [], [], align64(hdr_len)
    for tid, ndim, shape, nbytes, data, scales in blobs:
        off = align64(off)
        soff = 0
        if scales:
            soff = off
            off += len(scales)
            off = align64(off)
        data_off = off
        off += len(data)
        shp = list(shape) + [0] * (4 - len(shape))
        entries.append((tid, ndim, shp, nbytes, data_off, soff))
        placed.append((scales, data_off, soff))

    header = bytearray()
    header += MAGIC
    header += struct.pack("<I", VERSION)
    header += struct.pack("<8i", dim, n_layers, n_heads, n_kv, head_dim, hidden, vocab, max_seq)
    header += struct.pack("<2f", eps, theta)
    header += struct.pack("<I", len(blobs))
    # 1 = int8, 2 = int4 present (a mixed file has both; the engine tells them
    # apart per tensor from nbytes: rows*cols is int8, rows*cols/2 is int4).
    header += struct.pack("<I", 2 if q4_group else 1)
    header += struct.pack("<I", q4_group)               # 0 when not int4
    header += b"\x00" * 8
    for tid, ndim, shp, nbytes, doff, soff in entries:
        header += struct.pack("<ii4iIII", tid, ndim, *shp, nbytes, doff, soff)
    assert len(header) == hdr_len, (len(header), hdr_len)

    buf = bytearray(off)
    buf[0:len(header)] = header

    # write blobs (index-parallel to `placed`)
    for idx, (scales, data_off, soff) in enumerate(placed):
        data = blobs[idx][4]
        if scales:
            buf[soff:soff + len(scales)] = scales
        buf[data_off:data_off + len(data)] = data

    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "wb") as f:
        f.write(buf)

    n_params = sum(int(np.prod(b[2])) for b in blobs)
    print(f"wrote {outp} : {outp.stat().st_size/1e6:.2f} MB, "
          f"{len(blobs)} tensors, {n_params/1e6:.2f}M params")


if __name__ == "__main__":
    main()
