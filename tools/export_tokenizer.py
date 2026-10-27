#!/usr/bin/env python3
"""Convert a HF `tokenizer.json` (GPT-2 byte-level BPE) into tok.bin.

tok.bin layout (little-endian):
    magic "STOK" | u32 version | i32 n_vocab | i32 n_merges | i32 n_special | pad
    i32  byte_id[256]            raw byte -> token id (the 256 base tokens)
    u32  merge_keys[n_merges]    (a << 13) | b
    u32  merge_vals[n_merges]    resulting token id
    i32  special_ids[eos, im_start, im_end]
    vocab: for id in 0..n_vocab-1: u16 len + raw bytes

Tokens are stored as *raw* bytes (the ByteLevel unicode mapping inverted), so the
C side never needs the bytes<->unicode table: merges are keyed by token id and the
256 base tokens are single raw bytes.
"""
import json
import struct
import sys
from pathlib import Path

MAGIC = b"STOK"
VERSION = 1


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def main():
    if len(sys.argv) < 3:
        print("usage: export_tokenizer.py <tokenizer.json> <out tok.bin>")
        sys.exit(1)
    tj = json.loads(Path(sys.argv[1]).read_text())
    outp = Path(sys.argv[2])

    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}

    model = tj["model"]
    vocab_map = model["vocab"]                     # token string -> id
    n_vocab = max(vocab_map.values()) + 1

    def to_raw(s):
        try:
            return bytes(u2b[c] for c in s)
        except KeyError:
            return s.encode("utf-8")

    raw_of_id = [b""] * n_vocab
    for s, i in vocab_map.items():
        raw_of_id[i] = to_raw(s)

    # base byte tokens
    byte_id = [-1] * 256
    for s, i in vocab_map.items():
        r = to_raw(s)
        if len(r) == 1:
            byte_id[r[0]] = i
    missing = [b for b in range(256) if byte_id[b] < 0]
    if missing:
        raise SystemExit(f"base byte tokens missing for {missing[:10]}...")

    merges = model["merges"]
    mk, mv = [], []
    for m in merges:
        if isinstance(m, str):
            left, right = m.split(" ", 1)
        else:
            left, right = m[0], m[1]
        a = vocab_map[left]
        b = vocab_map[right]
        merged = vocab_map.get(left + right)
        if merged is None:
            print(f"  skip merge without target: {left!r}+{right!r}")
            continue
        if a > 8191 or b > 8191:
            raise SystemExit("token id exceeds 13 bits, packing would break")
        mk.append((a << 13) | b)
        mv.append(merged)

    added = {a["content"]: a["id"] for a in tj.get("added_tokens", [])}
    specials = [added.get("<|endoftext|>", 0),
                added.get("<|im_start|>", 1),
                added.get("<|im_end|>", 2)]

    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<I", VERSION)
    buf += struct.pack("<3i", n_vocab, len(mk), len(specials))
    buf += struct.pack("<i", 0)
    buf += struct.pack("<256i", *byte_id)
    buf += struct.pack(f"<{len(mk)}I", *mk)
    buf += struct.pack(f"<{len(mv)}I", *mv)
    buf += struct.pack(f"<{len(specials)}i", *specials)
    for i in range(n_vocab):
        r = raw_of_id[i]
        if len(r) > 65535:
            raise SystemExit("token too long")
        buf += struct.pack("<H", len(r)) + r

    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_bytes(buf)
    print(f"wrote {outp} : {len(buf)/1024:.1f} KB, vocab={n_vocab}, "
          f"merges={len(mk)}, specials={specials}")


if __name__ == "__main__":
    main()
