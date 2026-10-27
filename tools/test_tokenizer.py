#!/usr/bin/env python3
"""Compare the C tokenizer against the HF tokenizers library."""
import subprocess
import sys
from pathlib import Path

from tokenizers import Tokenizer

CASES = [
    "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "你好",
    "你好，世界！",
    "今天天气怎么样？",
    "我是一名人工智能助手，很高兴见到你。",
    "1+1=2, 3.14, 2024年",
    "iPhone 4S 上跑大模型",
    "line1 line2",
    "   leading spaces",
    "emoji 😀 test",
    "a",
    "《三体》是刘慈欣写的科幻小说。",
    "用三句话介绍深圳。",
    "def foo(x): return x*2",
    "价格是¥199.00",
    "ＡＢＣ全角",
]


def main():
    tokpath = sys.argv[1] if len(sys.argv) > 1 else "models/minimind2s/tokenizer.json"
    exe = sys.argv[2] if len(sys.argv) > 2 else "./llm4s"
    tok = Tokenizer.from_file(tokpath)

    tmp = Path("/tmp/tokcases.txt")
    tmp.write_text("\n".join(CASES) + "\n")

    out = subprocess.run([exe, "-m", "x", "-t", "models/minimind2s.tok",
                          "--tok-test", str(tmp)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr)
        sys.exit(1)

    lines = [l for l in out.stdout.splitlines() if l]
    bad = 0
    for case, line in zip(CASES, lines):
        text, ids = line.split("\t")
        got = [int(x) for x in ids.split(",")] if ids else []
        want = tok.encode(case, add_special_tokens=False).ids
        ok = got == want
        if not ok:
            bad += 1
            print(f"MISMATCH {case!r}")
            print(f"  want {want}")
            print(f"  got  {got}")
            print(f"  want_txt {[tok.decode([i]) for i in want]}")
            print(f"  got_txt  {[tok.decode([i]) for i in got]}")
    print(f"\n{len(CASES)-bad}/{len(CASES)} cases matched")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
