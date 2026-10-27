#!/usr/bin/env python3
"""Rewrite install names of SDK stubs for libraries that do not exist on iOS 6.

The iPhoneOS9.3 SDK splits libc into libsystem_c / _kernel / _m / _malloc /
_pthread / _compiler_rt ...  iOS 6.1.3's dyld shared cache only contains

    libsystem_blocks _c _dnssd _info _kernel _m _network _notify _sandbox

so anything else recorded in LC_LOAD_DYLIB makes dyld abort the launch with
"Library not loaded ... image not found" -- before a single instruction runs.
That is exactly how the first IPA failed.

Rather than dropping the symbols, we keep the SDK's stub (so linking still
resolves everything) and only rewrite its `install-name:` to the umbrella
`/usr/lib/libSystem.B.dylib`, which every iOS release has and which carries
all of those symbols.  The rewritten stubs go into a directory that is put
first on the linker search path.

usage: patch_stub_libs.py <sdk-root> <out-dir>
"""
import os
import re
import sys

ALLOWED = {"blocks", "c", "dnssd", "info", "kernel", "m",
           "network", "notify", "sandbox"}
UMBRELLA = "/usr/lib/libSystem.B.dylib"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    sdk, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)

    patched = []
    for d in ("usr/lib", "usr/lib/system"):
        src_dir = os.path.join(sdk, d)
        if not os.path.isdir(src_dir):
            continue
        for fn in sorted(os.listdir(src_dir)):
            if not fn.endswith(".tbd"):
                continue
            m = re.match(r"libsystem_([a-z0-9_]+)\.tbd$", fn)
            if not m:
                continue
            if m.group(1) in ALLOWED:
                continue
            path = os.path.join(src_dir, fn)
            text = open(path, "r", errors="replace").read()
            new, n = re.subn(r"(?m)^install-name:.*$",
                             "install-name: '%s'" % UMBRELLA, text, count=1)
            if n == 0:
                continue
            with open(os.path.join(outdir, fn), "w") as f:
                f.write(new)
            patched.append(fn)

    if not patched:
        print("patch_stub_libs: nothing to patch")
        return 0
    print("patch_stub_libs: %s -> %s" % (", ".join(p.replace('.tbd', '') for p in patched), UMBRELLA))
    return 0


if __name__ == "__main__":
    sys.exit(main())
