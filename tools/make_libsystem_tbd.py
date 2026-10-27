#!/usr/bin/env python3
"""Merge the SDK's per-library TBD stubs into a single libSystem.B.tbd.

Why: the iPhoneOS9.3 SDK splits libc into libsystem_c / _kernel / _m /
_malloc / _pthread.  Two of those do not exist on iOS 6 -- the dyld shared
cache of 6.1.3 only contains

    libsystem_blocks _c _dnssd _info _kernel _m _network _notify _sandbox

so a binary that records `LC_LOAD_DYLIB /usr/lib/system/libsystem_malloc.dylib`
dies in dyld with "image not found" before a single instruction runs.

On iOS 6 every one of those symbols is reachable through the umbrella
`/usr/lib/libSystem.B.dylib`, so we link against one synthetic stub that
carries the union of the symbol names and declares that install name.
The resulting binary has exactly one libc dependency, and it is one that
exists on every iOS release.

usage: make_libsystem_tbd.py <sdk-root> <out-dir>
"""
import os
import re
import sys

LIBS = ["libsystem_c", "libsystem_kernel", "libsystem_m",
        "libsystem_malloc", "libsystem_pthread", "libcompiler_rt",
        "libgcc_s.1", "libSystem"]
TARGETS = "armv7-ios, armv7s-ios"


def collect(path):
    """Pull every symbol token out of the `symbols: [ ... ]` blocks."""
    out = set()
    try:
        text = open(path, "r", errors="replace").read()
    except OSError:
        return out
    for m in re.finditer(r"symbols:\s*\[(.*?)\]", text, re.S):
        for tok in re.split(r"[\s,]+", m.group(1)):
            if tok.startswith("_") and not tok.startswith("_$"):
                out.add(tok)
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    sdk, outdir = sys.argv[1], sys.argv[2]
    syms = set()
    for lib in LIBS:
        for d in ("usr/lib", "usr/lib/system"):
            syms |= collect(os.path.join(sdk, d, lib + ".tbd"))
    if not syms:
        print("make_libsystem_tbd: no symbols found under %s" % sdk)
        return 1

    body = []
    body.append("--- !tapi-tbd")
    body.append("tbd-version: 4")
    body.append("targets: [ %s ]" % TARGETS)
    body.append("install-name: '/usr/lib/libSystem.B.dylib'")
    body.append("current-version: 1226.10.1")
    body.append("compatibility-version: 1.0.0")
    body.append("exports:")
    ordered = sorted(syms)
    for i in range(0, len(ordered), 200):
        chunk = ordered[i:i + 200]
        body.append("  - targets: [ %s ]" % TARGETS)
        body.append("    symbols: [ %s ]" % " ".join(chunk))
    body.append("...")
    text = "\n".join(body) + "\n"

    os.makedirs(outdir, exist_ok=True)
    # ld resolves `-lSystem` to libSystem.tbd / libSystem.dylib / libSystem.B.tbd
    # depending on version; ship identical copies under all three names.
    for name in ("libSystem.B.tbd", "libSystem.tbd"):
        with open(os.path.join(outdir, name), "w") as f:
            f.write(text)
    print("make_libsystem_tbd: %d symbols -> %s/libSystem.B.tbd" % (len(syms), outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
