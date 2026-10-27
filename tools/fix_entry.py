#!/usr/bin/env python3
"""Give the binary a 32-bit style entry point (LC_UNIXTHREAD).

Two separate things are wrong with what ld-prime emits for armv7:

1. It defaults the entry to `_main`.  On Darwin the entry has to be crt1's
   `start`, which reads argc/argv off the initial stack and calls __libc_init.
   Entering at main() leaves those registers undefined.

2. It uses LC_MAIN.  32-bit Darwin binaries use LC_UNIXTHREAD -- every armv7
   image on an iOS 6 device (Cydia, Activator, ...) has LC_UNIXTHREAD, and
   only their arm64 slice has LC_MAIN.  iOS 6's dyld does not understand
   LC_MAIN, so the process is entered at a garbage address and dies with
   SIGSEGV before it can print anything.

This rewrites the load commands: LC_MAIN is replaced by a full
ARM_THREAD_STATE with pc = `start`.  The load command area grows by 60 bytes,
which is taken out of the padding before the first section (ld always leaves
at least a page of header pad for 32-bit binaries).

usage: fix_entry.py <macho> [--entry start]
"""
import struct
import sys

LC_SEGMENT = 0x1
LC_SYMTAB = 0x2
LC_UNIXTHREAD = 0x5
LC_MAIN = 0x80000028

ARM_THREAD_STATE = 1
ARM_THREAD_STATE_COUNT = 17
THREAD_CMD_SIZE = 16 + ARM_THREAD_STATE_COUNT * 4   # 84
N_ARM_THUMB_DEF = 0x0008


def parse(d):
    ncmds, sizeofcmds = struct.unpack("<II", d[16:24])
    off, cmds = 28, []
    segs, symtab, first_sect_off = {}, None, None
    for i in range(ncmds):
        cmd, cmdsize = struct.unpack("<II", d[off:off + 8])
        cmds.append((i, off, cmd, cmdsize))
        if cmd == LC_SEGMENT:
            name = d[off + 8:off + 24].rstrip(b"\0").decode()
            _, _, fileoff, filesize = struct.unpack("<IIII", d[off + 24:off + 40])
            segs[name] = (fileoff, filesize)
            nsects = struct.unpack("<I", d[off + 48:off + 52])[0]
            for s in range(nsects):
                so = off + 56 + s * 68
                ssize, soff = struct.unpack("<II", d[so + 36:so + 44])
                if ssize and soff and (first_sect_off is None or soff < first_sect_off):
                    first_sect_off = soff
        elif cmd == LC_SYMTAB:
            symtab = struct.unpack("<IIII", d[off + 8:off + 24])
        off += cmdsize
    return ncmds, sizeofcmds, cmds, segs, symtab, first_sect_off


def find_symbol(d, symtab, want):
    symoff, nsyms, stroff, strsize = symtab
    strtab = d[stroff:stroff + strsize]
    for i in range(nsyms):
        e = symoff + i * 12
        strx, ntype, nsect, ndesc, nval = struct.unpack("<IBBHI", d[e:e + 12])
        end = strtab.find(b"\0", strx)
        if strtab[strx:end].decode("utf-8", "replace") == want:
            return nval, ndesc
    return None, 0


def main():
    path = sys.argv[1]
    want = "start"
    if "--entry" in sys.argv:
        want = sys.argv[sys.argv.index("--entry") + 1]

    d = bytearray(open(path, "rb").read())
    if struct.unpack("<I", d[:4])[0] != 0xFEEDFACE:
        print("fix_entry: not a 32-bit Mach-O")
        return 1
    ncmds, sizeofcmds, cmds, segs, symtab, first_sect = parse(d)
    if symtab is None:
        print("fix_entry: no LC_SYMTAB")
        return 1

    value, ndesc = find_symbol(d, symtab, want)
    if value is None:
        print("fix_entry: symbol '%s' not found -- crt1.o missing?" % want)
        return 1
    pc = value | (1 if ndesc & N_ARM_THUMB_DEF else 0)

    main_cmd = next((c for c in cmds if c[2] == LC_MAIN), None)
    if main_cmd is None:
        # already LC_UNIXTHREAD: just point pc at the right place
        ut = next((c for c in cmds if c[2] == LC_UNIXTHREAD), None)
        if ut is None:
            print("fix_entry: neither LC_MAIN nor LC_UNIXTHREAD")
            return 1
        struct.pack_into("<I", d, ut[1] + 16 + 15 * 4, pc)
        open(path, "wb").write(d)
        print("fix_entry: LC_UNIXTHREAD pc -> %#x" % pc)
        return 0

    _, off, _, cmdsize = main_cmd
    grow = THREAD_CMD_SIZE - cmdsize
    end_of_cmds = 28 + sizeofcmds
    slack = (first_sect - end_of_cmds) if first_sect else 0
    if slack < grow:
        print("fix_entry: only %d bytes of header pad, need %d" % (slack, grow))
        return 1

    tail = bytes(d[off + cmdsize:end_of_cmds])
    d[off + THREAD_CMD_SIZE:off + THREAD_CMD_SIZE + len(tail)] = tail

    st = struct.pack("<II", LC_UNIXTHREAD, THREAD_CMD_SIZE)
    st += struct.pack("<II", ARM_THREAD_STATE, ARM_THREAD_STATE_COUNT)
    regs = [0] * ARM_THREAD_STATE_COUNT
    regs[15] = pc                       # r0-r12, sp, lr, pc, cpsr
    st += b"".join(struct.pack("<I", r) for r in regs)
    d[off:off + THREAD_CMD_SIZE] = st

    struct.pack_into("<I", d, 20, sizeofcmds + grow)
    open(path, "wb").write(d)
    print("fix_entry: LC_MAIN -> LC_UNIXTHREAD, pc=%#x (+%d bytes of cmds)" % (pc, grow))
    return 0


if __name__ == "__main__":
    sys.exit(main())
