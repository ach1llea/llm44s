#!/bin/sh
# Generate a minimal armv7 TBD stub for libSystem.
#
# The theos iPhoneOS9.3.sdk ships text-based stubs (TBD v1/v2) that modern ld64
# misreads as "iOS-simulator", so `-lSystem` resolves to nothing and the link
# fails with "symbol(s) not found for architecture armv7". We regenerate just
# the slice we need, declaring the same install-name so the device loads its own
# /usr/lib/libSystem.B.dylib at runtime.
#
#   make_stub_tbd.sh <objdir> <outdir>
set -e
OBJDIR=${1:-build-ios}
OUTDIR=${2:-build-ios/sysroot}

mkdir -p "$OUTDIR/usr/lib"

nm -u "$OBJDIR"/*.o | awk '{print $NF}' | grep '^_' | sort -u > "$OUTDIR/syms.txt"

{
  echo "--- !tapi-tbd"
  echo "tbd-version: 4"
  echo "targets: [ armv7-ios ]"
  echo "install-name: '/usr/lib/libSystem.B.dylib'"
  echo "current-version: 1226.10.1"
  echo "compatibility-version: 1"
  echo "exports:"
  echo "  - targets: [ armv7-ios ]"
  echo "    symbols: [ $(tr '\n' ' ' < "$OUTDIR/syms.txt" | sed 's/ $//') ]"
  echo "..."
} > "$OUTDIR/libsystem.tbd"

cp "$OUTDIR/libsystem.tbd" "$OUTDIR/libSystem.tbd"
cp "$OUTDIR/libsystem.tbd" "$OUTDIR/usr/lib/libSystem.tbd"
cp "$OUTDIR/libsystem.tbd" "$OUTDIR/libSystem.B.tbd"
cp "$OUTDIR/libsystem.tbd" "$OUTDIR/usr/lib/libSystem.B.tbd"

echo "stub: $OUTDIR/libSystem.B.tbd ($(wc -l < "$OUTDIR/syms.txt" | tr -d ' ') symbols)"
