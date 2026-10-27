#!/bin/sh
# Cross-compile llm4s for iPhone 4S (armv7, iOS 6.0 deployment target).
#
#   ./build_ios6.sh
#
# Toolchain notes (this is the fiddly part, documented so it is reproducible):
#
#  * theos/sdks ships iPhoneOS9.3.sdk, the oldest redistributable SDK. Apple
#    never published a 6.1 SDK, but 9.3 is fine: we only call POSIX symbols that
#    exist on iOS 6 and set -miphoneos-version-min=6.0.
#  * that SDK contains TBD v1/v2 text stubs. Modern ld64 (ld-prime) reads their
#    `platform: ios` as "iOS Simulator", so re-exported symbols do not resolve.
#    Passing `-platform_version ios-simulator` satisfies the parser; the emitted
#    load command is still LC_VERSION_MIN_IPHONEOS 6.0, i.e. a device binary.
#  * the stubs never declare memcpy/memset/strcmp/... at all, so those are
#    provided by src/compat_ios.c (internal definitions, no dylib needed).
#  * libSystem re-exports /usr/lib/system/liblaunch.dylib which the SDK omits;
#    an empty stub is synthesised next to it (the only write into the SDK).
set -e
cd "$(dirname "$0")"

SRC_SDK="${IOS_SDK:-$HOME/.ios-sdks/iPhoneOS9.3.sdk}"
if [ ! -d "$SRC_SDK" ]; then
  cat <<EOF
SDK not found: $SRC_SDK

  git clone --filter=blob:none --no-checkout --depth 1 https://github.com/theos/sdks.git ~/.ios-sdks
  cd ~/.ios-sdks && git sparse-checkout init --cone
  git sparse-checkout set iPhoneOS9.3.sdk && git checkout
EOF
  exit 1
fi

CC=${CC:-clang}
BUILD=build-ios
SDK=$SRC_SDK
CRT=$(ls /Library/Developer/CommandLineTools/usr/lib/clang/*/lib/darwin/libclang_rt.ios.a 2>/dev/null | head -1)
if [ -z "$CRT" ]; then
  CRT=$(ls /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/*/lib/darwin/libclang_rt.ios.a 2>/dev/null | head -1)
fi
if [ -z "$CRT" ]; then echo "libclang_rt.ios.a not found (install Xcode or the Command Line Tools)"; exit 1; fi

mkdir -p "$BUILD"

# the only file we ever add to the SDK: an empty stub for liblaunch.dylib,
# which libSystem re-exports but theos/sdks does not ship
# the only file we ever add to the SDK: an empty stub for liblaunch.dylib,
# which libSystem re-exports but theos/sdks does not ship
stub() {
  p="$SDK${1%.dylib}.tbd"
  [ -f "$p" ] && return 0
  mkdir -p "$(dirname "$p")" || return 0 || return 0
  printf -- "--- !tapi-tbd\ntbd-version: 4\ntargets: [ armv7-ios, armv7s-ios, arm64-ios ]\ninstall-name: '%s'\ncurrent-version: 1\ncompatibility-version: 1\nexports:\n  - targets: [ armv7-ios, armv7s-ios, arm64-ios ]\n    symbols: [ ]\n...\n" "$1" > "$p"
}
stub /usr/lib/system/liblaunch.dylib

CFLAGS="-target armv7-apple-ios6.0 -isysroot $SDK -miphoneos-version-min=6.0 -mfpu=neon \
        -O3 -fno-math-errno -ffunction-sections -fdata-sections \
        -Wall -Wno-unused-parameter -DNDEBUG -std=gnu99"

echo "[1/5] compiling armv7 objects"
rm -f "$BUILD"/*.o "$BUILD/llm4s-ios"
for f in src/*.c; do
  b=$(basename "$f" .c)
  case $b in
    compat_ios)
      $CC $CFLAGS -fno-builtin-memcpy -fno-builtin-memset -fno-builtin-memmove -c "$f" -o "$BUILD/$b.o" ;;
    *)
      $CC $CFLAGS -c "$f" -o "$BUILD/$b.o" ;;
  esac
done

echo "[2/5] resolving system dylibs (rewriting ones iOS 6 does not have)"
SYSLIB=$BUILD/syslib
# libsystem_malloc / libsystem_pthread only exist from iOS 7 on; a binary that
# names them dies in dyld with "image not found".  Their stubs are copied with
# install-name rewritten to the umbrella libSystem.B.dylib and put first on the
# search path, so linking still resolves every symbol but the load command
# points at a library iOS 6 actually has.
python3 tools/patch_stub_libs.py "$SDK" "$SYSLIB"
LIBS=""
for sym in $(nm -u "$BUILD"/*.o | awk '{print $NF}' | sort -u); do
  case "$sym" in ___*) continue;; esac
  f=$(grep -lw -- "$sym" $SYSLIB/*.tbd $SDK/usr/lib/*.tbd $SDK/usr/lib/system/*.tbd 2>/dev/null \
      | grep -v "libSystem\.tbd\|libSystem\.B\.tbd" | head -1)
  [ -z "$f" ] && continue
  lib=$(basename "$f" .tbd | sed 's/^lib//')
  case " $LIBS " in *" -l$lib "*) ;; *) LIBS="$LIBS -l$lib" ;; esac
done
echo "     $LIBS"

echo "[3/5] linking"
# shellcheck disable=SC2086
ld -arch armv7 -platform_version ios-simulator 6.0 6.0 -syslibroot "$SDK" \
   -L"$SYSLIB" -L"$SDK/usr/lib/system" -L"$SDK/usr/lib" \
   -o "$BUILD/llm4s-ios" "$SDK/usr/lib/crt1.o" "$BUILD"/*.o $LIBS "$CRT" -lSystem \
   2>&1 | grep -v "building for iOS Simulator" || true
if [ ! -f "$BUILD/llm4s-ios" ]; then echo "link failed"; exit 1; fi
# ld defaults the entry to _main; without crt1's `start` the process would run
# main() with argc/argv uninitialised and die instantly.
python3 tools/fix_entry.py "$BUILD/llm4s-ios" || exit 1

echo "[4/5] signing"
if command -v ldid >/dev/null 2>&1; then
  ldid -S "$BUILD/llm4s-ios" && echo "      ldid -S ok"
else
  echo "      ldid not found: binary left unsigned."
  echo "      Unsigned binaries run on a jailbroken iOS 6 device with AppSync;"
  echo "      if AMFI refuses it, 'brew install ldid' and re-run."
fi

echo "[5/5] verifying"
file "$BUILD/llm4s-ios"
otool -l "$BUILD/llm4s-ios" | grep -A3 LC_VERSION_MIN_IPHONEOS | head -4
echo "  dependencies:"
otool -L "$BUILD/llm4s-ios" | tail -n +2 | sed 's/^/    /'
echo
echo "built: $BUILD/llm4s-ios  ($(wc -c < "$BUILD/llm4s-ios" | tr -d ' ') bytes)"
echo "next : ./deploy.sh <iphone-ip>"
