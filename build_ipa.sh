#!/bin/sh
# Build LLM44S.ipa - a self-contained iOS 6 / armv7 app with both models inside.
#
#   ./build_ipa.sh
#
# Same toolchain caveats as build_ios6.sh (TBD stubs misread as "iOS Simulator",
# missing _memcpy, missing liblaunch). The bundle is assembled in /tmp because
# this project lives on an exFAT volume that does not keep POSIX modes: a binary
# shipped as 0700 is not executable for the 'mobile' user the app runs as.
set -e
cd "$(dirname "$0")"
export COPYFILE_DISABLE=1

ROOT=$(pwd)
SDK="${IOS_SDK:-$HOME/.ios-sdks/iPhoneOS9.3.sdk}"
[ -d "$SDK" ] || { echo "SDK not found: $SDK  (see build_ios6.sh header)"; exit 1; }

CC=${CC:-clang}
OBJ=$ROOT/build-ipa/obj
WORK=$(mktemp -d /tmp/llm4s-ipa.XXXXXX)
APP=$WORK/Payload/LLM44S.app
trap 'rm -rf "$WORK"' EXIT

CRT=$(ls /Library/Developer/CommandLineTools/usr/lib/clang/*/lib/darwin/libclang_rt.ios.a 2>/dev/null | head -1)
[ -n "$CRT" ] || CRT=$(ls /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/*/lib/darwin/libclang_rt.ios.a 2>/dev/null | head -1)
[ -n "$CRT" ] || { echo "libclang_rt.ios.a not found"; exit 1; }

rm -rf "$ROOT/build-ipa"
mkdir -p "$OBJ" "$APP"

CFLAGS="-target armv7-apple-ios6.0 -isysroot $SDK -miphoneos-version-min=6.0 -mfpu=neon \
        -O3 -fno-math-errno -ffunction-sections -fdata-sections \
        -Wall -Wno-unused-parameter -DNDEBUG"

echo "[1/6] C engine"
for f in src/*.c; do
  b=$(basename "$f" .c)
  case $b in
    compat_ios) $CC $CFLAGS -fno-builtin-memcpy -fno-builtin-memset -fno-builtin-memmove \
                    -std=gnu99 -c "$f" -o "$OBJ/$b.o" ;;
    *)          $CC $CFLAGS -std=gnu99 -c "$f" -o "$OBJ/$b.o" ;;
  esac
done

echo "[2/6] Objective-C UI"
for f in app/*.m; do
  b=$(basename "$f" .m)
  $CC $CFLAGS -fobjc-arc -fblocks -Isrc -Iapp -c "$f" -o "$OBJ/$b.o"
done

echo "[3/6] linking"
# iOS 6 has no libsystem_malloc / libsystem_pthread (split out in iOS 7); their
# stubs are rewritten to point at the umbrella libSystem.B.dylib.
SYSLIB=$OBJ/syslib
python3 tools/patch_stub_libs.py "$SDK" "$SYSLIB"
p="$SDK/usr/lib/system/liblaunch.tbd"
if [ ! -f "$p" ]; then
  printf -- "--- !tapi-tbd\ntbd-version: 4\ntargets: [ armv7-ios ]\ninstall-name: '/usr/lib/system/liblaunch.dylib'\ncurrent-version: 1\ncompatibility-version: 1\nexports:\n  - targets: [ armv7-ios ]\n    symbols: [ ]\n...\n" > "$p" || true
fi

ld -arch armv7 -platform_version ios-simulator 6.0 6.0 -syslibroot "$SDK" \
   -L"$SYSLIB" -L"$SDK/usr/lib/system" -L"$SDK/usr/lib" -F"$SDK/System/Library/Frameworks" \
   -o "$APP/LLM44S" "$SDK/usr/lib/crt1.o" "$OBJ"/*.o \
   -framework UIKit -framework Foundation -framework CoreGraphics -framework QuartzCore \
   -lobjc -lsystem_c -lsystem_kernel -lsystem_malloc -lsystem_m -lsystem_pthread \
   "$CRT" -lSystem \
   2>&1 | grep -v "building for iOS Simulator" || true
[ -f "$APP/LLM44S" ] || { echo "link failed"; exit 1; }
python3 tools/fix_entry.py "$APP/LLM44S" || exit 1

echo "[4/6] bundling"
cp app/Info.plist                 "$APP/Info.plist"
cp app/Icon.png app/Icon@2x.png   "$APP/"
cp models/minimind2s_q8.sllm      "$APP/"
cp models/minimind2_i8.sllm       "$APP/"
cp models/minimind2.tok           "$APP/tokenizer.tok"
printf 'APPL????' > "$APP/PkgInfo"
chmod -R 755 "$APP"

echo "[5/6] signing"
if command -v ldid >/dev/null 2>&1; then
  cat > "$WORK/ent.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>application-identifier</key><string>com.achillea.llm44s</string>
  <key>com.apple.developer.team-identifier</key><string>LLM44S</string>
  <key>get-task-allow</key><true/>
  <key>keychain-access-groups</key><array><string>com.achillea.llm44s</string></array>
</dict></plist>
EOF
  ldid -S"$WORK/ent.plist" "$APP/LLM44S"
  echo "      ldid signed + entitlements"
else
  echo "      ldid missing - bundle left unsigned (needs AppSync to install)"
fi

echo "[6/6] packing"
# zip *updates* an existing archive: entries that are no longer on disk stay in
# it, so a model dropped from the bundle would still be shipped.  Start clean.
rm -f "$ROOT/LLM44S.ipa"
( cd "$WORK" && zip -qryX "$ROOT/LLM44S.ipa" Payload )

echo
file "$APP/LLM44S"
otool -l "$APP/LLM44S" | grep -A3 LC_VERSION_MIN_IPHONEOS | head -3
echo "  links:"
otool -L "$APP/LLM44S" | tail -n +2 | sed 's|^|    |' | cut -c1-72
echo
ls -lh "$ROOT/LLM44S.ipa" | awk '{print "ipa: "$9"  "$5}'
echo "install: ./install_ipa.sh <iphone-ip>"
