#!/bin/sh
# Build for the host machine (Apple Silicon / Intel macOS) - used for
# development and for generating the model files.
#
#   ./build_mac.sh
set -e
cd "$(dirname "$0")"
CC=${CC:-clang}
mkdir -p build-mac
for f in src/*.c; do
  b=$(basename "$f" .c)
  case $b in
    compat_ios) continue ;;   # iOS-only memcpy/strcmp shims
  esac
  $CC -O2 -std=gnu99 -Wall -Wextra -Wno-unused-parameter -c "$f" -o "build-mac/$b.o"
done
$CC -O2 -o llm4s build-mac/*.o -lpthread -lm
echo "built: llm4s"
