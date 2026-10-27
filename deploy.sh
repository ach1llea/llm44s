#!/bin/sh
# Copy the runtime onto the phone and (optionally) install a LaunchDaemon.
#
#   ./deploy.sh 192.168.1.23            # files only
#   ./deploy.sh 192.168.1.23 --daemon   # files + autostart on boot
#
# Requires OpenSSH from Cydia on the device (root / alpine by default - change it)
set -e
cd "$(dirname "$0")"

IP=${1:?usage: ./deploy.sh <iphone-ip> [--daemon]}
DAEMON=$2
DIR=/var/root/4sllm

for f in build-ios/llm4s-ios models/minimind2s_q8.sllm models/minimind2s.tok web/index.html; do
  [ -f "$f" ] || { echo "missing $f (run ./build_ios6.sh first)"; exit 1; }
done

echo "[1/3] creating $DIR"
ssh root@"$IP" "mkdir -p $DIR"

echo "[2/3] copying (58MB total, this is the slow part over wifi)"
scp build-ios/llm4s-ios models/minimind2s_q8.sllm models/minimind2s.tok web/index.html \
    root@"$IP":$DIR/
ssh root@"$IP" "chmod +x $DIR/llm4s-ios"

if [ "$DAEMON" = "--daemon" ]; then
  echo "[3/3] installing LaunchDaemon"
  scp deploy/com.4sllm.plist root@"$IP":/Library/LaunchDaemons/com.4sllm.plist
  ssh root@"$IP" "chown root:wheel /Library/LaunchDaemons/com.4sllm.plist; \
                  launchctl unload /Library/LaunchDaemons/com.4sllm.plist 2>/dev/null; \
                  launchctl load -w /Library/LaunchDaemons/com.4sllm.plist"
else
  echo "[3/3] skipped autostart"
fi

cat <<EOF

done. on the phone:

  ssh root@$IP
  cd $DIR
  ./llm4s-ios -m minimind2s_q8.sllm -t minimind2s.tok --serve 8080 --web . --threads 2

then open Safari on the phone:  http://127.0.0.1:8080

before starting the server, check how much RAM is free:
  vm_stat | head -5

to watch it from your mac, tunnel the port instead of exposing it on wifi:
  ssh -L 8080:127.0.0.1:8080 root@$IP
EOF
