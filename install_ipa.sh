#!/bin/sh
# Install LLM44S.ipa on a jailbroken device over ssh.
#
#   ./install_ipa.sh 192.168.x.x
#
# Tries, in order:
#   1. ipainstaller  (Cydia: "IPA Installer Console")
#   2. unzip into /Applications + uicache (always works on a jailbroken 6.x)
set -e
cd "$(dirname "$0")"
IP=${1:?usage: ./install_ipa.sh <iphone-ip>}
[ -f LLM44S.ipa ] || { echo "LLM44S.ipa not found (run ./build_ipa.sh)"; exit 1; }

echo "[1/2] uploading $(ls -lh LLM44S.ipa | awk '{print $5}') over wifi"
scp LLM44S.ipa root@"$IP":/var/root/LLM44S.ipa

echo "[2/2] installing"
ssh root@"$IP" '
set -e
cd /var/root
if command -v ipainstaller >/dev/null 2>&1; then
    echo "  using ipainstaller"
    ipainstaller LLM44S.ipa && echo "  installed" && exit 0
fi
echo "  falling back to /Applications copy"
rm -rf /tmp/ipax /Applications/LLM44S.app /Applications/LLM4S.app /Applications/Ember.app
mkdir -p /tmp/ipax
unzip -oq LLM44S.ipa -d /tmp/ipax
cp -R /tmp/ipax/Payload/LLM44S.app /Applications/LLM44S.app
chown -R root:wheel /Applications/LLM44S.app
chmod -R a+rX /Applications/LLM44S.app
chmod 755 /Applications/LLM44S.app/LLM44S
rm -rf /tmp/ipax
if command -v uicache >/dev/null 2>&1; then su mobile -c uicache; fi
killall SpringBoard 2>/dev/null || true
echo "  installed"
'

cat <<EOF

done. the "LLM 4 4s" icon should be on the springboard now.

if it launches into a black screen or crashes immediately, pull the log:
  ssh root@$IP "find /var/mobile/Applications -maxdepth 3 -name '*.log'"
  ssh root@$IP "find /var/mobile/Applications -maxdepth 3 -name crash.log -exec cat {} \\;"
EOF
