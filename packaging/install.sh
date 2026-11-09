#!/bin/sh
# Installs the memoryd LaunchAgent so the server starts at login, then checks it is up.
set -eu

label=com.g-varona.memoryd
plist="$HOME/Library/LaunchAgents/$label.plist"

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs/memoryd"
sed "s|HOME_DIR|$HOME|g" "$(dirname "$0")/$label.plist" > "$plist"
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$plist"
sleep 1
curl -fsS http://127.0.0.1:7788/health && echo
