#!/usr/bin/env bash
# One-time download of sentence-transformers/all-MiniLM-L6-v2 from Hugging Face.
#
# Pin on first use: the first run records the exact commit and SHA-256 of every file in
# scripts/model.lock, after checking model.safetensors against the SHA-256 that Hugging Face
# publishes for it (the X-Linked-Etag header of its Git LFS object). Later runs, e.g. on a
# new machine, download that same commit and refuse any file whose hash differs.
# memoryd itself never downloads anything.
set -euo pipefail

REPO="sentence-transformers/all-MiniLM-L6-v2"
FILES=(config.json tokenizer.json model.safetensors)
LOCK="$(cd "$(dirname "$0")" && pwd)/model.lock"
DEST="${MEMORYD_HOME:+$MEMORYD_HOME/data}"
DEST="${DEST:-$HOME/.local/share/memoryd}/model"

mkdir -p "$DEST"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if [[ -f "$LOCK" ]]; then
  rev="$(awk '$1 == "revision" {print $2}' "$LOCK")"
else
  rev="$(curl -fsSI "https://huggingface.co/$REPO/resolve/main/config.json" \
    | awk 'tolower($1) == "x-repo-commit:" {print $2}' | tr -d '\r')"
  [[ -n "$rev" ]] || { echo "could not resolve the current commit of $REPO" >&2; exit 1; }
  echo "pinning $REPO at $rev"
fi

for f in "${FILES[@]}"; do
  echo "downloading $f"
  curl -fL --proto '=https' --tlsv1.2 -o "$tmp/$f" "https://huggingface.co/$REPO/resolve/$rev/$f"
done

if [[ -f "$LOCK" ]]; then
  for f in "${FILES[@]}"; do
    want="$(awk -v f="$f" '$1 == f {print $2}' "$LOCK")"
    got="$(shasum -a 256 "$tmp/$f" | cut -d' ' -f1)"
    [[ "$want" == "$got" ]] || { echo "SHA-256 mismatch for $f: expected $want, got $got" >&2; exit 1; }
  done
  echo "hashes match scripts/model.lock"
else
  lfs_sha="$(curl -fsSI "https://huggingface.co/$REPO/resolve/$rev/model.safetensors" \
    | awk 'tolower($1) == "x-linked-etag:" {print $2}' | tr -d '\r"')"
  got="$(shasum -a 256 "$tmp/model.safetensors" | cut -d' ' -f1)"
  [[ -n "$lfs_sha" && "$lfs_sha" == "$got" ]] \
    || { echo "model.safetensors SHA-256 $got does not match Hugging Face's published ${lfs_sha:-<missing>}" >&2; exit 1; }
  echo "model.safetensors matches Hugging Face's published SHA-256"
  {
    echo "revision $rev"
    for f in "${FILES[@]}"; do echo "$f $(shasum -a 256 "$tmp/$f" | cut -d' ' -f1)"; done
  } > "$LOCK"
  echo "wrote $LOCK (commit it)"
fi

mv "$tmp"/* "$DEST"/
echo "model installed in $DEST"
