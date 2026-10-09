#!/bin/bash
# verify_kit.sh [--rope EXL3_MODEL_DIR TP2_SRC] [--bias ORIGINAL_CHECKPOINT_DIR]: check this kit against MANIFEST.
#
# Run from anywhere; it checks the kit folder it lives in (tools/..). Every file MANIFEST lists must be present with
# its sha256. Two kinds of file are generated rather than stored in git, and this script can make them first:
#   --rope EXL3_MODEL_DIR TP2_SRC   the four RoPE tables (rope-*.f32, 136 MB each), recomputed on the CPU by
#                                   tools/make_rope.py with the deepseek-v41-tp2 Python family (TP2_SRC = its src/
#                                   folder) and torch, e.g. inside nvcr.io/nvidia/pytorch:26.07-py3;
#   --bias ORIGINAL_CHECKPOINT_DIR  vision/gate_bias_vl.safetensors, copied by tools/make_bias_vl.py from your own
#                                   copy of deepseek-ai/DeepSeek-V4.1-Flash (standard library only).
# The Hugging Face mirror of this kit ships both, so a download needs neither option.
# Exit 0: every file matches. Exit 1: a file is missing or differs (each one is printed).
set -u
KIT=$(cd "$(dirname "$0")/.." && pwd)
cd "$KIT"
while [ $# -gt 0 ]; do
  case "$1" in
    --rope) PYTHONPATH="$3${PYTHONPATH:+:$PYTHONPATH}" python3 tools/make_rope.py "$2" "$KIT" || exit 1; shift 3 ;;
    --bias) python3 tools/make_bias_vl.py "$2" "$KIT/vision/gate_bias_vl.safetensors" || exit 1; shift 2 ;;
    *) echo "usage: $0 [--rope EXL3_MODEL_DIR TP2_SRC] [--bias ORIGINAL_CHECKPOINT_DIR]"; exit 2 ;;
  esac
done
bad=0; n=0
while read -r sum path; do
  n=$((n + 1))
  if [ ! -f "$path" ]; then echo "MISSING  $path"; bad=$((bad + 1)); continue; fi
  got=$(sha256sum "$path" | cut -d' ' -f1)
  [ "$got" = "$sum" ] || { echo "DIFFERS  $path"; bad=$((bad + 1)); }
done < MANIFEST
if [ $bad = 0 ]; then echo "kit OK: $n files match MANIFEST"; exit 0; fi
echo "kit NOT OK: $bad of $n files missing or different"; exit 1
