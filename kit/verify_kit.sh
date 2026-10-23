#!/bin/bash
# verify_kit.sh [--kit DIR] [--remake] MODEL [ORIGINAL_DIR]: checks the kernel kit against its MANIFEST, after making
# the files git does not hold. No GPU.
#   MODEL         the EXL3 checkpoint's folder, or just its config.json: tools/make_rope.py computes the four RoPE
#                 tables from it on the CPU (PyTorch; without a local torch it runs with docker in IMAGE, default
#                 nvcr.io/nvidia/pytorch:26.07-py3, the image whose PyTorch made the served tables)
#   ORIGINAL_DIR  DeepSeek's original checkpoint (deepseek-ai/DeepSeek-V4.1-Flash-1017): tools/make_bias_vl.py copies
#                 vision/gate_bias_vl.safetensors out of it. Without it that file must already be here (the kit/ folder
#                 on Hugging Face has it).
#   --kit DIR     the kit folder (default: this script's folder)
#   --remake      make the RoPE tables even when they are present
# Every file MANIFEST lists must be here with its size and sha256: exit 0 when all are, 1 otherwise.
set -uo pipefail
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
KIT=$HERE; REMAKE=0; ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --kit) KIT=$(cd "${2:?}" && pwd); shift 2 ;;
    --remake) REMAKE=1; shift ;;
    -h|--help) usage ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[ ${#ARGS[@]} -ge 1 ] && [ ${#ARGS[@]} -le 2 ] || usage
MODEL=${ARGS[0]}; ORIGINAL=${ARGS[1]:-}
IMAGE=${IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}
PYTHON=${PYTHON:-python3}
TOOLS=$HERE/tools; [ -f "$TOOLS/make_rope.py" ] || TOOLS=$HERE
[ -f "$KIT/MANIFEST" ] || { echo "no MANIFEST in $KIT" >&2; exit 1; }
CONFIG=$MODEL; [ -d "$MODEL" ] && CONFIG=$MODEL/config.json
[ -f "$CONFIG" ] || { echo "no $CONFIG: MODEL is the EXL3 checkpoint's folder or its config.json" >&2; exit 1; }
CONFIG=$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")
if command -v sha256sum >/dev/null; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

# the four RoPE tables: made here unless present (--remake: always)
need_rope=$REMAKE
for t in plain-cos plain-sin compressed-cos compressed-sin; do [ -f "$KIT/rope-$t.f32" ] || need_rope=1; done
if [ $need_rope = 1 ]; then
  echo "making the RoPE tables (tools/make_rope.py, CPU)" >&2
  if "$PYTHON" -c 'import torch' 2>/dev/null; then
    "$PYTHON" "$TOOLS/make_rope.py" "$CONFIG" "$KIT" >&2
  else
    docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$CONFIG:/model/config.json:ro" \
      -v "$KIT:/kit" -v "$TOOLS:/tools:ro" --entrypoint python3 "$IMAGE" /tools/make_rope.py /model/config.json /kit \
      >&2
  fi
fi
# the image bias: copied out of the original checkpoint when given
if [ -n "$ORIGINAL" ]; then
  echo "making vision/gate_bias_vl.safetensors (tools/make_bias_vl.py)" >&2
  mkdir -p "$KIT/vision"
  "$PYTHON" "$TOOLS/make_bias_vl.py" "$ORIGINAL" "$KIT/vision/gate_bias_vl.safetensors" >&2
fi

ok=0; bad=0; missing=0; git=0; rope=0; bias=0
while read -r sha bytes origin _ path; do
  case $sha in \#*|"") continue ;; esac
  f=$KIT/$path
  if [ ! -f "$f" ]; then
    missing=$((missing + 1)); echo "missing ($origin): $path"
    continue
  fi
  if [ "$(size "$f")" != "$bytes" ] || [ "$($SHA "$f" | cut -c1-64)" != "$sha" ]; then
    bad=$((bad + 1)); echo "DIFFERS ($origin): $path"
    continue
  fi
  ok=$((ok + 1))
  case $origin in git) git=$((git + 1)) ;; rope) rope=$((rope + 1)) ;; bias) bias=$((bias + 1)) ;; esac
done < "$KIT/MANIFEST"
total=$((ok + bad + missing))
echo "verify_kit: $ok of $total files match MANIFEST (git $git, rope $rope, bias $bias); $bad differ, $missing missing"
if [ $missing -gt 0 ] && [ ! -f "$KIT/vision/gate_bias_vl.safetensors" ]; then
  echo "vision/gate_bias_vl.safetensors: pass ORIGINAL_DIR (DeepSeek's original checkpoint)," \
    "or take the file from the kit/ folder on Hugging Face" >&2
fi
[ $bad = 0 ] && [ $missing = 0 ]
