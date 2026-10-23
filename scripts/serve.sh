#!/bin/bash
# The served DeepSeek-V4.1-Flash-1017 launch on two GB10 nodes (README "v0.7.0"), our served launch with the site
# values moved to config.env. Rank 1 starts `tf-dsv41-lanes` following rank 0, then rank 0 starts `tensorfold-native
# serve`; each runs in a container tp2-lane on its node (scripts/node_run.sh, fed over ssh, or run directly on the node
# that runs this script: no ssh key to itself needed).
#   scripts/serve.sh start      preflight, stop any tp2-lane, start the two ranks, wait for /v1/models, then check
#   scripts/serve.sh preflight  exit 0 when every file rank 0 opens at start is there and both nodes' RDMA devices have
#                               a RoCE v2 address at GID index 5 (RUNNING.md section 2), else name each miss; start
#                               runs it first and starts nothing when one is missing
#   scripts/serve.sh stop       remove tp2-lane on the two nodes
#   scripts/serve.sh check      exit 0 when rank 0 lists the model and answers a 4-token chat
#   scripts/serve.sh print      print what start runs on each node, run nothing
#   scripts/serve.sh inputs     make each node's weight file and token map in CACHE_DIR (ENGINE_DIR/node_inputs.sh on
#                               the two nodes at once; RUNNING.md section 2); the launch itself never runs it
set -u
here=$(cd "$(dirname "$0")" && pwd)
CFG=${DSV41_CONFIG:-$here/../config.env}
[ -r "$CFG" ] || { echo "no $CFG: copy config.env.example to config.env and fill it in" >&2; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
if grep -v '^[[:space:]]*#' "$CFG" | grep -q '<[A-Z0-9_]*>'; then echo "$CFG still holds a <PLACEHOLDER>" >&2; exit 1; fi
read -ra N <<< "$NODES"
[ ${#N[@]} = 2 ] || { echo "NODES must name two nodes, rank 0 first" >&2; exit 1; }
U=http://127.0.0.1:$PORT
PRINT=0
SITE="IMAGE=$IMAGE ENGINE_DIR=$ENGINE_DIR KIT_DIR=$KIT_DIR MODEL_DIR=$MODEL_DIR ENGRAM_DIR=$ENGRAM_DIR CACHE_DIR=$CACHE_DIR LOG_DIR=$LOG_DIR"
# Both ranks: the shape (rank 1's flags below say the same; the ranks' settings check stops a launch whose ranks
# differ), the served switches, and the link. The RDMA ring opens GID index 5 of each device; NCCL uses the same one.
SHAPE="TF_DS_STREAMS=$STREAMS TF_DS_ROUND_ROWS=$ROUND_ROWS TF_DS_WINDOW=$WINDOW $ENGINE_ENV"
LINK="TF_RDMA_DEVICES=$RDMA_HCA TF_DS_RDMA_KERNELS=/engine/rdma_gather.fatbin NCCL_IB_HCA=$RDMA_HCA NCCL_SOCKET_IFNAME=$NET_IF NCCL_IB_GID_INDEX=5"

# This host's names and addresses: a node in NODES that is one of them runs here, not over ssh to itself.
SELF=" localhost 127.0.0.1 ::1 $(hostname 2>/dev/null) $(hostname -s 2>/dev/null) $(hostname -f 2>/dev/null) $(hostname -I 2>/dev/null) "
is_local() {   # is_local NODE: NODE is this host (by name, by the name its ssh config gives it, or by its address)
  local real ip n
  real=$(ssh -G "$1" 2>/dev/null | awk '$1 == "hostname" { print $2; exit }')
  for n in "$1" "${real:-$1}"; do
    case "$SELF" in *" $n "*) return 0 ;; esac
    ip=$(getent ahosts "$n" 2>/dev/null | awk '{ print $1; exit }')
    [ -n "$ip" ] && case "$SELF" in *" $ip "*) return 0 ;; esac
  done
  return 1
}
via() {   # via NODE CMD: CMD on NODE (here when NODE is this host, else over ssh), stdin passed through
  if is_local "$1"; then bash -c "$2"; else ssh -o BatchMode=yes "$1" "$2"; fi
}
on() { via "$1" "$2" </dev/null; }
rank() {   # rank NODE ENVS CMD: CMD in tp2-lane on NODE with ENVS (VAR=value words without spaces) in its environment
  if [ $PRINT = 1 ]; then echo "$1: env $2 $SITE $LINK node_run.sh tp2-lane '$3'"; return 0; fi
  via "$1" "env $2 $SITE $LINK bash -s -- tp2-lane '$3'" < "$here/node_run.sh" >/dev/null
}
# Every file rank 0 opens at start (RUNNING.md section 2): the model folder's small files, each file the index maps a
# vision tower tensor to (model-00041-of-00041.safetensors in the download), the node's weight file, the token map, the
# kit's kernel set and the engine; and on both nodes, a RoCE v2 address at GID index 5 of each RDMA device. Checked
# before anything starts or stops.
preflight() {
  local miss h g bad=0
  miss=$(via "${N[0]}" "bash -s -- '$MODEL_DIR' '$CACHE_DIR' '$KIT_DIR' '$ENGINE_DIR' '$ENGRAM_DIR'" <<'EOF'
m=$1
for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja model.safetensors.index.json; do
  [ -s "$m/$f" ] || echo "MODEL_DIR/$f"
done
if [ -s "$m/model.safetensors.index.json" ]; then
  v=$(grep -oE '"(vision|aligner|image_(start|newline|end))[^"]*": *"[^"]+"' "$m/model.safetensors.index.json" |
      sed 's/.*: *"//; s/"$//' | sort -u)
  [ -n "$v" ] || echo "MODEL_DIR/model.safetensors.index.json names no vision tower file"
  for f in $v; do [ -s "$m/$f" ] || echo "MODEL_DIR/$f (the vision tower; rank 0 reads it at start)"; done
fi
ls "$2"/rank-cache/rank0of2-*.bin >/dev/null 2>&1 || echo "CACHE_DIR/rank-cache/rank0of2-<hash>.bin (scripts/serve.sh inputs)"
[ -s "$2/dsv41_token_map.json" ] || echo "CACHE_DIR/dsv41_token_map.json (scripts/serve.sh inputs)"
[ -s "$3/aot/aot.json" ] || echo "KIT_DIR/aot/aot.json"
for f in tensorfold-native rdma_gather.fatbin; do [ -s "$4/$f" ] || echo "ENGINE_DIR/$f"; done
[ -d "$5" ] || echo "ENGRAM_DIR $5"
EOF
  ) || { echo "preflight: could not reach rank 0's node ${N[0]}" >&2; return 1; }
  if [ -n "$miss" ]; then
    echo "preflight: rank 0 (${N[0]}) is missing these files, nothing was started or stopped:" >&2
    echo "$miss" | sed 's/^/  /' >&2
    bad=1
  fi
  for h in "${N[@]}"; do
    g=$(via "$h" "bash -s -- '$RDMA_HCA'" <<'EOF'
for d in ${1//,/ }; do
  p=/sys/class/infiniband/$d/ports/1
  t=$(cat "$p/gid_attrs/types/5" 2>/dev/null); a=$(cat "$p/gids/5" 2>/dev/null)
  if [ "$t" != "RoCE v2" ] || [ -z "$a" ] || [ "$a" = 0000:0000:0000:0000:0000:0000:0000:0000 ]; then
    echo "$d: GID index 5 is ${t:-not there}${a:+ ($a)}"
  fi
done
EOF
    ) || { echo "preflight: could not reach $h" >&2; return 1; }
    if [ -n "$g" ]; then
      echo "preflight: $h has no RoCE v2 address at GID index 5, which the engine's ring opens; nothing was started or stopped:" >&2
      echo "$g" | sed 's/^/  /' >&2
      bad=1
    fi
  done
  return $bad
}
check() {
  on "${N[0]}" "curl -sf -m 10 $U/v1/models | grep -q $SERVED_NAME && curl -sf -m 120 -H 'Content-Type: application/json' $U/v1/chat/completions -d '{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":4,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}' | grep -q choices"
}
stop() {
  local h
  for h in "${N[@]}"; do on "$h" "docker rm -f tp2-lane >/dev/null 2>&1; true"; done
}
inputs() {
  local g rc=0 pids=()
  for g in 0 1; do
    ( set -o pipefail; on "${N[$g]}" "IMAGE=$IMAGE $ENGINE_DIR/node_inputs.sh $MODEL_DIR $CACHE_DIR $g 2" 2>&1 | sed "s/^/rank $g: /" ) &
    pids+=($!)
  done
  for g in 0 1; do wait "${pids[$g]}" || { echo "rank $g: node_inputs.sh failed" >&2; rc=1; }; done
  return $rc
}
start() {
  local fam envs up=0
  if [ $PRINT = 0 ]; then preflight || return 1; stop; fi
  fam="--pool $WINDOW --context $CONTEXT --streams $STREAMS --round-rows $ROUND_ROWS --engram /engram"
  fam="$fam --token-map /cache/dsv41_token_map.json --graphs 1 --side 1 --prefetch 1 --aio 1"
  fam="$fam --rdma $RDMA_HCA --rdma-kernels /engine/rdma_gather.fatbin"
  rank "${N[1]}" "$SHAPE" "/engine/tf-dsv41-lanes /model /cache/rank-cache 1 2 $MASTER_IP $TP_PORT /kit $fam" \
    || { echo "start rank 1 failed" >&2; return 1; }
  envs="TF_TP_WORLD=2 TF_TP_MASTER=$MASTER_IP TF_TP_PORT=$TP_PORT TF_DS_RANK_CACHE=/cache/rank-cache TF_DS_KIT=/kit"
  envs="$envs TENSORFOLD_CUDA_KERNELS=/kit/aot TF_DS_ENGRAM=/engram TF_DS_TOKEN_MAP=/cache/dsv41_token_map.json $SHAPE"
  rank "${N[0]}" "$envs" "/engine/tensorfold-native serve /model --host $HOST --port $PORT --name $SERVED_NAME --context $CONTEXT --parallel $PARALLEL $SERVER_FLAGS${RANK0_FLAGS:+ $RANK0_FLAGS} --backend cuda" \
    || { echo "start rank 0 failed" >&2; return 1; }
  [ $PRINT = 1 ] && return 0
  for _ in $(seq 1 240); do
    sleep 5
    if on "${N[0]}" "curl -sf -m 5 $U/v1/models | grep -q $SERVED_NAME"; then up=1; break; fi
    on "${N[0]}" "docker ps --format '{{.Names}}' | grep -qx tp2-lane" || break
  done
  [ $up = 1 ] || { echo "rank 0 did not come up: see $LOG_DIR/tp2-lane.log on ${N[0]}" >&2; return 1; }
  check && echo "up and answering on ${N[0]}:$PORT" || { echo "up, but the chat check failed" >&2; return 1; }
}
case ${1:?start|stop|check|print|inputs|preflight} in
  start) start ;;
  preflight) preflight && echo "preflight: rank 0 (${N[0]}) has every file it opens at start; both nodes have a RoCE v2 address at GID index 5" ;;
  stop) stop ;;
  check) check ;;
  print) PRINT=1; start ;;
  inputs) inputs ;;
  *) echo "start|stop|check|print|inputs|preflight" >&2; exit 2 ;;
esac
