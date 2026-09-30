#!/usr/bin/env bash
# Fail-closed two-rank profile owner. Static provenance: r0b0tlab.hy3.w4a4.v3.owner-r12.v1
set -euo pipefail
PROFILE=${1:?base|mtp}; RUN_ID=${2:?run}; GPU_UTIL=${3:?util}; RUN_ROOT=${4:?artifact root}
[[ "$PROFILE" == base || "$PROFILE" == mtp ]]; [[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]
R0=hy3-head.local; R1=hy3-worker.local; NAME0="hy3-v025-${RUN_ID}-r0"; NAME1="hy3-v025-${RUN_ID}-r1"; T0="${NAME0}-owner"; T1="${NAME1}-owner"
P="$RUN_ROOT/performance/$PROFILE"; mkdir -p "$P/logs" "$P/telemetry"; rm -f "$P/READY" "$P/STOP" "$P/OWNER_FAILED" "$P/OWNER_COMPLETE"
cleanup(){ ssh "$R0" "tmux kill-session -t '$T0' 2>/dev/null || true; docker rm -f '$NAME0' >/dev/null 2>&1 || true" || true; ssh "$R1" "tmux kill-session -t '$T1' 2>/dev/null || true; docker rm -f '$NAME1' >/dev/null 2>&1 || true" || true; }
archive_logs(){ scp -q "$R0:/tmp/${NAME0}.log" "$P/logs/rank0-final.log" 2>/dev/null || true; scp -q "$R1:/tmp/${NAME1}.log" "$P/logs/rank1-final.log" 2>/dev/null || true; }
fail(){ printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$P/logs/owner.log"; touch "$P/OWNER_FAILED"; exit 1; }
trap 'archive_logs; cleanup' EXIT INT TERM
for h in "$R0" "$R1"; do ssh "$h" "test -f /opt/hy3/models/Hy3-NVFP4-w4a4-v3/model.safetensors.index.json && test \"\$(sha256sum /opt/hy3/models/Hy3-NVFP4-w4a4-v3/model.safetensors.index.json | cut -d' ' -f1)\" = 738e3ad2c8d16372e3d681d5793c78676fd256d7a4b43d9ca560f268d26adb26" || fail artifact_hash_$h; done
scp -q /opt/hy3/projects/hy3-balanced-chat8/campaign/launch_v025_profile.sh "$R0:/tmp/hy3-launch-profile.sh"
scp -q /opt/hy3/projects/hy3-balanced-chat8/campaign/launch_v025_profile.sh "$R1:/tmp/hy3-launch-profile.sh"
for h in "$R0" "$R1"; do ssh "$h" 'chmod +x /tmp/hy3-launch-profile.sh'; done
SEQ=32; BATCH=32768; [[ "$PROFILE" == mtp ]] && { SEQ=1; BATCH=2048; }
ssh "$R1" "tmux new-session -d -s '$T1' \"MAX_NUM_SEQS=$SEQ MAX_NUM_BATCHED_TOKENS=$BATCH KV_CACHE_MEMORY_BYTES=1536M SPEC_TOKENS=1 exec /tmp/hy3-launch-profile.sh 1 '$RUN_ID' '$PROFILE' '$GPU_UTIL' > /tmp/${NAME1}.log 2>&1\""
sleep 2
ssh "$R0" "tmux new-session -d -s '$T0' \"MAX_NUM_SEQS=$SEQ MAX_NUM_BATCHED_TOKENS=$BATCH KV_CACHE_MEMORY_BYTES=1536M SPEC_TOKENS=1 exec /tmp/hy3-launch-profile.sh 0 '$RUN_ID' '$PROFILE' '$GPU_UTIL' > /tmp/${NAME0}.log 2>&1\""
for x in $(seq 1 120); do a=$(ssh "$R0" "docker inspect -f '{{.State.Running}}' '$NAME0' 2>/dev/null || true"); b=$(ssh "$R1" "docker inspect -f '{{.State.Running}}' '$NAME1' 2>/dev/null || true"); [[ "$a" == true && "$b" == true ]] && break; sleep 2; done
[[ "${a:-}" == true && "${b:-}" == true ]] || fail containers_not_running
set_no_swap(){ h=$1; n=$2; cid=$(ssh "$h" "docker inspect -f '{{.Id}}' '$n'"); ssh "$h" "echo 0 | sudo -n tee /sys/fs/cgroup/system.slice/docker-${cid}.scope/memory.swap.max >/dev/null; test \"\$(cat /sys/fs/cgroup/system.slice/docker-${cid}.scope/memory.swap.max)\" = 0"; }
set_no_swap "$R0" "$NAME0"; set_no_swap "$R1" "$NAME1"
for x in $(seq 1 360); do
  a=$(ssh "$R0" "docker inspect -f '{{.State.Running}}' '$NAME0' 2>/dev/null || true")
  b=$(ssh "$R1" "docker inspect -f '{{.State.Running}}' '$NAME1' 2>/dev/null || true")
  [[ "$a" == true && "$b" == true ]] || fail rank_exited_before_api_ready
  curl -fsS --max-time 3 http://hy3-head.local:8004/v1/models > "$P/models.json.tmp" 2>/dev/null && { mv "$P/models.json.tmp" "$P/models.json"; break; }
  sleep 5
done
[[ -s "$P/models.json" ]] || fail api_not_ready
python3 - "$P/models.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); ids={row.get('id') for row in x.get('data',[])}
assert 'hy3-w4a4-v3' in ids,ids
PY
scp -q "$R0:/tmp/${NAME0}.log" "$P/logs/rank0-startup.log"; scp -q "$R1:/tmp/${NAME1}.log" "$P/logs/rank1-startup.log"
python3 - "$P/logs/rank0-startup.log" "$PROFILE" "$P/capacity.json" <<'PY'
import json,pathlib,re,sys
s=pathlib.Path(sys.argv[1]).read_text(errors='replace'); profile=sys.argv[2]
need=['HYV3ForCausalLM','modelopt_mixed','FLASHINFER_CUTLASS','FLASHINFER attention backend','Using fp8 data type to store kv cache','Skipping FlashInfer autotune because it is disabled','Using network IB','via NET/IB/0','via NET/IB/1','Application startup complete.']
if profile=='mtp': need.append('HYV3MTPModel')
missing=[x for x in need if x not in s]
if missing: raise SystemExit('missing startup markers: '+repr(missing))
for bad in ['FORBIDDEN_ARCHITECTURE_MARKER','FORBIDDEN_ARCHITECTURE_MARKER','FORBIDDEN_BACKEND_MARKER']:
 if bad in s: raise SystemExit('forbidden startup marker '+bad)
m=re.search(r'GPU KV cache size:\s*([0-9,]+) tokens',s)
if not m: raise SystemExit('missing measured KV token capacity')
cap=int(m.group(1).replace(',',''))
pathlib.Path(sys.argv[3]).write_text(json.dumps({'measured_kv_token_capacity':cap,'max_model_len':2048},indent=2)+'\n')
PY
read -r BSW0 BSW1 < <(printf '%s %s\n' "$(ssh "$R0" "awk '/SwapTotal/{t=\$2}/SwapFree/{f=\$2}END{print t-f}' /proc/meminfo")" "$(ssh "$R1" "awk '/SwapTotal/{t=\$2}/SwapFree/{f=\$2}END{print t-f}' /proc/meminfo")")
printf '{"profile":"%s","run_id":"%s","gpu_utilization":%s,"kv_cache":"1536M","max_num_seqs":%s,"speculative_tokens":%s,"host_swap_baseline_kb":{"rank0":%s,"rank1":%s}}\n' "$PROFILE" "$RUN_ID" "$GPU_UTIL" "$SEQ" "$([[ "$PROFILE" == mtp ]] && echo 1 || echo 0)" "$BSW0" "$BSW1" > "$P/profile.json"
printf 'timestamp,rank,mem_available_kb,host_swap_used_kb,container_swap_bytes,gpu_util_pct,temp_c,power_w\n' > "$P/telemetry/samples.csv"
touch "$P/READY"; printf '%s READY\n' "$(date -u +%FT%TZ)" >> "$P/logs/owner.log"
while [[ ! -e "$P/STOP" ]]; do
  a=$(ssh "$R0" "docker inspect -f '{{.State.Running}}' '$NAME0' 2>/dev/null || true"); b=$(ssh "$R1" "docker inspect -f '{{.State.Running}}' '$NAME1' 2>/dev/null || true"); [[ "$a" == true && "$b" == true ]] || fail rank_failure
  read -r m0 s0 c0 < <(ssh "$R0" "N='$NAME0' python3 -c 'import os,pathlib,subprocess; d={x.split()[0].rstrip(chr(58)):int(x.split()[1]) for x in open(chr(47)+\"proc/meminfo\") if len(x.split())>1}; cid=subprocess.check_output([\"docker\",\"inspect\",\"-f\",\"{{.Id}}\",os.environ[\"N\"]],text=True).strip(); cg=pathlib.Path(\"/sys/fs/cgroup/system.slice\")/(\"docker-\"+cid+\".scope\"); print(d[\"MemAvailable\"],d[\"SwapTotal\"]-d[\"SwapFree\"],(cg/\"memory.swap.current\").read_text().strip())'")
  read -r m1 s1 c1 < <(ssh "$R1" "N='$NAME1' python3 -c 'import os,pathlib,subprocess; d={x.split()[0].rstrip(chr(58)):int(x.split()[1]) for x in open(chr(47)+\"proc/meminfo\") if len(x.split())>1}; cid=subprocess.check_output([\"docker\",\"inspect\",\"-f\",\"{{.Id}}\",os.environ[\"N\"]],text=True).strip(); cg=pathlib.Path(\"/sys/fs/cgroup/system.slice\")/(\"docker-\"+cid+\".scope\"); print(d[\"MemAvailable\"],d[\"SwapTotal\"]-d[\"SwapFree\"],(cg/\"memory.swap.current\").read_text().strip())'")
  ts=$(date -u +%FT%TZ); g0=$(ssh "$R0" 'nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits'); g1=$(ssh "$R1" 'nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits'); printf '%s,rank0,%s,%s,%s,%s\n%s,rank1,%s,%s,%s,%s\n' "$ts" "$m0" "$s0" "$c0" "$g0" "$ts" "$m1" "$s1" "$c1" "$g1" >> "$P/telemetry/samples.csv"
  (( m0>=16777216 && m1>=16777216 && s0<=BSW0+1024*1024 && s1<=BSW1+1024*1024 && c0==0 && c1==0 )) || fail memory_hard_stop_m0_${m0}_m1_${m1}_s0_${s0}_s1_${s1}_c0_${c0}_c1_${c1}
  # Fail closed on fresh runtime/GPU fatal markers; read logs only and preserve them via cleanup.
  for spec in "$R0:$NAME0" "$R1:$NAME1"; do
    h=${spec%%:*}; n=${spec#*:}
    fatal=$(ssh "$h" "docker logs --since 20s '$n' 2>&1" | LC_ALL=C grep -Eim1 'Xid|NV_ERR_NO_MEMORY|CUDA out of memory|OutOfMemoryError|EngineDeadError|RPC call to (execute_model|sample_tokens) timed out' || true)
    [[ -z "$fatal" ]] || fail "fatal_runtime_marker_${h}_${fatal}"
  done
  sleep 10
done
archive_logs; cleanup
trap - EXIT INT TERM
touch "$P/OWNER_COMPLETE"
