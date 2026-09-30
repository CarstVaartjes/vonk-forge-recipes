#!/usr/bin/env bash
# Throughput + concurrency suite for xyz-aquila NVFP4 (vllm bench serve).
set -euo pipefail
ROOT="/home/r0b0tdgx/projects/xyz-aquila-mini-nvfp4"
CONTAINER="${CONTAINER:-xyz-aquila-nvfp4}"
PORT="${PORT:-18082}"
MODEL="${MODEL:-xyz-aquila-nvfp4}"
MODEL_PATH="${MODEL_PATH:-/models/model}"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="$ROOT/evidence/runs/perf-$RUN_ID/concurrency"
mkdir -p "$OUT"
# levels: respect server max_num_seqs (currently 4 on AST profile; allow override)
LEVELS="${LEVELS:-1 2 4 8}"
REPS="${REPS:-3}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"

{
  echo "run_id=$RUN_ID"
  echo "container=$CONTAINER port=$PORT model=$MODEL"
  echo "levels=$LEVELS reps=$REPS input=$INPUT_LEN output=$OUTPUT_LEN"
  date -u -Is
  curl -fsS "http://127.0.0.1:$PORT/v1/models" || true
} | tee "$OUT/preflight.txt"

for c in $LEVELS; do
  prompts=$(( c * 4 ))
  if (( prompts < 8 )); then prompts=8; fi
  for rep in $(seq 1 "$REPS"); do
    echo "=== c=${c} rep=${rep} prompts=${prompts} $(date -u -Is) ===" | tee -a "$OUT/progress.log"
    # health gate
    if ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
      echo "SERVER_DOWN before c${c}-r${rep}" | tee -a "$OUT/progress.log"
      exit 2
    fi
    set +e
    docker exec "$CONTAINER" /usr/local/bin/vllm bench serve \
      --backend openai-chat \
      --base-url "http://127.0.0.1:${PORT}" \
      --endpoint /v1/chat/completions \
      --model "$MODEL" \
      --tokenizer "$MODEL_PATH" \
      --dataset-name random \
      --random-input-len "$INPUT_LEN" \
      --random-output-len "$OUTPUT_LEN" \
      --num-prompts "$prompts" \
      --max-concurrency "$c" \
      --request-rate inf \
      --seed 0 \
      --ignore-eos \
      --temperature 0 \
      --percentile-metrics ttft,tpot,itl,e2el \
      --save-result \
      --result-dir /tmp \
      --result-filename "aquila-c${c}-r${rep}.json" \
      >"$OUT/c${c}-r${rep}.log" 2>&1
    rc=$?
    set -e
    echo "rc=$rc" | tee -a "$OUT/progress.log"
    if [[ $rc -ne 0 ]]; then
      tail -n 40 "$OUT/c${c}-r${rep}.log" | tee -a "$OUT/progress.log"
      # try copy anyway
    fi
    docker cp "$CONTAINER:/tmp/aquila-c${c}-r${rep}.json" "$OUT/c${c}-r${rep}.json" 2>/dev/null || true
  done
done

python3 - <<PY
import json, statistics, re
from pathlib import Path
out = Path("$OUT")
levels = [int(x) for x in "$LEVELS".split()]
rows = []
for c in levels:
    reps = []
    for rep in range(1, int("$REPS")+1):
        p = out / f"c{c}-r{rep}.json"
        if not p.exists():
            continue
        reps.append(json.loads(p.read_text()))
    if not reps:
        rows.append({"concurrency": c, "error": "no_results"})
        continue
    stable = reps[1:] if len(reps) > 1 else reps
    def mean_key(*names):
        vals=[]
        for r in stable:
            for n in names:
                if n in r and r[n] is not None:
                    vals.append(float(r[n])); break
        return statistics.mean(vals) if vals else None
    rows.append({
        "concurrency": c,
        "repetitions_present": len(reps),
        "warmup_rep_dropped": 1 if len(reps)>1 else 0,
        "request_throughput_rps": mean_key("request_throughput"),
        "output_throughput_tok_s": mean_key("output_throughput"),
        "total_throughput_tok_s": mean_key("total_token_throughput", "total_throughput"),
        "mean_ttft_ms": mean_key("mean_ttft_ms"),
        "median_ttft_ms": mean_key("median_ttft_ms", "p50_ttft_ms"),
        "p99_ttft_ms": mean_key("p99_ttft_ms"),
        "mean_tpot_ms": mean_key("mean_tpot_ms"),
        "p99_tpot_ms": mean_key("p99_tpot_ms"),
        "mean_itl_ms": mean_key("mean_itl_ms"),
        "p99_itl_ms": mean_key("p99_itl_ms"),
        "mean_e2el_ms": mean_key("mean_e2el_ms"),
        "p99_e2el_ms": mean_key("p99_e2el_ms"),
        "completed": sum(int(r.get("completed") or 0) for r in stable),
        "failed": sum(int(r.get("failed") or 0) for r in stable),
    })
payload = {
    "method": f"vllm bench serve random {$INPUT_LEN}in/{$OUTPUT_LEN}out ignore_eos temp0; levels={levels}; reps={$REPS} drop first",
    "server": {"port": $PORT, "model": "$MODEL", "container": "$CONTAINER"},
    "rows": rows,
}
(out/"summary.json").write_text(json.dumps(payload, indent=2)+"\n")
print(json.dumps(payload, indent=2))
PY
echo DONE > "$OUT/COMPLETE"
date -u -Is | tee "$OUT/finished.txt"
