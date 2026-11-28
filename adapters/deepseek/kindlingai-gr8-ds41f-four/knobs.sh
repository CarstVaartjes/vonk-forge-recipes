# The recipe's knobs as gr8's options, sourced by entrypoint.sh so every gr8 command sees the same.
#   MENTAT=0 (default): the four boxes from PEERS (their LAN addresses, head first) and, if set, IPS and IPS2 (each box's
#             fabric address on the first and second ConnectX root, in PEERS order). Nothing is announced.
#   MENTAT=1: find the other boxes through the host's mentat daemon (gr8's MENTAT=localhost) and announce the API in the
#             mentat cluster, in group GROUP (gr8-ds41-4x by default).
#   TUNE=speed (default): fast mode with gr8's default v2 release stack (12,288-row chunks, GR8_DS_PF_V2=1,
#             GR8_DS_PF_V2_W16K=1, GR8_DS_PF_V2_L20_TAIL=1, GR8_DS_PF_V2_SEND_FFN_OPT=3, GR8_DS_PF_V2_BF16_TC=0,
#             GR8_DS_ATTN_TC_TILE=2, GR8_S2_ADMIT_HOLD=1 and GR8_S2_TAIL_FIRST=1; gr8/scripts/engine_args.sh), plus
#             FIRST-BOOT.md Q3's two switches and Q4's output head, which change the output, the prefill setting
#             GR8_DS_ATTN_TC_MF=2, GR8_DS_CACHE_MIB=1024 and GR8_DS_ENGRAM_CACHE_MIB=2048 (gr8/docs/ENV.md), and,
#             unless set, SESS=32 and CAP=327680. gr8's published image (v1.3.1+) builds g6m4_ds41 with DS41_MX=48, so a
#             step holds 48 rows.
#   TUNE=context: FIRST-BOOT.md Q3 and Q4, GR8_DS_ATTN_TC_MF=2, GR8_DS_ATTN_TC_TILE=1, GR8_DS_MHC_ROW_SHARD=1,
#             GR8_DS_CACHE_MIB=2048 and, unless set, SESS=16 and CAP=1048576 (FIRST-BOOT.md Q2). GR8_DS_MHC_ROW_SHARD=1
#             selects gr8's old prefill path, not the release stack.
#   TUNE=quality: the exact model: gr8's EXACT=1 (the old prefill path) and, unless set, SESS=5 and CAP=262144. EXACT
#             set by hand is refused with a pointer here; TUNE=quality sets it.
#   SESS above 5 adds GR8_DS_SESS_DRAFTS=1: each chat drafts fewer tokens as more chats write (FIRST-BOOT.md Q1), and
#             on the image's 48-row build none when more than 24 write (gr8/docs/ENV.md).
# Engram is always on, from the token map in ENGRAM_PREFIX (entrypoint.sh). Settings already in EXTRA_ENV replace the
# recipe's value and come after these.
[ -z "${EXACT:-}" ] || { echo "EXACT=$EXACT: EXACT is now TUNE: EXACT=1 is TUNE=quality, EXACT=0 is TUNE=speed" >&2; exit 2; }
case "${MENTAT:-0}" in
  0) unset MENTAT
     n=$(tr , '\n' <<< "${PEERS:-}" | grep -c .) || true
     [ "$n" = 4 ] || { echo "PEERS=${PEERS:-}: want the four boxes' LAN addresses, head first, the same on every box (or MENTAT=1)" >&2; exit 2; } ;;
  1) unset PEERS; export MENTAT=localhost ;;
  localhost) unset PEERS ;;
  *) echo "MENTAT=$MENTAT: want 0 (the boxes in PEERS) or 1 (register with the mentat cluster)" >&2; exit 2 ;;
esac
ds_words=("GR8_DS_ENGRAM=1" "GR8_DS_ENGRAM_PREFIX=${ENGRAM_PREFIX:?}")
fast=(GR8_DS_RELAXED_ACCEPT=topp:0.9,ratio:0.1 GR8_DS_MOE_TOPK_DECODE=4 GR8_DS41_HEAD_MX=1 GR8_DS_ATTN_TC_MF=2)
case "${TUNE:-speed}" in
  speed) ds_words+=("${fast[@]}" GR8_DS_CACHE_MIB=1024 GR8_DS_ENGRAM_CACHE_MIB=2048); SESS=${SESS:-32} CAP=${CAP:-327680} ;;
  context) ds_words+=("${fast[@]}" GR8_DS_ATTN_TC_TILE=1 GR8_DS_MHC_ROW_SHARD=1 GR8_DS_CACHE_MIB=2048); SESS=${SESS:-16} CAP=${CAP:-1048576} ;;
  quality) export EXACT=1; SESS=${SESS:-5} CAP=${CAP:-262144} ;;
  *) echo "TUNE=$TUNE: want speed (fast mode, the default), context (one chat up to 1M tokens) or quality (the exact model)" >&2; exit 2 ;;
esac
export SESS CAP
case "$SESS" in
  ''|*[!0-9]*|0*) echo "SESS=$SESS: want a number of chats, 1 or more" >&2; exit 2 ;;
  [1-5]) ;;
  *) ds_words+=(GR8_DS_SESS_DRAFTS=1) ;;
esac
extra=${EXTRA_ENV:-}
read -ra extra_arr <<< "${extra//[$'\t\r\n']/ }"
extra=
for kv in ${extra_arr[@]+"${extra_arr[@]}"}; do
  case "$kv" in
    G6_DS41_CACHE_MIB=*) kv=GR8_DS_CACHE_MIB=${kv#*=} ;;
    G6_DS41_PF_V2_BF16_TC=*) kv=GR8_DS_PF_V2_BF16_TC=${kv#*=} ;;
  esac
  extra="${extra:+$extra }$kv"
done
kept=
for kv in "${ds_words[@]}"; do
  k=${kv%%=*}
  case " $extra " in
    *" $k="*|*" G6_${k#GR8_}="*) ;;
    *) kept="${kept:+$kept }$kv" ;;
  esac
done
export EXTRA_ENV="$kept${extra:+${kept:+ }$extra}"
unset ds_words fast extra extra_arr kept kv k n
