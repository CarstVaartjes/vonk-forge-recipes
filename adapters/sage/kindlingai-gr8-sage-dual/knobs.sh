# The recipe's TUNE and MENTAT knobs as gr8's options, sourced by entrypoint.sh and bin/gr8 so every gr8 command sees
# the same.
#   TUNE=speed (default): the release's K4 width manifest, four sessions sharing a 262,208-token KV pool, requests up
#             to 262,144 tokens, with MTP=2 and S2_SHARED_WATERFILL=1 (strict bit-exact speedup over MTP=1).
#   TUNE=context: K4, eight sessions sharing a 262,208-token KV pool, context-parallel KV (GR8_CP=1 in EXTRA_ENV),
#             requests up to 262,144 tokens, with MTP=2 and S2_SHARED_WATERFILL=1. With a smaller CAP, set
#             MAX_MODEL_LEN too: at most CAP less the guard rows.
#   TUNE=quality: K4, four sessions sharing a 262,208-token pool, requests up to 262,144 tokens, strict MTP=1.
#   A VARIANT, SESS, CAP, MAX_MODEL_LEN, MTP or S2_SHARED_WATERFILL set on its own keeps its value, and so does a
#             GR8_CP already in EXTRA_ENV.
#   Text-only is the default. The release includes dense/vision-bf16.safetensors (untested with this recipe); to opt
#             in, add GR8_GLM53F_VISION=1 to EXTRA_ENV on both boxes.
#   MENTAT=0 (default): the two boxes from PEERS (their LAN addresses, head first) and, if set, IPS and IPS2 (each
#             box's fabric address on the first and second ConnectX root, in PEERS order). Nothing is announced.
#   MENTAT=1: find the other box through the host's mentat daemon (gr8's MENTAT=localhost) and announce the API in the
#             mentat cluster, in group GROUP (gr8-sage-2x by default).
case "${TUNE:-speed}" in
  speed)
    VARIANT=${VARIANT:-k4} SESS=${SESS:-4} CAP=${CAP:-262208} MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
    MTP=${MTP:-2} S2_SHARED_WATERFILL=${S2_SHARED_WATERFILL:-1}
    export MTP S2_SHARED_WATERFILL ;;
  context)
    VARIANT=${VARIANT:-k4} SESS=${SESS:-8} CAP=${CAP:-262208} MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
    MTP=${MTP:-2} S2_SHARED_WATERFILL=${S2_SHARED_WATERFILL:-1}
    export MTP S2_SHARED_WATERFILL
    # gr8's Flash engine environment starts with GR8_CP=0 and ends with EXTRA_ENV, so the last GR8_CP wins
    case " ${EXTRA_ENV:-} " in *" GR8_CP="*) ;; *) export EXTRA_ENV="${EXTRA_ENV:+$EXTRA_ENV }GR8_CP=1" ;; esac ;;
  quality)
    VARIANT=${VARIANT:-k4} SESS=${SESS:-4} CAP=${CAP:-262208} MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
    MTP=${MTP:-1} S2_SHARED_WATERFILL=${S2_SHARED_WATERFILL:-0}
    export MTP S2_SHARED_WATERFILL ;;
  *) echo "TUNE=$TUNE: want speed (the default), context or quality (README.md, \"Pick what to prioritize\")" >&2; exit 2 ;;
esac
# gr8 takes k2, k3 and k4 by name, and any other manifest of the release as a path. The boot fetches the release's
# manifests/ before it reads VARIANT, so a custom manifest name becomes that file's path in the model cache for a
# MODEL pinned to a commit.
hf='^https://huggingface\.co/([^/@]+)/([^/@]+)@([0-9a-f]{40})$'
if [[ $VARIANT != k[234] && $VARIANT != */* && ${MODEL:-} =~ $hf ]]; then
  rel=${MODEL_CACHE:-/var/lib/gr8/cache}/hf/${BASH_REMATCH[1]}--${BASH_REMATCH[2]}/${BASH_REMATCH[3]}
  VARIANT=$rel/manifests/exlr8-manifest-$VARIANT.json
fi
unset hf rel
export VARIANT SESS CAP MAX_MODEL_LEN
case "${MENTAT:-0}" in
  0) unset MENTAT
     n=$(tr , '\n' <<< "${PEERS:-}" | grep -c .) || true
     [ "$n" = 2 ] || { echo "PEERS=${PEERS:-}: want the two boxes' LAN addresses, head first, the same on both boxes (or MENTAT=1)" >&2; exit 2; } ;;
  1) unset PEERS; export MENTAT=localhost ;;
  localhost) unset PEERS ;;
  *) echo "MENTAT=$MENTAT: want 0 (the boxes in PEERS) or 1 (register with the mentat cluster)" >&2; exit 2 ;;
esac
