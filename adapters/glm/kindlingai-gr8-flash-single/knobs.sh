# The recipe's TUNE and MENTAT knobs as gr8's options, sourced by entrypoint.sh and bin/gr8 so every gr8 command sees
# the same.
#   TUNE=speed (default): the settings measured in README.md, requests up to 262,144 tokens (MAX_MODEL_LEN=262144).
#   TUNE=context: one request up to 700,018 tokens (MAX_MODEL_LEN=700018: the 700,032-token pool less DFlash2's 14
#             guard rows). With a smaller CAP, set MAX_MODEL_LEN too.
#   TUNE=quality: the same as speed. K3 and K4 do not fit one box, and this recipe runs the int4 KV engine.
#   A MAX_MODEL_LEN set on its own keeps its value.
#   MENTAT=0 (default): this box on its own, PEERS=127.0.0.1. Nothing is announced.
#   MENTAT=1: find this box through the host's mentat daemon (gr8's MENTAT=localhost) and announce the API in the mentat
#             cluster, in group GROUP (gr8-glm53f-1x by default, so the box never joins a multi-box Flash cluster).
case "${TUNE:-speed}" in
  speed|quality) MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144} ;;
  context) MAX_MODEL_LEN=${MAX_MODEL_LEN:-700018} ;;
  *) echo "TUNE=$TUNE: want speed (the default), context or quality (README.md, \"Pick what to prioritize\")" >&2; exit 2 ;;
esac
export MAX_MODEL_LEN
case "${MENTAT:-0}" in
  0) unset MENTAT; export PEERS=${PEERS:-127.0.0.1} ;;
  1) unset PEERS; export MENTAT=localhost ;;
  localhost) unset PEERS ;;
  *) echo "MENTAT=$MENTAT: want 0 (this box alone) or 1 (register with the mentat cluster)" >&2; exit 2 ;;
esac
# gr8 takes GLM-5.3-Flash's k2, k3 and k4 by name, and any other manifest of the release as a path. The boot fetches
# the release's manifests/ before it reads VARIANT, so a name such as bit201 becomes that file's path in the model
# cache, for a MODEL pinned to a commit.
hf='^https://huggingface\.co/([^/@]+)/([^/@]+)@([0-9a-f]{40})$'
if [[ ${VARIANT:-} && $VARIANT != k[234] && $VARIANT != */* && ${MODEL:-} =~ $hf ]]; then
  export VARIANT=${MODEL_CACHE:-/var/lib/gr8/cache}/hf/${BASH_REMATCH[1]}--${BASH_REMATCH[2]}/${BASH_REMATCH[3]}/manifests/exlr8-manifest-$VARIANT.json
fi
unset hf
