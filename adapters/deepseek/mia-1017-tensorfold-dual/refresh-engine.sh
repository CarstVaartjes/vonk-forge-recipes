#!/bin/sh
set -eu
TF_REF=$(python3 -c 'import json; print(json.load(open("/opt/vonk-refresh/kit-lock.json"))["dependencies"]["engine"]["commit"])')
git init -q /tensorfold
git -C /tensorfold remote add origin https://github.com/bertholomus/TensorFold.git
git -C /tensorfold fetch -q --depth 1 origin "$TF_REF"
git -C /tensorfold checkout -q --detach FETCH_HEAD
test "$(git -C /tensorfold rev-parse HEAD)" = "$TF_REF"
cd /tensorfold
for patch in /opt/vonk-refresh/patches/*.patch; do
    git apply --check "$patch"
    git apply "$patch"
done
/opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast --prefix /opt/tensorfold -j2 install native
test -x /opt/tensorfold/native/bin/tensorfold-native
test -x /opt/tensorfold/bin/tf-dsv41-lanes
for file in LICENSE NOTICE ATTRIBUTION.md THIRD_PARTY_NOTICES.md LICENSES; do
    if [ -e "$file" ]; then cp -a "$file" /opt/tensorfold/share/doc/tensorfold/; fi
done
if [ -e tools/dsv41-zig/ATTRIBUTION.md ]; then
    cp tools/dsv41-zig/ATTRIBUTION.md /opt/tensorfold/share/doc/tensorfold/DSV41-ATTRIBUTION.md
fi
cd /
rm -rf /tensorfold /root/.cache/zig
# The rank-cache tool comes from the kit's second pin (the same fork's inputs branch) with its own patches, as zig/Dockerfile builds it.
INPUTS_REF=$(python3 -c 'import json; print(json.load(open("/opt/vonk-refresh/kit-lock.json"))["dependencies"]["inputs"]["commit"])')
git init -q /tf-inputs
git -C /tf-inputs remote add origin https://github.com/bertholomus/TensorFold.git
git -C /tf-inputs fetch -q --depth 1 origin "$INPUTS_REF"
git -C /tf-inputs checkout -q --detach FETCH_HEAD
test "$(git -C /tf-inputs rev-parse HEAD)" = "$INPUTS_REF"
cd /tf-inputs
for patch in /opt/vonk-refresh/patches-inputs/*.patch; do
    git apply --check "$patch"
    git apply "$patch"
done
/opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast --prefix /opt/inputs -j2 install
test -x /opt/inputs/bin/tf-dsv41-rank-cache
cp /opt/inputs/bin/tf-dsv41-rank-cache /opt/tensorfold/bin/
cd /
rm -rf /tf-inputs /opt/inputs /root/.cache/zig
