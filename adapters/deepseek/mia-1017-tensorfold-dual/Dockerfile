FROM ghcr.io/miaai-lab/deepseek-v4.1-flash-1017-2x-dgx-sparks-tensorfold@sha256:4ed15dceaf0a210052d5db89c74145794d8ee45f63a3436878ac3ec01cea38b5
LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks" \
      org.opencontainers.image.revision="1afc8910b76574aa6b844a08ec9643d14d57089a" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      ai.vonkforge.runtime-interface="v1"
RUN curl -fsSL https://codeload.github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/tar.gz/abe07c7d509a0199cc7fec15263be2aa52ac652f -o /tmp/kit.tar.gz \
 && echo '94cad134348a86dd49f93633cc70f3efc5f09142df8461bba119980987f7951f  /tmp/kit.tar.gz' | sha256sum -c - \
 && mkdir -p /tmp/kit /opt/vonk/bin \
 && tar -xzf /tmp/kit.tar.gz --strip-components=1 -C /tmp/kit \
 && cp -a /tmp/kit/kit /opt/tensorfold/kit \
 && echo '94cad134348a86dd49f93633cc70f3efc5f09142df8461bba119980987f7951f' > /opt/tensorfold/kit/VONK_SOURCE_SHA256 \
 && ln -s /opt/tensorfold/native/bin/tensorfold-native /opt/tensorfold/bin/tensorfold-native \
 && cp /opt/tensorfold/tools/inputs/*.py /opt/tensorfold/tools/ \
 && rm -rf /tmp/kit /tmp/kit.tar.gz
COPY requirements.lock /opt/tensorfold/requirements.lock
RUN python3 -m pip install --no-deps --require-hashes --no-cache-dir -r /opt/tensorfold/requirements.lock \
 && python3 -c 'import torch, tokenizers; print(tokenizers.__version__)' \
 && test -x /opt/tensorfold/bin/tensorfold-native \
 && test -x /opt/tensorfold/bin/tf-dsv41-lanes \
 && test -x /opt/tensorfold/bin/tf-dsv41-rank-cache \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
COPY licenses /opt/tensorfold/licenses/mia
COPY chat_template.jinja prepare_cache.py launcher.py /opt/vonk/bin/
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
ENV VONK_TF_WORLD=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home XDG_CACHE_HOME=/outputs/cache TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
