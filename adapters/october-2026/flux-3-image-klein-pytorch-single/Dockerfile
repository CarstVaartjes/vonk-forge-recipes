FROM docker.io/vllm/vllm-openai@sha256:4f63b83537c4cbd82822403965f395877054dc3b69612e7044ecd649a9badb02
LABEL org.opencontainers.image.source="https://github.com/black-forest-labs/flux-3-image-klein" \
      org.opencontainers.image.revision="bfc8889a4497f846416f253a3972a1941ac698b9"
RUN curl --fail --location --output /tmp/flux.tar.gz https://codeload.github.com/black-forest-labs/flux-3-image-klein/tar.gz/bfc8889a4497f846416f253a3972a1941ac698b9 \
    && printf '%s  %s\n' 672238637f0e656b9ffd17e79464b5186066561ef906fead445ea0ab1f15890c /tmp/flux.tar.gz | sha256sum --check --strict \
    && mkdir -p /opt/vonk/flux \
    && tar -xzf /tmp/flux.tar.gz -C /opt/vonk/flux --strip-components=1 \
    && rm /tmp/flux.tar.gz \
    && python3 -m pip install --no-cache-dir --no-deps /opt/vonk/flux \
    && python3 -m pip install --no-cache-dir transformers==5.8.1 accelerate==1.13.0 pillow==10.3.0 safetensors==0.8.0 \
    && python3 -c 'from flux3.generate import Models, generate'
COPY vonk_agent_protocol-4.1.0-py3-none-any.whl /opt/vonk/vonk_agent_protocol-4.1.0-py3-none-any.whl
RUN python3 -m pip install --break-system-packages --no-cache-dir /opt/vonk/vonk_agent_protocol-4.1.0-py3-none-any.whl \
    && install --directory --mode=0755 /opt/vonk/bin /inputs \
    && install --directory --mode=0770 --owner=10001 --group=10001 /state /outputs
COPY job.py /opt/vonk/job.py
COPY --chmod=0755 launcher /opt/vonk/bin/pytorch-pipeline
ENV HOME=/state HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
