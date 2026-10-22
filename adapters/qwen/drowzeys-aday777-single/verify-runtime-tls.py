"""Reject stale Python and system trust stores during the image build."""

import ssl

import certifi

assert certifi.__version__ == "2026.10.19", certifi.__version__
# urllib/aiohttp use system trust; requests/httpx normally use certifi.
for context in (ssl.create_default_context(), ssl.create_default_context(cafile=certifi.where())):
    roots = context.get_ca_certs()
    assert roots, "empty runtime CA store"
    assert not any(
        value == "emSign ECC Root CA - G3"
        for root in roots
        for rdn in root["subject"]
        for key, value in rdn
        if key == "commonName"
    ), "compromised emSign root remains trusted"
