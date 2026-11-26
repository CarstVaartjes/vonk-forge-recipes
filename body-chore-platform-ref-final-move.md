## Summary
- Move every `platform_ref` / `validate-recipe-library.yml@` literal (publish.yml x2, validate.yml x2) from 3dacd2e6 to vonk-forge main 25c4dde1 (deployed; locks contracts 5.7.0). No other literal exists (grep).
- Scripts the workflows call at the new SHA (`build-recipe-library`, `recipe-image-plan`, `build-control-wheel`) exist at 25c4dde1.
- Python 3.15: the setup-tooling action (python-version and its trust-bundle `uv venv`) moves to 3.15, which the pinned forge requires.
- Folds #3077 (uv 0.13.10: action version and `required-version`), applied cleanly.
- AGENTS.md: states this is the last manual move (RG removes the pin).
- Follow-ups rebased onto this branch: #3095 (BiRefNet) and #3116 (BPT).

## Identity churn
"Plan images" changed-key list: see the PR's CI run (filled in below once read). Not merge-armed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01DaJVtqsPAPpx9YBbXxQfdK
