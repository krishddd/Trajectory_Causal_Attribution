# Deploying this Space to Hugging Face

These three files (`app.py`, `requirements.txt`, `README.md`) are a complete
Gradio Space. To publish under your account (`Harishkris`):

## Option A — web UI + git

1. Create the Space: <https://huggingface.co/new-space>
   - Owner: `Harishkris`, Space name: `agent-replay`
   - SDK: **Gradio**, hardware: CPU basic (free) is enough.
2. Clone it and copy these files in:
   ```bash
   git clone https://huggingface.co/spaces/Harishkris/agent-replay
   cp huggingface/app.py huggingface/requirements.txt huggingface/README.md agent-replay/
   cd agent-replay
   git add app.py requirements.txt README.md
   git commit -m "agent-replay interactive demo"
   git push
   ```
   You'll be prompted for your HF username and an access token
   (<https://huggingface.co/settings/tokens>, "write" scope) as the password.

## Option B — huggingface_hub CLI

```bash
pip install huggingface_hub
huggingface-cli login                      # paste a write token
huggingface-cli repo create agent-replay --type space --space_sdk gradio
huggingface-cli upload Harishkris/agent-replay huggingface . --repo-type space
```

The Space builds automatically (installs `agent-replay` from this GitHub repo per
`requirements.txt`) and goes live at
`https://huggingface.co/spaces/Harishkris/agent-replay`.

## Note on hosting the *code* on Hugging Face

Hugging Face is git-based, so you *can* mirror the whole repo into a Model or
Dataset repo, but that is not what the Hub is for — it hosts models, datasets, and
Spaces (apps), not general source code. GitHub remains the code home and PyPI the
distribution channel; the Space above is the idiomatic HF presence — a live demo
that points back to both.
