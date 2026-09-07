# GenAI Deepfake Detection (Voice & Video)

Hackathon entry (HK4SDG) building deepfake detection for voice and video
calls. See **[CLAUDE.md](./CLAUDE.md)** for the full project brief, all
confirmed technical findings, and current status — that file is the source
of truth for project decisions and should be updated as the project evolves.

## Structure

```
src/audio/          Phase 1 (voice) pipeline — download, extraction, model
                     loading, inference, evaluation. Real logic lives here,
                     not in notebook cells, so it's version-controlled and
                     editable in VS Code / Claude Code.
notebooks/           Thin Colab driver notebooks that import from src/ and
                     run on Colab's GPU.
external/            Third-party repos added as git submodules (e.g.
                     DeepfakeBench for Phase 2). Not our code — see their
                     own licenses before any commercial use.
```

## Why this split

GPU compute (model inference/training) needs Colab; the surrounding
engineering (repo structure, preprocessing pipelines, debugging) is better
done locally in VS Code with Claude Code, where files persist, get real
version control, and don't require re-pasting code into a notebook cell
every time something changes. The workflow is:

1. Write/edit code in `src/` locally (Claude Code can do this directly).
2. `git commit` and `git push`.
3. In Colab, `git pull` (or clone) to get the latest code, then run the
   driver notebook, which imports from `src/` and executes on the GPU.
4. Bring anything worth keeping (metrics, findings — not raw data or model
   checkpoints) back into the repo and into CLAUDE.md.

## Local setup (VS Code + Claude Code)

```bash
git clone https://github.com/Ayat1807/ElderTrustShield.git
cd ElderTrustShield
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Colab setup

1. Open `notebooks/phase1_audio_colab.ipynb` in Colab (upload it, or open
   directly from GitHub via Colab's File > Open notebook > GitHub tab).
2. Edit the `REPO_URL` variable in the first code cell to your repo's path.
3. **If your repo is private**, generate a GitHub personal access token
   (Settings > Developer settings > Personal access tokens > Fine-grained,
   scoped to just this repo, read-only is enough) and store it in Colab's
   **Secrets** panel (key icon, left sidebar) under the name `GITHUB_TOKEN`.
   Never paste the token directly into a cell — the notebook reads it from
   Secrets automatically if present.
4. Runtime > Change runtime type > **T4 GPU**, then run all cells.

## Adding Phase 2 (DeepfakeBench)

DeepfakeBench pins **Python 3.7.2** and its own dependency versions —
deliberately kept in its own environment, not merged into `requirements.txt`,
to avoid version conflicts with the audio pipeline. Add it as a submodule
rather than copying its code in, so upstream updates stay trackable and its
license (CC BY-NC-4.0, non-commercial) stays clearly attributed to its own
directory:

```bash
git submodule add https://github.com/SCLBD/DeepfakeBench.git external/DeepfakeBench
cd external/DeepfakeBench
conda create -n DeepfakeBench python=3.7.2
conda activate DeepfakeBench
sh install.sh
```

Anyone else cloning this repo afterward needs
`git clone --recurse-submodules <repo-url>` (or `git submodule update --init`
after a normal clone) to actually pull DeepfakeBench's contents.

## Current status

See CLAUDE.md. Short version: Phase 1 zero-shot AASIST3 baseline confirmed
at 19.23% EER; Phase 2 (video) kickoff in progress.
