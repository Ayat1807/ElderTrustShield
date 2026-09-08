# GenAI Deepfake Detection (Voice & Video) — project brief

(Internal folder/Drive naming stays `ElderTrustShield` for continuity with
existing code and paths — this is a naming/branding note only, not a scope
constraint.)

## What this is
A hackathon entry (2026 HK4SDG GenAI Hackathon) building GenAI-assisted
detection for deepfake voices and videos on calls. The core deliverable is
the detection and notification technology itself — models that flag whether
a voice or video is likely AI-generated. Specific end applications (e.g.
protecting individuals from scam calls, protecting companies from
deepfake-enabled fraud like the 2024 Arup Hong Kong incident) are open and
to be decided later, not a fixed constraint on the technical build.

## Note on project history (folder reconciliation)
Before the Colab-based work documented below, there was an earlier local
VS Code workspace (`Scam_Detection/`) with its own CLAUDE.md and a set of
dataset-handling scripts (`download_flac_D_aa.py`, `build_dev_shard_csv.py`,
`match_flac_D_aa_dev.py`, and others). That attempt predates essentially
everything below — model confirmation, the label-inversion fix, the
confirmed EER result — and was abandoned after confirming the local network
couldn't reliably download large files at all (documented in that old file
too). Reviewed and reconciled: the FaceForensics++ c40/DFD plan above was
recovered from it as genuinely new; everything else in that old file
(elder-specific-only framing, "no deadline" framing, untested model
candidates) is superseded by what's below and was not carried forward.
That folder is being left in place, untouched, not merged into the new repo.

## Repository
Phase 1 is version-controlled at `github.com/Ayat1807/ElderTrustShield`
(private). Contains: `src/audio/` (the confirmed Phase 1 pipeline as
importable modules, including the baked-in label-inversion fix),
`notebooks/phase1_audio_colab.ipynb` (thin Colab driver importing from
`src/`), `notebooks/phase1_audio_working.ipynb` (the actual full run
history — extraction, the 80.77%→19.23% EER fix, calibration — kept
alongside the clean version as the real evidence trail, not superseded by
it). Local clone lives at `C:\Users\Acer\Desktop\ElderTrustShield`, worked
on via VS Code + Claude Code; GPU execution still happens via Colab
(cloning/pulling this repo with a fine-grained read-only PAT stored in
Colab Secrets, since the repo is private).

## Phase 2 (video) — UCF confirmed working end to end
Confirmed via a smoke test (random noise input, not yet real images): UCF
loads with its real trained checkpoint and produces a valid forward pass.
`predictions['cls']` returns raw logits (can be negative — confirmed by
seeing an actual negative value in output, not assumed); apply
`torch.softmax(cls, dim=1)` to get real probabilities. Per `ucf.yaml`'s own
`label_dict` (`FF-real: 0`, all fake categories nonzero): **index 0 =
probability real/bonafide, index 1 = probability fake** — natural
convention, unlike AASIST3's reversed one, but not yet independently
verified on real labeled images the way AASIST3's convention was checked;
do that before trusting it fully.

Getting here required ten rounds of real fixes, in order — worth keeping
so this doesn't need rediscovering. Importing `from detectors import
DETECTOR` eagerly imports code for all ~36 detectors (via
`training/detectors/__init__.py`), not just UCF, so most of these are
unrelated to UCF itself but block the import regardless:
1. Missing `fvcore`, `iopath` (video detectors' SlowFast utilities)
2. Circular import inside `detectors/utils/slowfast/utils/env.py` — fixed
   by commenting out the 11 video/CLIP-detector import lines in
   `detectors/__init__.py` (`slowfast`, `videomae`, `clip`, `timesformer`,
   `xclip`, `ftcn`, `i3d`, `altfreezing`, `stil`, `tall`, `effort`) — none
   needed for UCF (a Spatial detector, per the repo's own category list)
3. Missing `efficientnet_pytorch` (networks/__init__.py backbone import)
4. Missing `kornia` (multi_attention_detector.py)
5. Missing `imgaug` (bi_online_generation.py)
6. `imgaug` itself breaks on NumPy 2.0 — its code calls the removed
   `np.sctypes` attribute. Fixed with a monkeypatch restoring
   `np.sctypes` as a plain dict of type lists, run before any DeepfakeBench
   import. Real value, not a hack: `np.sctypes` was a static reference
   table, never used in computation, so restoring it can't change any
   numeric result anywhere.
7. Missing `lmdb`
8. Missing data file: `preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat`
   (dlib face-landmark model, loaded eagerly at import time by
   fwa_blend.py). Downloaded from
   `github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/shape_predictor_81_face_landmarks.dat`
   — this is also the exact file needed for the deferred self-preprocessing
   c40 plan, so this download has dual value, not just an import-fix.
9. Missing `training/pretrained/xception-b5690688.pth` (ImageNet Xception
   backbone — UCFDetector.__init__ loads this unconditionally as a
   starting point even though the real trained checkpoint overwrites it
   moments later). Downloaded from
   `github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip`.
10. That zip's contents were nested one level deeper than expected
    (`pretrained/pretrained/*.pth` instead of `pretrained/*.pth`) — fixed
    by moving the files up one directory level after extraction.

None of Colab's own preinstalled versions (numpy, torch, opencv,
scikit-learn) were downgraded to match install.sh's old pins — deliberate:
Colab's modern torch (2.11.0+cu128, CUDA confirmed working) was kept as-is
rather than replaced with install.sh's CUDA-11.3-linked 1.12.0 build, since
GPU compute runs on Colab, not the local machine, and downgrading a
working modern stack for an old pin risked breaking more than it fixed.

## Working norms
- Distinguish clearly between something confirmed from actual
  evidence/documentation versus something inferred or assumed — never
  present a guess as a verified fact.
- When reviewing status updates or debugging output, check whether a stated
  conclusion is actually supported by the evidence shown, rather than
  accepting a confident-sounding claim at face value.
- Flag conceptual mismatches (e.g. a model built for the wrong task) as more
  urgent than surface-level bugs.

## Build order
1. Voice/audio model first. This is the primary deliverable and current
   focus.
2. Video deepfake detection — start once voice is solid.
3. Fusion of voice + video signals — once both individual models work.
4. A lightweight Gradio demo wrapping the finished model(s) — last step.

## Pretrained models — status
Check for and prefer existing pretrained models over training from scratch.
Candidates evaluated so far, with direct evidence rather than assumption:

- **`MTUCI/AASIST3`** (also mirrored at `lab260/AASIST3`, which now flags
  itself as a deprecated pointer to `lab260/Spectra-AASIST3`) — **confirmed
  correct model, in active use.** Model card explicitly states: "AASIST3:
  KAN-Enhanced AASIST Speech Deepfake Detection," "pipeline_tag:
  audio-classification," and "prediction: 0 = bonafide, 1 = spoof." Loads via
  a custom `PyTorchModelHubMixin` interface, not a standard
  `transformers.AutoModel`:
  ```
  git clone https://github.com/mtuciru/AASIST3.git
  from model import aasist3
  model = aasist3.from_pretrained("MTUCI/AASIST3")
  ```
  Expects mono 16kHz audio, padded/truncated to 64,600 samples (~4s).
- **`JYP2024/Wedefense_ASV2025_WavLM_Base_Pruning`** — **eliminated, wrong
  task.** Confirmed via its source paper (a model-compression paper reporting
  Vox1-O/E/H EER numbers) that this is a *speaker-verification* /
  compression model, not spoof detection. It surfaced on the ASVspoof 5 HF
  page via Track 2 (spoofing-robust ASV), not Track 1 (pure spoof
  detection). Could be relevant later for a "does this voice match the
  enrolled contact" feature, but is not a bonafide/spoof classifier.
- **`kaz17/WavLM-AASIST-ASVspoof5-v1` / `v2`** — **unconfirmed, not pursued.**
  Both returned 404 on Hugging Face at time of checking; no model card or
  config available. Not enough evidence to treat as valid candidates.

## Phase 2 (video) — kickoff findings, not yet acted on
- **Critical, confirmed constraint: FaceForensics++ in-distribution scores do
  not predict real-world performance.** Published cross-dataset results show
  Xception trained/tested on FaceForensics++ itself scores 99.7% AUC, but
  the identical trained model scores only 65.3% AUC on Celeb-DF and 72.2% on
  DFDC — barely above chance on unseen datasets. A separate documented case
  showed a model trained to 96.98% accuracy / 0.99 AUC on FaceForensics++
  misclassifying every tested real-world internet deepfake (Obama PSA, Tom
  Cruise face-swaps) as genuine, because FF++'s manipulation techniques are
  circa-2019 and don't represent how modern deepfakes are actually made.
  This is the video-domain analog of the Phase 1 EER-vs-accuracy lesson and
  matters more here given the project's real-world (scam-call) target use
  case. Design constraint going forward: any video model's real evaluation
  must include held-out footage outside FaceForensics++ itself (mirrors the
  existing audio requirement for a realistic, non-benchmark held-out set) —
  a high in-distribution FF++ number alone should not be treated as evidence
  of real-world readiness.
- **Candidate pretrained-model source, not yet selected/loaded:**
  `SCLBD/DeepfakeBench` (GitHub) — an actively maintained toolkit bundling
  33 detection methods (27 image-level, 6 video-level: Xception,
  EfficientNet-family, I3D, FTCN, AltFreezing, SBI, SLADD, LSDA, IID, etc.)
  with released pretrained weights and consistent benchmarking across
  methods. Preferred over picking a single small/hobbyist Hugging Face
  checkpoint in isolation, several of which show unverified or
  suspiciously specific self-reported numbers with little documentation —
  same overclaiming pattern already seen and treated with caution in the
  scam-classifier search (Q3 discussion below) and in the AASIST3
  label-inversion incident. Next step: review DeepfakeBench's own
  cross-dataset benchmark numbers to select a specific method, rather than
  defaulting to whichever scores highest on FaceForensics++ alone, given
  the generalization-gap finding above.
- **Research lead, not confirmed usable:** a March 2026 paper ("Deepfake
  Detection that Generalizes Across Benchmarks," method "GenD") targets
  this exact generalization problem directly — lightweight fine-tuning
  (only LayerNorm parameters, ~0.03% of weights) of a pretrained vision
  encoder plus a hyperspherical feature-manifold constraint, evaluated
  across 14 benchmarks spanning 2019–2025. Relevant to the project's
  best-possible-performance goal, but no public/loadable checkpoint has
  been confirmed yet — needs checking before relying on it.

## Phase 2 (video) — decisions made
- **Detector chosen: UCF**, over SPSL — UCF has the best within-domain
  average (0.9527 vs SPSL's 0.9408) and ties for most cross-dataset top-3
  finishes (5 vs 3) in DeepfakeBench's own published table; SPSL's raw
  cross-domain average is marginally higher (0.7875 vs 0.7801) but driven
  by exceptional performance on just two datasets (Celeb-DF-v1/v2) rather
  than broad consistency. Confirmed checkpoint available and downloadable:
  `ucf_best.pth` (188.0 MB), from
  `github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1`. `spsl_best.pth`
  (87.8 MB) also confirmed available from the same release if reconsidered.
- **Preprocessing: using DeepfakeBench's own preprocessed data at c23**
  compression for the prototype, not the c40 plan recovered from the old
  pre-Colab CLAUDE.md (chosen there specifically to match real-world call
  quality). Deliberate trade-off, not an oversight: DeepfakeBench provides
  ready-to-use preprocessed data (cropped faces, landmarks, masks) plus
  matching JSON configs at c23 only — using it skips building a face-
  detection/cropping pipeline entirely, matching the "presentable prototype
  now" priority. **Revisit later:** self-preprocessing raw FF++ at c40 (own
  dlib pipeline) remains the documented path to better real-world-call-
  quality matching, deferred alongside the other fine-tuning/improvement
  work already queued after the prototype ships.
- Detector's own trained-checkpoint weights (`./training/weights/`, from
  the v1.0.1 release) are a separate thing from the empty
  `./training/pretrained/` folder (ImageNet backbones, for training
  from scratch only) — do not confuse the two when following the repo's
  own docs; this distinction wasn't obvious from the folder names alone.

## Tech stack
- Python throughout.
- Audio: torchaudio, librosa, audiomentations (augmentation),
  huggingface_hub (targeted dataset/model downloads), Hugging Face
  transformers (wav2vec2 / WavLM), SpeechBrain (ECAPA-TDNN for speaker
  verification, if the voice-match feature gets built later).
- Video (phase 2): OpenCV, mediapipe (face detection/cropping), timm or
  DeepfakeBench (pretrained Xception/EfficientNet forgery classifiers).
- Fusion (phase 3): scikit-learn — a simple meta-classifier on top of the
  individual models' scores, not a new deep model.
- Demo: Gradio.
- Training/inference compute: Google Colab (free GPU tier, T4). GPU runtime
  is required for AASIST3 inference at any reasonable speed — confirm
  Runtime > Change runtime type > GPU before running the model-loading cell.
- **Large file downloads (roughly >1GB): must go through Colab, not the
  local machine.** Confirmed 2026-09-03: a 6.65GB Hugging Face file stalled
  completely locally (no bytes, no error) with and without HF's Xet backend,
  across multiple retries, while the identical file downloaded via Colab in
  under 2 minutes at ~69MB/s. Small files (a few MB) work fine locally — this
  is specifically a large-transfer issue on the local network. Set
  `os.environ["HF_HUB_DISABLE_XET"] = "1"` before importing `huggingface_hub`
  if a transfer stalls with no progress and no error even inside Colab.

## Datasets
- **Audio, primary: ASVspoof 5**, hosted on Hugging Face at
  `jungjee/asvspoof5`. Large dataset (142GB across many `.tar` shards,
  train/dev/eval) — never download the whole thing. Use `huggingface_hub` to
  list files, then pull only the protocol file and specific dev shards
  (`flac_D_*.tar`). Skip eval shards (official challenge submission only).
  - Protocol file format confirmed against a published paper describing the
    challenge: space-separated columns —
    `speaker_id, filename, gender, [reserved fields], attack_configuration
    (AC1/AC2/AC3), attack_algorithm (A##), label (bonafide/spoof),
    trailing placeholder field`. In code, `parts[1]` = filename, `parts[-2]`
    = label, `parts[-3]` = attack_algorithm — this indexing is confirmed
    correct against that schema, not a guess from example rows.
  - First shard (`flac_D_aa.tar`, 6.65GB) fully downloaded and extracted:
    **47,400 utterances — 36,700 spoof / 10,700 bonafide.** Labels CSV built
    and matched cleanly against this schema.
- **Audio, secondary: ASVspoof 2021 (DF track)** — adds "in the wild"
  channel/codec variety. Requires a short registration form, then direct
  Zenodo download. Not yet started.
- **Audio, fallback only: ASVspoof 2019 (LA)** — six-year-old synthesis
  methods; use only to reproduce a published baseline, not as a primary
  source. Available on Kaggle/HF mirrors with no wait.
- **Video (phase 2): FaceForensics++** (access approved), **FakeAVCeleb**
  (video + matched audio, pending approval) — not yet started; requests
  submitted early so approval isn't a bottleneck when phase 2 begins.
  Recovered from an earlier pre-Colab planning file, not yet acted on:
  primary compression level should be **c40** (not the cleaner c23),
  deliberately chosen because it matches real-world phone/WhatsApp call
  quality — the same reasoning already applied to the audio augmentation
  choices. Download command:
  `python download-FaceForensics.py <output_path> -d all -c c40 -t videos
  --server EU2` (run with `--num_videos 20` first as a sanity check before
  the full pull). Also pull the **Deep Fake Detection Dataset (DFD)** —
  bundled via the same script (`-d DeepFakeDetection` and
  `-d DeepFakeDetection_original`) — and use it specifically as an
  **independent generalization test set**: train on FaceForensics++'s 4
  manipulation methods, evaluate separately on DFD, which was generated
  with Google/Jigsaw's own separate pipeline. This is a concrete, ready
  answer to the FF++ generalization-gap concern already documented above
  under "Phase 2 kickoff findings" — a real non-FF++ held-out check,
  available through the same download step, not something to source
  separately.

## Data storage
- **Pattern: bulk raw data stays ephemeral on Colab's local disk
  (`/content/...`); only small derived outputs (labels CSVs, results,
  metrics, checkpoints) get copied to Drive for persistence.** Re-downloading
  a shard from Hugging Face via Colab takes under 2 minutes, so nothing is
  lost by not persisting raw audio.
- Google Drive is on a **200GB paid tier** (upgraded from the free 15GB,
  which had been hit by keeping full raw shards in Drive). This solved the
  storage-quota problem specifically — it does **not** address Colab runtime
  disconnects or Colab's own local-disk/session limits (see Current status
  below — these are separate problems).
- Folder structure: `data/audio/asvspoof5/`, `data/audio/asvspoof2021/`,
  `data/audio/asvspoof2019/`, `data/video/faceforensics/`,
  `data/video_audio/fakeavceleb/` (Drive, for small persisted outputs only).
- Kaggle-hosted sets: attach via Kaggle's data browser in a Kaggle Notebook,
  or the Kaggle API from Colab.
- Google-form-gated sets (FakeAVCeleb, FaceForensics++): mount via Drive
  shortcut if given a share link, or run the provided download script inside
  Colab pointed at the working data folder.

## Data handling requirements
- Resample all audio to 16kHz, mono.
- Trim silence, normalize volume.
- Cut/pad into fixed-length windows (AASIST3 expects 64,600 samples ≈ 4s).
- Fix class imbalance (this shard is ~77% spoof / 23% bonafide) via class
  weighting or resampling — don't train or evaluate as if it's balanced.
- Augment training data with realistic phone-call/compression artifacts and
  background noise (audiomentations) — models trained only on clean lab
  audio generalize poorly to real-world recordings.
- Keep a held-out test set of self-recorded, realistic (noisy/compressed)
  samples, separate from the clean benchmark split, to check real-world
  generalization honestly.

## Evaluation
- Primary metric: Equal Error Rate (EER), the field-standard metric for this
  task.
- Also report a confusion matrix and accuracy on the realistic held-out set —
  never just the clean-benchmark numbers.
- Be explicit about where the model fails (which attack/synthesis types it
  misses) — the existing inference script already breaks results down by
  `attack_algorithm`, keep this.

## Ethics and framing — keep this in any user-facing output
- Never claim or imply the model "guarantees" scam/fraud prevention. Outputs
  are a risk score/signal, not a verdict.
- User-facing text should read as calm and reassuring, not alarming.

## Background (for code style and approach suggestions)
- Comfortable with Python, scikit-learn, TensorFlow, and ensemble methods
  (bagging/boosting/model averaging) — prior experience building a
  98%-accuracy ensemble classifier for SQL-injection detection.
- Prefer fine-tuning or directly reusing pretrained checkpoints over training
  architectures from scratch, especially now that a confirmed working
  pretrained model (AASIST3) exists for this exact task.

## Decision: Phase 1 frozen for prototype purposes, moving to Phase 2
Zero-shot AASIST3 (19.23% EER, 80.79% calibrated accuracy) is accepted as
good enough for a presentable hackathon prototype. Deliberately deferred
until after a working end-to-end prototype exists (not abandoned — queued):
- Fine-tuning AASIST3's back-end further / dataset expansion / feature
  engineering, targeting a stated 90%+ accuracy goal (metric definition —
  at what threshold, weighted how given asymmetric error costs — still
  needs pinning down when this is revisited).
- Suspicious-timestamp highlighting via attention/Grad-CAM-style saliency
  (confirmed technically real for this architecture family via published
  work, but only localizes *where*, not human-readable *why* — models in
  this space are documented to often attend to non-semantic features, so
  semantic explanations like "sounds too generic" would need a separate,
  clearly-labeled heuristic acoustic-feature layer, not something read out
  of AASIST3 itself).
- Speech-to-text + scam-content classification as a second, independent
  signal, fused with voice-authenticity at the decision layer (same
  late-fusion pattern as the planned Phase 3 audio+video fusion). Confirmed
  real scam-classifiers exist (e.g. BERT/BiLSTM/CNN models fine-tuned on
  labeled scam-call transcripts) but they operate on transcripts, not raw
  audio — this adds an ASR step, plus added latency and a more sensitive
  privacy posture (transcribing call content) than acoustic-only analysis.
  Some published metrics for these classifiers look like small-eval-set
  overfitting (e.g. 100% accuracy/recall on an 80-example test set) and
  would need independent re-verification, not blind trust, before use —
  same caution that caught the AASIST3 label-inversion issue.
- Confidence-score *display*, however, is cheap and not blocked: the model
  already outputs a continuous score; presenting it as a risk tier (Low/
  Medium/High, banded around the confirmed EER threshold) rather than a
  raw hard label is a UI change, not new modeling work, and can go in the
  prototype now if useful. A properly calibrated literal percentage (e.g.
  via Platt scaling) is the deferred, more-work version of this.

## Current status (as of this update)
- Dev shard `flac_D_aa` (47,400 files, 36,700 spoof / 10,700 bonafide)
  downloaded via Colab, extracted successfully, labels CSV built and
  matched against the confirmed protocol schema.
- Redundant/buggy verification cell (checked a Drive path the extraction
  step never wrote to — leftover from an earlier Drive-direct-extraction
  design) removed. Extraction made resumable (copy tar to local disk first,
  skip already-extracted members) after Colab disconnects during extraction
  were traced to reading the large `.tar` over the Drive FUSE mount.
- **AASIST3 zero-shot inference completed on the full dev shard. Confirmed
  result: EER = 19.23%, accuracy (default 0.5 threshold) = 81.87%.**
  This is the baseline number to beat with any further fine-tuning.
- **Critical confirmed finding — do not re-assume the model card's label
  convention:** the AASIST3 model card states "0 = bonafide, 1 = spoof," but
  for this loaded checkpoint (`MTUCI/AASIST3` via the `aasist3` custom
  loader) the actual output order is inverted from that description. Initial
  inference using the card's stated convention produced EER = 80.77% (a
  result mathematically impossible to read as "the model is just bad" —
  EER above 50% indicates strong separation read backward, not weak
  separation). Swapping the two output indices (`predicted = 1 -
  raw_predicted`, `bonafide_score = 1 - raw_bonafide_score`) dropped EER to
  19.23% and flipped per-attack-type accuracy from ~0% to ~100% across every
  spoof type (A09–A16) — strong, multi-signal confirmation this was the
  correct fix, not a coincidence. Any future code loading this same
  checkpoint must apply this swap; do not trust the model card's stated
  index order at face value for this specific pipeline.
- **Open finding, not yet resolved:** at the default 0.5 softmax threshold,
  bonafide recall is poor (24.2% — most genuine audio is wrongly flagged as
  fake) even though spoof recall is near-perfect (98.7%) and overall EER is
  reasonable. This reflects threshold miscalibration at the default cutoff,
  not necessarily a flaw in the model's underlying score separation —
  EER is threshold-independent and already accounts for this, which is why
  published work on this exact dataset treats EER as the trustworthy metric
  and raw argmax accuracy as unreliable under this class imbalance (77%
  spoof / 23% bonafide in this shard). Recomputing predictions at the
  EER-optimal threshold (rather than default argmax) is the next step if a
  single calibrated accuracy number is needed for a real notify/don't-notify
  decision later — not yet done.
- Benchmark context for the confirmed 19.23% EER (from published results on
  this exact dataset, not this project's own work): challenge-baseline
  systems without a pretrained SSL front-end scored 25–40%+ EER on
  ASVspoof5; fine-tuned Wav2Vec2/HuBERT-front-end + AASIST-family back-ends
  in published papers reached roughly 1.5–8% EER. 19.23% zero-shot sits
  above the fine-tuned range, as expected for a checkpoint not fine-tuned
  on this specific data — fine-tuning only the back-end classifier (keeping
  the Wav2Vec2 front-end frozen, per a published ablation showing frozen
  clearly outperforms end-to-end fine-tuning on this exact dataset: 8.8%
  vs 21.7% EER) is the evidence-backed next lever if a lower EER is wanted,
  rather than training a new architecture from scratch.
