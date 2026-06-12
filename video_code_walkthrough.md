# DiffTrack — 5-Minute Code Walkthrough Guide

> **Purpose:** Screen-by-screen guide for a 5-minute video explanation of the code.  
> **Emphasis:** What we added/changed to generate results, not just what the paper describes.  
> Each section lists the file + lines to show on screen and the talking points.

---

## [0:00 — 0:30] · Overview: What the codebase is

**Show on screen:** File tree in terminal
```
DiffTrack/
├── evaluate_tapvid.py          ← our main experiment runner
├── diffusers/
│   └── src/diffusers/
│       ├── models/
│       │   ├── attention_processor.py   ← modified: Q/K capture hook
│       │   └── transformers/cogvideox_transformer_3d.py  ← modified: layer routing
│       └── pipelines/cogvideo/
│           └── pipeline_cogvideox_tracking.py  ← modified: timestep hook
├── results/
│   └── *.log                   ← our raw experimental output
└── motion_guidance.py          ← CAG demo (separate script)
```

**Say:** "The project has three layers. The frozen CogVideoX model lives in diffusers. We modified three specific files inside it — the attention processor, the transformer forward pass, and the pipeline denoising loop — to intercept query and key tensors without touching the weights. `evaluate_tapvid.py` is the script we ran for all 13 experiments by changing just two command-line flags."

---

## [0:30 — 1:15] · Layer 1: Attention processor hook
**File:** `diffusers/src/diffusers/models/attention_processor.py`  
**Lines:** ~3324–3326  

**Show on screen:**
```python
# ADDED: conditional Q/K capture
if args['query_key']:
    self.query = query
    self.key = key
```

**Say:** "This is the lowest-level change. The standard diffusers attention processor just computes attention and throws Q and K away. We added this three-line block: if the `query_key` flag is set, store Q and K as attributes on the processor object so the pipeline can read them later. That's it — no weight changes, no new modules, just a conditional save. The `args` dict is passed all the way through the transformer forward pass so we can toggle this per-layer."

**Show on screen:** Also point out line ~3767:
```python
# Same pattern in the HunyuanVideo processor (separate class)
self.query = query
self.key = key
```
"There's an analogous save in the HunyuanVideo processor — same idea, different model family."

---

## [1:15 — 2:00] · Layer 2: Transformer layer routing
**File:** `diffusers/src/diffusers/models/transformers/cogvideox_transformer_3d.py`  
**Lines:** ~500–507  

**Show on screen:**
```python
for i, block in enumerate(self.transformer_blocks):  # 42 blocks total
    # ADDED: only enable Q/K capture for the target layer(s)
    save_descriptor = True if i in args['matching_layer'] else False

    args['feature'] = save_descriptor
    args['query_key'] = save_descriptor     # ← turns on the hook above

    hidden_states, encoder_hidden_states = block(
        hidden_states=hidden_states,
        ...
        args=args
    )
```

**Say:** "CogVideoX-2B has 42 transformer blocks. We added four lines in the forward loop. Each iteration, `save_descriptor` is True only when block index `i` is in the target layer list. That flips `args['query_key']` on or off before calling the block. So only the nominated layer stores Q and K — the rest run normally. This is how `--matching_layer 17` or `--matching_layer 8` works: it just changes which block index evaluates to True."

**Show on screen:** Show terminal, point to the args dict being passed through:
```python
PARAMS = {
    'trajectory': False,
    'attn_weight': False,
    'query_key': True,         # master switch
    'head_matching_layer': -1
}
```
"This dict in `evaluate_tapvid.py` is the master config that gets handed to the transformer every forward pass."

---

## [2:00 — 2:50] · Layer 3: Pipeline timestep hook
**File:** `diffusers/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_tracking.py`  
**Lines:** ~762–796  

**Show on screen:**
```python
queries, keys = [], []

for i, t in enumerate(timesteps):    # 50 denoising steps total
    if i < inverse_step or i > max(matching_timestep):
        continue                      # skip steps outside the window

    noise_pred = self.transformer(... args=params)  # forward pass

    if i in matching_timestep:        # ADDED: harvest Q/K at this step
        with torch.no_grad():
            for l in matching_layer:
                blk = self.transformer.transformer_blocks[l]
                Q = blk.attn1.processor.query[1]  # [1] = video tokens only
                K = blk.attn1.processor.key[1]
                queries.append(Q)
                keys.append(K)
                del blk.attn1.processor.query     # free VRAM immediately
                del blk.attn1.processor.key

return (video, queries, keys, text_seq_len)
```

**Say:** "The pipeline runs 50 denoising steps. We added the block after the transformer call: if the current step index `i` is in `matching_timestep`, read Q and K off the processor we saved a moment ago — index `[1]` strips the text conditioning tokens, keeping only video tokens. Then delete them to free VRAM. This is what `--matching_timestep 49` or `--matching_timestep 30` controls: step 49 is near-clean latents, step 1 is near-maximum noise — same loop, different index."

---

## [2:50 — 3:40] · Core computation: QK cross-frame attention
**File:** `evaluate_tapvid.py`  
**Lines:** ~305–325  

**Show on screen:**
```python
B, head_dim, qk_len, _, _ = query_frames.shape

for k in range(1, qk_len):            # k = each non-reference frame
    # Frame 0 → Frame k (forward correspondence)
    attn_tts = torch.einsum(
        "b h i d, b h j d -> b h i j",
        query_frames[:, :, 0, :, :],   # Q from frame 0
        key_frames[:, :, k, :, :]      # K from frame k
    ) / math.sqrt(head_dim)

    # Frame k → Frame 0 (backward correspondence)
    attn_stt = torch.einsum(
        "b h i d, b h j d -> b h i j",
        query_frames[:, :, k, :, :],
        key_frames[:, :, 0, :, :]
    ) / math.sqrt(head_dim)

    attn_tts = attn_tts.softmax(dim=-1).mean(1)  # average over heads
    attn_stt = attn_stt.softmax(dim=-1).mean(1)

    # Symmetrize: average forward and backward
    correlation = (attn_tts + attn_stt.T_like()) / 2
```

**Say:** "This is the mathematical core. Q and K have shape [batch, heads, spatial, dim]. The einsum computes the dot product between every patch in frame 0 and every patch in frame k — this is a dense correspondence map. We compute it both ways (frame 0 → k and k → 0), softmax each, average over heads, then symmetrize by averaging both directions. The result is a heatmap: for each query point in frame 0, the peak of this heatmap is the best matching location in frame k."

---

## [3:40 — 4:15] · How we ran the experiments: the CLI
**Show on screen:** Terminal with example commands

```bash
# Paper default (our baseline reproduction)
python evaluate_tapvid.py \
    --model cogvideox_t2v_2b \
    --matching_layer 17 \
    --matching_timestep 49 \
    --chunk_frame_interval --average_overlapped_corr \
    --tapvid_root /path/to/davis

# Layer ablation — change ONE flag, everything else identical
python evaluate_tapvid.py --matching_layer 8  --matching_timestep 49 ...
python evaluate_tapvid.py --matching_layer 5  --matching_timestep 49 ...
python evaluate_tapvid.py --matching_layer 29 --matching_timestep 49 ...

# Timestep ablation — change ONE flag
python evaluate_tapvid.py --matching_layer 17 --matching_timestep 1  ...
python evaluate_tapvid.py --matching_layer 17 --matching_timestep 10 ...
python evaluate_tapvid.py --matching_layer 17 --matching_timestep 30 ...

# Chunking experiment — add/remove ONE flag
python evaluate_tapvid.py --matching_layer 17 --matching_timestep 49 \
    --chunk_frame_interval ...              # with chunking: 47.0
python evaluate_tapvid.py --matching_layer 17 --matching_timestep 49 \
    # (no flag) ...                         # without chunking: 46.4
```

**Say:** "Every one of our 13 experiments is a single command where we changed exactly one argument. That's what makes this a clean ablation study: the code is identical — only the layer index or timestep index changes. The output directory is auto-named after the parameters, so `layer17_timestep49_noiseFalse/log.txt` contains the full metrics for that run. Show the results directory — each folder is one experiment."

**Show on screen:** `ls results/` output showing log file names

---

## [4:15 — 4:35] · New experiments: proving limitations with code changes
**Files:** `utils/evaluation.py` lines 149, 177–178 · `evaluate_tapvid.py` lines 390–412

**Show on screen — `utils/evaluation.py` change (1 line):**
```python
# BEFORE (zero_shot mode suppressed AJ and OA):
message = f"Video {self.cnt} ({video_len} frames)| delta_avg: {delta:.2f} | ..."

# AFTER — we un-suppressed AJ and OA:
message = f"Video {self.cnt} ({video_len} frames)| delta_avg: {delta:.2f} | ... | AJ: {aj:.2f} | OA: {oa:.2f}"
```

**Show on screen — new per-frame analysis block in `evaluate_tapvid.py`:**
```python
# ADDED: per-frame-distance analysis
bins = [(1, 10), (11, 20), (21, 30), (31, 50), (51, num_frames - 1)]
for b_start, b_end in bins:
    for t in range(b_start, b_end + 1):
        vis_t = ~gt_occluded[0, :, t]
        dist_sq = np.sum((pred_tracks[0, :, t, :] - gt_tracks[0, :, t, :]) ** 2, axis=-1)
        total_correct += int(np.sum((dist_sq < 16) & vis_t))  # delta_4: 4^2=16
        total_visible += int(np.sum(vis_t))
```

**Show on screen — result from `results/limitation_proof_vis/…/frame_distance_log.txt`:**
```
Video 0 (69 frames) | per-frame-distance delta_4:
  frames  1-10: delta_4 = 63.2%   ← near reference: good
  frames 11-20: delta_4 = 39.2%
  frames 21-30: delta_4 = 33.3%
  frames 31-50: delta_4 = 29.2%
  frames 51-68: delta_4 = 14.7%   ← 50+ frames away: 4× worse
```

**Say:** "We wrote two sets of code changes — about 25 lines total. The first un-suppressed metrics that the evaluator was already computing but not logging. That gave us AJ: 28.2 vs delta_avg: 45.9 — a 38% gap, proving the method is blind to occlusion. The second added per-frame binned accuracy. Twenty-five lines of new code, two GPU runs, two provable limitations. That's the whole contribution of the limitations section."

---

## [4:35 — 4:55] · Reading a result: new log format
**File:** `results/limitation_proof_vis/layer[17]_timestep[49]_noiseFalse/log.txt`

**Show on screen:**
```
Video 0 (69 frames)| delta_avg: 38.18 | delta_1: 1.87 | delta_2: 11.10 |
                     delta_4: 36.03 | delta_8: 65.09 | delta_16: 76.81 |
                     AJ: 20.07 | OA: 65.52    ← new columns

Mean delta_avg: 45.9
Mean AJ: 28.2          ← 38% below delta_avg
Mean OA: 74.5%         ← 25.5% of occluded frames predicted wrong
```

**Say:** "The new log format shows AJ and OA alongside the existing delta metrics. Mean AJ of 28.2 tells us the method would rank much lower on the full TAP-Vid leaderboard if AJ were the primary metric. OA of 74.5% sounds okay, but it means 1 in 4 occluded frames is confidently predicted as visible — which could cause real downstream failures in robotics or depth-estimation pipelines."

---

## [4:55 — 5:00] · Summary of what we changed

**Show on screen:** Side-by-side diff summary (can just be the three-bullet list below in large text)

```
What we added / changed (full list):

  diffusers/attention_processor.py          →  2 lines: store Q and K when flag is set
  diffusers/cogvideox_transformer_3d.py     →  4 lines: per-layer routing of the flag
  diffusers/pipeline_cogvideox_tracking.py  →  15 lines: harvest stored Q/K after each step
  utils/evaluation.py                       →  3 lines: un-suppress AJ and OA in zero_shot mode
  evaluate_tapvid.py                        →  22 lines: per-frame-distance binned accuracy

Everything else (correlation math, TAP-Vid metrics, visualization)
is original paper code in evaluate_tapvid.py and utils/.

13 experiments = 1 script × changing 1–2 flags per run.
```

**Say:** "The paper's key insight is that tracking is already inside a frozen diffusion model — you just need to reach in and pull out Q and K at the right layer and timestep. Our implementation makes that concrete: two lines to save, four lines to route, fifteen lines to harvest. Every experiment result on the project page comes directly from these three files and a log entry. That's the whole system."

---

> **Notes for recording:**  
> - Use a dark terminal theme (matches the project page aesthetic)  
> - Show actual file paths in the terminal so viewers can find the code  
> - For the CLI section, run `ls results/` live to show the experiment folders  
> - The QK einsum block is the single most important thing to linger on — zoom in if possible