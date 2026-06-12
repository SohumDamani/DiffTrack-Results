# 5-Minute Presentation Script — DiffTrack EE243 Final Project

---

## [0:00 — 0:40 · Hero]

"Our project is a reproduction and ablation study of DiffTrack, a NeurIPS 2025 paper that asks: can a video diffusion model track points in a video without ever being trained to do so?

The idea is elegant — inside a frozen CogVideoX-2B model, the query-key attention matrices in the middle transformer layers happen to encode which pixels correspond across frames. No labels. No fine-tuning. Just emergent geometry from a model trained only to generate video.

You can see it working right here in the hero — colored trajectories following real objects zero-shot. Three numbers summarize our 13 experiments: we reproduced the paper's baseline at 47.0, observed a complete collapse at ts=1, and measured a 12× accuracy spread across precision thresholds."

---

## [0:40 — 1:00 · Setup]

"Quick setup note before results. All experiments use TAP-Vid DAVIS — 30 densely annotated real-world video sequences. Due to compute constraints, each ablation run used a fixed 4-video subset. The primary metric, delta_avg, is the average accuracy at five distance thresholds: within 1, 2, 4, 8, and 16 pixels of ground truth. Higher is better. Our 4-video baseline came in at 47.0, vs the paper's 46.3 on all 30 — within expected variance for a small sample."

---

## [1:00 — 2:00 · Part A — Layer Ablation]

"We ran ablations over two axes: which transformer layer to read attention from, and which diffusion timestep to use.

For layers, we tested five. Watch the three videos side by side. Layer 5 is shallow — misses semantic structure, gets 31.3. Layer 8 has a positional bias problem: RoPE embeddings dominate, so it matches the *same spatial location* instead of the actual moving object — gets 41.9, but the trajectories are wrong in a specific way. Layer 17 is the paper's default and is confirmed best at 47.0 — sharp, precise trajectories that follow real objects.

The bar chart shows the full picture. Mid-network layers win because they balance spatial precision with learned semantic abstraction. Too shallow — no semantics. Too deep — spatial detail is gone."

---

## [2:00 — 2:50 · Part A — Timestep + Chunking]

"For timestep, the paper runs at ts=49 — near the end of denoising, almost clean latents. We swept from ts=1 all the way to ts=49 across six points.

Top-left video is ts=1 — completely dead. Top-right is ts=10 — barely functional at 13.1. Bottom row, ts=30 and ts=49, both produce solid tracking in the 47–48 range. They're in an effective plateau — the 0.8-point difference is within small-sample noise.

The curve independently replicates paper Figure 4(c) — sharp failure cliff below ts=10, stable plateau above ts=30. Paper's ts=49 is confirmed.

Experiment 3 is quick: we tested the chunk_frame_interval windowing heuristic the paper uses for long videos. Without it: 46.4. With it: 47.0. A 0.6-point difference — marginal, within noise. The windowing can't fix the absence of temporal memory, which brings us to Part B."

---

## [2:50 — 4:10 · Part B — Limitations]

"Four limitations — the first two reproduce what the paper already knows, the last two are findings we generated ourselves with new experiments.

**Limitation 1**: ts=1. The pipeline feeds clean video latents to a transformer conditioned at t=980 — distribution mismatch. Result: every metric is 0.0 across all four videos. Trajectories freeze or scatter randomly. Correspondence is emergent from denoising; remove the noise and there is nothing to track.

**Limitation 2**: Layer sensitivity. 16-point spread across five layers with no principled selection criterion. Layer 8 tracks the position where the object *started at frame zero* — RoPE positional bias, paper Figure 6. Layer 29 is diffuse and scattered — paper Figure A.18.

**Limitation 3 — our new experiment**: Frame-distance degradation. We added 20 lines of code to compute delta_4 accuracy in bins of 10 frames. Near frame 0: 63% accuracy. Beyond frame 50: 15% accuracy. A 4× drop over a single 69-frame video. This directly proves no temporal memory — the method re-matches from scratch against frame 0 every time, and correspondences degrade as objects drift away. Look at the right video and watch the trajectories diverge from GT as the video progresses.

**Limitation 4 — our new experiment**: Occlusion blindness. Line 385 of evaluate_tapvid.py: `pred_occluded = torch.zeros_like(visibility)` — hardcoded all-zeros, always predicts visible. The standard delta metric ignores this because it only evaluates on visible ground-truth frames. But Average Jaccard — the stricter TAP-Vid metric — penalizes wrong visibility predictions. We un-suppressed AJ reporting and re-ran. Mean AJ = 28.2 vs delta_avg = 45.9 — a 38% gap. Videos 0 and 2 drop to 20–23 AJ because they have real occlusions that the method confidently tracks through."

---

## [3:50 — 4:35 · Part C — CAG Demo]

"Part C is Cross-Frame Attention Guidance — the same attention maps repurposed for motion transfer rather than tracking.

Left video is unguided generation. The model produces plausible but uncontrolled motion. Right video has CAG applied — during denoising, the cross-frame keys and values from a reference video replace those in the target generation. The output motion steers toward the reference pattern. Same frozen model, zero training, zero labels.

This demonstrates the generality of the mechanism: the query-key correspondence in video DiTs isn't a tracking-specific artifact — it's a fundamental property of how these models organize temporal information. That's why it transfers directly to motion guidance."

---

## [4:35 — 5:00 · Close]

"To summarize — DiffTrack is real and reproducible. We confirmed the paper's baseline at 47.0 and ran 13 ablation experiments. Beyond reproducing the paper, we ran two original experiments: we proved frame-distance degradation empirically — 4× accuracy drop from near to far frames — and we un-suppressed AJ reporting to show a 38% gap between delta_avg and the stricter occlusion-aware metric. Those are not just observations; they required writing new evaluation code and running new GPU jobs.

All code, result videos, and raw logs are on GitHub. Every number is verified against raw run logs. Thanks."
