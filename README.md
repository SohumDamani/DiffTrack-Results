<div align="center">
<h1>
<img src="assets/difftrack.png" alt="Project Logo" style="height: 1em; vertical-align: middle; margin-right: 10px;">
 Emergent Temporal Correspondences from Video Diffusion Transformers</h1>

[**Jisu Nam**](https://scholar.google.com/citations?hl=&user=xakYe8MAAAAJ)<sup>*1</sup>, [**Soowon Son**](https://scholar.google.com/citations?hl=&user=Eo87mRsAAAAJ)<sup>*1</sup>, [**Dahyun Chung**](https://scholar.google.com/citations?hl=&user=EU52riMAAAAJ)<sup>2</sup>, [**Jiyoung Kim**](https://scholar.google.co.kr/citations?hl=&user=DqG-ybIAAAAJ)<sup>1</sup>, [**Siyoon Jin**](https://scholar.google.com/citations?hl=&user=rXRHxkwAAAAJ)<sup>1</sup>, [**Junhwa Hur**](https://scholar.google.com/citations?hl=&user=z4dNJdkAAAAJ)<sup>&dagger;3</sup>, [**Seungryong Kim**](https://scholar.google.com/citations?hl=&user=cIK1hS8AAAAJ)<sup>&dagger;1</sup>

<sup>1</sup>KAIST AI&emsp;&emsp;&emsp;&emsp;<sup>2</sup>Korea University&emsp;&emsp;&emsp;&emsp;<sup>3</sup>Google DeepMind


<sup>*</sup> Equal contribution. <sup>&dagger;</sup>Co-corresponding author.

<h3>NeurIPS 2025</h3>

<a href="https://arxiv.org/abs/2506.17220"><img src='https://img.shields.io/badge/arXiv-DiffTrack-red' alt='Paper PDF'></a>
<a href="https://cvlab-kaist.github.io/DiffTrack/"><img src='https://img.shields.io/badge/Project_Page-DiffTrack-green' alt='Project Page'></a>

**Final project dashboard:** https://sohumdamani.github.io/DiffTrack-Results/

**GitHub repository:** https://github.com/SohumDamani/DiffTrack-Results



</div>

### 🔍 How do Video Diffusion Transformers (Video DiTs) learn and represent temporal correspondences across frames?

To address this fundamental question, we present 

**DiffTrack** - a unified framework for uncovering and exploiting emergent temporal correspondences in video diffusion models. DiffTrack introduces:

**📊 Novel Evaluation Metrics** specifically designed to quantify temporal correspondence in video DiTs.

**🚀 Two Practical Applications**
- [**Zero-shot Point Tracking**](#2-zero-shot-point-tracking)  achieving state-of-the-art (SOTA) performance.
- [**Motion-Enhanced Video Generation**](#3-cross-attention-guidance-cag) via a novel Cross-Attention Guidance (CAG) technique.




## Installation

```bash
git clone https://github.com/cvlab-kaist//DiffTrack.git
cd DiffTrack

conda create -n difftrack python=3.10 -y
conda activate difftrack
pip install -r requirements.txt

cd diffusers
pip install -e .
```
</br>

# 1. Correspondence Analysis in Video DiTs

### Analysis on Generated Videos
We provide correspondence analysis across several video backbone models: CogVideoX-2B, CogVideoX-5B, HunyuanVideo, CogVideoX-2B-I2V, and CogVideoX-5B-I2V.

Additional analysis scripts are available in the `scripts/analysis` directory. 

```bash
model=cogvideox_t2v_2b
scene=fg
python analyze_generation.py \
    --output_dir ./output \
    --model $model --video_mode $scene --num_inference_steps 50 \
    --matching_accuracy --conf_attn_score \
    --vis_timesteps 49 --vis_layers 17 \
    --vis_attn_map --pos_h 16 24 --pos_w 16 36 --vis_track \
    --txt_path ./dataset/$model/$scene/prompt.txt \
    --track_dir ./dataset/$model/$scene/tracks \
    --visibility_dir ./dataset/$model/$scene/visibility \
    --device cuda:0
```

#### Key Options

- `--model`: Supported models include `cogvideox_t2v_2b`, `cogvideox_t2v_5b`, `cogvideox_i2v_2b`, `cogvideox_i2v_5b`, `hunyuan_t2v`.
- `--video_mode`: Set to `fg` for object-centric or `bg` for scenic videos.
- `--matching_accuracy`: Computes matching accuracy using both query-key and intermediate features.
- `--conf_attn_score`: Computes confidence score and attention score.
- `--vis_attn_map`: Aggregates cost maps for attention visualization.
- `--vis_track`: Visualizes trajectory using query-key descriptors.



*This script should reproduce videos in the `sample` directory.*

</br>

### Analysis on Real Videos (TAP-Vid-DAVIS)
We provide correspondence analysis across several video backbone models: CogVideoX-2B, CogVideoX-5B. 

Additional analysis scripts are available in the `scripts/analysis` directory. 

```bash
python analyze_real.py \
    --output_dir ./output \
    --model cogvideox_t2v_2b --num_inference_steps 50 \
    --matching_accuracy --confidence_attention_score \
    --resize_h 480 --resize_w 720 \
    --eval_dataset davis_first --tapvid_root /path/to/data \
    --device cuda:0
```
</br>


# 2. Zero-Shot Point Tracking

### Download Evaluation Dataset

```bash
wget https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip
unzip tapvid_davis.zip
```

For TAP-Vid-Kinetics, please refer to the [TAP-Vid GitHub](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid).


### Run Evaluation
We provide across several video backbone models: CogVideoX-2B, CogVideoX-5B, HunyuanVideo.

Additional evaluation scripts are available in the `scripts/point_tracking` directory.

```bash
model=cogvideox_t2v_2b
python evaluate_tapvid.py \
    --model $model \
    --matching_layer 17 --matching_timestep 49 --inverse_step 49 \
    --output_dir ./output \
    --eval_dataset davis_first --tapvid_root /path/to/data \
    --resize_h 480 --resize_w 720 \
    --chunk_frame_interval --average_overlapped_corr \
    --vis_video --tracks_leave_trace 15 \
    --pipe_device cuda:0
```

#### Chunking Options

- `--chunk_len`: Number of frames per chunk. (default: `13`)
- `--chunk_frame_interval`: Interleave frames to reduce temporal gap.
- `--chunk_stride`: Stride for sliding window. (default: `1`)
- `--average_overlapped_corr`: Average overlapping correlation maps.

#### Cost Map Aggregation

- `--matching_layer`: Transformer layers for descriptor extraction. (e.g., `17` for cogvideox_t2v_2b).
- `--matching_timestep`: Denoising timesteps for descriptor extraction. (e.g., `49` for cogvideox_t2v_2b).

#### Dataset Options

- `--tapvid_root`: Path to TAP-Vid dataset.
- `--eval_dataset`: Choose from `davis_first` and `kinetics_first`
- `--resize_h` / `--resize_w`: Resize video resolution.
- `--video_max_len`: Max length of input video.
- `--do_inversion` / `--add_noise`: Modify inversion strategy.

#### Visualization Options

- `--vis_video`: Visualize trajectories on video.
- `--tracks_leave_trace`: Number of frames for trajectory trail.


</br>


# 3. Cross-Attention Guidance (CAG)

We provide across several video backbone models: CogVideoX-2B, CogVideoX-5B.

Additional motion guidance scripts are available in the `scripts/motion_guidance` directory.

```bash
CUDA_VISIBLE_DEVICES=0 python motion_guidance.py \
    --output_dir ./output \
    --model_version 2b \
    --txt_path ./dataset/cag_prompts.txt \
    --pag_layers 13 17 21 \
    --pag_scale 1 \ 
    --cfg_scale 6
```
#### Key Options
- `--model_version`: Supported cogvideox models include `2b`, `5b`.
- `--pag_layers`: Layers where CAG is applied (e.g., `[13, 17, 21]` for 2B, `[15, 17, 18]` for 5B).
- `--pag_scale`: Cross attention guidance scale (default: `1.0`).
- `--cfg_scale`: Classifier-Free Guidance scale (default: `6.0`).


---

## CV Final Project — Experiment Results (Sohum Damani)

This fork reproduces and extends DiffTrack on TAP-Vid-DAVIS using CogVideoX-2B.

### Result Logs & Metrics
All experiment metric logs (`log.txt` per run) are committed in `results/`. delta_avg = % of tracked points within distance threshold (higher = better).

**Timestep convention**: Code's `--matching_timestep` is a 0-indexed denoising step; `ts=49` = paper's `t=1` (near-clean, optimal). `ts=1` = paper's `t=50` (near-pure noise).

#### Layer and Timestep Ablation (l=17, CogVideoX-2B, TAP-Vid DAVIS)

| Config | Mean delta_avg | Notes |
|---|---|---|
| layer=17, ts=30 | 47.8 | In plateau zone with ts=49 — 0.8pt difference within small-sample noise (paper Fig 4c) |
| layer=17, ts=49 | 47.0 | Paper's recommended config (l=17, t=1); paper reports 46.3 on full DAVIS |
| layer=17, ts=49, no-chunk | 46.4 | No chunked sliding window; marginal regression |
| layer=17, ts=20 | 33.4 | Degraded at moderate noise |
| layer=17, ts=10 | 13.1 | High noise — severe degradation (paper Fig 4c) |
| layer=17, ts=5 | 0.8 | Near-noise — near-collapse (paper Fig 4c) |
| layer=17, ts=1 | 0.0 | Near-pure noise — complete failure (paper Fig 4c, t≈50) |

#### Layer Ablation (ts=49 = paper's t=1)

| Config | Mean delta_avg | Notes |
|---|---|---|
| layer=5, ts=49 | 31.3 | Shallow layer — less semantic correspondence |
| layer=8, ts=49 | 41.9 | Positional bias due to RoPE (paper Fig 6) |
| layer=17, ts=49 | 47.0 | Optimal layer (paper Table 1, Fig A.18) |
| layer=27, ts=49 | 37.0 | Over-deep layer — degraded |
| layer=29, ts=49 | 38.0 | Bottom-3 layer, diffuse attention (paper Fig A.18) |

### Video Visualizations
Tracking visualizations (~111 MP4 files, ~6.5 GB) are hosted on Google Drive:

**[View Result Videos on Google Drive](https://drive.google.com/drive/folders/1XM9ZF2WiHAUeXJieYLo7NUz8q9MpGAOv?usp=sharing)**

Videos are organized by experiment: `param_study/`, `limitations/`, `new_experiments/`, `full_eval_davis/`, and `cag_demo/`.

---

## Research Lineage

DiffTrack is best understood as a convergence point rather than a flat list of related papers. Three research threads flow into it, and several successor threads flow out of it:

```text
Self-supervised temporal correspondence
        |
        |  Space-Time Correspondence as a Contrastive Random Walk
        |  TAP-Vid, CoTracker, CoTracker3, Particle Video Revisited
        v
Diffusion feature reuse  ----->  DiffTrack  ----->  Point prompting and tracking control
        |                       /    |    \         DiTFlow motion transfer
        |                      /     |     \        ZeroTrail and weighted cross-frame attention
        v                     v      v      v       restoration and deraining consistency
Semantic correspondence   layer choice   timestep choice
```

At a graduate level, the important point is that DiffTrack does not merely apply a pretrained model to tracking. It reinterprets cross-frame attention inside a video diffusion transformer as a transport operator over visual evidence. That operator can move point identity, motion cues, and temporal correspondence through the denoising process without supervised tracker training. This is why the ablation results are meaningful: layer selection, timestep selection, and chunking are not just engineering knobs; they probe where the model stores geometry, when denoising produces useful correspondence, and whether temporal consistency is actually solved by the architecture.

### Predecessor Papers: What DiffTrack Inherits

**Emergent Correspondence from Image Diffusion (Tang et al., NeurIPS 2023)** gives the direct representational foundation. It shows that diffusion features can contain geometric correspondence before any explicit tracker is trained. DiffTrack extends this idea from image-level correspondence into video sequences.

**Space-Time Correspondence as Contrastive Random Walk (Jabri et al., NeurIPS 2020)** supplies the temporal logic. A correspondence is not strong simply because it matches locally; it must remain coherent when propagated across time. This motivates evaluating long-range consistency instead of only short-range similarity.

**Semantics Meets Temporal Correspondence (Qian et al., ICCV 2023)** clarifies the semantic tradeoff. Semantic abstraction can preserve object identity, but it can also blur point-level location. That tension directly explains why layer choice matters in DiffTrack.

**Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance (Ahn et al., ECCV 2024)** contributes the idea that attention can be perturbed or guided during diffusion sampling. DiffTrack's Cross-Attention Guidance belongs to this family of attention-control methods.

**Diffusion Model for Dense Matching (Nam et al., 2023)** and **Unsupervised Semantic Correspondence via Stable Diffusion (Hedlin et al., NeurIPS 2023)** motivate query-key analysis. They support the claim that diffusion representations are not only generative features; they can also serve as correspondence descriptors.

**TAP-Vid, CoTracker, CoTracker3, and Particle Video Revisited** provide the tracking context. They define the benchmark pressure: long-range point tracking needs stability through occlusion, deformation, and appearance change. DiffTrack is interesting because it attempts this without supervised point-tracker training.

**CATs: Cost Aggregation Transformers (Cho et al., NeurIPS 2021)** and **Neural Matching Fields (Hong et al., NeurIPS 2022)** supply the broader correspondence machinery: transformer-based cost aggregation, implicit matching fields, and confidence reasoning.

### Forward Lineage: What Later Work Extracts

**Point Prompting: Counterfactual Tracking with Video Diffusion Models (Shrivastava et al., ICLR 2026)** turns DiffTrack's implicit tracking signal into an interactive point-propagation mechanism.

**DiTFlow: Video Motion Transfer with Diffusion Transformers (Pondaven et al., CVPR 2025)** extracts motion flow from cross-frame attention maps. This directly develops the idea that attention is not merely explanatory; it can become a motion representation.

**ZeroTrail: Zero-Shot Trajectory Control for Video Diffusion Models (Lu et al., NeurIPS Workshop)** extends Cross-Attention Guidance into a trajectory-control system. In other words, the attention mechanism becomes a steering interface.

**Enhancing Video Consistency in Zero-Shot T2V via Weighted Cross-Frame Attention (Wang et al., 2025)** weights cross-frame attention to stabilize long generated videos, using the same core insight that temporal identity can be managed through attention.

**Zero-Shot Video Restoration with Video DiMs (Cao et al., 2026)** and **Zero-Shot Video Deraining with Video Diffusion Models (Varanka et al., WACV)** apply temporal correspondence to restoration tasks. Instead of tracking for its own sake, they use correspondence to preserve consistency under degradation.

**Investigating Cross-Attention for Zero-Shot Editing of T2V Models (Motamed et al., CVPR Workshop 2024)** and **VDT: General-Purpose Video Diffusion Transformers via Mask Modeling (Lu et al., 2025)** show the broader architectural consequence: once cross-frame attention is treated as a temporal structure carrier, it influences editing, generation, and model design.

The takeaway is that DiffTrack sits at a methodological bridge. Before it, the central question was whether representations contain correspondence. After it, the question becomes how cross-frame attention can be controlled, weighted, prompted, or extracted to move visual evidence across time.

---

### Citing this Work
Please use the following bibtex to cite our work:
```
@misc{nam2025emergenttemporalcorrespondencesvideo,
    title={Emergent Temporal Correspondences from Video Diffusion Transformers},
    author={Jisu Nam and Soowon Son and Dahyun Chung and Jiyoung Kim and Siyoon Jin and Junhwa Hur and Seungryong Kim},
    year={2025},
    eprint={2506.17220},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2506.17220},
}
```
