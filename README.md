# SWAG-Video LLM

**Spatiotemporal Warping with Attention Guidance for Video Multimodal Large Language Models**

SWAG is a training-free preprocessing method for video question answering. It extracts query-conditioned spatial attention from a frozen Video-LLM, stabilizes the attention maps across frames with dense optical flow, and warps each frame so that more of the model's visual-token budget is allocated to task-relevant regions without cropping away the surrounding scene.

Developed at the University of California, Irvine by **Kantha Vikas Gowda, Adithya Rajendra, and Sai Chandu Tammineni**.

[Read the project paper](paper/SWAG_VideoMLLM.pdf)

## Why SWAG?

Applying image attention warping independently to every video frame creates temporal flicker: the emphasized region can move, expand, or disappear even when the underlying object moves smoothly. SWAG addresses this by propagating attention with Farneback optical flow and blending it with each frame's current attention estimate.

## Pipeline

1. Sample frames from a Video-MME clip.
2. Capture query-to-vision saliency from the middle third of Qwen2.5-VL's language layers.
3. Select a high-confidence attention anchor frame.
4. Propagate and blend attention maps across time using dense optical flow.
5. Apply attention-guided spatial warping while preserving the complete frame.
6. Run a second model pass on the warped frames and compare against the original and per-frame-warp baselines.

## Reported results

The paper reports the following three-way ablation on 286 questions from the short-video subset of Video-MME using Qwen2.5-VL-3B-Instruct:

| Condition | Accuracy | Change vs. original |
|---|---:|---:|
| Original frames | 66.67% | — |
| Independent per-frame AttWarp | 68.91% | +2.24 pp |
| **SWAG flow-stabilized warp** | **71.83%** | **+5.16 pp** |

The CSV and text files under `results/` are retained experiment artifacts from development runs and may represent smaller subsets than the final paper evaluation.

## Repository contents

| Path | Purpose |
|---|---|
| `test.py` | Middle-layer saliency and anchor/bounding-box evaluation pipeline |
| `demo.py` | Keyframe and flow-stabilized three-way ablation variant |
| `main.py` | Original, vanilla-warp, and flow-warp evaluation pipeline |
| `new_method.py` | Attention-guided spatial warping implementation and utilities |
| `videos_filter.py` | Utility for identifying short local videos |
| `data/` | Video-MME metadata and selected-video identifiers |
| `results/` | Saved predictions and ablation summaries |
| `paper/` | Final report and retained draft |
| `attwarp.yaml` | Conda environment for the AttWarp/LLaVA utility path |

## Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone <repository-url>
cd SWAG-Video-LLM

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model weights are downloaded by Transformers on first use. A CUDA GPU or Apple Silicon system with substantial unified memory is recommended; the evaluation scripts are not intended for low-memory CPU execution.

## Dataset setup

Download Video-MME separately and arrange the local files as follows:

```text
dataset/
├── test-00000-of-00001.parquet
└── test_data/
    ├── <video-id>.mp4
    └── ...
```

The dataset and model checkpoints are intentionally excluded from Git. You can keep them elsewhere by setting:

```bash
export VIDEOMME_VIDEO_DIR=/absolute/path/to/test_data
export VIDEOMME_METADATA_PATH=/absolute/path/to/test-00000-of-00001.parquet
```

## Run an evaluation

For the anchor-based middle-layer experiment:

```bash
python test.py
```

For the original/vanilla/flow comparison:

```bash
python main.py
```

Results are written incrementally to CSV, so a partially completed run can still be inspected if evaluation is interrupted.

## Reproducibility notes

- Default model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Frame sampling: 2 FPS in the anchor-based experiment
- Frame cap: 32 frames
- Middle-layer range: 33%–67% of decoder depth
- Flow: OpenCV Farneback dense optical flow
- Default temporal blend weight: `alpha = 0.6`
- Salient-region fraction: top 20% of spatial tokens

## Paper citation

If you use this work, please cite the included report:

```text
Gowda, K. V., Rajendra, A., and Tammineni, S. C.
SWAG-Video LLM: Spatiotemporal Warping with Attention Guidance for Video MLLMs.
University of California, Irvine, 2026.
```

## Acknowledgements

This project builds on attention-guided image warping ideas from AttWarp and evaluates on the Video-MME benchmark. See the report for the complete related-work discussion and references.
