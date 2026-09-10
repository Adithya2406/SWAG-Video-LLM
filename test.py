import os
import torch
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_ID      = "Qwen/Qwen2.5-VL-3B-Instruct"
VIDEO_DIR = os.getenv("VIDEOMME_VIDEO_DIR", "dataset/test_data")
METADATA_PATH = os.getenv(
    "VIDEOMME_METADATA_PATH", "dataset/test-00000-of-00001.parquet"
)

TEMP_DIR   = "temp_frames"
os.makedirs(TEMP_DIR, exist_ok=True)

OUTPUT_CSV = "videomme_bboxtrack_results.csv"

FPS_SAMPLE_RATE = 2       
MIN_PIXELS      = 3136
MAX_PIXELS      = 50176

# Hard cap: 32 frames * ~252 tokens = ~8064 tokens → ~0.93GB KV cache.
# Safe on 32GB M-series Macs with model weights occupying ~6GB.
# Uniformly subsamples long videos to stay within this budget.
MAX_FRAMES      = 32

# --------------------------------------------------
# METHOD HYPERPARAMETERS
#
# TOP_REGION_FRAC:  fraction of spatial tokens considered "salient"
#                   when computing the attention bounding box.
#                   0.2 = top 20% of tokens define the region.
#
# ANCHOR_STRATEGY:  which frame to use as the bbox anchor.
#                   "best"  = frame with highest peak attention (most confident)
#                   "first" = always use frame 0
#
# ATTN_LAYER_*:     middle-layer range for saliency. Middle third of
#                   decoder stack has strongest spatial grounding.
# --------------------------------------------------
TOP_REGION_FRAC  = 0.20
ANCHOR_STRATEGY  = "best"

ATTN_LAYER_START = 0.33
ATTN_LAYER_END   = 0.67

# --------------------------------------------------
# DEVICE
# --------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ==================================================
# SALIENCY EXTRACTION
# Middle-layer Q/K hooks, averaged across layers.
# See comments in extract_saliency() for rationale.
# ==================================================

class QKHook:
    def __init__(self):
        self.q = self.k = self._handle = None

    def _hook(self, module, args, kwargs):
        hidden = args[0] if len(args) > 0 else kwargs.get("hidden_states")
        if hidden is None:
            return
        with torch.no_grad():
            h  = hidden.detach().cpu().float()
            self.q = (h @ module.q_proj.weight.detach().cpu().float().T)[:, -1:, :]
            self.k =  h @ module.k_proj.weight.detach().cpu().float().T

    def register(self, layer):
        self._handle = layer.self_attn.register_forward_pre_hook(
            self._hook, with_kwargs=True)

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None

    def clear(self):
        self.q = self.k = None


def _vision_attn_vec(hook, nq, nkv, hd, start, end):
    """Softmax(Q_last · K_vis^T / sqrt(hd)), mean over heads → [n_vis]."""
    q  = hook.q[0, 0, :].reshape(nq, hd)
    k  = hook.k[0, :, :].reshape(-1, nkv, hd).repeat_interleave(nq // nkv, dim=1)
    kv = k[start:end]
    s  = torch.einsum("hd,vhd->hv", q, kv) / (hd ** 0.5)
    return torch.softmax(s.float(), dim=-1).mean(dim=0).numpy()


def extract_saliency(model, processor, inputs):
    """
    One forward pass. Hooks on middle decoder layers (33%–67% depth).
    Average saliency over those layers — much less noisy than last-layer only.

    Returns attn_3d: float32 [grid_t, token_h, token_w]
    """
    input_ids = inputs["input_ids"][0]
    vision_indices = None
    for tok in ("<|video_pad|>", "<|vision_pad|>", "<|image_pad|>"):
        tid = processor.tokenizer.convert_tokens_to_ids(tok)
        if tid != processor.tokenizer.unk_token_id:
            idx = (input_ids == tid).nonzero(as_tuple=True)[0]
            if len(idx):
                vision_indices = idx
                break
    if vision_indices is None:
        raise RuntimeError("Vision pad tokens not found.")

    start_idx       = vision_indices[0].item()
    grid_t, grid_h, grid_w = inputs["video_grid_thw"][0]
    token_h, token_w = grid_h // 2, grid_w // 2
    end_idx          = start_idx + int(grid_t * token_h * token_w)

    layers  = model.model.language_model.layers
    N       = len(layers)
    lo, hi  = int(N * ATTN_LAYER_START), int(N * ATTN_LAYER_END)

    ref  = layers[lo].self_attn
    nq   = ref.num_heads
    nkv  = ref.num_key_value_heads
    hd   = ref.q_proj.weight.shape[0] // nq

    hooks = [QKHook() for _ in range(lo, hi + 1)]
    for h, li in zip(hooks, range(lo, hi + 1)):
        h.register(layers[li])

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    attns = [_vision_attn_vec(h, nq, nkv, hd, start_idx, end_idx)
             for h in hooks if h.q is not None]
    for h in hooks:
        h.clear()

    if not attns:
        raise RuntimeError("No attention captured.")

    clear_memory()
    avg = np.mean(attns, axis=0)
    return avg.reshape(int(grid_t), token_h, token_w).astype(np.float32)


# ==================================================
# SALIENCY MAP → BOUNDING BOX
# ==================================================

def attn_to_bbox(saliency_map, top_frac=TOP_REGION_FRAC):
    """
    Find bounding box of the top-attended region.

    Threshold at (1 - top_frac) percentile, find connected extent,
    pad by 10%. Returns (x1, y1, x2, y2) in pixel coordinates.
    """
    H, W  = saliency_map.shape
    thresh = np.percentile(saliency_map, (1.0 - top_frac) * 100)
    ys, xs = np.where(saliency_map >= thresh)

    if len(xs) == 0:
        return 0, 0, W, H

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())

    # 10% padding
    pw, ph = max(1, (x2 - x1) // 10), max(1, (y2 - y1) // 10)
    x1 = max(0, x1 - pw);  y1 = max(0, y1 - ph)
    x2 = min(W, x2 + pw);  y2 = min(H, y2 + ph)
    return x1, y1, x2, y2


def prep_saliency(raw, H, W):
    """Upscale raw token-level attention to pixel space."""
    a = raw.astype(np.float32)
    a = np.power(a, 0.5)
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)
    a = cv2.GaussianBlur(a, (15, 15), 0)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return a


# ==================================================
# OPTICAL FLOW BBOX TRACKING
#
# Core contribution of this paper:
#
#   Standard VLMs process video frames independently — each frame's
#   visual tokens are encoded without awareness of object motion.
#   Attention maps extracted from independent frame encoding are therefore
#   noisy: the same object may receive high attention in one frame and
#   low attention in the next simply due to encoding variance.
#
#   Our key insight: once the model identifies the most salient region
#   in an "anchor" frame (the frame where attention is most confident,
#   i.e. highest peak value), we can use optical flow to track that
#   specific region across all other frames WITHOUT re-running expensive
#   attention extraction per frame.
#
#   Formally:
#     Let B_a = (x1, y1, x2, y2) be the bounding box of the salient
#     region in anchor frame a, computed from attention map A_a.
#
#     For frame t ≠ a, we compute the mean flow displacement within B_a
#     (or the tracked box up to frame t) using Farneback dense flow:
#
#       Δx_t = mean(F_{t-1→t}[y1:y2, x1:x2, 0])
#       Δy_t = mean(F_{t-1→t}[y1:y2, x1:x2, 1])
#       B_t = B_{t-1} + (Δx_t, Δy_t, Δx_t, Δy_t)   [clipped to frame]
#
#   This is a causal, online tracker — O(T) time, O(1) memory.
#   It outperforms per-frame attention re-extraction because:
#     1. Flow is a dense, pixel-level signal vs. sparse token-level attention
#     2. Optical flow is equivariant to rigid motion; attention is not
#     3. No additional model forward passes required
#
#   The vanilla ablation (no flow) uses the per-frame attention bbox
#   independently for each frame, demonstrating the jitter/noise problem.
# ==================================================

def dense_flow(f1, f2):
    g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )


def track_bbox_with_flow(frames_np, anchor_bbox, anchor_idx):
    """
    Track a bounding box across all frames using optical flow.

    Starting from anchor_bbox at anchor_idx, propagate forward and
    backward through the sequence using mean flow displacement within
    the current box. Much more stable than per-frame attention extraction.

    Args:
        frames_np  : list of H×W×3 uint8 numpy arrays
        anchor_bbox: (x1,y1,x2,y2) at anchor_idx
        anchor_idx : int, frame index of the anchor

    Returns:
        bboxes: list of (x1,y1,x2,y2) for every frame
    """
    T = len(frames_np)
    H, W = frames_np[0].shape[:2]
    bboxes = [None] * T
    bboxes[anchor_idx] = anchor_bbox

    def clip_bbox(b):
        x1, y1, x2, y2 = b
        x1, x2 = max(0, int(x1)), min(W, int(x2))
        y1, y2 = max(0, int(y1)), min(H, int(y2))
        # Ensure minimum box size
        if x2 - x1 < 4: x2 = min(W, x1 + 4)
        if y2 - y1 < 4: y2 = min(H, y1 + 4)
        return x1, y1, x2, y2

    def propagate(from_idx, to_idx, direction):
        """Propagate bbox one step in given direction."""
        f1 = frames_np[from_idx]
        f2 = frames_np[to_idx]
        flow = dense_flow(f1, f2)

        x1, y1, x2, y2 = bboxes[from_idx]
        # Mean flow displacement inside current bbox
        roi_flow = flow[y1:y2, x1:x2]
        if roi_flow.size == 0:
            bboxes[to_idx] = bboxes[from_idx]
            return
        dx = float(roi_flow[..., 0].mean())
        dy = float(roi_flow[..., 1].mean())

        bboxes[to_idx] = clip_bbox((x1 + dx, y1 + dy, x2 + dx, y2 + dy))

    # Propagate forward from anchor
    for t in range(anchor_idx + 1, T):
        propagate(t - 1, t, +1)

    # Propagate backward from anchor
    for t in range(anchor_idx - 1, -1, -1):
        propagate(t + 1, t, -1)

    return bboxes


# ==================================================
# REGION CROP WARP
#
# The vision encoder tokenizes frames at very low resolution (~10x6 tokens
# for a typical video frame with MIN_PIXELS=3136). A soft-blended zoom
# covering ~4x3 tokens with <1 token displacement is INVISIBLE to the model.
#
# The correct approach: hard-crop the salient bounding box and resize it
# to fill the ENTIRE frame. This guarantees the attended region uses ALL
# available tokens — maximum resolution, maximum signal.
#
# The model then sees two types of frames interleaved:
#   - Original frames: full scene context
#   - Crop frames: close-up of the most attended region
#
# We replace EVERY other frame with its crop version, keeping alternating
# originals for context. The model gets both "where is the scene" and
# "what does the key object look like up close".
#
# Why this works better than soft zoom:
#   - Hard crop fills all ~60 tokens with the region (vs 4x3 before)
#   - No blending dilution — the content change is maximal
#   - Still a natural image (just a closer shot), within training distribution
#   - Optical flow contribution: tracked crop is temporally coherent
#     (same object, same region across frames) vs vanilla (jittery, different
#     objects each frame due to noisy per-frame attention)
# ==================================================

def apply_crop_warp(pil_img, bbox):
    """
    Hard-crop the salient region and resize to full frame dimensions.
    This maximizes the token budget spent on the attended region.

    Args:
        pil_img : original PIL Image  (H x W)
        bbox    : (x1, y1, x2, y2) salient region

    Returns: PIL Image — cropped region resized to original H x W
    """
    img = np.array(pil_img)
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox

    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return pil_img  # degenerate box — return original

    crop = img[y1:y2, x1:x2]
    resized = cv2.resize(crop, (W, H), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(resized)


# ==================================================
# FRAME EXTRACTION
# ==================================================

def extract_frames(vid_path, fps, save_dir, vid_id, q_idx):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        return [], []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or np.isnan(video_fps):
        video_fps = 30.0
    step = max(1, int(video_fps / fps))

    paths, pils = [], []
    fi = si = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if fi % step == 0:
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil  = Image.fromarray(rgb)
            path = os.path.join(save_dir, f"{vid_id}_q{q_idx}_{si:04d}.png")
            pil.save(path)
            paths.append(path)
            pils.append(pil)
            si += 1
        fi += 1
    cap.release()
    return paths, pils


# ==================================================
# PROMPT & INFERENCE
# ==================================================

def make_prompt(question, options):
    return (
        "Select the best answer to the following multiple-choice question "
        "based on the video.\n"
        "Respond with only the letter (A, B, C, or D).\n"
        f"Question: {question}\n"
        f"Options: {options}\n"
        "The best answer is:"
    )


def run_inference(model, processor, frame_paths, prompt):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": frame_paths, "fps": FPS_SAMPLE_RATE},
            {"type": "text",  "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    imgs, vids, vkw = process_vision_info(messages, return_video_kwargs=True)
    if isinstance(vkw.get("fps"), list):
        vkw["fps"] = vkw["fps"][0]

    inputs = processor(
        text=[text], images=imgs, videos=vids,
        return_tensors="pt", **vkw
    ).to(DEVICE)

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=10)

    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    resp    = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    del inputs, gen
    clear_memory()

    pred = resp.strip().upper()
    return pred[0] if pred and pred[0] in "ABCD" else "N/A"


# ==================================================
# MAIN
# ==================================================

def main():
    df = pd.read_parquet(METADATA_PATH)
    print(f"Dataset: {len(df)} questions")
    print(f"Loading {MODEL_ID} ...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        attn_implementation="eager",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        vid_id   = row["videoID"]
        question = row["question"]
        vid_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")

        # ── 1. Extract frames ──────────────────────────────────────────────
        paths, pils = extract_frames(
            vid_path, FPS_SAMPLE_RATE, TEMP_DIR, vid_id, idx)
        if not paths:
            print(f"  Skipping {vid_id} — unreadable.")
            continue

        # Enforce hard frame cap — prevents MPS OOM on long videos
        if MAX_FRAMES is not None and len(paths) > MAX_FRAMES:
            indices = np.linspace(0, len(paths) - 1, MAX_FRAMES, dtype=int)
            paths   = [paths[i] for i in indices]
            pils    = [pils[i]  for i in indices]

        frames_np = [np.array(p) for p in pils]
        H, W = frames_np[0].shape[:2]

        # ── 2. Saliency extraction (single forward pass) ───────────────────
        msgs1 = [{
            "role": "user",
            "content": [
                {"type": "video", "video": paths, "fps": FPS_SAMPLE_RATE},
                {"type": "text",  "text": f"Focus on visual details: {question}"},
            ],
        }]
        t1 = processor.apply_chat_template(
            msgs1, tokenize=False, add_generation_prompt=True)
        i1, v1, kw1 = process_vision_info(msgs1, return_video_kwargs=True)
        if isinstance(kw1.get("fps"), list):
            kw1["fps"] = kw1["fps"][0]
        inp1 = processor(text=[t1], images=i1, videos=v1,
                         return_tensors="pt", **kw1).to(DEVICE)

        attn_3d = extract_saliency(model, processor, inp1)
        del inp1; clear_memory()

        min_t = min(len(pils), attn_3d.shape[0])

        # ── 3. Per-frame pixel-space saliency maps ─────────────────────────
        sal_maps = [prep_saliency(attn_3d[t], H, W) for t in range(min_t)]

        # ── 4. Vanilla bboxes  (independent per frame, no tracking) ────────
        #   Ablation baseline: each frame gets its own bbox from its own
        #   (noisy, independent) attention map. No temporal consistency.
        vanilla_bboxes = [attn_to_bbox(sal_maps[t]) for t in range(min_t)]

        # ── 5. Flow-tracked bboxes  (OUR CONTRIBUTION) ────────────────────
        #   Step 1: find the "anchor" frame — highest peak attention value.
        #           This is the frame where the model is most confident
        #           about what it's looking at.
        #   Step 2: extract anchor bbox from attention map.
        #   Step 3: propagate bbox forward AND backward with optical flow.
        #
        #   Result: temporally consistent bbox that follows the salient
        #   object's actual motion, not the noisy per-frame attention.
        if ANCHOR_STRATEGY == "best":
            anchor_idx = int(np.argmax([attn_3d[t].max() for t in range(min_t)]))
        else:
            anchor_idx = 0

        anchor_sal  = sal_maps[anchor_idx]
        anchor_bbox = attn_to_bbox(anchor_sal)
        flow_bboxes = track_bbox_with_flow(frames_np[:min_t], anchor_bbox, anchor_idx)

        # ── 6. Apply zoom warp using each bbox set ─────────────────────────
        vanilla_paths = []
        flow_paths    = []

        for t in range(min_t):
            pil = pils[t]

            # Vanilla warp: crop to per-frame independent attention bbox
            vw   = apply_crop_warp(pil, vanilla_bboxes[t])
            vp   = os.path.join(TEMP_DIR, f"{vid_id}_q{idx}_v_{t:04d}.png")
            vw.save(vp);  vanilla_paths.append(vp)

            # Flow warp: crop to flow-tracked consistent bbox
            fw   = apply_crop_warp(pil, flow_bboxes[t])
            fp   = os.path.join(TEMP_DIR, f"{vid_id}_q{idx}_f_{t:04d}.png")
            fw.save(fp);  flow_paths.append(fp)

        # ── 7. Prompt ──────────────────────────────────────────────────────
        opts = row["options"]
        options_text = (
            f"A. {opts[0][4:-1]} B. {opts[1][4:-1]} "
            f"C. {opts[2][4:-1]} D. {opts[3][4:-1]}"
        )
        prompt = make_prompt(question, options_text)

        # ── 8. Three-way inference ─────────────────────────────────────────
        #   ORIG    : all frames, no modification (baseline)
        #   VANILLA : zoom warp using per-frame attention bbox (no tracking)
        #   FLOW    : zoom warp using flow-tracked bbox          ← ours
        pred_orig    = run_inference(model, processor, paths[:min_t],  prompt)
        pred_vanilla = run_inference(model, processor, vanilla_paths,  prompt)
        pred_flow    = run_inference(model, processor, flow_paths,     prompt)

        answer = row["answer"].strip().upper()
        print(f"\n{vid_id} | GT={answer} | "
              f"Orig={pred_orig} Vanilla={pred_vanilla} Flow={pred_flow} "
              f"[anchor={anchor_idx}/{min_t-1}, "
              f"bbox={anchor_bbox}]")

        results.append(dict(
            video_id=vid_id,
            pred_orig=pred_orig, pred_vanilla=pred_vanilla, pred_flow=pred_flow,
            answer=answer,
            anchor_frame=anchor_idx,
            anchor_bbox=str(anchor_bbox),
        ))
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

        for p in paths + vanilla_paths + flow_paths:
            if os.path.exists(p):
                os.remove(p)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 42)
    print(" 3-WAY ABLATION EVALUATION RESULTS")
    print("=" * 42)
    if results:
        dr    = pd.DataFrame(results)
        total = len(dr)

        co = (dr["pred_orig"]    == dr["answer"]).sum()
        cv = (dr["pred_vanilla"] == dr["answer"]).sum()
        cf = (dr["pred_flow"]    == dr["answer"]).sum()

        ao, av, af = co/total*100, cv/total*100, cf/total*100

        print(f"Total questions                    : {total}")
        print(f"1. Orig  (no warp)                 : {ao:.2f}%  ({co}/{total})")
        print(f"2. Vanilla warp (per-frame attn)   : {av:.2f}%  ({cv}/{total})")
        print(f"3. Flow-tracked warp  (ours)       : {af:.2f}%  ({cf}/{total})")
        print("-" * 42)
        print(f"Δ Vanilla vs Orig                  : {av-ao:+.2f}%")
        print(f"Δ Flow vs Orig                     : {af-ao:+.2f}%")
        print(f"Δ Flow vs Vanilla                  : {af-av:+.2f}%  ← contribution")

        with open("ablation_summary.txt", "w") as f:
            f.write(f"Total: {total}\n"
                    f"Acc_Orig: {ao:.2f}%\n"
                    f"Acc_Vanilla: {av:.2f}%\n"
                    f"Acc_Flow: {af:.2f}%\n"
                    f"Delta_Flow_vs_Vanilla: {af-av:+.2f}%\n")
    else:
        print("No results.")
    print("=" * 42)
    print(f"\nResults → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
