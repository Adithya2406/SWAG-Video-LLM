import os
import torch
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- IMPORTS ---
from new_method import save_warped_image

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
VIDEO_DIR = os.getenv("VIDEOMME_VIDEO_DIR", "dataset/test_data")
METADATA_PATH = os.getenv(
    "VIDEOMME_METADATA_PATH", "dataset/test-00000-of-00001.parquet"
)  # Parquet metadata from Video-MME
TEMP_WARPED_DIR = "temp_warped_frames"
os.makedirs(TEMP_WARPED_DIR, exist_ok=True)
FPS_SAMPLE_RATE = 1

# --- VRAM SAFEGUARDS ---
FPS_SAMPLE_RATE = 1.0
MAX_FRAMES = 16  
MIN_PIXELS = 3136        
MAX_PIXELS = 50176
OUTPUT_CSV = "videomme_qwen_vanilla_results.csv"

def get_dense_flow(prev_frame, next_frame):
    """Calculates Farneback dense optical flow between two RGB frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
    
    # Standard Farneback parameters for robust motion tracking
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow

def temporal_flow_stabilization(orig_frames, raw_attn_3d, alpha=0.6):
    """
    Stabilizes a sequence of attention maps using optical flow.
    orig_frames: List of raw PIL Images or numpy arrays.
    raw_attn_3d: The (T, H, W) numpy array extracted from Qwen.
    alpha: Weight of the current frame's raw attention (0.0 to 1.0).
    """
    num_frames = len(orig_frames)
    
    # Convert PIL images to numpy arrays if necessary
    frames_np = [np.array(img) if not isinstance(img, np.ndarray) else img for img in orig_frames]
    orig_h, orig_w = frames_np[0].shape[:2]
    
    stabilized_maps = []
    
    # ---------------------------------------------------------
    # Frame 0: No previous frame to flow from, just format it
    # ---------------------------------------------------------
    prev_attn = raw_attn_3d[0, :, :]
    prev_attn = np.power(prev_attn, 0.5) # Soften the peaks
    prev_attn = cv2.resize(prev_attn, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    prev_attn = cv2.GaussianBlur(prev_attn, (15, 15), 0)
    prev_attn = (prev_attn - prev_attn.min()) / (prev_attn.max() - prev_attn.min() + 1e-8)
    
    stabilized_maps.append(prev_attn)
    
    # ---------------------------------------------------------
    # Frame 1 to T: Apply Flow Stabilization
    # ---------------------------------------------------------
    for t in range(1, num_frames):
        # 1. Format the current raw attention
        curr_attn = raw_attn_3d[t, :, :]
        curr_attn = np.power(curr_attn, 0.5)
        curr_attn = cv2.resize(curr_attn, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        curr_attn = cv2.GaussianBlur(curr_attn, (15, 15), 0)
        curr_attn = (curr_attn - curr_attn.min()) / (curr_attn.max() - curr_attn.min() + 1e-8)
        
        # 2. Compute Optical Flow from t-1 to t
        flow = get_dense_flow(frames_np[t-1], frames_np[t])
        
        # 3. Warp the PREVIOUS stabilized map using the flow field
        # We use backward mapping (x - flow) to pull pixels to their new locations
        map_x, map_y = np.meshgrid(np.arange(orig_w), np.arange(orig_h))
        map_x = (map_x - flow[..., 0]).astype(np.float32) 
        map_y = (map_y - flow[..., 1]).astype(np.float32)
        
        propagated_attn = cv2.remap(
            stabilized_maps[t-1], 
            map_x, map_y, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # 4. Blend the propagated map with the current raw map
        blended_attn = (alpha * curr_attn) + ((1.0 - alpha) * propagated_attn)
        
        # 5. Normalize safely to [0, 1] for the MarginalNet warper
        blended_attn = (blended_attn - blended_attn.min()) / (blended_attn.max() - blended_attn.min() + 1e-8)
        
        stabilized_maps.append(blended_attn)
        
    return stabilized_maps

def get_official_prompt(question, options):
    return (
        f"Select the best answer to the following multiple-choice question based on the video. "
        f"Respond with only the letter (A, B, C, or D) of the correct option.\n"
        f"Question: {question}\n"
        f"Options: {options}\n"
        f"The best answer is:"
    )

def extract_video_spatial_attention(model, processor, inputs, outputs):
    last_layer_attn = outputs.attentions[-1]
    mean_attn = last_layer_attn.mean(dim=1)
    
    video_pad_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    if video_pad_token_id is None:
        video_pad_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        
    input_ids = inputs['input_ids'][0]
    vision_indices = (input_ids == video_pad_token_id).nonzero(as_tuple=True)[0]
    
    if len(vision_indices) == 0:
        raise ValueError("Could not locate video tokens in the sequence.")
        
    start_idx = vision_indices[0].item()
    grid_t, grid_h, grid_w = inputs['video_grid_thw'][0]
    
    token_t = grid_t
    token_h = grid_h // 2
    token_w = grid_w // 2
    num_vision_tokens = token_t * token_h * token_w
    end_idx = start_idx + num_vision_tokens
    
    text_to_vision_attn = mean_attn[0, -1, start_idx:end_idx]
    spatiotemporal_map_3d = text_to_vision_attn.reshape(token_t, token_h, token_w)
    
    return spatiotemporal_map_3d.to(torch.float32).cpu().numpy()

def extract_frames_cv2(vid_path, fps_sample_rate, max_frames, save_dir, vid_id, q_idx):
    """Robustly extracts frames using OpenCV to bypass torchvision metadata errors."""
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        return [], []
        
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or np.isnan(video_fps):
        video_fps = 30.0 # Fallback for completely broken metadata
        
    frame_step = max(1, int(video_fps / fps_sample_rate))
    
    orig_paths = []
    orig_pils = []
    frame_idx = 0
    saved_count = 0
    
    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
            
        if frame_idx % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Save original frame so Qwen can read it
            path = os.path.join(save_dir, f"{vid_id}_q{q_idx}_orig_{saved_count:04d}.png")
            pil_img.save(path)
            
            orig_paths.append(path)
            orig_pils.append(pil_img)
            saved_count += 1
            
        frame_idx += 1
        
    cap.release()
    return orig_paths, orig_pils

def main():
    df = pd.read_parquet(METADATA_PATH)
    # existing_vids = [f.replace('.mp4', '') for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    # df = df[df['video_id'].isin(existing_vids)]
    print(f"\nFiltered dataset: Found {len(df)} questions matching local videos.")
    
    print(f"Loading {MODEL_ID} with eager attention...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto", attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    final_results = []
    
    print("\n--- STARTING UNIFIED PIPELINE ---")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        vid_id = row['videoID']
        question = row['question']
        vid_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")
        
        orig_paths, orig_pils = extract_frames_cv2(vid_path, FPS_SAMPLE_RATE, MAX_FRAMES, TEMP_WARPED_DIR, vid_id, idx)
        if not orig_paths:
            print(f"Skipping {vid_id} (could not read video).")
            continue
        
        # ==========================================
        # STAGE 1: EXTRACT ATTENTION
        # ==========================================
        messages_s1 = [{
            "role": "user",
            "content": [
                {
                    "type": "video", "video": orig_paths, # Feed exactly the frames we just extracted
                    "fps": FPS_SAMPLE_RATE, "max_frames": MAX_FRAMES,
                    "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS
                }, 
                {"type": "text", "text": f"Focus on visual details to answer: {question}"}
            ]
        }]
        
        text_s1 = processor.apply_chat_template(messages_s1, tokenize=False, add_generation_prompt=True)
        image_inputs_s1, video_inputs_s1 = process_vision_info(messages_s1)
        inputs_s1 = processor(text=[text_s1], images=image_inputs_s1, videos=video_inputs_s1, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs_s1 = model(**inputs_s1, output_attentions=True)
            
        attn_3d = extract_video_spatial_attention(model, processor, inputs_s1, outputs_s1)
        grid_t = attn_3d.shape[0] 
        
        del outputs_s1, inputs_s1, image_inputs_s1, video_inputs_s1
        torch.cuda.empty_cache()

        # ==========================================
        # STAGE 1.5: TRIPLE-WARP GENERATION
        # ==========================================
        min_t = min(grid_t, len(orig_pils))
        
        vanilla_warped_paths = []
        flow_warped_paths = []
        
        # 1. Pre-compute the Flow-Stabilized maps for the whole sequence
        stabilized_maps = temporal_flow_stabilization(
            orig_frames=orig_pils[:min_t], 
            raw_attn_3d=attn_3d[:min_t], 
            alpha=0.6
        )
        
        # 2. Loop through frames and apply BOTH warping methods
        for t in range(min_t):
            pil_img = orig_pils[t]
            orig_w, orig_h = pil_img.size
            
            # --- VANILLA MAP (Raw, smoothed) ---
            raw_attn = attn_3d[t, :, :]
            vanilla_map = np.power(raw_attn, 0.5) 
            vanilla_map = cv2.resize(vanilla_map, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            vanilla_map = cv2.GaussianBlur(vanilla_map, (15, 15), 0)
            vanilla_map = (vanilla_map - vanilla_map.min()) / (vanilla_map.max() - vanilla_map.min() + 1e-8)
            
            # --- FLOW MAP (Stabilized) ---
            flow_map = stabilized_maps[t]
            
            # --- SAVE VANILLA WARP ---
            vanilla_path = os.path.join(TEMP_WARPED_DIR, f"{vid_id}_q{idx}_vanilla_{t:04d}.png")
            save_warped_image(
                image_path=pil_img, att_map=vanilla_map,
                original_image_save_path=None, masked_overlay_save_path=None,
                output_path=vanilla_path, vis_path=None,
                width=orig_w, height=orig_h, transform="identity"
            )
            vanilla_warped_paths.append(vanilla_path)
            
            # --- SAVE FLOW WARP ---
            flow_path = os.path.join(TEMP_WARPED_DIR, f"{vid_id}_q{idx}_flow_{t:04d}.png")
            save_warped_image(
                image_path=pil_img, att_map=flow_map,
                original_image_save_path=None, masked_overlay_save_path=None,
                output_path=flow_path, vis_path=None,
                width=orig_w, height=orig_h, transform="identity"
            )
            flow_warped_paths.append(flow_path)
        

        # ==========================================
        # STAGE 2: DUAL INFERENCE (Orig vs Warped)
        # ==========================================
        print(row['options'])
        options_text = f"A. {row['options'][0][4:-1]} B. {row['options'][1][4:-1]} C. {row['options'][2][4:-1]} D. {row['options'][3][4:-1]}"
        prompt = get_official_prompt(question, options_text)
        
        # Helper function to run inference and clear VRAM safely
        def evaluate_frames(frame_paths):
            messages_s2 = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": frame_paths, "fps": FPS_SAMPLE_RATE}, 
                    {"type": "text", "text": prompt}
                ]
            }]
            text_s2 = processor.apply_chat_template(messages_s2, tokenize=False, add_generation_prompt=True)
            image_inputs_s2, video_inputs_s2 = process_vision_info(messages_s2)
            inputs_s2 = processor(text=[text_s2], images=image_inputs_s2, videos=video_inputs_s2, return_tensors="pt").to(model.device)

            # --- ADD THIS SWEEP ---
            del messages_s2, text_s2, image_inputs_s2, video_inputs_s2
            torch.cuda.empty_cache()
            # ----------------------

            with torch.no_grad():
                gen_ids = model.generate(**inputs_s2, max_new_tokens=10)
                
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_s2.input_ids, gen_ids)]
            response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            
            # VRAM Cleanup
            del inputs_s2, gen_ids
            torch.cuda.empty_cache()
            
            prediction = response.strip().upper()
            if len(prediction) > 0 and prediction[0] in ['A', 'B', 'C', 'D']:
                return prediction[0]
            return "N/A"

        # Run all THREE inferences!
        pred_orig = evaluate_frames(orig_paths)
        pred_vanilla = evaluate_frames(vanilla_warped_paths)
        pred_flow = evaluate_frames(flow_warped_paths)
        
        answer = row['answer'].strip().upper()
        
        print(f"\nVideo: {vid_id} | Q: {question[:30]}...")
        print(f"  -> Actual: {answer} | Orig: {pred_orig} | Vanilla: {pred_vanilla} | Flow: {pred_flow}")
        
        final_results.append({
            "video_id": vid_id,
            "question": question,
            "pred_orig": pred_orig,
            "pred_vanilla": pred_vanilla,
            "pred_flow": pred_flow,
            "answer": answer
        })
        
        pd.DataFrame(final_results).to_csv(OUTPUT_CSV, index=False)

        # Cleanup ALL THREE image sets from hard drive
        for p in orig_paths + vanilla_warped_paths + flow_warped_paths:
            if os.path.exists(p):
                os.remove(p)

    # ==========================================
    # QUANTITATIVE EVALUATION SUMMARY
    # ==========================================
    print("\n======================================")
    print(" 3-WAY ABLATION EVALUATION RESULTS")
    print("======================================")
    if len(final_results) > 0:
        df_results = pd.DataFrame(final_results)
        
        correct_orig = (df_results['pred_orig'] == df_results['answer']).sum()
        correct_vanilla = (df_results['pred_vanilla'] == df_results['answer']).sum()
        correct_flow = (df_results['pred_flow'] == df_results['answer']).sum()
        total = len(df_results)
        
        acc_orig = (correct_orig / total) * 100
        acc_vanilla = (correct_vanilla / total) * 100
        acc_flow = (correct_flow / total) * 100
        
        print(f"Total Questions Processed : {total}")
        print(f"1. Original Unwarped      : {acc_orig:.2f}% ({correct_orig}/{total})")
        print(f"2. Vanilla Warp (No Flow) : {acc_vanilla:.2f}% ({correct_vanilla}/{total})")
        print(f"3. Flow-Stabilized Warp   : {acc_flow:.2f}% ({correct_flow}/{total})")
        print("--------------------------------------")
        
        delta_v_vs_o = acc_vanilla - acc_orig
        delta_f_vs_o = acc_flow - acc_orig
        delta_f_vs_v = acc_flow - acc_vanilla
        
        print(f"Delta (Vanilla vs Orig)   : {delta_v_vs_o:+.2f}%")
        print(f"Delta (Flow vs Orig)      : {delta_f_vs_o:+.2f}%")
        print(f"Delta (Flow vs Vanilla)   : {delta_f_vs_v:+.2f}%  <-- Your Contribution!")
        
        with open("videomme_qwen_ablation_summary.txt", "w") as f:
            f.write(f"Total: {total}\n")
            f.write(f"Acc_Orig: {acc_orig:.2f}%\n")
            f.write(f"Acc_Vanilla: {acc_vanilla:.2f}%\n")
            f.write(f"Acc_Flow: {acc_flow:.2f}%\n")
            f.write(f"Delta_Flow_vs_Vanilla: {delta_f_vs_v:+.2f}%\n")
    else:
        print("No results to evaluate.")
    print("======================================\n")
    
    print(f"Pipeline complete! Results fully saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
