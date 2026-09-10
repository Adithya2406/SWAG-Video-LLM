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
)
TEMP_WARPED_DIR = "temp_warped_frames"
os.makedirs(TEMP_WARPED_DIR, exist_ok=True)

# --- VRAM SAFEGUARDS ---
FPS_SAMPLE_RATE = 1.0
MAX_FRAMES = 16  
NUM_KEYFRAMES = 8
MIN_PIXELS = 3136        
MAX_PIXELS = 50176
OUTPUT_CSV = "videomme_qwen_vanilla_results.csv"

def get_dense_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow

def temporal_flow_stabilization(orig_frames, raw_attn_3d, alpha=0.6, protect_subtitles=True):
    num_frames = len(orig_frames)
    frames_np = [np.array(img) if not isinstance(img, np.ndarray) else img for img in orig_frames]
    orig_h, orig_w = frames_np[0].shape[:2]
    stabilized_maps = []
    
    prev_attn = raw_attn_3d[0, :, :]
    prev_attn = np.power(prev_attn, 0.5) 
    prev_attn = cv2.resize(prev_attn, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    prev_attn = cv2.GaussianBlur(prev_attn, (15, 15), 0)
    prev_attn = (prev_attn - prev_attn.min()) / (prev_attn.max() - prev_attn.min() + 1e-8)
    stabilized_maps.append(prev_attn)
    
    for t in range(1, num_frames):
        curr_attn = raw_attn_3d[t, :, :]
        curr_attn = np.power(curr_attn, 0.5)
        curr_attn = cv2.resize(curr_attn, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        curr_attn = cv2.GaussianBlur(curr_attn, (15, 15), 0)
        curr_attn = (curr_attn - curr_attn.min()) / (curr_attn.max() - curr_attn.min() + 1e-8)
        
        flow = get_dense_flow(frames_np[t-1], frames_np[t])
        
        if protect_subtitles:
            subtitle_horizon = int(orig_h * 0.65)
            flow[subtitle_horizon:, :, :] = 0.0
        
        map_x, map_y = np.meshgrid(np.arange(orig_w), np.arange(orig_h))
        map_x = (map_x - flow[..., 0]).astype(np.float32) 
        map_y = (map_y - flow[..., 1]).astype(np.float32)
        
        propagated_attn = cv2.remap(
            stabilized_maps[t-1], map_x, map_y, 
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        
        blended_attn = (alpha * curr_attn) + ((1.0 - alpha) * propagated_attn)
        blended_attn = (blended_attn - blended_attn.min()) / (blended_attn.max() - blended_attn.min() + 1e-8)
        stabilized_maps.append(blended_attn)
        
    return stabilized_maps

def get_attention_keyframes(attn_3d_expanded, top_k=8, strategy="binned"):
    temporal_scores = np.mean(attn_3d_expanded, axis=(1, 2))
    total_frames = len(temporal_scores)
    top_k = min(top_k, total_frames)
    
    if strategy == "top_k":
        selected_indices = np.argsort(temporal_scores)[-top_k:]
    elif strategy == "binned":
        selected_indices = []
        bin_size = total_frames / top_k
        for i in range(top_k):
            start = int(i * bin_size)
            end = int(min((i + 1) * bin_size, total_frames))
            if start >= total_frames: break
            local_max_idx = start + np.argmax(temporal_scores[start:end])
            selected_indices.append(local_max_idx)
            
    return np.sort(selected_indices).astype(int)

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
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        return [], []
        
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or np.isnan(video_fps):
        video_fps = 30.0 
        
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
            
            path = os.path.join(save_dir, f"{vid_id}q{q_idx}_orig{saved_count:04d}.png")
            pil_img.save(path)
            
            orig_paths.append(path)
            orig_pils.append(pil_img)
            saved_count += 1
            
        frame_idx += 1
        
    cap.release()
    return orig_paths, orig_pils

def main():
    df = pd.read_parquet(METADATA_PATH)
    df.to_csv('data.csv')

    print(f"\nFiltered dataset: Found {len(df)} questions matching local videos.")
    
    print(f"Loading {MODEL_ID} with eager attention...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto", attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    final_results = []
    filtered_df = df[df['duration'] == 'short']
    grouped_df = filtered_df.groupby('videoID')
    
    filtered_videos = open("short_videos_list.txt").read().splitlines()
    for vid_id, group in tqdm(grouped_df, total=len(grouped_df), desc="Processing Videos"):
        if vid_id not in filtered_videos:
            continue

        vid_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")
        
        orig_paths, orig_pils = extract_frames_cv2(vid_path, FPS_SAMPLE_RATE, MAX_FRAMES, TEMP_WARPED_DIR, vid_id, "base")
        if not orig_paths:
            print(f"Skipping {vid_id} (could not read video).")
            continue
        
        for idx, row in group.iterrows():
            question = row['question']
            
            # STAGE 1: EXTRACT ATTENTION
            messages_s1 = [{
                "role": "user",
                "content": [
                    {
                        "type": "video", "video": orig_paths,
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

            # STAGE 1.5: KEYFRAME SELECTION & WARPING
            frames_per_token = max(1, len(orig_pils) // grid_t)
            attn_3d_expanded = np.repeat(attn_3d, frames_per_token, axis=0)
            attn_3d_expanded = attn_3d_expanded[:len(orig_pils)]
            
            selected_indices = get_attention_keyframes(attn_3d_expanded, top_k=NUM_KEYFRAMES, strategy="binned")
            print(f"  -> Extracted Keyframes: {selected_indices}")
            
            vanilla_warped_paths = []
            flow_warped_paths = []
            selected_orig_paths = []
            
            stabilized_maps = temporal_flow_stabilization(
                orig_frames=orig_pils,             
                raw_attn_3d=attn_3d_expanded,      
                alpha=0.6
            )
            
            for t in selected_indices:          
                pil_img = orig_pils[t]
                orig_w, orig_h = pil_img.size
                
                selected_orig_paths.append(orig_paths[t])
                
                # --- VANILLA MAP ---
                raw_attn = attn_3d_expanded[t, :, :]
                vanilla_map = np.power(raw_attn, 0.5) 
                vanilla_map = cv2.resize(vanilla_map, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                vanilla_map = cv2.GaussianBlur(vanilla_map, (15, 15), 0)
                vanilla_map = (vanilla_map - vanilla_map.min()) / (vanilla_map.max() - vanilla_map.min() + 1e-8)
                
                vanilla_path = os.path.join(TEMP_WARPED_DIR, f"{vid_id}q{idx}_vanilla{t:04d}.png")
                save_warped_image(
                    image_path=pil_img, att_map=vanilla_map,
                    original_image_save_path=None, masked_overlay_save_path=None,
                    output_path=vanilla_path, vis_path=None, width=orig_w, height=orig_h, transform="identity"
                )
                vanilla_warped_paths.append(vanilla_path)
                
                # --- FLOW MAP ---
                flow_map = stabilized_maps[t]
                flow_path = os.path.join(TEMP_WARPED_DIR, f"{vid_id}q{idx}_flow{t:04d}.png")
                save_warped_image(
                    image_path=pil_img, att_map=flow_map,
                    original_image_save_path=None, masked_overlay_save_path=None,
                    output_path=flow_path, vis_path=None, width=orig_w, height=orig_h, transform="identity"
                )
                flow_warped_paths.append(flow_path)
            
            # STAGE 2: INFERENCE
            options_text = f"A. {row['options'][0][4:-1]} B. {row['options'][1][4:-1]} C. {row['options'][2][4:-1]} D. {row['options'][3][4:-1]}"
            prompt = get_official_prompt(question, options_text)
            
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

                del messages_s2, text_s2, image_inputs_s2, video_inputs_s2
                torch.cuda.empty_cache()

                with torch.no_grad():
                    gen_ids = model.generate(**inputs_s2, max_new_tokens=10)
                    
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_s2.input_ids, gen_ids)]
                response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
                
                del inputs_s2, gen_ids
                torch.cuda.empty_cache()
                
                prediction = response.strip().upper()
                if len(prediction) > 0 and prediction[0] in ['A', 'B', 'C', 'D']:
                    return prediction[0]
                return "N/A"

            pred_orig    = evaluate_frames(selected_orig_paths)
            pred_vanilla = evaluate_frames(vanilla_warped_paths)
            pred_flow    = evaluate_frames(flow_warped_paths)
            
            answer = row['answer'].strip().upper()
            
            print(f"\nVideo: {vid_id} | Q: {question[:30]}...")
            print(f"  -> Actual: {answer} | Orig: {pred_orig} | Vanilla: {pred_vanilla} | Flow: {pred_flow}")
            
            final_results.append({
                "video_id": vid_id,
                "question": question,
                "answer": answer,
                "pred_orig": pred_orig,
                "correct_orig": pred_orig == answer,
                "pred_vanilla": pred_vanilla,
                "correct_vanilla": pred_vanilla == answer,
                "pred_flow": pred_flow,
                "correct_flow": pred_flow == answer
            })
            
            pd.DataFrame(final_results).to_csv(OUTPUT_CSV, index=False)

            # NOTE: Cleanup is intentionally DISABLED so that original,
            # vanilla-warped, and flow-warped frames are all preserved
            # in temp_warped_frames/ for qualitative analysis and paper figures.

    # SUMMARY
    print("\n======================================")
    print(" 3-WAY ABLATION EVALUATION RESULTS")
    print("======================================")
    if len(final_results) > 0:
        df_results = pd.DataFrame(final_results)
        
        correct_orig    = (df_results['pred_orig']    == df_results['answer']).sum()
        correct_vanilla = (df_results['pred_vanilla'] == df_results['answer']).sum()
        correct_flow    = (df_results['pred_flow']    == df_results['answer']).sum()
        total = len(df_results)
        
        acc_orig    = (correct_orig    / total) * 100
        acc_vanilla = (correct_vanilla / total) * 100
        acc_flow    = (correct_flow    / total) * 100
        
        print(f"Total Questions Processed : {total}")
        print(f"1. Original Unwarped      : {acc_orig:.2f}% ({correct_orig}/{total})")
        print(f"2. Vanilla Warp (No Flow) : {acc_vanilla:.2f}% ({correct_vanilla}/{total})")
        print(f"3. Flow-Stabilized Warp   : {acc_flow:.2f}% ({correct_flow}/{total})")
        print("--------------------------------------")
        
        delta_v_vs_o = acc_vanilla - acc_orig
        delta_f_vs_o = acc_flow    - acc_orig
        delta_f_vs_v = acc_flow    - acc_vanilla
        
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
