import cv2
import os

def find_short_videos(folder_path, output_txt_path, max_frames=45):
    """
    Scans a folder for videos with fewer than `max_frames` and writes their paths to a text file.
    """
    # Tuple of common video extensions to look for
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    
    # Check if the provided folder path actually exists
    if not os.path.exists(folder_path):
        print(f"Error: The directory '{folder_path}' does not exist.")
        return

    short_videos_found = 0

    print(f"Scanning '{folder_path}' for videos with less than {max_frames} frames...\n")

    # Open the output text file in write mode
    with open(output_txt_path, 'w') as file:
        
        # Iterate through all files in the given directory
        # print(os.listdir(folder_path))
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                video_id = filename[:-4]  # Remove the file extension to get the video ID
                video_path = os.path.join(folder_path, filename)
                
                try:
                    # Open the video file
                    cap = cv2.VideoCapture(video_path)
                    
                    if not cap.isOpened():
                        print(f"Warning: Could not read video data for {filename}")
                        continue
                        
                    # Extract the total frame count
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Check if it meets the filter criteria
                    if total_frames < max_frames:
                        print(f"Match found: {filename} ({total_frames} frames)")
                        file.write(f"{video_id}\n")
                        short_videos_found += 1
                        
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                
                finally:
                    # Always release the video capture object to free up resources
                    cap.release()

    print(f"\nDone! Found {short_videos_found} short videos.")
    print(f"The list has been saved to: {output_txt_path}")

# --- Configuration ---
if __name__ == "__main__":
    TARGET_FOLDER = os.getenv("VIDEOMME_VIDEO_DIR", "dataset/data")
    OUTPUT_FILE = os.getenv("SHORT_VIDEOS_OUTPUT", "short_videos_list.txt")
    
    find_short_videos(TARGET_FOLDER, OUTPUT_FILE, max_frames=1000)
