import glob
import os
from src.preprocessing import VideoSplitter


def run_batch_preprocessing(raw_dir: str, output_base_dir: str, frame_rate: int = 1):
    
    splitter = VideoSplitter(frame_rate_fps=frame_rate, scene_cut_threshold=35.0)

    video_extensions = ["*.mp4", "*.mkv", "*.avi", "*.mov"]
    video_paths = []

    for ext in video_extensions:
        video_paths.extend(glob.glob(os.path.join(raw_dir, ext)))

    if not video_paths:
        print(f"No videos found in {raw_dir}")
        return

    print(f"Found {len(video_paths)} video(s) total. Commencing scan...")

    for video_path in video_paths:
        video_file_name = os.path.basename(video_path)

        video_slug = os.path.splitext(video_file_name)[0]

        target_output_dir = os.path.join(output_base_dir, video_slug)

        if os.path.exists(target_output_dir) and os.listdir(target_output_dir):
            print(f"⏩ Skipping '{video_file_name}' - Already processed.")
            continue

        # 5. Process the new video
        print(f"Processing '{video_file_name}'")
        try:
            metadata = splitter.process_video(
                video_path=video_path, output_dir=output_base_dir
            )
            print(
                f"Completed. Extracted {metadata['sampled_frames_written']} frames. "
                f"Cuts found: {metadata['scene_cuts_detected']}"
            )

            # Optional: Warning flag for team review
            if metadata["scene_cuts_detected"] > 0:
                print(
                    f"Warning: {metadata['scene_cuts_detected']} video cuts detected!"
                )

        except Exception as e:
            print(f"Failed to process {video_file_name}. Error: {e}")



if __name__ == "__main__":
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed/frames"

    run_batch_preprocessing(raw_dir=RAW_DATA_DIR, output_base_dir=PROCESSED_DATA_DIR, frame_rate=2)