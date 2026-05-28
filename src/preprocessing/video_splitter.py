import os
import cv2
import numpy as np


class VideoSplitter:

    def __init__(
        self, frame_rate_fps: int = 1, scene_cut_threshold: float = 30.0
    ):
        """
        Args:
            frame_rate_fps (int): How many frames to save per second of video.
            scene_cut_threshold (float): Threshold percentage change in pixel
              intensity to flag a scene cut.
        """
        self.frame_rate_fps = frame_rate_fps
        self.scene_cut_threshold = scene_cut_threshold

    def process_video(self, video_path: str, output_dir: str) -> dict:
        """        
        Samples frames from a video and checks for abrupt cuts.
        """
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_output_path = os.path.join(output_dir, video_name)
        os.makedirs(frame_output_path, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if source_fps == 0:
            raise ValueError(
                f"Could not read FPS from video file: {video_path}"
            )

        frame_interval = max(1, int(source_fps / self.frame_rate_fps))

        prev_frame_gray = None
        scene_cuts = []
        sampled_count = 0
        current_frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp_sec = current_frame_idx / source_fps

            if prev_frame_gray is not None:
                frame_diff = cv2.absdiff(gray_frame, prev_frame_gray)
                mean_diff = np.mean(frame_diff)

                if mean_diff > self.scene_cut_threshold:
                    scene_cuts.append(
                        {
                            "frame_index": current_frame_idx,
                            "timestamp_seconds": round(timestamp_sec, 2),
                            "severity": round(float(mean_diff), 2),
                        }
                    )

            prev_frame_gray = gray_frame

            if current_frame_idx % frame_interval == 0:
                frame_filename = os.path.join(
                    frame_output_path, f"frame_{current_frame_idx:04d}.jpg"
                )
                cv2.imwrite(frame_filename, frame)
                sampled_count += 1

            current_frame_idx += 1

        cap.release()

        return {
            "video_name": video_name,
            "original_fps": round(source_fps, 2),
            "total_frames": total_frames,
            "sampled_frames_written": sampled_count,
            "scene_cuts_detected": len(scene_cuts),
            "cut_manifest": scene_cuts,
        }