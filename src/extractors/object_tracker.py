import os
from ultralytics import YOLO


class ObjectTracker:

    def __init__(self, model_weight: str = "yolo11n.pt"):
        """Initializes the object tracking model."""
        self.model = YOLO(model_weight)

    def track_objects_in_frames(self, frame_folder: str) -> dict:
        """
        Scans the sampled frames to inventory all distinct detected objects.
        Returns dict: A unique list of objects identified over the course of the video.
        """
        detected_items_set = set()
        frame_files = sorted(
            [
                os.path.join(frame_folder, f)
                for f in os.listdir(frame_folder)
                if f.endswith((".jpg", ".png", ".jpeg"))
            ]
        )

        for frame_path in frame_files:
            # Run inference on individual frames
            results = self.model(frame_path, verbose=False)

            for result in results:
                # Extract detected class names (e.g., 'cell phone', 'laptop', 'brick')
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    detected_items_set.add(label)

        return {
            "extracted_objects_inventory": list(detected_items_set),
            "total_processed_frames": len(frame_files),
        }