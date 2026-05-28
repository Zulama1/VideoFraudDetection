import os
import easyocr


class OCREngine:

    def __init__(self, languages: list = ["en"]):
        self.reader = easyocr.Reader(languages, gpu=False)

    def extract_text_from_labels(self, frame_folder: str) -> dict:
        """
        Looks through frames to capture texts matching shipping IDs or trackingnumbers.
        """
        all_detected_text_snippets = []
        frame_files = sorted(
            [
                os.path.join(frame_folder, f)
                for f in os.listdir(frame_folder)
                if f.endswith((".jpg", ".png", ".jpeg"))
            ]
        )

        for frame_path in frame_files:
            results = self.reader.readtext(frame_path)

            for bbox, text, confidence in results:
                cleaned_text = text.strip()
                if len(cleaned_text) > 4 and confidence > 0.4:
                    if cleaned_text not in all_detected_text_snippets:
                        all_detected_text_snippets.append(cleaned_text)

        return {
            "raw_text_metadata": all_detected_text_snippets,
            "label_ocr_success": len(all_detected_text_snippets) > 0,
        }