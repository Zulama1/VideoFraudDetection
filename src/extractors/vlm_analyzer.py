import base64
import json
import os
import cv2
import requests


class VLMAnalyzer:

    def __init__(self, api_key: str = None):
        """Initializes the VLM Analyzer client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def _encode_image(self, image_path: str) -> str:
        """Encodes a local image frame to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def analyze_unboxing_frames(
        self, frame_folder: str, prompt_template: str = None
    ) -> dict:
        """Sends a subset of extracted frames to the VLM to analyze video

        contents.
        """
        if not self.api_key:
            raise ValueError(
                "API Key is missing. Set OPENAI_API_KEY environment variable."
            )

        # Gather and sort frame images
        all_frames = sorted(
            [
                os.path.join(frame_folder, f)
                for f in os.listdir(frame_folder)
                if f.endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        if not all_frames:
            return {"error": "No frames found to analyze."}

        # Subsample frames to fit within API context windows (e.g., max 10 keyframes)
        max_keyframes = 10
        step = max(1, len(all_frames) // max_keyframes)
        selected_keyframes = all_frames[::step][:max_keyframes]

        # Structure the payload content list with the text prompt first
        default_prompt = (
            "Analyze these chronological frames from an e-commerce unboxing video. "
            "Determine if the shipping package state is initially fully sealed, identify the items extracted, "
            "and assess whether there is any visual evidence of fraud/tampering. "
            "Respond strictly with a JSON object containing keys: 'package_initially_sealed' (bool), "
            "'items_detected' (list of strings), 'tampering_detected' (bool), and 'confidence_score' (float)."
        )

        content_payload = [{"type": "text", "text": prompt_template or default_prompt}]

        # Append base64 encoded images to the payload
        for frame_path in selected_keyframes:
            base64_image = self._encode_image(frame_path)
            content_payload.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                }
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": content_payload}],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        try:
            response = requests.post(
                self.api_url, headers=headers, json=payload
            )
            response.raise_for_status()
            result_json = response.json()["choices"][0]["message"]["content"]
            return json.loads(result_json)
        except Exception as e:
            return {"error": f"VLM API request failed: {str(e)}"}