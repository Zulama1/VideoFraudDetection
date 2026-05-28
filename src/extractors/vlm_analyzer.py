import os
from typing import List
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# 1. Define the desired output schema using Pydantic for guaranteed structured JSON
class UnboxingAnalysis(BaseModel):
    package_initially_sealed: bool = Field(
        description="Determine if the shipping package state is initially fully sealed."
    )
    items_detected: List[str] = Field(
        description="Identify the list of items extracted from the package."
    )
    tampering_detected: bool = Field(
        description="Assess whether there is any visual evidence of fraud/tampering."
    )
    confidence_score: float = Field(
        description="Confidence score of the assessment between 0.0 and 1.0."
    )


class VLMAnalyzer:

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        self.client = genai.Client(api_key=self.api_key)

    def _prepare_image_part(self, image_path: str) -> types.Part:
        """Reads a local image frame and converts it into a Gemini API Part object."""
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        # Determine mime type from extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"

        # Return the official Part object from bytes
        return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    def analyze_unboxing_frames(self, frame_folder: str, prompt_template: str = None) -> dict:
        """Sends a subset of extracted frames to the Gemini VLM to analyze video contents."""

        all_frames = sorted(
            [
                os.path.join(frame_folder, f)
                for f in os.listdir(frame_folder)
                if f.endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        if not all_frames:
            return {"error": "No frames found to analyze."}

        max_keyframes = 10
        step = max(1, len(all_frames) // max_keyframes)
        selected_keyframes = all_frames[::step][:max_keyframes]

        
        default_prompt = (
            "Analyze these chronological frames from an e-commerce unboxing video. "
            "Determine if the shipping package state is initially fully sealed, identify the items extracted, "
            "and assess whether there is any visual evidence of fraud/tampering."
        )

        contents_payload = [prompt_template or default_prompt]

        for frame_path in selected_keyframes:
            image_part = self._prepare_image_part(frame_path)
            contents_payload.append(image_part)

        try:
            # Generate structured content using Gemini
            # Using 'gemini-2.5-flash' for optimal multi-modal processing speed and cost
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents_payload,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=UnboxingAnalysis,
                    temperature=0.2,
                ),
            )

            # When response_schema is passed, the SDK automatically parses the JSON text
            # into an object accessible via response.parsed
            if response.parsed:
                return response.parsed.model_dump()
            else:
                return {"error": "Failed to parse structured JSON from model."}

        except Exception as e:
            return {"error": f"Gemini VLM API request failed: {str(e)}"}