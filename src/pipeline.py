import glob
import os
import json
from src.preprocessing.video_splitter import VideoSplitter
from src.extractors.vlm_analyzer import VLMAnalyzer
from src.extractors.object_tracker import ObjectTracker
from src.extractors.ocr_engine import OCREngine
from src.risk_engine.classifier import FraudClassifier
from src.risk_engine.rules import RuleEngine


class ECommerceUnboxingPipeline:

    def __init__(self, gemini_api_key: str = None):
        """Initializes all three processing stages of the unboxing verification pipeline."""
        # Stage 1: Preprocessing
        self.splitter = VideoSplitter(frame_rate_fps=2, scene_cut_threshold=35.0)

        # Stage 2: AI Feature Extraction
        self.vlm_analyzer = VLMAnalyzer(api_key=gemini_api_key)
        self.object_tracker = ObjectTracker(model_weight="yolo11n.pt")
        self.ocr_engine = OCREngine(languages=["en"])

        # Stage 3: Decision Making
        self.rule_engine = RuleEngine(weight_cuts=0.4, weight_seal=0.6)
        self.classifier = FraudClassifier()  # Fallback logic used since model is untrained

    def process_single_video(self, video_path: str, output_base_dir: str, order_manifest: dict) -> dict:
        video_file_name = os.path.basename(video_path)
        video_slug = os.path.splitext(video_file_name)[0]
        
        # Frame extraction destination directory
        # video_splitter handles nesting the slug folder internally, so we pass base directory
        print(f"\nExtracting frames & detecting cuts for: {video_file_name}")
        preprocessing_meta = self.splitter.process_video(
            video_path=video_path, output_dir=output_base_dir
        )
        
        # Build exact frame path target to pass over to extractor engines
        frame_folder = os.path.join(output_base_dir, video_slug)
        
        # 1. Gemini VLM Analysis
        print("Calling Gemini VLM (2.5-Flash):")
        vlm_res = self.vlm_analyzer.analyze_unboxing_frames(frame_folder)
        
        # 2. YOLO Object Tracking
        print("Running YOLO Object Tracker:")
        yolo_res = self.object_tracker.track_objects_in_frames(frame_folder)
        
        # 3. Shipping Label OCR Engine
        print("Extracting text with EasyOCR:")
        ocr_res = self.ocr_engine.extract_text_from_labels(frame_folder)

        # Unified aggregate payload construction to resolve the data structural mismatch
        combined_extractor_meta = {
            "package_initially_sealed": vlm_res.get("package_initially_sealed", True),
            "items_detected": vlm_res.get("items_detected", []),
            "raw_text_metadata": ocr_res.get("raw_text_metadata", []),
            "yolo_objects_inventory": yolo_res.get("extracted_objects_inventory", []),
            "vlm_confidence": vlm_res.get("confidence_score", 1.0),
            "vlm_tampering_flag": vlm_res.get("tampering_detected", False)
        }

        # Computing final risk scores
        # Evaluate deterministic rules
        rule_results = self.rule_engine.evaluate_static_rules(
            preprocessing_meta=preprocessing_meta, 
            extractor_meta=combined_extractor_meta
        )

        # Infer risk using ML Classifier
        classifier_results = self.classifier.predict_risk(
            preprocessing_meta=preprocessing_meta,
            extractor_meta=combined_extractor_meta,
            order_manifest=order_manifest
        )

        # Assemble full comprehensive audit trail package
        pipeline_audit_report = {
            "video_identity": preprocessing_meta["video_name"],
            "order_reference": order_manifest.get("order_id", "UNKNOWN"),
            "stage_1_preprocessing": preprocessing_meta,
            "stage_2_raw_extracted_features": combined_extractor_meta,
            "stage_3_risk_evaluation": {
                "heuristic_rules": rule_results,
                "ml_classifier": classifier_results
            }
        }

        return pipeline_audit_report


def run_batch_pipeline(raw_dir: str, output_base_dir: str, mock_manifests: dict):
    """Orchestrates continuous flow for all videos discovered inside raw folder data."""
    # Initialize the entire processing engine pipeline
    pipeline = ECommerceUnboxingPipeline()

    video_extensions = ["*.mp4", "*.mkv", "*.avi", "*.mov"]
    video_paths = []

    for ext in video_extensions:
        video_paths.extend(glob.glob(os.path.join(raw_dir, ext)))

    if not video_paths:
        print(f"No videos found in raw source folder: {raw_dir}")
        return

    print(f"Found {len(video_paths)} video(s). Commencing Orchestrator Execution Pipeline")

    for video_path in video_paths:
        video_file_name = os.path.basename(video_path)
        video_slug = os.path.splitext(video_file_name)[0]

        # Fetch matching order item requirements or fallback to generic layout mock manifest
        order_manifest = mock_manifests.get(
            video_file_name, 
            {
                "order_id": f"MOCK_{video_slug.upper()}",
                "tracking_number": "TRK123456789",
                "expected_items": ["cell phone", "laptop"]
            }
        )

        try:
            report = pipeline.process_single_video(
                video_path=video_path, 
                output_base_dir=output_base_dir, 
                order_manifest=order_manifest
            )
            
            # Print explicit summary log data
            print(f"\n AUDIT REPORT FOR: {video_file_name}")
            print(f"Fraud Probability (ML): {report['stage_3_risk_evaluation']['ml_classifier']['fraud_probability'] * 100}%")
            print(f"Heuristic Risk Score:  {report['stage_3_risk_evaluation']['heuristic_rules']['heuristic_risk_score']}")
            print(f"Requires Manual Review: {report['stage_3_risk_evaluation']['heuristic_rules']['requires_manual_review']}")
            print(f"Flagged Violations:     {report['stage_3_risk_evaluation']['heuristic_rules']['rule_violations']}")

        except Exception as e:
            print(f"Critical system breakdown during parsing of {video_file_name}. Error: {e}")


if __name__ == "__main__":
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed/frames"

    # Mock DB linking video source filenames to actual backend e-commerce customer transaction history
    MOCK_ORDER_DATABASE = {
        "fake_unboxing_scam.mp4": {
            "order_id": "ORD98124",
            "tracking_number": "1Z999AA10123456784",
            "expected_items": ["iPhone 15 Pro", "Charging Cable"]
        },
        "legit_unboxing.mp4": {
            "order_id": "ORD55210",
            "tracking_number": "LX882311245US",
            "expected_items": ["Mechanical Keyboard", "Gaming Mouse"]
        }
    }

    # Ensure local directory folders are ready before starting execution paths
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    run_batch_pipeline(
        raw_dir=RAW_DATA_DIR, 
        output_base_dir=PROCESSED_DATA_DIR, 
        mock_manifests=MOCK_ORDER_DATABASE
    )