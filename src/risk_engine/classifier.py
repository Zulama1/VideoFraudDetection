import numpy as np
import xgboost as xgb


class FraudClassifier:

    def __init__(self, model_path: str = None):
        self.model = xgb.XGBClassifier()
        if model_path:
            self.load_model(model_path)
        else:
            # Placeholder initialization for an untrained model configuration
            self.is_trained = False

    def _engineer_features(self, preprocessing_meta: dict, extractor_meta: dict, order_manifest: dict) -> np.ndarray:
        
        # 1. Video edits feature
        scene_cuts = float(preprocessing_meta.get("scene_cuts_detected", 0))

        # 2. Base seal integrity feature
        is_sealed = (
            1.0 if extractor_meta.get("package_initially_sealed", True) else 0.0
        )

        # 3. Text overlap match feature 
        expected_tracking = order_manifest.get("tracking_number", "").lower()
        ocr_texts = [
            t.lower() for t in extractor_meta.get("raw_text_metadata", [])
        ]

        label_match = 0.0
        if expected_tracking:
            for text in ocr_texts:
                if expected_tracking in text or text in expected_tracking:
                    label_match = 1.0
                    break

        # 4. Item content validation feature 
        expected_items = set(
            [item.lower() for item in order_manifest.get("expected_items", [])]
        )
        detected_items = set(
            [item.lower() for item in extractor_meta.get("items_detected", [])]
        )

        item_mismatch_count = 0.0
        if expected_items:
            missing_items = expected_items - detected_items
            item_mismatch_count = float(len(missing_items))

        return np.array(
            [[scene_cuts, is_sealed, label_match, item_mismatch_count]],
            dtype=np.float32,
        )

    def predict_risk(self, preprocessing_meta: dict, extractor_meta: dict, order_manifest: dict) -> dict:
        features = self._engineer_features(
            preprocessing_meta, extractor_meta, order_manifest
        )

        if not self.is_trained:
            base_prob = 0.1
            if features[0][0] > 0:
                base_prob += 0.4  # Add risk for cuts
            if features[0][1] == 0:
                base_prob += 0.4  # Add risk for open box
            return {
                "fraud_probability": min(0.99, base_prob),
                "model_status": "untrained_fallback_estimation",
            }

        probability = self.model.predict_proba(features)[0][1]

        return {
            "fraud_probability": round(float(probability), 4),
            "model_status": "inferred_by_xgboost",
        }

    def save_model(self, path: str):
        self.model.save_model(path)

    def load_model(self, path: str):
        self.model.load_model(path)
        self.is_trained = True