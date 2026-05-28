class RuleEngine:

    def __init__(self, weight_cuts: float = 0.4, weight_seal: float = 0.6):
        """Initializes rule weights for deterministic anomaly scoring."""
        self.weight_cuts = weight_cuts
        self.weight_seal = weight_seal

    def evaluate_static_rules(
        self, preprocessing_meta: dict, extractor_meta: dict
    ) -> dict:
        """Evaluates extracted visual metadata against strict risk rules.

        Args:
            preprocessing_meta (dict): Output from your VideoSplitter.
            extractor_meta (dict): Merged output from your VLM / Trackers.

        Returns:
            dict: Structured rule violations and a calculated heuristic risk
            score.
        """
        violations = []
        heuristic_score = 0.0

        # Rule 1: Check for edited footage (Scene cuts are a major red flag)
        cuts = preprocessing_meta.get("scene_cuts_detected", 0)
        if cuts > 0:
            violations.append(f"CRITICAL: {cuts} video cuts/edits detected.")
            heuristic_score += self.weight_cuts * min(1.0, cuts * 0.5)

        # Rule 2: Verify if package was unsealed at the start of filming
        was_sealed = extractor_meta.get("package_initially_sealed", True)
        if not was_sealed:
            violations.append(
                "CRITICAL: Package was already unsealed or open at frame zero."
            )
            heuristic_score += self.weight_seal

        # Normalize score bounds
        heuristic_score = min(1.0, heuristic_score)

        return {
            "rule_violations": violations,
            "heuristic_risk_score": round(heuristic_score, 2),
            "requires_manual_review": heuristic_score >= 0.4,
        }