"""
explanation_engine.py
=====================
Phase 7 - Explainability: Natural Language Explanation Generator

PURPOSE:
    Generates human-readable, natural language explanations for every
    infrastructure recommendation made by the SmartCityAI system.

    City planners and government officials cannot act on a black-box
    "Zone XYZ: Priority 91". They need:
    - WHY this zone was selected
    - WHAT metrics drove the decision
    - HOW confident the AI is
    - WHAT-IF alternative scenarios

EXPLANATION COMPONENTS:
    1. Feature-level explanation (SHAP / Decision Tree rules)
    2. Natural language summary template
    3. Severity classification (CRITICAL / HIGH / MODERATE / LOW)
    4. Comparison with city average
    5. Confidence level

EXAMPLE OUTPUT:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ ZONE H3-8928308280FFFFF — HOSPITAL RECOMMENDATION                  │
    │                                                                     │
    │ RECOMMENDATION: BUILD NEW HOSPITAL                                  │
    │ PRIORITY SCORE: 91/100 (HIGH PRIORITY)                             │
    │                                                                     │
    │ WHY THIS LOCATION WAS SELECTED:                                    │
    │ ✗ Population Density:    32,450 /km² [CRITICAL — 2.1× city avg]   │
    │ ✗ Nearest Hospital:      6.2 km     [CRITICAL — WHO: ≤5 km]       │
    │ ✗ Coverage Gap:          72%        [SEVERE — no hospital in 5km]  │
    │ ✓ Road Accessibility:    0.74       [GOOD — arterial road nearby]  │
    │ ✗ Elderly Population:    14.2%      [HIGH — frequent hospital use] │
    │ ✗ Income Level:          Low        [EQUITY PRIORITY]              │
    │                                                                     │
    │ KEY DECISION RULE (Decision Tree):                                  │
    │   IF coverage_gap > 0.6 AND pop_density > 20000                    │
    │   AND elderly_ratio > 0.1 → HIGH PRIORITY (confidence: 94%)       │
    │                                                                     │
    │ ESTIMATED IMPACT: ~89,000 residents newly within 5km of hospital   │
    └─────────────────────────────────────────────────────────────────────┘

METHODS:
    - SHAP (SHapley Additive exPlanations): Feature attribution for SVM
    - Decision Tree rule extraction: Path from root to leaf for DT
    - Statistical comparison: z-score vs city mean for each feature
    - Template-based NL generation: f-string templates with severity labels

WHY SHAP?
    - Game-theoretic foundation (Shapley values from cooperative game theory)
    - Locally faithful (explains individual predictions, not averages)
    - Works with ANY model (SVM, KMeans, Ridge, etc.)
    - Additive: feature contributions sum to (prediction - base_rate)
    - Industry standard for ML explainability (Airbnb, Microsoft, Zillow)

USAGE:
    from explanation_engine import ExplanationEngine
    engine = ExplanationEngine(df=zone_df, model=svm_model, feature_cols=features)
    explanation = engine.explain(h3_id="89283082803ffff", infra_type="hospital")
    print(explanation.to_text())
    print(explanation.to_html())
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("explanation_engine")

BASE_DIR      = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"

# Feature severity thresholds (for natural language labels)
FEATURE_THRESHOLDS: Dict[str, Dict] = {
    "coverage_gap": {
        "unit": "",
        "critical": 0.70,
        "high":     0.50,
        "moderate": 0.30,
        "low":      0.10,
        "good_if":  "low",  # Low gap = good
        "label":    "Coverage Gap",
        "description": "fraction of zone not within service radius of any facility",
    },
    "population_density": {
        "unit": "/km²",
        "critical": 30000,
        "high":     15000,
        "moderate": 8000,
        "low":      3000,
        "good_if":  "high",  # High density = high need
        "label":    "Population Density",
        "description": "persons per square kilometer",
    },
    "dist_nearest_hospital": {
        "unit": "km",
        "critical": 7.0,
        "high":     5.0,
        "moderate": 3.0,
        "low":      1.5,
        "good_if":  "low",  # Low distance = well-served
        "label":    "Distance to Nearest Hospital",
        "description": "Haversine distance to closest hospital/clinic",
    },
    "elderly_ratio": {
        "unit": "%",
        "critical": 0.15,
        "high":     0.10,
        "moderate": 0.07,
        "low":      0.04,
        "good_if":  "low",
        "label":    "Elderly Population Ratio",
        "description": "fraction of population aged 60+",
        "multiplier": 100,  # Display as percentage
    },
    "income_bracket_norm": {
        "unit": "",
        "critical": 0.20,  # Very poor (inverted: low = bad)
        "high":     0.35,
        "moderate": 0.55,
        "low":      0.75,
        "good_if":  "high",
        "label":    "Income Level (normalized)",
        "description": "normalized median household income [0=low, 1=high]",
    },
    "road_accessibility_index": {
        "unit": "",
        "critical": 0.15,  # Low access = bad (inverted)
        "high":     0.30,
        "moderate": 0.50,
        "low":      0.70,
        "good_if":  "high",
        "label":    "Road Accessibility Index",
        "description": "weighted road density [0=poor, 1=excellent]",
    },
    "emergency_response_time_min": {
        "unit": " min",
        "critical": 12.0,
        "high":     8.0,
        "moderate": 6.0,
        "low":      4.0,
        "good_if":  "low",
        "label":    "Emergency Response Time",
        "description": "estimated travel time for emergency vehicle",
    },
    "vulnerability_index": {
        "unit": "",
        "critical": 0.70,
        "high":     0.50,
        "moderate": 0.30,
        "low":      0.15,
        "good_if":  "low",
        "label":    "Vulnerability Index",
        "description": "compound social vulnerability score [0=low, 1=high]",
    },
}

# Severity labels and icons
SEVERITY = {
    "CRITICAL":  "🔴",
    "HIGH":      "🟠",
    "MODERATE":  "🟡",
    "LOW":       "🟢",
    "EXCELLENT": "✅",
    "N/A":       "⚪",
}

# Infrastructure-specific recommendations
INFRA_RECOMMENDATIONS = {
    "hospital":     "Build new primary hospital / polyclinic",
    "school":       "Establish new government school",
    "ev_station":   "Install EV charging hub (6-8 charging points)",
    "fire_station": "Construct new fire station",
}

# Benchmark values (WHO/NFPA standards)
BENCHMARKS = {
    "dist_nearest_hospital":      ("WHO guideline", 5.0, "km"),
    "emergency_response_time_min": ("NFPA 1710",    6.0, "min"),
    "hospital_coverage_ratio":    ("WHO target",    0.9, ""),
    "dist_nearest_school":        ("Norm standard", 2.0, "km"),
}


# ── Explanation Data Classes ──────────────────────────────────────────────────

@dataclass
class FeatureExplanation:
    """Explanation for a single feature's contribution."""
    feature:   str
    value:     float
    severity:  str         # CRITICAL / HIGH / MODERATE / LOW / EXCELLENT
    label:     str
    unit:      str
    description: str
    city_avg:  Optional[float] = None
    z_score:   Optional[float] = None
    shap_value: Optional[float] = None
    benchmark: Optional[Tuple[str, float, str]] = None  # (source, value, unit)
    contribution: float = 0.0  # Share of priority score driven by this feature


@dataclass
class ZoneExplanation:
    """
    Complete explanation for a zone recommendation.

    Includes:
    - Overall recommendation and priority score
    - Feature-level breakdown
    - Decision rule from Decision Tree
    - Natural language summary
    - Estimated impact
    """
    h3_id:            str
    infra_type:       str
    priority_score:   float
    priority_class:   str
    recommendation:   str
    confidence:       float
    features:         List[FeatureExplanation] = field(default_factory=list)
    decision_rule:    str = ""
    nl_summary:       str = ""
    estimated_impact: str = ""
    lat:              float = 0.0
    lon:              float = 0.0
    shap_values:      Optional[Dict] = None

    def to_text(self) -> str:
        """Generate plain text explanation."""
        lines = [
            "=" * 70,
            f"ZONE {self.h3_id} — {self.infra_type.upper()} RECOMMENDATION",
            "=" * 70,
            f"RECOMMENDATION:  {self.recommendation.upper()}",
            f"PRIORITY SCORE:  {self.priority_score:.0f}/100 ({self.priority_class.upper()} PRIORITY)",
            f"CONFIDENCE:      {self.confidence*100:.0f}%",
            "",
            "WHY THIS LOCATION WAS SELECTED:",
            "-" * 40,
        ]

        for fe in self.features:
            icon = SEVERITY.get(fe.severity, "⚪")
            val_display = fe.value * fe.unit.count("%") if "%" not in fe.unit else fe.value * 100
            val_str = f"{fe.value:.1f}{fe.unit}" if "%" not in fe.unit else f"{fe.value*100:.1f}%"
            avg_str = f" (city avg: {fe.city_avg:.2f})" if fe.city_avg is not None else ""
            lines.append(f"  {icon} {fe.label:30s}: {val_str:12s} [{fe.severity}]{avg_str}")

        if self.decision_rule:
            lines += ["", "KEY DECISION RULE:", f"  {self.decision_rule}"]

        if self.estimated_impact:
            lines += ["", "ESTIMATED IMPACT:", f"  {self.estimated_impact}"]

        if self.nl_summary:
            lines += ["", "SUMMARY:", f"  {self.nl_summary}"]

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "h3_id":           self.h3_id,
            "infra_type":      self.infra_type,
            "priority_score":  self.priority_score,
            "priority_class":  self.priority_class,
            "recommendation":  self.recommendation,
            "confidence":      self.confidence,
            "lat":             self.lat,
            "lon":             self.lon,
            "decision_rule":   self.decision_rule,
            "nl_summary":      self.nl_summary,
            "estimated_impact": self.estimated_impact,
            "features": [
                {
                    "feature":    fe.feature,
                    "value":      fe.value,
                    "severity":   fe.severity,
                    "label":      fe.label,
                    "unit":       fe.unit,
                    "city_avg":   fe.city_avg,
                    "z_score":    fe.z_score,
                    "contribution": fe.contribution,
                }
                for fe in self.features
            ],
        }

    def to_html(self) -> str:
        """Generate HTML explanation card (for Streamlit)."""
        severity_colors = {
            "CRITICAL":  "#E74C3C",
            "HIGH":      "#E67E22",
            "MODERATE":  "#F39C12",
            "LOW":       "#27AE60",
            "EXCELLENT": "#27AE60",
        }

        priority_color = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#27AE60"}.get(
            self.priority_class, "#95A5A6"
        )

        rows = ""
        for fe in self.features:
            color = severity_colors.get(fe.severity, "#95A5A6")
            val_str = f"{fe.value:.3f}{fe.unit}"
            avg_str = f"(avg: {fe.city_avg:.3f})" if fe.city_avg else ""
            rows += f"""
            <tr>
                <td style='padding:4px'>{fe.label}</td>
                <td style='padding:4px;font-weight:bold'>{val_str}</td>
                <td style='padding:4px;color:{color};font-weight:bold'>{fe.severity}</td>
                <td style='padding:4px;color:#999;font-size:0.85em'>{avg_str}</td>
            </tr>"""

        return f"""
        <div style='border:2px solid {priority_color};border-radius:12px;padding:16px;margin:8px 0'>
          <h3 style='color:{priority_color};margin:0'>
            Zone {self.h3_id[:15]}... — {self.infra_type.title()}
          </h3>
          <div style='display:flex;gap:16px;margin:8px 0'>
            <span style='background:{priority_color};color:white;padding:4px 12px;border-radius:20px;font-weight:bold'>
              Priority: {self.priority_score:.0f}/100
            </span>
            <span style='color:{priority_color};font-weight:bold'>{self.priority_class.upper()} PRIORITY</span>
            <span style='color:#666'>Confidence: {self.confidence*100:.0f}%</span>
          </div>
          <p style='color:#666;margin:4px 0'>{self.nl_summary}</p>
          <table style='width:100%;border-collapse:collapse;margin:8px 0'>
            <tr style='background:#f0f0f0'>
              <th style='padding:4px;text-align:left'>Feature</th>
              <th style='padding:4px;text-align:left'>Value</th>
              <th style='padding:4px;text-align:left'>Status</th>
              <th style='padding:4px;text-align:left'>vs City Average</th>
            </tr>
            {rows}
          </table>
          {'<p style="color:#444;font-style:italic;font-size:0.9em">Rule: ' + self.decision_rule + '</p>' if self.decision_rule else ''}
          <p style='color:#2980B9;font-size:0.9em'>📍 Impact: {self.estimated_impact}</p>
        </div>"""


# ── Feature Severity Classifier ───────────────────────────────────────────────

def classify_severity(
    feature: str,
    value: float,
    city_avg: Optional[float] = None,
) -> str:
    """
    Classify a feature value as CRITICAL/HIGH/MODERATE/LOW/EXCELLENT.

    Uses predefined thresholds for known features.
    Falls back to z-score comparison for unknown features.

    Args:
        feature:  Feature name
        value:    Feature value
        city_avg: Optional city-wide average for context

    Returns:
        Severity string
    """
    if feature not in FEATURE_THRESHOLDS:
        if city_avg is not None:
            ratio = value / (city_avg + 1e-9)
            if ratio > 2.0:   return "CRITICAL"
            if ratio > 1.5:   return "HIGH"
            if ratio > 1.0:   return "MODERATE"
            if ratio > 0.7:   return "LOW"
            return "EXCELLENT"
        return "N/A"

    thresh = FEATURE_THRESHOLDS[feature]
    good_if = thresh.get("good_if", "low")

    if good_if == "low":
        # Low value = good (distances, gaps)
        if value >= thresh["critical"]: return "CRITICAL"
        if value >= thresh["high"]:     return "HIGH"
        if value >= thresh["moderate"]: return "MODERATE"
        if value >= thresh["low"]:      return "LOW"
        return "EXCELLENT"
    else:
        # High value = good (income, accessibility)
        if value <= thresh["critical"]: return "CRITICAL"
        if value <= thresh["high"]:     return "HIGH"
        if value <= thresh["moderate"]: return "MODERATE"
        if value <= thresh["low"]:      return "LOW"
        return "EXCELLENT"


# ── NL Summary Generator ──────────────────────────────────────────────────────

def generate_nl_summary(
    zone: pd.Series,
    infra_type: str,
    priority_score: float,
    priority_class: str,
    feature_explanations: List[FeatureExplanation],
) -> str:
    """
    Generate a human-readable natural language summary.

    Uses template-based generation with dynamic values.
    More robust and explainable than LLM-generated text.

    Args:
        zone:                Zone feature row
        infra_type:          Infrastructure type
        priority_score:      Computed priority score
        priority_class:      High/Medium/Low
        feature_explanations: List of FeatureExplanation objects

    Returns:
        Natural language explanation string
    """
    # Extract key metrics
    pop_density = zone.get("population_density", 0)
    coverage_gap = zone.get("coverage_gap", zone.get(f"{infra_type}_coverage_gap", 0))
    dist_hosp = zone.get("dist_nearest_hospital", 0)
    elderly = zone.get("elderly_ratio", 0) * 100
    income = zone.get("income_bracket_norm", 0.5)

    # Priority description
    priority_desc = {
        "High":   "urgent attention",
        "Medium": "timely attention",
        "Low":    "monitoring",
    }.get(priority_class, "review")

    # Coverage gap phrase
    if coverage_gap > 0.7:
        gap_phrase = f"has a critical coverage gap ({coverage_gap*100:.0f}% of population unserved)"
    elif coverage_gap > 0.5:
        gap_phrase = f"has a significant coverage gap ({coverage_gap*100:.0f}% unserved)"
    elif coverage_gap > 0.3:
        gap_phrase = f"has moderate infrastructure deficit ({coverage_gap*100:.0f}% unserved)"
    else:
        gap_phrase = f"has manageable coverage needs ({coverage_gap*100:.0f}% unserved)"

    # Population phrase
    if pop_density > 20000:
        pop_phrase = f"extremely dense ({pop_density:,.0f}/km²)"
    elif pop_density > 10000:
        pop_phrase = f"highly dense ({pop_density:,.0f}/km²)"
    elif pop_density > 5000:
        pop_phrase = f"moderately dense ({pop_density:,.0f}/km²)"
    else:
        pop_phrase = f"lower density ({pop_density:,.0f}/km²)"

    # Equity phrase
    equity_phrase = ""
    if income < 0.3:
        equity_phrase = " This is a low-income area requiring equitable infrastructure access."
    elif elderly > 12:
        equity_phrase = f" The high elderly population ({elderly:.1f}%) significantly increases demand."

    # Infrastructure specific
    infra_context = {
        "hospital":     f"The nearest healthcare facility is {dist_hosp:.1f} km away.",
        "school":       f"School-age population requires improved access to education.",
        "ev_station":   f"Growing EV adoption in this zone is underserved.",
        "fire_station": f"Emergency response time exceeds NFPA 1710 standards.",
    }.get(infra_type, "")

    summary = (
        f"This zone requires {priority_desc} for {infra_type.replace('_', ' ')} infrastructure. "
        f"The area is {pop_phrase} and {gap_phrase}. "
        f"{infra_context}{equity_phrase}"
        f" Priority Score: {priority_score:.0f}/100."
    )

    return summary


# ── Decision Rule Extractor ───────────────────────────────────────────────────

def extract_decision_rule_for_zone(
    zone_features: np.ndarray,
    dt_model_path: Optional[Path] = None,
    feature_names: Optional[List[str]] = None,
) -> str:
    """
    Extract the specific Decision Tree path for this zone.

    Traverses from root to leaf, collecting the split conditions
    that led to this zone's classification.

    Args:
        zone_features: 1D feature array for this zone
        dt_model_path: Path to saved DT model pickle
        feature_names: Feature column names

    Returns:
        Human-readable IF-THEN rule string
    """
    if dt_model_path is None:
        dt_model_path = MODELS_DIR / "decision_tree_model.pkl"

    if not dt_model_path.exists():
        return "Decision Tree model not found. Run decision_tree.py first."

    try:
        with open(dt_model_path, "rb") as f:
            dt_data = pickle.load(f)
        model   = dt_data["model"]
        features = dt_data.get("features", feature_names or [])

        # Get decision path
        tree = model.tree_
        node_id = 0
        conditions = []

        while tree.feature[node_id] != -2:  # -2 = leaf node
            feat_idx  = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feat_name = features[feat_idx] if feat_idx < len(features) else f"f{feat_idx}"
            feat_val  = zone_features[feat_idx] if feat_idx < len(zone_features) else 0

            if feat_val <= threshold:
                conditions.append(f"{feat_name} ≤ {threshold:.4f} (actual: {feat_val:.4f})")
                node_id = tree.children_left[node_id]
            else:
                conditions.append(f"{feat_name} > {threshold:.4f} (actual: {feat_val:.4f})")
                node_id = tree.children_right[node_id]

        # Leaf class
        class_counts = tree.value[node_id][0]
        total = sum(class_counts)
        best_class_idx = int(np.argmax(class_counts))
        classes = model.classes_
        predicted_class = classes[best_class_idx]
        confidence = class_counts[best_class_idx] / total if total > 0 else 0

        rule = "IF " + " AND ".join(conditions[:3])  # Top 3 conditions
        rule += f" → {predicted_class.upper()} (confidence: {confidence*100:.0f}%)"
        return rule

    except Exception as exc:
        log.warning(f"Could not extract DT rule: {exc}")
        return "Decision rule extraction failed."


# ── SHAP Integration ──────────────────────────────────────────────────────────

def compute_shap_values(
    zone_features: np.ndarray,
    model_pipeline,  # sklearn Pipeline
    feature_names: List[str],
) -> Optional[Dict[str, float]]:
    """
    Compute SHAP values for a single zone.

    SHAP (SHapley Additive exPlanations):
    - φ_i = contribution of feature i to prediction vs baseline
    - Σ φ_i = (prediction - base_value)
    - Game-theoretic (Shapley): unique fair attribution

    For SVM: Uses KernelExplainer (model-agnostic, ~100 background samples)
    For Tree: Uses TreeExplainer (exact, fast)

    Args:
        zone_features: 1D feature array
        model_pipeline: Fitted sklearn Pipeline (scaler + SVM/Tree)
        feature_names: Feature column names

    Returns:
        Dict of feature → SHAP value, or None if SHAP unavailable
    """
    try:
        import shap

        # Use KernelExplainer for SVM (model-agnostic)
        # Background: small random sample from training data
        if hasattr(model_pipeline, "named_steps"):
            scaler = model_pipeline.named_steps.get("scaler")
            if scaler:
                zone_scaled = scaler.transform(zone_features.reshape(1, -1))
            else:
                zone_scaled = zone_features.reshape(1, -1)
        else:
            zone_scaled = zone_features.reshape(1, -1)

        explainer = shap.KernelExplainer(
            model_pipeline.predict_proba,
            shap.sample(zone_scaled, 50),  # 50 background samples
        )

        shap_vals = explainer.shap_values(zone_scaled, nsamples=200)

        if isinstance(shap_vals, list):
            # Multi-class: take values for "High" class
            high_class_idx = 0  # Adjust based on label encoding
            shap_feature_vals = shap_vals[high_class_idx][0]
        else:
            shap_feature_vals = shap_vals[0]

        return {feat: float(val) for feat, val in zip(feature_names, shap_feature_vals)}

    except ImportError:
        log.info("SHAP not installed. pip install shap for SHAP explanations.")
        return None
    except Exception as exc:
        log.debug(f"SHAP computation failed: {exc}")
        return None


# ── Main Explanation Engine ───────────────────────────────────────────────────

class ExplanationEngine:
    """
    Master explanation engine for SmartCityAI.

    Combines SHAP, Decision Tree rules, statistical comparison,
    and template NL generation into comprehensive zone explanations.

    Usage:
        engine = ExplanationEngine.from_saved_models()
        explanation = engine.explain_zone("89283082803ffff", "hospital")
        print(explanation.to_text())
    """

    def __init__(
        self,
        df: pd.DataFrame,
        svm_pipeline=None,
        dt_model_path: Optional[Path] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        self.df           = df
        self.svm_pipeline = svm_pipeline
        self.dt_path      = dt_model_path or MODELS_DIR / "decision_tree_model.pkl"
        self.feature_cols = feature_cols or self._default_features()
        self._compute_city_stats()

    def _default_features(self) -> List[str]:
        return [c for c in FEATURE_THRESHOLDS.keys() if c in self.df.columns]

    def _compute_city_stats(self) -> None:
        """Compute city-wide mean/std for each feature."""
        numeric = self.df.select_dtypes(include=np.number)
        self.city_mean = numeric.mean().to_dict()
        self.city_std  = numeric.std().to_dict()

    @classmethod
    def from_saved_models(cls, df: pd.DataFrame) -> "ExplanationEngine":
        """Load saved models and create ExplanationEngine."""
        svm_pipeline = None
        svm_path = MODELS_DIR / "svm_model.pkl"
        if svm_path.exists():
            with open(svm_path, "rb") as f:
                data = pickle.load(f)
            svm_pipeline = data.get("pipeline")
            log.info("Loaded SVM model for SHAP")

        return cls(df=df, svm_pipeline=svm_pipeline)

    def explain_zone(
        self,
        h3_id: str,
        infra_type: str = "hospital",
    ) -> ZoneExplanation:
        """
        Generate complete explanation for a zone.

        Args:
            h3_id:      H3 zone identifier
            infra_type: Infrastructure type

        Returns:
            ZoneExplanation object
        """
        # Find zone row
        if "h3_id" in self.df.columns:
            zone_rows = self.df[self.df["h3_id"] == h3_id]
        else:
            zone_rows = self.df.iloc[[0]]  # Fallback

        if len(zone_rows) == 0:
            log.warning(f"Zone {h3_id} not found in data")
            zone = self.df.iloc[0].copy()
        else:
            zone = zone_rows.iloc[0]

        # Feature explanations
        feature_explanations = []
        for feat_name, config in FEATURE_THRESHOLDS.items():
            if feat_name not in zone.index:
                continue

            val = float(zone[feat_name])
            city_avg = self.city_mean.get(feat_name)
            city_std = self.city_std.get(feat_name)
            z_score = (val - city_avg) / (city_std + 1e-9) if city_avg and city_std else None

            severity = classify_severity(feat_name, val, city_avg)
            multiplier = config.get("multiplier", 1)

            fe = FeatureExplanation(
                feature=feat_name,
                value=val * multiplier,
                severity=severity,
                label=config.get("label", feat_name),
                unit=config.get("unit", ""),
                description=config.get("description", ""),
                city_avg=(city_avg * multiplier) if city_avg else None,
                z_score=z_score,
                benchmark=BENCHMARKS.get(feat_name),
            )
            feature_explanations.append(fe)

        # Priority score
        priority_score = float(zone.get("priority_composite_100",
                                 zone.get("priority_score", 50.0)))
        priority_class = "High" if priority_score >= 66 else ("Medium" if priority_score >= 33 else "Low")

        if "priority_class" in zone.index:
            priority_class = str(zone["priority_class"])

        # Decision rule
        zone_array = np.array([float(zone.get(f, 0)) for f in self.feature_cols])
        decision_rule = extract_decision_rule_for_zone(zone_array, self.dt_path, self.feature_cols)

        # SHAP values
        shap_dict = None
        if self.svm_pipeline:
            shap_dict = compute_shap_values(zone_array, self.svm_pipeline, self.feature_cols)

        # Natural language summary
        nl_summary = generate_nl_summary(
            zone, infra_type, priority_score, priority_class, feature_explanations
        )

        # Estimated impact
        pop = float(zone.get("population_total", 0))
        gap = float(zone.get("coverage_gap", 0))
        newly_covered = int(pop * gap * 0.7)  # 70% of uncovered pop within new service radius
        estimated_impact = f"~{newly_covered:,} residents newly within {infra_type.replace('_',' ')} service area"

        # Confidence
        confidence = min(0.99, priority_score / 100.0 * 1.2)

        return ZoneExplanation(
            h3_id=h3_id,
            infra_type=infra_type,
            priority_score=priority_score,
            priority_class=priority_class,
            recommendation=INFRA_RECOMMENDATIONS.get(infra_type, f"Build {infra_type}"),
            confidence=confidence,
            features=feature_explanations,
            decision_rule=decision_rule,
            nl_summary=nl_summary,
            estimated_impact=estimated_impact,
            lat=float(zone.get("lat", 0)),
            lon=float(zone.get("lon", 0)),
            shap_values=shap_dict,
        )

    def explain_top_n(
        self,
        infra_type: str = "hospital",
        n: int = 10,
        ranked_csv: Optional[Path] = None,
    ) -> List[ZoneExplanation]:
        """
        Explain the top-N recommended zones.

        Args:
            infra_type:  Infrastructure type
            n:           Number of explanations
            ranked_csv:  Path to ranked_sites CSV from BFS

        Returns:
            List of ZoneExplanation objects
        """
        explanations = []

        if ranked_csv and ranked_csv.exists():
            ranked_df = pd.read_csv(ranked_csv)
            zone_ids = ranked_df["h3_id"].head(n).tolist()
        else:
            # Use priority_composite_100 ranking
            if "priority_composite_100" in self.df.columns:
                zone_ids = self.df.nlargest(n, "priority_composite_100")["h3_id"].tolist()
            elif "h3_id" in self.df.columns:
                zone_ids = self.df["h3_id"].head(n).tolist()
            else:
                zone_ids = [str(i) for i in range(min(n, len(self.df)))]

        for zone_id in zone_ids:
            expl = self.explain_zone(zone_id, infra_type)
            explanations.append(expl)

        return explanations

    def save_explanations(
        self,
        explanations: List[ZoneExplanation],
        output_path: Optional[Path] = None,
    ) -> None:
        """Save explanations as JSON."""
        output_path = output_path or PROCESSED_DIR / "explanations.json"
        data = [e.to_dict() for e in explanations]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        log.info(f"Explanations saved → {output_path} ({len(data)} zones)")


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate explanations for SmartCityAI")
    parser.add_argument("--infra",    default="hospital")
    parser.add_argument("--zone-id",  default=None, help="Specific H3 zone ID to explain")
    parser.add_argument("--top-n",    type=int, default=10)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SmartCityAI — Explanation Engine")
    log.info("=" * 60)

    # Load data
    csv_path = PROCESSED_DIR / "zone_features.csv"
    if not csv_path.exists():
        log.error("zone_features.csv not found. Run create_zones.py first.")
        return

    df = pd.read_csv(csv_path)

    # Merge predictions
    for pred_file in ["svm_predictions.csv", "dt_predictions.csv"]:
        path = PROCESSED_DIR / pred_file
        if path.exists():
            df = df.merge(pd.read_csv(path), on="h3_id", how="left")

    # Create engine
    engine = ExplanationEngine.from_saved_models(df)

    if args.zone_id:
        expl = engine.explain_zone(args.zone_id, args.infra)
        print(expl.to_text())
    else:
        ranked_csv = PROCESSED_DIR / f"ranked_sites_{args.infra}.csv"
        explanations = engine.explain_top_n(args.infra, args.top_n, ranked_csv)

        for expl in explanations[:3]:
            print(expl.to_text())

        engine.save_explanations(explanations)

    log.info("Explanation engine complete!")


if __name__ == "__main__":
    main()
