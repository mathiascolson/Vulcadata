import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATE_RESULT = "reports/retraining/candidate_training_result.json"
DEFAULT_CHAMPION_DECISION = "configs/final_model_decision.json"
DEFAULT_DRIFT_SUMMARY = "reports/retraining/evidently/candidate_drift_summary.json"
DEFAULT_OUTPUT_JSON = "reports/retraining/candidate_vs_champion_comparison.json"


MetricDict = dict[str, float | None]
RuleList = list[dict[str, Any]]


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {json_path}")

    return payload


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def get_nested(payload: dict[str, Any], keys: list[str]) -> Any:
    current: Any = payload

    for key in keys:
        if not isinstance(current, dict):
            return None
        if key not in current:
            return None
        current = current[key]

    return current


def to_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_number(payload: dict[str, Any], paths: list[list[str]]) -> float | None:
    for path in paths:
        value = to_float(get_nested(payload, path))
        if value is not None:
            return value

    return None


def get_command_arg(command: Any, flag: str) -> str | None:
    if not isinstance(command, list):
        return None

    for index, item in enumerate(command):
        if not isinstance(item, str):
            continue

        if item == flag and index + 1 < len(command):
            return str(command[index + 1])

        prefix = flag + "="
        if item.startswith(prefix):
            return item.split("=", 1)[1]

    return None


def get_candidate_epochs(candidate_result: dict[str, Any]) -> int | None:
    direct_value = first_number(
        candidate_result,
        [
            ["epochs"],
            ["training_parameters", "epochs"],
            ["hyperparameters", "epochs"],
        ],
    )

    if direct_value is not None:
        return int(direct_value)

    command_value = get_command_arg(candidate_result.get("command"), "--epochs")
    if command_value is None:
        return None

    try:
        return int(command_value)
    except ValueError:
        return None


def get_champion_section(champion_decision: dict[str, Any]) -> dict[str, Any]:
    classification_candidate = champion_decision.get("classification_candidate")
    if isinstance(classification_candidate, dict):
        return classification_candidate

    champion = champion_decision.get("champion")
    if isinstance(champion, dict):
        return champion

    return champion_decision


def extract_candidate_metrics(candidate_result: dict[str, Any]) -> MetricDict:
    return {
        "business_score_classification": first_number(
            candidate_result,
            [
                ["metrics", "test", "business_score_classification"],
                ["metrics", "business_score_classification"],
                ["test_business_score_classification"],
                ["business_score_classification"],
            ],
        ),
        "alert_24h_f1": first_number(
            candidate_result,
            [
                ["metrics", "test", "alert_24h_f1"],
                ["metrics", "alert_24h_f1"],
                ["alert_24h_f1"],
                ["f1_alert_24h"],
            ],
        ),
        "alert_24h_recall": first_number(
            candidate_result,
            [
                ["metrics", "test", "alert_24h_recall"],
                ["metrics", "alert_24h_recall"],
                ["alert_24h_recall"],
                ["recall_alert_24h"],
            ],
        ),
        "alert_24h_precision": first_number(
            candidate_result,
            [
                ["metrics", "test", "alert_24h_precision"],
                ["metrics", "alert_24h_precision"],
                ["alert_24h_precision"],
                ["precision_alert_24h"],
            ],
        ),
        "class_5_f1": first_number(
            candidate_result,
            [
                ["metrics", "test", "class_5_f1"],
                ["metrics", "class_5_f1"],
                ["test_class_5_f1"],
                ["class_5_f1"],
            ],
        ),
    }


def extract_champion_metrics(champion_decision: dict[str, Any]) -> MetricDict:
    champion = get_champion_section(champion_decision)

    return {
        "business_score_classification": first_number(
            champion,
            [
                ["test_business_score_classification"],
                ["business_score_classification"],
                ["metrics", "test", "business_score_classification"],
                ["metrics", "business_score_classification"],
            ],
        ),
        "alert_24h_f1": first_number(
            champion,
            [
                ["f1_alert_24h"],
                ["alert_24h_f1"],
                ["metrics", "test", "alert_24h_f1"],
                ["metrics", "alert_24h_f1"],
            ],
        ),
        "alert_24h_recall": first_number(
            champion,
            [
                ["recall_alert_24h"],
                ["alert_24h_recall"],
                ["metrics", "test", "alert_24h_recall"],
                ["metrics", "alert_24h_recall"],
            ],
        ),
        "alert_24h_precision": first_number(
            champion,
            [
                ["precision_alert_24h"],
                ["alert_24h_precision"],
                ["metrics", "test", "alert_24h_precision"],
                ["metrics", "alert_24h_precision"],
            ],
        ),
        "class_5_f1": first_number(
            champion,
            [
                ["test_class_5_f1"],
                ["class_5_f1"],
                ["metrics", "test", "class_5_f1"],
                ["metrics", "class_5_f1"],
            ],
        ),
    }


def add_rule(
    passed_rules: RuleList,
    failed_rules: RuleList,
    name: str,
    passed: bool,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    rule = {
        "name": name,
        "passed": passed,
        "message": message,
        "details": details or {},
    }

    if passed:
        passed_rules.append(rule)
    else:
        failed_rules.append(rule)


def add_skipped_rule(
    skipped_rules: RuleList,
    name: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    skipped_rules.append(
        {
            "name": name,
            "passed": None,
            "message": message,
            "details": details or {},
        }
    )


def compare_values_with_minimum(
    passed_rules: RuleList,
    failed_rules: RuleList,
    rule_name: str,
    candidate_value: float | None,
    minimum_allowed: float,
    message: str,
) -> None:
    add_rule(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        name=rule_name,
        passed=candidate_value is not None and candidate_value >= minimum_allowed,
        message=message,
        details={
            "candidate": candidate_value,
            "minimum_allowed": minimum_allowed,
        },
    )


def compare_candidate_against_champion(
    passed_rules: RuleList,
    failed_rules: RuleList,
    rule_name: str,
    candidate_value: float | None,
    champion_value: float | None,
    max_allowed_drop: float,
    message: str,
) -> None:
    minimum_allowed = champion_value - max_allowed_drop if champion_value is not None else None
    passed = (
        candidate_value is not None
        and champion_value is not None
        and minimum_allowed is not None
        and candidate_value >= minimum_allowed
    )

    add_rule(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        name=rule_name,
        passed=passed,
        message=message,
        details={
            "candidate": candidate_value,
            "champion": champion_value,
            "max_allowed_drop": max_allowed_drop,
            "minimum_allowed": minimum_allowed,
        },
    )


def validate_required_metrics(
    passed_rules: RuleList,
    failed_rules: RuleList,
    candidate_metrics: MetricDict,
    champion_metrics: MetricDict,
) -> None:
    required_metric_names = [
        "business_score_classification",
        "alert_24h_f1",
        "alert_24h_recall",
        "alert_24h_precision",
    ]
    missing_metrics = []

    for metric_name in required_metric_names:
        if candidate_metrics.get(metric_name) is None:
            missing_metrics.append("candidate." + metric_name)
        if champion_metrics.get(metric_name) is None:
            missing_metrics.append("champion." + metric_name)

    add_rule(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        name="required_metrics_available",
        passed=len(missing_metrics) == 0,
        message="Les métriques minimales doivent être disponibles côté candidat et côté champion.",
        details={"missing_metrics": missing_metrics},
    )


def validate_optional_class_5_rule(
    passed_rules: RuleList,
    failed_rules: RuleList,
    skipped_rules: RuleList,
    candidate_metrics: MetricDict,
    champion_metrics: MetricDict,
    max_class_5_f1_drop: float,
) -> None:
    candidate_value = candidate_metrics.get("class_5_f1")
    champion_value = champion_metrics.get("class_5_f1")

    if candidate_value is None or champion_value is None:
        add_skipped_rule(
            skipped_rules=skipped_rules,
            name="class_5_f1_not_degraded",
            message="Contrôle classe 5 ignoré car la métrique est absente côté candidat ou champion.",
            details={"candidate": candidate_value, "champion": champion_value},
        )
        return

    compare_candidate_against_champion(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        rule_name="class_5_f1_not_degraded",
        candidate_value=candidate_value,
        champion_value=champion_value,
        max_allowed_drop=max_class_5_f1_drop,
        message="Le F1 de la classe 5 ne doit pas se dégrader au-delà de la tolérance.",
    )


def validate_drift_summary(
    passed_rules: RuleList,
    failed_rules: RuleList,
    drift_summary: dict[str, Any],
) -> None:
    add_rule(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        name="drift_summary_success",
        passed=drift_summary.get("status") == "success",
        message="Le rapport de drift doit être disponible et terminé avec succès.",
        details={"status": drift_summary.get("status")},
    )

    critical_drift_detected = bool(drift_summary.get("critical_drift_detected"))
    candidate_rejected_by_drift_check = bool(drift_summary.get("candidate_rejected_by_drift_check"))

    add_rule(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        name="no_critical_drift_detected",
        passed=not critical_drift_detected,
        message="Aucun drift critique ne doit être détecté.",
        details={
            "critical_drift_detected": critical_drift_detected,
            "dataset_drift": drift_summary.get("dataset_drift"),
            "target_drift": drift_summary.get("target_drift"),
            "prediction_drift": drift_summary.get("prediction_drift"),
            "reason": drift_summary.get("reason"),
        },
    )

    add_rule(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        name="candidate_not_rejected_by_drift_check",
        passed=not candidate_rejected_by_drift_check,
        message="Le candidat ne doit pas être rejeté par le contrôle de drift.",
        details={
            "candidate_rejected_by_drift_check": candidate_rejected_by_drift_check,
            "reason": drift_summary.get("reason"),
        },
    )


def build_comparison(args: argparse.Namespace) -> dict[str, Any]:
    candidate_result = read_json(args.candidate_result)
    champion_decision = read_json(args.champion_decision)
    drift_summary = read_json(args.drift_summary)

    candidate_metrics = extract_candidate_metrics(candidate_result)
    champion_metrics = extract_champion_metrics(champion_decision)
    candidate_epochs = get_candidate_epochs(candidate_result)

    passed_rules: RuleList = []
    failed_rules: RuleList = []
    skipped_rules: RuleList = []

    add_rule(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        name="candidate_training_success",
        passed=candidate_result.get("status") == "success",
        message="Le candidat doit provenir d'un entraînement terminé avec succès.",
        details={"candidate_status": candidate_result.get("status")},
    )

    add_rule(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        name="candidate_not_dry_run",
        passed=candidate_result.get("dry_run") is False,
        message="Le candidat ne doit pas provenir d'un dry-run.",
        details={"dry_run": candidate_result.get("dry_run")},
    )

    add_rule(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        name="minimum_epochs_for_promotion",
        passed=candidate_epochs is not None and candidate_epochs >= args.min_epochs_for_promotion,
        message="Le candidat doit avoir été entraîné suffisamment longtemps pour être promouvable.",
        details={
            "candidate_epochs": candidate_epochs,
            "min_epochs_for_promotion": args.min_epochs_for_promotion,
        },
    )

    validate_required_metrics(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
    )

    compare_candidate_against_champion(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        rule_name="business_score_not_lower_than_champion",
        candidate_value=candidate_metrics.get("business_score_classification"),
        champion_value=champion_metrics.get("business_score_classification"),
        max_allowed_drop=args.max_business_score_drop,
        message="Le business score candidat ne doit pas être inférieur au champion.",
    )

    compare_candidate_against_champion(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        rule_name="alert_24h_f1_not_degraded",
        candidate_value=candidate_metrics.get("alert_24h_f1"),
        champion_value=champion_metrics.get("alert_24h_f1"),
        max_allowed_drop=args.max_alert_24h_f1_drop,
        message="Le F1 alerte 24h ne doit pas se dégrader au-delà de la tolérance.",
    )

    compare_values_with_minimum(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        rule_name="alert_24h_recall_minimum",
        candidate_value=candidate_metrics.get("alert_24h_recall"),
        minimum_allowed=args.min_alert_24h_recall,
        message="Le recall alerte 24h doit rester au-dessus du seuil opérationnel.",
    )

    compare_values_with_minimum(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        rule_name="alert_24h_precision_minimum",
        candidate_value=candidate_metrics.get("alert_24h_precision"),
        minimum_allowed=args.min_alert_24h_precision,
        message="La précision alerte 24h doit rester au-dessus du seuil opérationnel.",
    )

    validate_optional_class_5_rule(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        skipped_rules=skipped_rules,
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        max_class_5_f1_drop=args.max_class_5_f1_drop,
    )

    validate_drift_summary(
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        drift_summary=drift_summary,
    )

    eligible_for_promotion = len(failed_rules) == 0

    if eligible_for_promotion:
        decision = "promote_candidate"
        decision_reason = "Le candidat respecte toutes les règles de comparaison et de drift."
    else:
        decision = "reject_candidate"
        failed_rule_names = [rule["name"] for rule in failed_rules]
        decision_reason = "Candidat rejeté par les règles suivantes : " + ", ".join(failed_rule_names)

    return {
        "status": "success",
        "decision": decision,
        "eligible_for_promotion": eligible_for_promotion,
        "decision_reason": decision_reason,
        "candidate_result_path": str(args.candidate_result),
        "champion_decision_path": str(args.champion_decision),
        "drift_summary_path": str(args.drift_summary),
        "candidate_training_run_id": candidate_result.get("training_run_id"),
        "candidate_mlflow_run_id": candidate_result.get("mlflow_run_id"),
        "candidate_epochs": candidate_epochs,
        "candidate_metrics": candidate_metrics,
        "champion_metrics": champion_metrics,
        "drift_summary": {
            "status": drift_summary.get("status"),
            "critical_drift_detected": drift_summary.get("critical_drift_detected"),
            "candidate_rejected_by_drift_check": drift_summary.get("candidate_rejected_by_drift_check"),
            "dataset_drift": drift_summary.get("dataset_drift"),
            "target_drift": drift_summary.get("target_drift"),
            "prediction_drift": drift_summary.get("prediction_drift"),
            "reason": drift_summary.get("reason"),
            "output_html": drift_summary.get("output_html"),
            "output_json": drift_summary.get("output_json"),
        },
        "comparison_policy": {
            "min_epochs_for_promotion": args.min_epochs_for_promotion,
            "max_business_score_drop": args.max_business_score_drop,
            "max_alert_24h_f1_drop": args.max_alert_24h_f1_drop,
            "min_alert_24h_recall": args.min_alert_24h_recall,
            "min_alert_24h_precision": args.min_alert_24h_precision,
            "max_class_5_f1_drop": args.max_class_5_f1_drop,
        },
        "passed_rules": passed_rules,
        "failed_rules": failed_rules,
        "skipped_rules": skipped_rules,
        "generated_at_utc": utc_now_iso(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a retrained candidate model against the champion and the drift report."
    )
    parser.add_argument("--candidate-result", default=DEFAULT_CANDIDATE_RESULT)
    parser.add_argument("--champion-decision", default=DEFAULT_CHAMPION_DECISION)
    parser.add_argument("--drift-summary", default=DEFAULT_DRIFT_SUMMARY)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--min-epochs-for-promotion", type=int, default=2)
    parser.add_argument("--max-business-score-drop", type=float, default=0.0)
    parser.add_argument("--max-alert-24h-f1-drop", type=float, default=0.02)
    parser.add_argument("--min-alert-24h-recall", type=float, default=0.70)
    parser.add_argument("--min-alert-24h-precision", type=float, default=0.40)
    parser.add_argument("--max-class-5-f1-drop", type=float, default=0.05)
    parser.add_argument("--fail-on-reject", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = build_comparison(args)
    write_json(comparison, args.output_json)

    if args.print_json:
        print(json.dumps(comparison, indent=2, ensure_ascii=False))

    if args.fail_on_reject and comparison["decision"] != "promote_candidate":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
