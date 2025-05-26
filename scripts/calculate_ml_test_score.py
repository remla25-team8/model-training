"""
ML Test Score Calculator

This script calculates the ML Test Score based on the methodology described in
"The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction"

The score is calculated across four main categories:
1. Features and Data (0-10 points)
2. Model Development (0-7 points)
3. Infrastructure (0-11 points)
4. Monitoring (0-4 points)

Total possible score: 32 points (displayed as percentage)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Test categories and their descriptions
TEST_CATEGORIES = {
    "features_and_data": {
        "name": "Features and Data",
        "max_score": 10,
        "tests": [
            "test_feature_expectations",
            "test_feature_schema_validation",
            "test_data_pipeline_integration",
            "test_data_distribution_stability",
            "test_feature_importance_consistency",
            "test_training_serving_skew",
            "test_data_validation_comprehensive",
            "test_feature_coverage",
            "test_data_freshness",
            "test_feature_correlation_stability",
        ],
    },
    "model_development": {
        "name": "Model Development",
        "max_score": 7,
        "tests": [
            "test_model_specifications",
            "test_model_implementation_offline",
            "test_model_quality_validation",
            "test_model_debugging_instrumentation",
            "test_model_unit_tests",
            "test_integration_test_model_api",
            "test_model_staleness_validation",
        ],
    },
    "infrastructure": {
        "name": "ML Infrastructure",
        "max_score": 11,
        "tests": [
            "test_training_reproducibility",
            "test_model_deployment_automation",
            "test_model_server_load_capacity",
            "test_model_quality_pipeline",
            "test_model_rollback_capability",
            "test_data_invariant_checking",
            "test_training_data_model_provenance",
            "test_model_prediction_audit_logging",
            "test_pipeline_jerk_detection",
            "test_model_numerical_stability",
            "test_computing_performance_regression",
        ],
    },
    "monitoring": {
        "name": "Monitoring",
        "max_score": 4,
        "tests": [
            "test_dependency_change_notification",
            "test_data_invariant_change_detection",
            "test_training_performance_monitoring",
            "test_serving_performance_monitoring",
        ],
    },
}


def discover_implemented_tests() -> Dict[str, List[str]]:
    """Discover which ML Test Score tests have been implemented."""
    implemented_tests = {category: [] for category in TEST_CATEGORIES.keys()}

    test_dir = Path("tests")
    if not test_dir.exists():
        print(f"Warning: Test directory {test_dir} does not exist")
        return implemented_tests

    for test_file in test_dir.glob("**/*.py"):
        if test_file.name.startswith("test_"):
            try:
                with open(test_file, "r", encoding="utf-8") as f:
                    content = f.read()

                    for category, category_info in TEST_CATEGORIES.items():
                        for test_name in category_info["tests"]:
                            if (
                                f"def {test_name}" in content
                                or f"def {test_name.replace('test_', '')}" in content
                            ):
                                if test_name not in implemented_tests[category]:
                                    implemented_tests[category].append(test_name)

            except Exception as e:
                print(f"Warning: Could not read test file {test_file}: {e}")

    return implemented_tests


def calculate_ml_test_score(verbose: bool = False) -> Dict:
    """Calculate the complete ML Test Score."""
    implemented_tests = discover_implemented_tests()

    results = {
        "categories": {},
        "total_score": 0,
        "max_possible_score": 32,
        "percentage": 0,
        "grade": "F",
    }

    total_achieved = 0
    total_possible = 0

    if verbose:
        print("\n" + "=" * 60)
        print("🧪 ML TEST SCORE CALCULATION")
        print("=" * 60)

    for category_name, category_info in TEST_CATEGORIES.items():
        max_score = category_info["max_score"]
        total_tests = len(category_info["tests"])
        implemented_count = len(implemented_tests[category_name])

        achieved = (
            int((implemented_count / total_tests) * max_score) if total_tests > 0 else 0
        )

        results["categories"][category_name] = {
            "name": category_info["name"],
            "achieved_score": achieved,
            "max_score": max_score,
            "implemented_tests": implemented_tests[category_name],
            "total_tests": total_tests,
            "percentage": (achieved / max_score * 100) if max_score > 0 else 0,
        }

        total_achieved += achieved
        total_possible += max_score

        if verbose:
            print(f"\n📋 {category_info['name']}")
            print(f"   Score: {achieved}/{max_score} ({achieved / max_score * 100:.1f}%)")
            print(f"   Tests implemented: {implemented_count}/{total_tests}")
            for test in implemented_tests[category_name]:
                print(f"   ✅ {test}")
            missing_tests = set(category_info["tests"]) - set(
                implemented_tests[category_name]
            )
            for test in missing_tests:
                print(f"   ❌ {test}")

    percentage = (total_achieved / total_possible * 100) if total_possible > 0 else 0
    grade = (
        "A"
        if percentage >= 90
        else (
            "B"
            if percentage >= 80
            else "C" if percentage >= 70 else "D" if percentage >= 60 else "F"
        )
    )

    results.update(
        {
            "total_score": total_achieved,
            "percentage": round(percentage, 1),
            "grade": grade,
        }
    )

    if verbose:
        print("\n📊 FINAL RESULTS")
        print(f"Total Score: {total_achieved}/{total_possible} ({percentage:.1f}%)")
        print(f"Grade: {grade}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Calculate ML Test Score")
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed breakdown"
    )
    parser.add_argument("--output-json", type=str, help="Output results to JSON file")
    parser.add_argument(
        "--fail-below", type=float, default=0, help="Exit with error if below threshold"
    )

    args = parser.parse_args()

    try:
        results = calculate_ml_test_score(verbose=args.verbose)

        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.output_json}")

        if results["percentage"] < args.fail_below:
            print(
                f"\n❌ ML Test Score ({results['percentage']}%) is below threshold ({args.fail_below}%)"
            )
            sys.exit(1)

        if not args.verbose:
            print(
                f"ML Test Score: {results['percentage']}% (Grade: {results['grade']})"
            )

    except Exception as e:
        print(f"❌ Error calculating ML Test Score: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
