"""
ML Test Score Calculator

This script calculates the ML Test Score based on the methodology described in
"The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction"
by Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley (Google, 2017)

The score is calculated across four main categories with exactly 28 tests:
1. Features and Data (7 tests, max 7 points)
2. Model Development (7 tests, max 7 points) 
3. Infrastructure (7 tests, max 7 points)
4. Monitoring (7 tests, max 7 points)

Max possible score: 7
Final score: minimum of the 4 category scores (as per original paper)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Official 28 tests from the Google ML Test Score paper
TEST_CATEGORIES = {
    "features_and_data": {
        "name": "Features and Data",
        "max_score": 7,
        "tests": [
            "test_feature_distributions",
            "test_feature_benefits", 
            "test_feature_costs",
            "test_feature_policies",
            "test_data_privacy",
            "test_new_feature_time",
            "test_feature_creation",
        ],
    },
    "model_development": {
        "name": "Model Development", 
        "max_score": 7,
        "tests": [
            "test_model_code_review",
            "test_offline_online_metrics",
            "test_tune_hyperparameters",
            "test_model_staleness",
            "test_baseline_model",
            "test_quality_on_slices",
            "test_inclusiveness",
        ],
    },
    "infrastructure": {
        "name": "ML Infrastructure",
        "max_score": 7,
        "tests": [
            "test_training_reproducibility",
            "test_unit_test_model",
            "test_integration_test",
            "test_quality_before_serving",
            "test_debugger",
            "test_canary_before_serving",
            "test_serving_model_rollback",
        ],
    },
    "monitoring": {
        "name": "Monitoring",
        "max_score": 7,
        "tests": [
            "test_monitor_dependencies",
            "test_monitor_data_invariants",
            "test_monitor_training_serving_skew",
            "test_monitor_model_staleness",
            "test_monitor_numerical_stability",
            "test_monitor_performance_regression",
            "test_monitor_quality_regression",
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
    """Calculate the official ML Test Score as per Google's paper."""
    implemented_tests = discover_implemented_tests()

    results = {
        "categories": {},
        "category_scores": {},
        "final_score": 0,
        "max_possible_score": 28,
        "interpretation": "",
    }

    category_scores = []

    if verbose:
        print("\n" + "=" * 70)
        print("🧪 OFFICIAL ML TEST SCORE CALCULATION")
        print("Based on Google's 'ML Test Score' paper (Breck et al., 2017)")
        print("=" * 70)

    for category_name, category_info in TEST_CATEGORIES.items():
        max_score = category_info["max_score"]
        total_tests = len(category_info["tests"])
        implemented_count = len(implemented_tests[category_name])

        # Calculate score: 1 point per implemented test (assuming automated)
        # In practice, you'd need to distinguish manual (0.5) vs automated (1.0)
        achieved = implemented_count  # Assuming all implemented tests are automated

        results["categories"][category_name] = {
            "name": category_info["name"],
            "achieved_score": achieved,
            "max_score": max_score,
            "implemented_tests": implemented_tests[category_name],
            "total_tests": total_tests,
            "percentage": (achieved / max_score * 100) if max_score > 0 else 0,
        }

        results["category_scores"][category_name] = achieved
        category_scores.append(achieved)

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

    # Final score is the MINIMUM of all category scores (as per original paper)
    final_score = min(category_scores) if category_scores else 0
    
    # Interpretation based on original paper
    if final_score == 0:
        interpretation = "More of a research project than a productionized system."
    elif 0 < final_score <= 1:
        interpretation = "Not totally untested, but serious holes in reliability likely exist."
    elif 1 < final_score <= 2:
        interpretation = "First pass at basic productionization, but additional investment needed."
    elif 2 < final_score <= 3:
        interpretation = "Reasonably tested, but more automation may be beneficial."
    elif 3 < final_score <= 5:
        interpretation = "Strong levels of automated testing, appropriate for mission-critical systems."
    else:
        interpretation = "Exceptional levels of automated testing and monitoring."

    results.update({
        "final_score": final_score,
        "interpretation": interpretation,
    })

    if verbose:
        print("\n📊 FINAL RESULTS")
        print(f"Category Scores: {dict(zip(TEST_CATEGORIES.keys(), category_scores))}")
        print(f"Final ML Test Score: {final_score}/7 (minimum of category scores)")
        print(f"Interpretation: {interpretation}")
        
        print(f"\n📚 ORIGINAL 28 TESTS BREAKDOWN:")
        print(f"• Features and Data: 7 tests")
        print(f"• Model Development: 7 tests") 
        print(f"• Infrastructure: 7 tests")
        print(f"• Monitoring: 7 tests")
        print(f"• Total: 28 tests, max score 28 points")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Official ML Test Score (Google's 28-test rubric)"
    )
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

        if results["final_score"] < args.fail_below:
            print(
                f"\n❌ ML Test Score ({results['final_score']}) is below threshold ({args.fail_below})"
            )
            sys.exit(1)

        if not args.verbose:
            print(f"Official ML Test Score: {results['final_score']}/7")
            print(f"Interpretation: {results['interpretation']}")

    except Exception as e:
        print(f"❌ Error calculating ML Test Score: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
