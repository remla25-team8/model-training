#!/usr/bin/env python3
"""
README Badge Updater

Automatically updates badges in README.md with latest CI/CD metrics
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


def create_badge_url(label: str, message: str, color: str) -> str:
    """Create a shields.io badge URL."""
    label = label.replace(" ", "%20")
    message = message.replace(" ", "%20").replace("%", "%25")
    return f"https://img.shields.io/badge/{label}-{message}-{color}"


def create_badge_markdown(
    label: str, message: str, color: str, link: Optional[str] = None
) -> str:
    """Create markdown for a badge."""
    badge_url = create_badge_url(label, message, color)
    if link:
        return f"[![{label}]({badge_url})]({link})"
    else:
        return f"![{label}]({badge_url})"


def find_badge_section(content: str) -> tuple[int, int]:
    """Find badges section in README content."""
    lines = content.split("\n")
    start_patterns = [r"<!-- BADGES:START -->", r"## Badges", r"# Badges"]
    end_patterns = [r"<!-- BADGES:END -->", r"<!-- /badges -->"]

    start_line = end_line = -1

    for i, line in enumerate(lines):
        for pattern in start_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                start_line = i
                break
        if start_line != -1:
            break

    if start_line != -1:
        for i in range(start_line + 1, len(lines)):
            for pattern in end_patterns:
                if re.search(pattern, lines[i], re.IGNORECASE):
                    end_line = i
                    break
            if end_line != -1:
                break
            if lines[i].startswith("#") and i > start_line + 1:
                end_line = i - 1
                break

    return start_line, end_line


def update_or_add_badge(content: str, badge_label: str, new_badge: str) -> str:
    """Update an existing badge or add a new one."""
    pattern = rf"!\[{re.escape(badge_label)}\]\(https://img\.shields\.io/badge/{re.escape(badge_label)}-[^)]+\)"

    if re.search(pattern, content):
        content = re.sub(pattern, new_badge, content)
        return content

    # If badge doesn't exist, add it to badges section
    start_line, end_line = find_badge_section(content)
    lines = content.split("\n")

    if start_line != -1:
        if end_line != -1:
            lines.insert(end_line, new_badge)
        else:
            lines.insert(start_line + 1, new_badge)
    else:
        # Create new badges section after title
        title_line = -1
        for i, line in enumerate(lines):
            if line.startswith("# "):
                title_line = i
                break

        if title_line != -1:
            # Insert badges section after title
            badges_section = [
                "",
                "<!-- BADGES:START -->",
                new_badge,
                "<!-- BADGES:END -->",
                "",
            ]
            for j, badge_line in enumerate(badges_section):
                lines.insert(title_line + 1 + j, badge_line)
        else:
            # Insert at very beginning
            lines.insert(0, "<!-- BADGES:START -->")
            lines.insert(1, new_badge)
            lines.insert(2, "<!-- BADGES:END -->")
            lines.insert(3, "")

    return "\n".join(lines)


def load_ml_test_score_from_json(json_path: str) -> Optional[tuple]:
    """Load ML Test Score from JSON output and return score and color."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        final_score = data.get('final_score', 0)
        max_score = 7  # ML Test Score is out of 7
        
        return final_score, max_score
    except Exception as e:
        print(f"Warning: Could not load ML Test Score from {json_path}: {e}")
        return None


def get_ml_test_score_color(score: float) -> str:
    """Get appropriate color for ML Test Score badge based on raw score (0-7)."""
    if score >= 5:
        return "green"      # 5-7: Strong/Exceptional testing
    elif score >= 3:
        return "yellow"     # 3-4: Reasonably tested
    elif score >= 1:
        return "orange"     # 1-2: Basic productionization
    else:
        return "red"        # 0: Research project level


def update_readme_badges(readme_path: str, **kwargs) -> bool:
    """Update README.md with new badge metrics."""
    readme_file = Path(readme_path)

    if not readme_file.exists():
        print(f"❌ README file not found: {readme_path}")
        return False

    try:
        with open(readme_file, "r", encoding="utf-8") as f:
            original_content = f.read()

        content = original_content
        modified = False

        # Update PyLint badge
        if "pylint_score" in kwargs and "pylint_color" in kwargs:
            pylint_badge = create_badge_markdown(
                "PyLint", f"{kwargs['pylint_score']:.1f}/10", kwargs["pylint_color"]
            )
            content = update_or_add_badge(content, "PyLint", pylint_badge)
            modified = True
            print(
                f"✅ Updated PyLint badge: {kwargs['pylint_score']:.1f}/10 ({kwargs['pylint_color']})"
            )

        # Update Coverage badge
        if "coverage_percent" in kwargs and "coverage_color" in kwargs:
            coverage_badge = create_badge_markdown(
                "Coverage",
                f"{kwargs['coverage_percent']:.0f}%",
                kwargs["coverage_color"],
            )
            content = update_or_add_badge(content, "Coverage", coverage_badge)
            modified = True
            print(
                f"✅ Updated Coverage badge: {kwargs['coverage_percent']:.0f}% ({kwargs['coverage_color']})"
            )

        # Update ML Test Score badge
        if "ml_test_score" in kwargs and "ml_test_color" in kwargs:
            ml_test_badge = create_badge_markdown(
                "ML%20Test%20Score",
                f"{kwargs['ml_test_score']:.1f}/7",
                kwargs["ml_test_color"],
            )
            # For ML Test Score, we need to handle the URL encoding in the pattern
            pattern = r"!\[ML%20Test%20Score\]\(https://img\.shields\.io/badge/ML%20Test%20Score-[^)]+\)"
            if re.search(pattern, content):
                content = re.sub(pattern, ml_test_badge, content)
            else:
                content = update_or_add_badge(content, "ML Test Score", ml_test_badge)
            modified = True
            print(
                f"✅ Updated ML Test Score badge: {kwargs['ml_test_score']:.1f}/7 ({kwargs['ml_test_color']})"
            )

        # Write updated content if modified
        if modified and content != original_content:
            with open(readme_file, "w", encoding="utf-8") as f:
                f.write(content)
            print("📝 README.md updated successfully")
            return True
        else:
            print("ℹ️  No changes needed to README.md")
            return False

    except Exception as e:
        print(f"❌ Error updating README: {e}")
        return False


def main():
    """Main function to handle command line arguments and update README badges."""
    parser = argparse.ArgumentParser(
        description="Update README.md badges with latest CI/CD metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python update_readme_badges.py --pylint-score 8.5 --pylint-color green
  python update_readme_badges.py --coverage-percent 85 --coverage-color yellow
  python update_readme_badges.py --ml-test-score 75 --ml-test-color green

  # Load ML Test Score from JSON (auto-calculates percentage and color)
  python update_readme_badges.py --ml-test-json ml_test_results.json

  # Update all badges at once
  python update_readme_badges.py \\
    --pylint-score 8.5 --pylint-color green \\
    --coverage-percent 85 --coverage-color yellow \\
    --ml-test-json ml_test_results.json

  # GitHub workflow integration
  python scripts/calculate_ml_test_score.py --output-json ml_test_results.json
  python scripts/update_readme_badges.py --ml-test-json ml_test_results.json
        """,
    )

    parser.add_argument(
        "--readme-path",
        default="README.md",
        help="Path to README.md file (default: README.md)",
    )

    parser.add_argument("--pylint-score", type=float, help="PyLint score (0-10)")

    parser.add_argument(
        "--pylint-color",
        choices=["red", "yellow", "green", "blue", "orange", "lightgrey"],
        help="Color for PyLint badge",
    )

    parser.add_argument(
        "--coverage-percent", type=float, help="Test coverage percentage (0-100)"
    )

    parser.add_argument(
        "--coverage-color",
        choices=["red", "yellow", "green", "blue", "orange", "lightgrey"],
        help="Color for coverage badge",
    )

    parser.add_argument(
        "--ml-test-score", type=float, help="ML Test Score percentage (0-100)"
    )

    parser.add_argument(
        "--ml-test-color",
        choices=["red", "yellow", "green", "blue", "orange", "lightgrey"],
        help="Color for ML Test Score badge",
    )

    parser.add_argument(
        "--ml-test-json",
        type=str,
        help="Path to ML Test Score JSON output file (auto-calculates score and color)",
    )

    args = parser.parse_args()

    # Load ML Test Score from JSON if provided
    if args.ml_test_json:
        ml_score_percentage = load_ml_test_score_from_json(args.ml_test_json)
        if ml_score_percentage is not None:
            args.ml_test_score = ml_score_percentage[0]
            args.ml_test_color = get_ml_test_score_color(args.ml_test_score)
            print(f"📊 Loaded ML Test Score: {args.ml_test_score:.1f}/7 ({args.ml_test_color})")

    # Validate that if score is provided then color is also provided
    if args.pylint_score is not None and args.pylint_color is None:
        print("❌ Error: --pylint-color is required when --pylint-score is provided")
        sys.exit(1)

    if args.coverage_percent is not None and args.coverage_color is None:
        print(
            "❌ Error: --coverage-color is required when --coverage-percent is provided"
        )
        sys.exit(1)

    if args.ml_test_score is not None and args.ml_test_color is None and not args.ml_test_json:
        print("❌ Error: --ml-test-color is required when --ml-test-score is provided")
        sys.exit(1)

    try:
        # Update README badges
        kwargs = {
            k: v for k, v in vars(args).items() if v is not None and k != "readme_path"
        }

        success = update_readme_badges(args.readme_path, **kwargs)

        if success:
            print("🎉 README badges updated successfully!")
        else:
            print("ℹ️  No changes were needed")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
