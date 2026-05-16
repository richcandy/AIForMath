import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "mathlib4_1bc7728a050fc18ca2683f614c531cd7050ff063" / "novel_premises"
DEFAULT_OUTPUT_ROOT = ROOT / "Deepseek_highschool_data"

STRICT_INCLUDE_PATHS = [
    "Mathlib/Algebra/Polynomial",
    "Mathlib/RingTheory/Polynomial",
    "Mathlib/Algebra/Ring",
    "Mathlib/Algebra/Field",
    "Mathlib/FieldTheory",
]

STRICT_EXCLUDE_PATHS = [
    "LinearAlgebra",
    "Homology",
    "Lie",
    "AlgebraicGeometry",
    "CStarAlgebra",
    "RootSystem",
    "Category",
    "Topology",
    "MeasureTheory",
    "Probability",
    "Manifold",
    "DifferentialGeometry",
    "Order/Filter",
    "SetTheory",
    "RepresentationTheory",
    "AlgebraicTopology",
]

STRICT_TEXT_PATTERNS = [
    r"\bpolynomial\b",
    r"\bsqrt\b",
    r"\bpow\b",
    r"\broot\b",
    r"\bfactor",
    r"\bring\b",
    r"\bfield\b",
    r"\bineq",
    r"ℝ",
    r"ℚ",
    r"ℤ",
    r"\breal\b",
    r"\brat\b",
    r"\bint\b",
]

MEDIUM_INCLUDE_PATHS = [
    "Mathlib/Algebra/Polynomial",
    "Mathlib/RingTheory/Polynomial",
    "Mathlib/Algebra/Ring",
    "Mathlib/Algebra/Field",
    "Mathlib/FieldTheory",
    "Mathlib/Data/Polynomial",
    "Mathlib/Data/Int",
    "Mathlib/Data/Rat",
    "Mathlib/Analysis/SpecialFunctions/Pow",
    "Mathlib/Analysis/SpecialFunctions/Sqrt",
    "Mathlib/NumberTheory",
]

MEDIUM_EXCLUDE_PATHS = STRICT_EXCLUDE_PATHS + [
    "Algebra/Homology",
    "Algebra/Lie",
    "LinearAlgebra",
]

MEDIUM_TEXT_PATTERNS = STRICT_TEXT_PATTERNS + [
    r"\bdiv\b",
    r"\bmul\b",
    r"\badd\b",
    r"\bsub\b",
    r"\bnorm_num\b",
    r"\bring_nf\b",
    r"\blinarith\b",
    r"\bnlinarith\b",
    r"\bint\.sqrt\b",
    r"\bquadratic\b",
]

RECOMMENDED_INCLUDE_PATHS = [
    "Mathlib/Data/Nat",
    "Mathlib/Data/Int",
    "Mathlib/Data/Rat",
    "Mathlib/Data/Real",
    "Mathlib/Data/Complex",
    "Mathlib/Algebra/Polynomial",
    "Mathlib/RingTheory/Polynomial",
    "Mathlib/Algebra/Ring",
    "Mathlib/Algebra/Field",
    "Mathlib/Algebra/Order/Ring",
    "Mathlib/Algebra/Order/Field",
    "Mathlib/RingTheory/Binomial",
]

RECOMMENDED_EXCLUDE_PATHS = [
    "Mathlib/CategoryTheory",
    "Mathlib/Topology",
    "Mathlib/Analysis",
    "Mathlib/Geometry",
    "Mathlib/MeasureTheory",
    "Mathlib/AlgebraicGeometry",
    "Mathlib/Probability",
    "Mathlib/LinearAlgebra",
    "Mathlib/Algebra/Homology",
    "Mathlib/Algebra/Lie",
    "Mathlib/Algebra/MvPolynomial",
    "Mathlib/RingTheory/MvPolynomial",
    "Mathlib/RingTheory/PowerSeries",
    "Mathlib/RingTheory/MvPowerSeries",
    "Mathlib/RingTheory/WittVector",
    "Mathlib/RingTheory/HahnSeries",
    "Mathlib/RingTheory/LaurentSeries",
    "Mathlib/Algebra/Module",
]

RECOMMENDED_TACTICS = {
    "rw",
    "simp",
    "simp_rw",
    "ring",
    "ring_nf",
    "ring1",
    "linarith",
    "nlinarith",
    "norm_num",
    "field_simp",
    "positivity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a high-school algebra focused subset from mathlib traced-tactics data."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--variant",
        choices=["strict", "medium", "recommended"],
        default="recommended",
        help="Subset rule variant.",
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def tactic_head(tactic: str) -> str:
    stripped = tactic.strip()
    if not stripped:
        return ""
    return stripped.split()[0]


def row_matches(
    row: dict[str, Any],
    include_paths: list[str],
    exclude_paths: list[str],
    text_patterns: list[str],
    require_path_match: bool,
    allowed_tactics: set[str] | None,
) -> bool:
    file_path = str(row.get("file_path") or "")
    full_name = str(row.get("full_name") or "")
    theorem_statement = str(row.get("theorem_statement") or "")
    traced_tactics = row.get("traced_tactics") or []

    if not theorem_statement.strip() or not traced_tactics:
        return False
    if any(fragment in file_path for fragment in exclude_paths):
        return False

    path_ok = any(file_path.startswith(prefix) for prefix in include_paths)
    if require_path_match and not path_ok:
        return False

    if allowed_tactics is not None:
        tactic_names = {
            tactic_head(str(step.get("tactic") or ""))
            for step in traced_tactics
            if step.get("tactic")
        }
        if not (tactic_names & allowed_tactics):
            return False
        if require_path_match:
            return True

    text = " | ".join([file_path, full_name, theorem_statement])
    text_ok = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in text_patterns)
    return path_ok or text_ok if not require_path_match else text_ok


def get_variant_rules(variant: str) -> tuple[list[str], list[str], list[str], bool, set[str] | None]:
    if variant == "strict":
        return STRICT_INCLUDE_PATHS, STRICT_EXCLUDE_PATHS, STRICT_TEXT_PATTERNS, True, None
    if variant == "medium":
        return MEDIUM_INCLUDE_PATHS, MEDIUM_EXCLUDE_PATHS, MEDIUM_TEXT_PATTERNS, False, None
    if variant == "recommended":
        return RECOMMENDED_INCLUDE_PATHS, RECOMMENDED_EXCLUDE_PATHS, [], True, RECOMMENDED_TACTICS
    raise ValueError(f"Unknown variant: {variant}")


def filter_split(rows: list[dict[str, Any]], variant: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    include_paths, exclude_paths, text_patterns, require_path_match, allowed_tactics = get_variant_rules(variant)
    filtered = []
    stats = {
        "theorems": 0,
        "tactics": 0,
    }
    for row in rows:
        if not row_matches(
            row,
            include_paths,
            exclude_paths,
            text_patterns,
            require_path_match,
            allowed_tactics,
        ):
            continue
        filtered.append(row)
        stats["theorems"] += 1
        stats["tactics"] += len(row.get("traced_tactics", []))
    return filtered, stats


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    if args.output_root == DEFAULT_OUTPUT_ROOT and args.variant == "medium":
        args.output_root = ROOT / "training_data" / "highschool_algebra_mathlib_medium"
    if args.output_root == DEFAULT_OUTPUT_ROOT and args.variant == "strict":
        args.output_root = ROOT / "training_data" / "highschool_algebra_mathlib"
    args.output_root = args.output_root.resolve()
    include_paths, exclude_paths, text_patterns, require_path_match, allowed_tactics = get_variant_rules(args.variant)

    split_names = ["train", "val", "test"]
    summary: dict[str, Any] = {
        "variant": args.variant,
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "created_at_unix": int(time.time()),
        "rules": {
            "include_paths": include_paths,
            "exclude_paths": exclude_paths,
            "text_patterns": text_patterns,
            "allowed_tactics": sorted(allowed_tactics) if allowed_tactics is not None else None,
            "require_path_match": require_path_match,
            "require_nonempty_theorem_statement": True,
            "require_nonempty_traced_tactics": True,
        },
        "splits": {},
    }

    for split_name in split_names:
        source_path = args.source_root / f"{split_name}.json"
        rows = load_json(source_path)
        filtered_rows, stats = filter_split(rows, args.variant)
        output_path = args.output_root / f"{split_name}.json"
        atomic_write_json(output_path, filtered_rows)
        summary["splits"][split_name] = {
            "source_path": str(source_path),
            "output_path": str(output_path),
            "source_theorems": len(rows),
            **stats,
        }
        print(
            f"{split_name}: source_theorems={len(rows)} kept_theorems={stats['theorems']} kept_tactics={stats['tactics']}",
            flush=True,
        )

    manifest_path = args.output_root / "manifest.json"
    atomic_write_json(manifest_path, summary)
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
