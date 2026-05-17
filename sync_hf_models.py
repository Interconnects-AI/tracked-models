# /// script
# dependencies = [
#   "huggingface_hub>=1,<2",
#   "pyyaml>=6,<7",
# ]
# ///

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parent
CSV_NAMES = ("models.csv", "extra_models.csv")
ALLOW_UNTAGGED_TAGS = {"transformers", "llama.cpp", "vllm", "diffusers"}
HF_METADATA_SUFFIXES = (
    "config.json",
    "adapter_config.json",
    "model_index.json",
    "preprocessor_config.json",
    "processor_config.json",
)
WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".keras",
    ".msgpack",
)
SKIP_FORMAT_TOKENS = {"gguf", "onnx", "mlx", "openvino", "ov"}
VARIANT_MODIFIER_TOKENS = {"block", "dynamic", "static"}
VARIANT_TOKEN_PATTERN = re.compile(
    r"^(?:fp\d+(?:\.\d+)?|bf\d+(?:\.\d+)?|int\d+|uint\d+|nf\d+|mxfp\d+|nvfp\d+|w\d+a\d+|q\d+(?:_[a-z0-9]+)*|\d+bit)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Decision:
    bucket: str
    reason: str


def tokenize(text: str) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return {token.lower() for token in re.findall(r"[A-Za-z]{4,}", text)}


def matches_any(patterns: list[str] | None, text: str) -> str:
    for pattern in patterns or []:
        if re.search(pattern, text):
            return pattern
    return ""


def format_skip_reason(model_name: str, tags: set[str], file_names: list[str]) -> str:
    lowered_name = model_name.lower()
    if "gguf" in lowered_name:
        return "gguf"
    if any(name.endswith(".gguf") for name in file_names):
        return "gguf"

    if "onnx" in lowered_name:
        return "format"
    if any(
        name.endswith(".onnx") or name.startswith("onnx/") or "/onnx/" in name
        for name in file_names
    ):
        return "format"

    if re.search(r"(^|[-_])mlx($|[-_])", lowered_name):
        return "format"

    if "openvino" in lowered_name:
        return "format"
    if re.search(r"(^|[-_])ov($|[-_])", lowered_name):
        return "format"
    if any("openvino" in name for name in file_names):
        return "format"

    return ""


def is_numeric_variant_suffix(suffix: str) -> bool:
    tokens = [token.lower() for token in suffix.split("-") if token]
    if not tokens:
        return False

    has_numeric_token = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in SKIP_FORMAT_TOKENS:
            return False
        if token in VARIANT_MODIFIER_TOKENS:
            index += 1
            continue
        if token.isdigit() and index + 1 < len(tokens) and tokens[index + 1] == "bit":
            has_numeric_token = True
            index += 2
            continue
        if VARIANT_TOKEN_PATTERN.fullmatch(token):
            has_numeric_token = True
            index += 1
            continue
        return False

    return has_numeric_token


def strip_numeric_variant_suffix(model_name: str) -> str:
    parts = model_name.split("-")
    for index in range(1, len(parts)):
        suffix = "-".join(parts[index:])
        if is_numeric_variant_suffix(suffix):
            return "-".join(parts[:index])

    return ""


def find_variant_path(
    org: str,
    model_name: str,
    tags: set[str],
    tracked_models_by_org: dict[str, list[tuple[str, str]]],
) -> str:
    base_tags: set[str] = set()
    for tag in tags:
        if not tag.startswith("base_model:"):
            continue
        base_model_id = tag.rsplit(":", 1)[-1]
        if "/" not in base_model_id:
            continue
        tag_org, base_name = base_model_id.split("/", 1)
        if tag_org == org:
            base_tags.add(base_name.lower())
            stem = strip_numeric_variant_suffix(base_name)
            if stem:
                base_tags.add(stem.lower())

    for base_name, csv_name in tracked_models_by_org[org]:
        if not model_name.startswith(base_name + "-"):
            continue
        if base_tags and base_name.lower() not in base_tags:
            continue
        if is_numeric_variant_suffix(model_name[len(base_name) + 1 :]):
            return csv_name

    return ""


def load_catalog() -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[str]],
    dict[str, datetime],
    dict[str, set[str]],
    set[str],
]:
    rows_by_path: dict[str, list[dict[str, str]]] = {}
    orgs_by_path: dict[str, list[str]] = {}
    cutoffs: dict[str, datetime] = {}
    tokens_by_org: dict[str, set[str]] = defaultdict(set)
    existing_model_ids: set[str] = set()

    for csv_name in CSV_NAMES:
        csv_path = REPO_ROOT / csv_name
        with csv_path.open() as handle:
            rows = list(csv.DictReader(handle))

        rows_by_path[csv_name] = rows
        orgs_by_path[csv_name] = sorted({row["org"] for row in rows}, key=str.lower)

        for row in rows:
            existing_model_ids.add(row["modelId"])
            tokens_by_org[row["org"]].update(tokenize(row["model"]))

        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", csv_name],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        stamp = result.stdout.strip()
        if stamp:
            cutoffs[csv_name] = datetime.fromisoformat(stamp)
            continue

        cutoffs[csv_name] = datetime.fromtimestamp(
            csv_path.stat().st_mtime
        ).astimezone()

    # Shared orgs stay on the primary list so the two CSVs do not diverge.
    shared_orgs = set(orgs_by_path["models.csv"]) & set(
        orgs_by_path["extra_models.csv"]
    )
    orgs_by_path["extra_models.csv"] = [
        org for org in orgs_by_path["extra_models.csv"] if org not in shared_orgs
    ]
    return rows_by_path, orgs_by_path, cutoffs, tokens_by_org, existing_model_ids


def is_related_untagged_model(model_name: str, org_tokens: set[str]) -> bool:
    for token in tokenize(model_name):
        for existing_token in org_tokens:
            if token == existing_token:
                return True
            if len(token) < 5 or len(existing_token) < 5:
                continue
            if token.startswith(existing_token) or existing_token.startswith(token):
                return True
    return False


def classify_model(
    model,
    policy: dict,
    org_tokens: set[str],
) -> Decision:
    org, model_name = model.id.split("/", 1)
    global_policy = policy["global"]
    org_policy = policy.get("orgs", {}).get(org, {})
    tags = {tag.lower() for tag in (model.tags or [])}
    file_names = [sibling.rfilename.lower() for sibling in (model.siblings or [])]
    searchable_files = "\n".join([model_name, *file_names])

    pattern = matches_any(global_policy.get("reject_name_patterns"), model_name)
    if pattern:
        return Decision("reject", f"global reject pattern: `{pattern}`")

    reject_pipeline_tags = set(global_policy.get("reject_pipeline_tags", []))
    if model.pipeline_tag in reject_pipeline_tags:
        return Decision("reject", f"rejected pipeline tag: `{model.pipeline_tag}`")

    format_reason = format_skip_reason(model_name, tags, file_names)
    if format_reason:
        return Decision("reject", f"format artifact: {format_reason}")

    pattern = matches_any(global_policy.get("reject_format_patterns"), searchable_files)
    if pattern:
        return Decision("reject", f"format reject pattern: `{pattern}`")

    has_weights = any(name.endswith(WEIGHT_SUFFIXES) for name in file_names)
    has_hf_metadata = any(name.endswith(HF_METADATA_SUFFIXES) for name in file_names)
    if not has_weights or not has_hf_metadata:
        return Decision("reject", "no HF weight files or metadata")

    pattern = matches_any(org_policy.get("reject_name_patterns"), model_name)
    if pattern:
        return Decision("reject", f"org reject pattern: `{pattern}`")

    pattern = matches_any(org_policy.get("primary_family_patterns"), model_name)
    if pattern:
        downloads = model.downloads or 0
        min_downloads = global_policy.get("primary_min_downloads", 100000)
        if downloads >= min_downloads:
            return Decision(
                "primary",
                f"primary family match `{pattern}` and {downloads:,} downloads",
            )
        return Decision(
            "primary",
            f"primary family match `{pattern}`, recent official family release",
        )

    pattern = matches_any(org_policy.get("extra_family_patterns"), model_name)
    if pattern:
        return Decision("extra", f"extra family match `{pattern}`")

    allowed_pipeline_tags = set(global_policy.get("primary_allowed_pipeline_tags", []))
    allowed_pipeline_tags.update(
        global_policy.get("conditional_allowed_pipeline_tags", [])
    )
    if model.pipeline_tag in allowed_pipeline_tags:
        downloads = model.downloads or 0
        min_downloads = global_policy.get("primary_min_downloads", 100000)
        if downloads >= min_downloads:
            return Decision(
                "review",
                f"valid LLM/VLM tag and {downloads:,} downloads, no family match",
            )
        return Decision("review", "valid LLM/VLM tag but no known family match")

    if model.pipeline_tag is None and tags & ALLOW_UNTAGGED_TAGS:
        if is_related_untagged_model(model_name, org_tokens):
            return Decision("review", "untagged related model with HF metadata")

    return Decision("reject", "not in LLM/VLM scope")


def write_rows(
    csv_path: Path,
    rows: list[dict[str, str]],
    additions: list[tuple[str, str, str, str]],
) -> None:
    merged_rows = rows + [
        {"org": org, "model": model_name, "modelId": model_id}
        for org, model_name, model_id, _ in additions
    ]
    merged_rows.sort(
        key=lambda row: (
            row["org"].lower(),
            row["model"].lower(),
            row["modelId"].lower(),
        )
    )

    with csv_path.open("w", newline="\n") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["org", "model", "modelId"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(merged_rows)


def validate_csv(csv_path: Path) -> None:
    seen: set[str] = set()
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["org", "model", "modelId"]:
            raise ValueError(f"{csv_path} has unexpected columns: {reader.fieldnames}")
        for line_number, row in enumerate(reader, start=2):
            expected_model_id = f"{row['org']}/{row['model']}"
            if row["modelId"] != expected_model_id:
                raise ValueError(f"{csv_path}:{line_number} has mismatched modelId")
            if row["modelId"] in seen:
                raise ValueError(
                    f"{csv_path}:{line_number} duplicates {row['modelId']}"
                )
            seen.add(row["modelId"])


def markdown_table(rows: list[tuple[str, str]]) -> str:
    if not rows:
        return "_None._"

    lines = ["| modelId | reason |", "|---|---|"]
    for model_id, reason in rows:
        model_id = model_id.replace("|", "\\|").replace("\n", " ")
        reason = reason.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{model_id}` | {reason} |")
    return "\n".join(lines)


def sorted_additions(
    additions: list[tuple[str, str, str, str]],
) -> list[tuple[str, str, str, str]]:
    return sorted(additions, key=lambda row: (row[0].lower(), row[1].lower(), row[2]))


def write_report(
    report_path: Path,
    sync_after: datetime | None,
    additions_by_path: dict[str, list[tuple[str, str, str, str]]],
    review_rows: list[tuple[str, str]],
    rejected_by_reason: dict[str, list[str]],
    mention: str,
) -> None:
    now = datetime.now(timezone.utc)
    rejected_rows = [
        (model_id, reason)
        for reason, model_ids in sorted(rejected_by_reason.items())
        for model_id in sorted(model_ids, key=str.lower)
    ]
    lines = ["# Weekly Hugging Face model sync", ""]
    if sync_after:
        lines.append(f"Window: {sync_after.date()} to {now.date()} UTC")
    else:
        lines.append("Window: since the last CSV commit")
    if mention:
        lines.extend(["", f"Reviewer: {mention}"])
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Added to `models.csv`: {len(additions_by_path['models.csv'])}",
            f"- Added to `extra_models.csv`: {len(additions_by_path['extra_models.csv'])}",
            f"- Needs manual review: {len(review_rows)}",
            f"- Rejected: {len(rejected_rows)}",
            "",
            "## Added to models.csv",
            "",
            markdown_table(
                [
                    (model_id, reason)
                    for _, _, model_id, reason in sorted_additions(
                        additions_by_path["models.csv"]
                    )
                ]
            ),
            "",
            "## Added to extra_models.csv",
            "",
            markdown_table(
                [
                    (model_id, reason)
                    for _, _, model_id, reason in sorted_additions(
                        additions_by_path["extra_models.csv"]
                    )
                ]
            ),
            "",
            "## Needs manual review",
            "",
            markdown_table(sorted(review_rows, key=lambda row: row[0].lower())),
            "",
            "## Rejected",
            "",
            markdown_table(rejected_rows),
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def update_readme_changelog(
    additions_by_path: dict[str, list[tuple[str, str, str, str]]],
    review_rows: list[tuple[str, str]],
    rejected_by_reason: dict[str, list[str]],
) -> None:
    model_additions = additions_by_path["models.csv"]
    extra_additions = additions_by_path["extra_models.csv"]
    if not model_additions and not extra_additions:
        return

    today = datetime.now(timezone.utc).date().isoformat()
    rejected_count = sum(len(model_ids) for model_ids in rejected_by_reason.values())
    block = [
        f"### {today}",
        "- Weekly Hugging Face sync",
    ]
    if model_additions:
        orgs = ", ".join(
            sorted({org for org, _, _, _ in model_additions}, key=str.lower)
        )
        block.append(f"- Added {len(model_additions)} models to `models.csv`: {orgs}")
    if extra_additions:
        orgs = ", ".join(
            sorted({org for org, _, _, _ in extra_additions}, key=str.lower)
        )
        block.append(
            f"- Added {len(extra_additions)} models to `extra_models.csv`: {orgs}"
        )
    block.append(
        f"- {rejected_count} candidates rejected, "
        f"{len(review_rows)} left for manual review"
    )
    new_block = "\n".join(block) + "\n\n"

    readme_path = REPO_ROOT / "README.md"
    readme = readme_path.read_text()
    marker = "## Changelog\n\n"
    if marker not in readme:
        readme_path.write_text(readme.rstrip() + "\n\n" + marker + new_block)
        return

    date_marker = f"### {today}\n"
    if date_marker in readme:
        start = readme.index(date_marker)
        next_changelog = readme.find("\n### ", start + len(date_marker))
        next_section = readme.find("\n## ", start + len(date_marker))
        candidates = [index for index in (next_changelog, next_section) if index != -1]
        end = min(candidates) if candidates else len(readme)
        readme_path.write_text(readme[:start] + new_block + readme[end:].lstrip("\n"))
        return

    readme_path.write_text(readme.replace(marker, marker + new_block, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write", action="store_true", help="Update the CSV files in place."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write CSV files. This is the default.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print rejected model ids by reason."
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=REPO_ROOT / "policy" / "model_policy.yml",
        help="YAML policy file used to classify candidates.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write a Markdown report for the PR body.",
    )
    parser.add_argument(
        "--since-days",
        type=int,
        help="Only consider Hugging Face models created in the last N days.",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Insert or replace today's README changelog entry when rows are added.",
    )
    parser.add_argument(
        "--mention",
        default="",
        help="GitHub handle to mention in the generated report.",
    )
    args = parser.parse_args()
    if args.dry_run:
        args.write = False

    policy = yaml.safe_load(args.policy.read_text())
    rows_by_path, orgs_by_path, cutoffs, tokens_by_org, existing_model_ids = (
        load_catalog()
    )
    sync_after = None
    if args.since_days is not None:
        sync_after = datetime.now(timezone.utc) - timedelta(days=args.since_days)

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    additions_by_path: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    rejected_by_reason: dict[str, list[str]] = defaultdict(list)
    review_rows: list[tuple[str, str]] = []
    default_path_by_org: dict[str, str] = {}
    tracked_models_by_org: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for csv_name in CSV_NAMES:
        for org in orgs_by_path[csv_name]:
            default_path_by_org[org] = csv_name
        for row in rows_by_path[csv_name]:
            tracked_models_by_org[row["org"]].append((row["model"], csv_name))
            stem = strip_numeric_variant_suffix(row["model"])
            if stem:
                tracked_models_by_org[row["org"]].append((stem, csv_name))

    for tracked_models in tracked_models_by_org.values():
        tracked_models.sort(key=lambda row: (-len(row[0]), row[0].lower(), row[1]))

    for org in sorted(default_path_by_org, key=str.lower):
        default_path = default_path_by_org[org]
        for model in api.list_models(author=org, sort="created_at", full=True):
            if model.id in existing_model_ids:
                continue

            created_at = model.created_at
            if created_at and created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            if sync_after and (created_at is None or created_at <= sync_after):
                continue

            model_name = model.id.split("/", 1)[1]
            tags = {tag.lower() for tag in (model.tags or [])}
            variant_path = find_variant_path(
                org, model_name, tags, tracked_models_by_org
            )

            if not sync_after and not variant_path:
                if created_at is None or created_at <= cutoffs[default_path]:
                    continue

            decision = classify_model(model, policy, tokens_by_org[org])
            if decision.bucket == "reject":
                rejected_by_reason[decision.reason].append(model.id)
                continue
            if decision.bucket == "review":
                review_rows.append((model.id, decision.reason))
                continue

            target_path = (
                default_path if decision.bucket == "primary" else "extra_models.csv"
            )
            if variant_path:
                target_path = variant_path
            additions_by_path[target_path].append(
                (org, model_name, model.id, decision.reason)
            )
            existing_model_ids.add(model.id)
            tokens_by_org[org].update(tokenize(model_name))
            tracked_models_by_org[org].append((model_name, target_path))
            stem = strip_numeric_variant_suffix(model_name)
            if stem:
                tracked_models_by_org[org].append((stem, target_path))
            tracked_models_by_org[org].sort(
                key=lambda row: (-len(row[0]), row[0].lower(), row[1])
            )

    for csv_name in CSV_NAMES:
        additions = sorted_additions(additions_by_path[csv_name])
        print(f"{csv_name}: {len(additions)} additions")
        for _, _, model_id, reason in additions:
            print(f"{model_id} -- {reason}")
        print()

        if not args.write or not additions:
            continue
        write_rows(REPO_ROOT / csv_name, rows_by_path[csv_name], additions)

    if args.write and args.update_readme:
        update_readme_changelog(additions_by_path, review_rows, rejected_by_reason)

    if args.write:
        for csv_name in CSV_NAMES:
            validate_csv(REPO_ROOT / csv_name)

    if args.report:
        write_report(
            args.report,
            sync_after,
            additions_by_path,
            review_rows,
            rejected_by_reason,
            args.mention,
        )

    print("Needs manual review:")
    for model_id, reason in sorted(review_rows, key=lambda row: row[0].lower()):
        print(f"{model_id} -- {reason}")
    print()

    print("Rejected candidates:")
    for reason in sorted(rejected_by_reason):
        print(f"{reason}: {len(rejected_by_reason[reason])}")
        if not args.verbose:
            continue
        for model_id in rejected_by_reason[reason]:
            print(model_id)
        print()


if __name__ == "__main__":
    main()
