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
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import yaml
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from sync_hf_models import (
    Decision,
    classify_model,
    load_catalog,
    markdown_table,
    validate_csv,
    write_rows,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DATA_DIR = REPO_ROOT.parent / "artifacts-models" / "data"
MODEL_ID_PATTERN = re.compile(r"^[^/\s]+/[^/\s]+$")
NON_TEXT_NAME_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(^|[-_])(vl|vlm|vla|mm|omni)($|[-_])",
        r"(vision|visual|image|video|audio|speech|asr|tts|whisper|ocr)",
        r"(embedding|embed|reranker|reward|classifier)",
    )
)
NON_TEXT_TAGS = {
    "models/llms/multi-modal",
    "models/embedding",
    "models/image-generation",
    "models/video-generation",
    "models/audio-generation",
}


@dataclass(frozen=True)
class ArtifactModel:
    model_id: str
    title: str
    path: Path
    tags: tuple[str, ...]
    url: str


@dataclass(frozen=True)
class CandidateDecision:
    artifact: ArtifactModel
    decision: Decision
    downloads: int
    pipeline_tag: str


def parse_frontmatter(path: Path) -> dict:
    text = path.read_text(errors="replace")
    if not text.startswith("---"):
        return {}

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}

    loaded = yaml.safe_load(parts[1]) or {}
    return loaded if isinstance(loaded, dict) else {}


def model_id_from_url(url: str) -> str:
    if not url:
        return ""

    parsed = urlparse(url)
    if parsed.netloc != "huggingface.co":
        return ""

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2 or parts[0] in {"datasets", "spaces", "collections", "papers"}:
        return ""

    model_id = f"{parts[0]}/{parts[1]}"
    return model_id if MODEL_ID_PATTERN.fullmatch(model_id) else ""


def has_llm_tag(tags: tuple[str, ...]) -> bool:
    return any(tag == "models/llms" or tag.startswith("models/llms/") for tag in tags)


def looks_text_first(artifact: ArtifactModel) -> bool:
    lowered_tags = {tag.lower() for tag in artifact.tags}
    if lowered_tags & NON_TEXT_TAGS:
        return False

    searchable = " ".join([artifact.model_id, artifact.title, artifact.path.name])
    return not any(pattern.search(searchable) for pattern in NON_TEXT_NAME_PATTERNS)


def iter_artifact_models(
    artifacts_data_dir: Path,
    *,
    include_multimodal: bool,
) -> tuple[list[ArtifactModel], list[tuple[str, str]]]:
    models: list[ArtifactModel] = []
    skipped: list[tuple[str, str]] = []
    seen_model_ids: set[str] = set()

    for path in sorted(artifacts_data_dir.glob("**/*.md")):
        if "datasets" in path.parts:
            continue

        frontmatter = parse_frontmatter(path)
        tags = tuple(str(tag) for tag in frontmatter.get("tags") or [])
        if not has_llm_tag(tags):
            continue

        model_id = model_id_from_url(str(frontmatter.get("url") or ""))
        if not model_id:
            skipped.append((str(path), "no Hugging Face model URL"))
            continue
        if model_id in seen_model_ids:
            skipped.append((model_id, "duplicate artifacts model ID"))
            continue

        artifact = ArtifactModel(
            model_id=model_id,
            title=str(frontmatter.get("name") or frontmatter.get("title") or path.stem),
            path=path,
            tags=tags,
            url=str(frontmatter.get("url") or ""),
        )
        if not include_multimodal and not looks_text_first(artifact):
            skipped.append((model_id, "outside text-first LLM scope"))
            continue

        models.append(artifact)
        seen_model_ids.add(model_id)

    return models, skipped


def is_addable_review(decision: Decision, downloads: int, min_downloads: int) -> bool:
    if decision.bucket in {"primary", "extra"}:
        return True
    return decision.bucket == "review" and downloads >= min_downloads


def classify_artifact_candidates(
    artifacts: list[ArtifactModel],
    *,
    api: HfApi,
    policy: dict,
    tokens_by_org: dict[str, set[str]],
    existing_model_ids: set[str],
    min_downloads: int,
) -> tuple[
    list[CandidateDecision],
    list[CandidateDecision],
    list[tuple[str, str]],
    list[ArtifactModel],
]:
    additions: list[CandidateDecision] = []
    review: list[CandidateDecision] = []
    rejected: list[tuple[str, str]] = []
    already_tracked: list[ArtifactModel] = []

    for artifact in artifacts:
        if artifact.model_id in existing_model_ids:
            already_tracked.append(artifact)
            continue

        org = artifact.model_id.split("/", 1)[0]
        try:
            model = api.model_info(artifact.model_id, files_metadata=False)
        except (HfHubHTTPError, RepositoryNotFoundError) as error:
            rejected.append((artifact.model_id, f"Hugging Face lookup failed: {error}"))
            continue

        decision = classify_model(model, policy, tokens_by_org[org])
        downloads = model.downloads or 0
        candidate = CandidateDecision(
            artifact=artifact,
            decision=decision,
            downloads=downloads,
            pipeline_tag=model.pipeline_tag or "",
        )
        if decision.bucket == "reject":
            rejected.append((artifact.model_id, decision.reason))
            continue
        if is_addable_review(decision, downloads, min_downloads):
            additions.append(candidate)
            existing_model_ids.add(artifact.model_id)
            tokens_by_org[org].update({artifact.model_id.rsplit("/", 1)[-1].lower()})
            continue

        review.append(candidate)

    return additions, review, rejected, already_tracked


def sorted_additions(
    additions: list[CandidateDecision],
) -> list[tuple[str, str, str, str]]:
    rows = []
    for candidate in additions:
        org, model_name = candidate.artifact.model_id.split("/", 1)
        rows.append(
            (
                org,
                model_name,
                candidate.artifact.model_id,
                (
                    "artifacts-models LLM candidate; "
                    f"{candidate.decision.reason}; "
                    f"source `{candidate.artifact.path}`"
                ),
            )
        )
    return sorted(rows, key=lambda row: (row[0].lower(), row[1].lower(), row[2]))


def compact_artifact_path(path: Path, artifacts_data_dir: Path) -> str:
    try:
        return str(path.relative_to(artifacts_data_dir.parent))
    except ValueError:
        return str(path)


def write_report(
    report_path: Path,
    artifacts_data_dir: Path,
    additions: list[CandidateDecision],
    review: list[CandidateDecision],
    rejected: list[tuple[str, str]],
    skipped: list[tuple[str, str]],
    already_tracked: list[ArtifactModel],
    mention: str,
) -> None:
    now = datetime.now(timezone.utc).date().isoformat()
    skipped_by_reason: dict[str, list[str]] = defaultdict(list)
    for model_id, reason in skipped:
        skipped_by_reason[reason].append(model_id)

    lines = [
        "# Artifacts model sync",
        "",
        f"Generated: {now} UTC",
        f"Source: `{artifacts_data_dir}`",
    ]
    if mention:
        lines.extend(["", f"Reviewer: {mention}"])

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Added to `extra_models.csv`: {len(additions)}",
            f"- Already tracked: {len(already_tracked)}",
            f"- Needs manual review: {len(review)}",
            f"- Rejected by tracked-models policy: {len(rejected)}",
            f"- Skipped before HF lookup: {len(skipped)}",
            "",
            "## Added to extra_models.csv",
            "",
            markdown_table(
                [
                    (
                        candidate.artifact.model_id,
                        (
                            f"{candidate.downloads:,} downloads; "
                            f"`{candidate.pipeline_tag or 'no pipeline_tag'}`; "
                            f"{candidate.decision.reason}; "
                            f"source `{compact_artifact_path(candidate.artifact.path, artifacts_data_dir)}`"
                        ),
                    )
                    for candidate in sorted(
                        additions, key=lambda item: item.artifact.model_id.lower()
                    )
                ]
            ),
            "",
            "## Needs manual review",
            "",
            markdown_table(
                [
                    (
                        candidate.artifact.model_id,
                        (
                            f"{candidate.downloads:,} downloads; "
                            f"`{candidate.pipeline_tag or 'no pipeline_tag'}`; "
                            f"{candidate.decision.reason}; "
                            f"source `{compact_artifact_path(candidate.artifact.path, artifacts_data_dir)}`"
                        ),
                    )
                    for candidate in sorted(
                        review, key=lambda item: item.artifact.model_id.lower()
                    )
                ]
            ),
            "",
            "## Rejected",
            "",
            markdown_table(sorted(rejected, key=lambda row: row[0].lower())),
            "",
            "## Skipped before HF lookup",
            "",
            markdown_table(
                [
                    (reason, f"{len(model_ids)} candidates")
                    for reason, model_ids in sorted(skipped_by_reason.items())
                ]
            ),
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def update_readme_changelog(additions: list[CandidateDecision], review_count: int) -> None:
    if not additions:
        return

    today = datetime.now(timezone.utc).date().isoformat()
    orgs = ", ".join(
        sorted(
            {candidate.artifact.model_id.split("/", 1)[0] for candidate in additions},
            key=str.lower,
        )
    )
    block = "\n".join(
        [
            f"### {today}",
            "- Artifacts model sync",
            f"- Added {len(additions)} models to `extra_models.csv`: {orgs}",
            f"- {review_count} candidates left for manual review",
            "",
        ]
    )

    readme_path = REPO_ROOT / "README.md"
    readme = readme_path.read_text()
    marker = "## Changelog\n\n"
    if marker not in readme:
        readme_path.write_text(readme.rstrip() + "\n\n" + marker + block + "\n")
        return

    readme_path.write_text(readme.replace(marker, marker + block + "\n", 1))


def write_artifacts_snapshot(
    snapshot_path: Path,
    additions: list[CandidateDecision],
    review: list[CandidateDecision],
    rejected: list[tuple[str, str]],
    skipped: list[tuple[str, str]],
) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open("w", newline="\n") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bucket",
                "modelId",
                "downloads",
                "pipeline_tag",
                "reason",
                "source",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for candidate in sorted(
            additions, key=lambda item: item.artifact.model_id.lower()
        ):
            writer.writerow(
                {
                    "bucket": "add_extra",
                    "modelId": candidate.artifact.model_id,
                    "downloads": candidate.downloads,
                    "pipeline_tag": candidate.pipeline_tag,
                    "reason": candidate.decision.reason,
                    "source": candidate.artifact.path,
                }
            )
        for candidate in sorted(
            review, key=lambda item: item.artifact.model_id.lower()
        ):
            writer.writerow(
                {
                    "bucket": "manual_review",
                    "modelId": candidate.artifact.model_id,
                    "downloads": candidate.downloads,
                    "pipeline_tag": candidate.pipeline_tag,
                    "reason": candidate.decision.reason,
                    "source": candidate.artifact.path,
                }
            )
        for model_id, reason in sorted(rejected, key=lambda row: row[0].lower()):
            writer.writerow(
                {
                    "bucket": "rejected",
                    "modelId": model_id,
                    "downloads": "",
                    "pipeline_tag": "",
                    "reason": reason,
                    "source": "",
                }
            )
        for model_id, reason in sorted(skipped, key=lambda row: row[0].lower()):
            writer.writerow(
                {
                    "bucket": "skipped",
                    "modelId": model_id,
                    "downloads": "",
                    "pipeline_tag": "",
                    "reason": reason,
                    "source": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DATA_DIR,
        help="Path to the artifacts-models data directory.",
    )
    parser.add_argument("--write", action="store_true", help="Update extra_models.csv.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not update CSV files. This is the default.",
    )
    parser.add_argument(
        "--include-multimodal",
        action="store_true",
        help="Include artifacts tagged as multimodal LLMs or VLM-like models.",
    )
    parser.add_argument(
        "--min-downloads",
        type=int,
        default=100000,
        help="Minimum downloads required for unknown-family review candidates.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=REPO_ROOT / "policy" / "model_policy.yml",
        help="YAML policy file used to classify candidates.",
    )
    parser.add_argument("--report", type=Path, help="Write a Markdown PR report.")
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Write a CSV snapshot of add/review/reject decisions.",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Insert an artifacts sync entry into the README changelog.",
    )
    parser.add_argument(
        "--mention",
        default="",
        help="GitHub handle to mention in the generated report.",
    )
    args = parser.parse_args()
    if args.dry_run:
        args.write = False

    artifacts_data_dir = args.artifacts_dir.resolve()
    if not artifacts_data_dir.exists():
        raise FileNotFoundError(
            f"Artifacts data directory not found: {artifacts_data_dir}"
        )

    policy = yaml.safe_load(args.policy.read_text())
    rows_by_path, _, _, tokens_by_org, existing_model_ids = load_catalog()
    artifacts, skipped = iter_artifact_models(
        artifacts_data_dir,
        include_multimodal=args.include_multimodal,
    )
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    additions, review, rejected, already_tracked = classify_artifact_candidates(
        artifacts,
        api=api,
        policy=policy,
        tokens_by_org=tokens_by_org,
        existing_model_ids=existing_model_ids,
        min_downloads=args.min_downloads,
    )

    print(f"Artifacts LLM candidates: {len(artifacts)}")
    print(f"Already tracked: {len(already_tracked)}")
    print(f"Additions for extra_models.csv: {len(additions)}")
    print(f"Needs manual review: {len(review)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Skipped: {len(skipped)}")

    if args.write and additions:
        write_rows(
            REPO_ROOT / "extra_models.csv",
            rows_by_path["extra_models.csv"],
            sorted_additions(additions),
        )
        validate_csv(REPO_ROOT / "extra_models.csv")
        if args.update_readme:
            update_readme_changelog(additions, len(review))

    if args.report:
        write_report(
            args.report,
            artifacts_data_dir,
            additions,
            review,
            rejected,
            skipped,
            already_tracked,
            args.mention,
        )
    if args.snapshot:
        write_artifacts_snapshot(args.snapshot, additions, review, rejected, skipped)


if __name__ == "__main__":
    main()
