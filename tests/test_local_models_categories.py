"""Independent tests for modification categories relative to tracked parents."""

from scripts.local_models import CatalogError, discover_local_models, validate_catalog
import pytest


def parent(model_id="Root/Base", kind="models"):
    org, model = model_id.split("/")
    return {"org": org, "model": model, "modelId": model_id, "parent_list": kind}


def model(
    model_id,
    base=None,
    relation="quantized",
    formats=("gguf",),
    explicit=False,
    **extra,
):
    card = {"base_model_relation": relation} if explicit and relation else {}
    return {
        "id": model_id,
        "tags": list(formats),
        "base_model_ids": [base] if base else [],
        "base_model_relation": relation,
        "card_data": card,
        **extra,
    }


def discover(rows, parents=None, overrides=None):
    parents = parents or [parent()]
    result = discover_local_models(
        rows, parents, {"root": "china", "team": "us"}, overrides
    )
    validate_catalog(
        result["rows"],
        result["evidence"],
        parents,
        {"root": "china", "team": "us"},
        overrides,
    )
    return result


def row_for(result, model_id):
    return next(row for row in result["rows"] if row["modelId"] == model_id)


def assert_category(row, category, modified=True):
    assert row["variant_category"] == category
    assert row["is_modified"] == str(modified).lower()
    lineage = row["parent_lineage"].split(";")
    assert lineage[0] == row["modelId"]
    assert lineage[-1] == row["parent_model_id"]
    assert len(row["lineage_categories"].split(";")) == len(lineage) - 1


def test_conversion_of_nonlocal_finetune_preserves_training_edge_and_original_origin():
    leaf = "Quant/CustomAssistant-GGUF"
    result = discover(
        [
            model(
                "Team/CustomAssistant",
                "Root/Base",
                "finetune",
                formats=(),
                explicit=True,
            ),
            model(leaf, "Team/CustomAssistant", explicit=True),
        ]
    )
    row = row_for(result, leaf)
    assert_category(row, "finetune")
    assert row["parent_model_id"] == "Root/Base"
    assert row["parent_region"] == "china"
    assert row["base_model_id"] == "Team/CustomAssistant"
    assert row["parent_lineage"] == f"{leaf};Team/CustomAssistant;Root/Base"
    assert row["lineage_categories"] == "conversion;finetune"
    assert row["modification_categories"] == "finetune"
    assert "Team/CustomAssistant" not in {row["modelId"] for row in result["rows"]}


@pytest.mark.parametrize(
    "relation,category",
    [
        ("finetune", "finetune"),
        ("pruned", "pruned"),
        ("adapter", "adapter"),
        ("merge", "merge"),
        ("distilled", "distilled"),
    ],
)
def test_explicit_weight_modification_relation_has_its_own_category(relation, category):
    result = discover(
        [model("Local/Variant-GGUF", "Root/Base", relation, explicit=True)]
    )
    assert_category(row_for(result, "Local/Variant-GGUF"), category)


def test_pruned_intermediate_can_change_parameter_name_before_local_conversion():
    leaf = "Quant/Base-264B-GGUF"
    result = discover(
        [
            model("Team/Base-264B", "Root/Base", "pruned", formats=(), explicit=True),
            model(leaf, "Team/Base-264B", explicit=True),
        ]
    )
    row = row_for(result, leaf)
    assert_category(row, "pruned")
    assert row["lineage_categories"] == "conversion;pruned"


@pytest.mark.parametrize(
    "suffix,category", [("REAP-pruned", "pruned"), ("abliterated", "abliterated")]
)
def test_modification_name_is_not_erased_by_quantized_card_relation(suffix, category):
    leaf = f"Quant/Base-{suffix}-GGUF"
    result = discover([model(leaf, "Root/Base", explicit=True)])
    assert_category(row_for(result, leaf), category)


def test_finetuned_tracked_checkpoint_is_the_baseline_not_an_upstream_modification():
    tracked = "Team/Base-Instruct"
    leaf = "Quant/Base-Instruct-GGUF"
    result = discover(
        [
            model(tracked, "Root/Base", "finetune", formats=(), explicit=True),
            model(leaf, tracked, explicit=True),
        ],
        [parent(), parent(tracked, "extra_models")],
    )
    row = row_for(result, leaf)
    assert_category(row, "conversion", modified=False)
    assert row["parent_model_id"] == tracked
    assert row["parent_list"] == "extra_models"
    assert row["parent_region"] == "us"
    assert row["parent_lineage"] == f"{leaf};{tracked}"
    assert row["modification_categories"] == ""


def test_tracked_pruned_baseline_does_not_mark_its_ordinary_conversion_modified():
    tracked = "Team/Base-REAP-pruned"
    leaf = "Quant/Base-REAP-pruned-GGUF"
    result = discover(
        [
            model(tracked, "Root/Base", "pruned", formats=(), explicit=True),
            model(leaf, tracked, explicit=True),
        ],
        [parent(), parent(tracked)],
    )
    assert_category(row_for(result, leaf), "conversion", modified=False)


def test_distinct_modifications_accumulate_to_mixed_category():
    leaf = "Quant/PrunedAssistant-GGUF"
    result = discover(
        [
            model("Team/Assistant", "Root/Base", "finetune", formats=(), explicit=True),
            model(
                "Team/PrunedAssistant",
                "Team/Assistant",
                "pruned",
                formats=(),
                explicit=True,
            ),
            model(leaf, "Team/PrunedAssistant", explicit=True),
        ]
    )
    row = row_for(result, leaf)
    assert_category(row, "mixed")
    assert set(row["modification_categories"].split(";")) == {"finetune", "pruned"}
    assert row["lineage_categories"] == "conversion;pruned;finetune"


def test_real_multi_parent_merge_stays_ambiguous_even_if_branches_share_tracked_origin():
    leaf = model("Local/Blended-GGUF", relation="merge", explicit=True)
    leaf["base_model_ids"] = ["Team/A", "Team/B"]
    result = discover(
        [
            model("Team/A", "Root/Base", "finetune", formats=(), explicit=True),
            model("Team/B", "Root/Base", "finetune", formats=(), explicit=True),
            leaf,
        ]
    )
    assert "Local/Blended-GGUF" not in {row["modelId"] for row in result["rows"]}
    candidate = next(
        row for row in result["candidates"] if row["model_id"] == "Local/Blended-GGUF"
    )
    assert candidate["reason"] == "ambiguous_parent"


def test_unknown_unique_lineage_is_conservatively_flagged_not_declared_pure_conversion():
    result = discover([model("Local/Renamed-GGUF", "Root/Base", relation=None)])
    assert_category(row_for(result, "Local/Renamed-GGUF"), "unknown")


def test_modification_still_requires_actual_local_format_evidence():
    result = discover(
        [
            model(
                "Local/Base-finetuned-GGUF",
                "Root/Base",
                "finetune",
                formats=(),
                explicit=True,
            )
        ]
    )
    assert not result["rows"]


@pytest.mark.parametrize("flag", ["private", "disabled"])
def test_ineligible_training_intermediate_cannot_supply_a_lineage(flag):
    result = discover(
        [
            model(
                "Team/Assistant",
                "Root/Base",
                "finetune",
                formats=(),
                explicit=True,
                **{flag: True},
            ),
            model("Local/Assistant-GGUF", "Team/Assistant", explicit=True),
        ]
    )
    assert not result["rows"]


def test_cycle_of_training_edges_does_not_resolve_to_a_guessed_parent():
    result = discover(
        [
            model("Team/A", "Team/B", "finetune", formats=(), explicit=True),
            model("Team/B", "Team/A", "finetune", formats=(), explicit=True),
            model("Local/A-GGUF", "Team/A", explicit=True),
        ]
    )
    assert not result["rows"]
    assert (
        next(row for row in result["candidates"] if row["model_id"] == "Local/A-GGUF")[
            "reason"
        ]
        == "lineage_cycle"
    )


@pytest.mark.parametrize(
    "tag,category",
    [
        ("pruned", "pruned"),
        ("expert-pruning", "pruned"),
        ("abliterated", "abliterated"),
    ],
)
def test_explicit_modification_tag_is_not_hidden_by_ordinary_name_and_quantized_relation(
    tag, category
):
    leaf = model("Local/Base-GGUF", "Root/Base", explicit=True)
    leaf["tags"].append(tag)
    result = discover([model("Root/Base", formats=(), relation=None), leaf])
    assert_category(row_for(result, leaf["id"]), category)


def test_inherited_pruning_tag_does_not_modify_tracked_pruned_baseline_again():
    root = model("Root/Base", formats=(), relation=None)
    root["tags"] = ["expert-pruning"]
    leaf = model("Local/Base-GGUF", "Root/Base", explicit=True)
    leaf["tags"].append("expert-pruning")
    result = discover([root, leaf])
    assert_category(row_for(result, leaf["id"]), "conversion", modified=False)


def test_generic_finetuned_tag_alone_is_not_new_training_evidence():
    leaf = model("Local/Base-GGUF", "Root/Base", explicit=True)
    leaf["tags"].append("finetuned")
    result = discover([leaf])
    assert_category(row_for(result, leaf["id"]), "conversion", modified=False)


def test_modified_tracked_local_format_checkpoint_remains_the_nearest_baseline():
    tracked = "Team/Base-SFT-GGUF"
    leaf = "Local/Base-SFT-GGUF-MLX"
    result = discover(
        [
            model(
                tracked,
                "Root/Base",
                formats=("gguf",),
                card_data={"base_model_relation": "finetune"},
            ),
            model(leaf, tracked, formats=("mlx",), explicit=True),
        ],
        [parent(), parent(tracked, "extra_models")],
    )
    row = row_for(result, leaf)
    assert_category(row, "conversion", modified=False)
    assert row["parent_model_id"] == tracked
    assert row["parent_region"] == "us"
    assert row["parent_lineage"] == f"{leaf};{tracked}"


def test_validator_rejects_reviewed_exclusion_as_an_intermediate():
    metadata = [
        model("Team/Assistant", "Root/Base", "finetune", formats=(), explicit=True),
        model("Local/Assistant-GGUF", "Team/Assistant", explicit=True),
    ]
    result = discover(metadata)
    exclusion = {
        "model_id": "Team/Assistant",
        "action": "exclude",
        "reason": "Reviewed source has an ambiguous multi-parent lineage.",
        "source_urls": [
            "https://huggingface.co/Team/Assistant/blob/"
            "0123456789012345678901234567890123456789/README.md"
        ],
    }
    # Both the registry and persisted evidence carry the exclusion: agreeing
    # copies must not make a previously generated lineage eligible again.
    result["evidence"][0]["intermediate_evidence"][0]["override"] = exclusion
    assert not discover(metadata, overrides=[exclusion])["rows"]
    with pytest.raises(CatalogError, match="exclu"):
        validate_catalog(
            result["rows"],
            result["evidence"],
            [parent()],
            {"root": "china", "team": "us"},
            [exclusion],
        )
