from __future__ import annotations

from jascue_video_lab.editing_capabilities import (
    EditingCapabilityCatalog,
    simple_production_capability_catalog,
)


def test_simple_capability_catalog_is_stable_and_full_bleed_first() -> None:
    catalog = simple_production_capability_catalog()
    restored = EditingCapabilityCatalog.model_validate(
        catalog.model_dump(mode="json")
    )

    assert restored.definition_sha256() == catalog.definition_sha256()
    assert catalog.vertical_delivery_preference == "full_bleed_first"
    assert catalog.prohibited_automatic_delivery == [
        "solid_fit",
        "blurred_background",
    ]
    assert len(catalog.definition_sha256()) == 64


def test_capability_catalog_exposes_execution_without_geometry_output() -> None:
    catalog = simple_production_capability_catalog()
    ids = {item.capability_id for item in catalog.capabilities}

    assert {
        "tracked_full_bleed_crop",
        "phase_virtual_camera",
        "hard_cut_between_views",
        "controlled_semantic_clip",
        "alternate_candidate",
    } <= ids
    assert catalog.planner_boundary == (
        "semantic_intent_only_no_exact_time_or_geometry"
    )
