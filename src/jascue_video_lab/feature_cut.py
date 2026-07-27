from __future__ import annotations

import base64
import hashlib
import html
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from time import monotonic
from typing import Any, Collection, Literal, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .auto_reframe import (
    AutoReframePolicy,
    CandidatePreflight,
    FailureCode,
    RegionAssessment,
    SemanticCheckpointStatus,
    audit_auto_bounded_clip,
    choose_recovery,
)
from .billing import summarize_usage_and_list_price, summarize_usage_files
from .gemini import (
    EDITORIAL_SYSTEM_INSTRUCTION,
    GeminiLabClient,
    GroundingIdentityReference,
    MODEL_ID,
    VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
    canonicalize_selected_vertical_framing_output,
)
from .grounding_selection import (
    require_grounding_request_match,
    require_tracking_seed_candidate,
)
from .identity_checkpoints import (
    IdentityCheckpointExecution,
    execute_identity_checkpoints,
    plan_identity_checkpoints,
)
from .media import (
    extract_frame,
    extract_frame_at_pts,
    has_audio_stream,
    last_decoded_video_frame_pts,
    probe_video,
    sha256_file,
)
from .music import MusicMapLock
from .multi_tracking import validate_segmentation_track_alignment
from .models import (
    EvidenceApprovalSource,
    EvidenceAspectConstraintV2,
    EvidenceClaimSource,
    EvidenceFramingObligationsV2,
    EvidenceIdentityContractV2,
    EvidencePredicateContractV2,
    EvidenceQueryApprovalProvenance,
    EvidenceQueryLockV2,
    EvidenceQueryProposalV2,
    EvidenceQueryProvenanceV2,
    EvidenceTargetIdentityV2,
    EvidenceTargetVisibilityConstraintV2,
    EligibilityGateStatus,
    FeatureChapterBrief,
    FeatureChapterSelect,
    FeatureCutEditorialContract,
    FeatureCutEligibilityReport,
    FeatureCutExecutionProfile,
    FeatureCutRunState,
    FeatureEditBrief,
    FeatureEditPlan,
    FeatureVerticalCandidate,
    FramingRegionIntent,
    GeminiNativeGroundingProposal,
    GroundingProposal,
    MediaInfo,
    RushClip,
    RushFrame,
    RushesCatalog,
    RhythmPlan,
    SegmentationTrack,
    SelectedVerticalFramingProposal,
    ShotQualityMap,
    SharedSam21AnalysisFramesManifest,
    SharedSam21BBoxSeed,
    SharedSam21SessionManifest,
    TrackingState,
    TrimIntentDecision,
    VerticalVirtualCameraPhase,
    VerticalVirtualCameraPlan,
    VerticalVirtualCameraProposal,
    VirtualCameraKeyframe,
    VirtualCameraPlan,
    approve_evidence_query_proposal_v2,
)
from .overlay import draw_grounding_overlay
from .reframe_policy import (
    REFRAME_POLICY_BINDING_ORIGIN,
    validate_reframe_policy_bundle,
)
from .rushes import _segment_bounds, validate_rushes_catalog_sources
from .sam_tracking import (
    SAM21_CONFIG,
    SAM21_IMPLEMENTATION_REVISION,
    SAM21_TINY_MODEL_ID,
    pad_normalized_box,
    require_bbox_track_request_match,
    track_bbox_sam21,
    track_bboxes_shared_sam21,
)
from .schema import gemini_response_schema
from .shots import ShotManifest, detect_shots_ffmpeg
from .shot_quality import (
    build_quality_safe_intervals,
    build_render_quality_report,
    load_shot_quality_map,
    scan_shot_quality,
)
from .editorial_planning import (
    build_attention_profile,
    build_rhythm_plan,
    reconcile_attention_delivery_floor,
)
from .storage import read_json, utc_now, write_json


_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
)
_RENDER_PIPELINE_VERSION = "feature-cut-v15-pts-topk-delivery-chain"
RenderAspect = Literal["both", "9x16", "16x9"]
_TRACKING_MAX_SIDE = 960
_TRACKING_DEVICE = "cpu"
_TRACKING_SEED_BOX_PADDING_RATIO = 0.04
_PORTRAIT_PHASE_MAX_ZOOM = 1.12
_PORTRAIT_DEADBAND_VIEWPORT_FRACTION = 0.08
_PORTRAIT_MAX_SPEED_PX_S = 720.0
_PORTRAIT_MAX_ACCELERATION_PX_S2 = 1800.0
_PORTRAIT_MAX_JERK_PX_S3 = 7200.0
_FEATURE_PLAN_BINDING_VERSION = "feature-plan-binding-v1"
_EXTERNAL_PROJECTION_SIDECAR_VERSION = "external-feature-plan-projection-v1"
_EXTERNAL_PROJECTION_POINTER_NAME = "feature-plan.external-projection.json"
_EXTERNAL_PROJECTION_CONTRACTS = {
    "clip-card-open-edit-v1": {
        "source": "validated open edit plan selected from Clip Card evidence",
        "transform": "project_feature_contracts maps ordered shots into brief and feature plan",
        "target": "FeatureEditPlan",
        "module": "scripts.plan_clip_card_open_edit",
        "source_model": "OpenEditPlan",
        "projector": "reproject_external_feature_plan",
        "raw_output_role": "source_raw_output",
        "required_artifact_roles": [
            "source_raw_interaction",
            "source_raw_output",
        ],
    },
    "clip-card-open-edit-v2": {
        "source": (
            "validated no-brief open edit plan with Top-K aspect candidates"
        ),
        "transform": (
            "project_feature_contracts preserves selected-first runtime candidate "
            "lists without mutating the editorial source plan"
        ),
        "target": "FeatureEditPlan",
        "module": "scripts.plan_clip_card_open_edit",
        "source_model": "OpenEditPlan",
        "projector": "reproject_external_feature_plan_v2",
        "raw_output_role": "source_raw_output",
        "required_artifact_roles": [
            "source_raw_interaction",
            "source_raw_output",
        ],
    },
    "clip-card-feature-cut-v1": {
        "source": "validated brief-aware Clip Card feature plan",
        "transform": "chapter selections are projected in order into FeatureChapterSelect",
        "target": "FeatureEditPlan",
        "module": "scripts.plan_clip_card_feature_cut",
        "source_model": "ClipCardFeaturePlan",
        "projector": "reproject_external_feature_plan",
        "raw_output_role": "source_raw_output",
        "required_artifact_roles": [
            "source_raw_interaction",
            "source_raw_output",
        ],
    },
    "clip-card-feature-cut-v2": {
        "source": (
            "validated brief-aware Clip Card feature plan with hash-bound Top-K "
            "aspect candidates and entity-resolved framing regions"
        ),
        "transform": (
            "selected-first candidate lists and legacy rank-1 fields are "
            "deterministically projected into FeatureEditPlan"
        ),
        "target": "FeatureEditPlan",
        "module": "scripts.plan_clip_card_feature_cut",
        "source_model": "ClipCardFeaturePlanV2",
        "projector": "reproject_external_feature_plan_v2",
        "raw_output_role": "source_raw_output",
        "required_artifact_roles": [
            "source_raw_interaction",
            "source_raw_output",
        ],
    },
    "clip-card-feature-cut-v3": {
        "source": (
            "selection-only brief-aware Top-K choices plus hash-bound local "
            "Clip Card evidence"
        ),
        "transform": (
            "rank-one mirrors, target descriptions, and executable framing "
            "regions are derived locally from selected entity priorities and "
            "the bound Clip Card evidence"
        ),
        "target": "FeatureEditPlan",
        "module": "scripts.plan_clip_card_feature_cut",
        "source_model": "ClipCardFeaturePlanV3",
        "projector": "reproject_external_feature_plan_v3",
        "raw_output_role": "source_raw_output",
        "required_artifact_roles": [
            "source_raw_interaction",
            "source_raw_output",
            "selected_clip_card_evidence",
        ],
    },
    "direct-video-edit-plan-v1": {
        "source": (
            "compact rank-based editorial intent over hash-bound bounded "
            "candidate videos and supplied music"
        ),
        "transform": (
            "chapter, candidate, and entity integer indices are resolved "
            "locally into a validated ClipCardFeaturePlanV3 before the "
            "existing deterministic renderer projection"
        ),
        "target": "FeatureEditPlan",
        "module": "scripts.plan_clip_card_feature_cut",
        "source_model": "DirectVideoEditPlan",
        "source_plan_has_model_provenance": False,
        "projector": "reproject_direct_video_edit_plan",
        "raw_output_role": "source_raw_output",
        "required_artifact_roles": [
            "source_raw_interaction",
            "source_raw_output",
            "derived_clip_card_feature_plan",
            "selected_clip_card_evidence",
            "feature_shortlist",
            "candidate_video_evidence_manifest",
        ],
    },
    "clip-card-feature-music-rerank-v1": {
        "source": (
            "music-aware aspect selections over an existing validated Top-K "
            "Clip Card feature plan"
        ),
        "transform": (
            "only selected horizontal and vertical candidate IDs are changed; "
            "all candidate evidence, framing roles, and executable regions are "
            "reused from hash-bound upstream artifacts"
        ),
        "target": "FeatureEditPlan",
        "module": "scripts.rerank_clip_card_feature_plan_with_music",
        "source_model": "MusicAwareTopKSelection",
        "projector": "reproject_music_aware_topk_selection",
        "raw_output_role": "source_raw_output",
        "required_artifact_roles": [
            "source_raw_interaction",
            "source_raw_output",
            "source_music",
            "upstream_source_plan",
            "selected_clip_card_evidence",
        ],
    },
    "open-edit-candidate-overrides-v1": {
        "source": "validated upstream open edit plan plus human-reviewed candidate patch",
        "transform": "only named aspect candidates are replaced before project_feature_contracts",
        "target": "FeatureEditPlan",
        "module": "scripts.apply_open_edit_candidate_overrides",
        "source_model": "OpenEditPlan",
        "projector": "reproject_external_feature_plan",
        "raw_output_role": None,
        "required_artifact_roles": [
            "input_open_edit_plan",
            "candidate_override_patch",
            "candidate_override_audit",
            "upstream_projection_pointer",
            "upstream_projection_record",
        ],
    },
    "open-edit-candidate-overrides-v2": {
        "source": (
            "validated upstream open edit plan plus human-reviewed candidate "
            "patch with Top-K runtime candidates"
        ),
        "transform": (
            "only named aspect candidates are replaced before selected-first "
            "candidate-preserving project_feature_contracts"
        ),
        "target": "FeatureEditPlan",
        "module": "scripts.apply_open_edit_candidate_overrides",
        "source_model": "OpenEditPlan",
        "projector": "reproject_external_feature_plan_v2",
        "raw_output_role": None,
        "required_artifact_roles": [
            "input_open_edit_plan",
            "candidate_override_patch",
            "candidate_override_audit",
            "upstream_projection_pointer",
            "upstream_projection_record",
        ],
    },
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _prompt_binds_sha256(prompt: str, field_name: str, expected_sha256: str) -> bool:
    """Accept explicit field bindings without coupling provenance to punctuation."""

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field_name):
        raise ValueError("invalid prompt binding field name")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("invalid expected SHA-256")
    pattern = (
        rf"(?<![A-Za-z0-9_]){re.escape(field_name)}"
        rf"(?:\s*(?:=|:|：)|\s+必須原樣回傳\s*(?:=|:|：))"
        rf"\s*{expected_sha256}(?![0-9a-f])"
    )
    return re.search(pattern, prompt) is not None


def _external_projection_contract_sha256(contract_id: str) -> str:
    contract = _EXTERNAL_PROJECTION_CONTRACTS.get(contract_id)
    if contract is None:
        raise ValueError(f"unsupported external feature plan projection: {contract_id}")
    return _sha256_json({"contract_id": contract_id, **contract})


def _validate_external_projection_semantics(
    *,
    projection_contract_id: str,
    catalog_path: Path,
    brief_path: Path,
    feature_plan_path: Path,
    source_plan_path: Path,
    source_request_path: Path,
    source_artifacts: Mapping[str, Path],
) -> None:
    """Reparse source evidence and deterministically reproduce the projection."""

    contract = _EXTERNAL_PROJECTION_CONTRACTS.get(projection_contract_id)
    if contract is None:
        raise ValueError(
            f"unsupported external feature plan projection: {projection_contract_id}"
        )
    module_name = contract.get("module")
    source_model_name = contract.get("source_model")
    projector_name = contract.get("projector")
    if not all(
        isinstance(value, str)
        for value in (module_name, source_model_name, projector_name)
    ):
        raise ValueError("external projection registry entry is incomplete")
    required_roles = contract.get("required_artifact_roles")
    if not isinstance(required_roles, list) or not all(
        isinstance(role, str) for role in required_roles
    ):
        raise ValueError("external projection registry artifact contract is invalid")
    missing_roles = sorted(set(required_roles) - set(source_artifacts))
    if missing_roles:
        raise ValueError(
            "external projection is missing required source artifacts: "
            + ", ".join(missing_roles)
        )

    module = importlib.import_module(module_name)
    source_model = getattr(module, source_model_name, None)
    projector = getattr(module, projector_name, None)
    if source_model is None or not callable(projector):
        raise ValueError("external projection registry implementation is unavailable")
    source_plan = source_model.model_validate(read_json(source_plan_path))
    request = read_json(source_request_path)
    response_format = request.get("response_format") if isinstance(request, dict) else None
    request_schema = (
        response_format.get("schema") if isinstance(response_format, dict) else None
    )
    expected_source_schema = gemini_response_schema(source_model)
    if request_schema != expected_source_schema:
        raise ValueError(
            "external projection source request schema does not match its registered model"
        )

    raw_output_role = contract.get("raw_output_role")
    if raw_output_role is not None:
        if not isinstance(raw_output_role, str) or raw_output_role not in source_artifacts:
            raise ValueError("external projection raw output artifact is missing")
        raw_output = read_json(source_artifacts[raw_output_role])
        output_text = raw_output.get("output_text") if isinstance(raw_output, dict) else None
        if not isinstance(output_text, str):
            raise ValueError("external projection raw output has no output_text")
        raw_plan = source_model.model_validate_json(output_text)
        if contract.get("source_plan_has_model_provenance", True):
            source_interaction_id = source_plan.model_provenance.interaction_id
            normalized_raw_plan = raw_plan.model_copy(
                update={
                    "model_provenance": raw_plan.model_provenance.model_copy(
                        update={"interaction_id": source_interaction_id}
                    )
                }
            )
        else:
            normalized_raw_plan = raw_plan
        if normalized_raw_plan.model_dump(mode="json") != source_plan.model_dump(
            mode="json"
        ):
            raise ValueError(
                "external projection source plan differs from validated raw model output"
            )

    catalog = RushesCatalog.model_validate(read_json(catalog_path))
    brief = FeatureEditBrief.model_validate(read_json(brief_path))
    expected_brief, expected_plan = projector(
        source_plan=source_plan,
        catalog=catalog,
        brief=brief,
        source_artifacts=dict(source_artifacts),
    )
    if not isinstance(expected_brief, FeatureEditBrief) or not isinstance(
        expected_plan, FeatureEditPlan
    ):
        raise ValueError("external projection projector returned an invalid contract")
    actual_plan = FeatureEditPlan.model_validate(read_json(feature_plan_path))
    if expected_brief.model_dump(mode="json") != brief.model_dump(mode="json"):
        raise ValueError(
            "external projection brief differs from deterministic projector output"
        )
    if expected_plan.model_dump(mode="json") != actual_plan.model_dump(mode="json"):
        raise ValueError(
            "external FeatureEditPlan differs from deterministic projector output"
        )


def _external_request_claims(request_path: Path) -> dict[str, str]:
    request = read_json(request_path)
    if not isinstance(request, dict):
        raise ValueError("external projection source request must be an object")
    model_id = request.get("model")
    system_instruction = request.get("system_instruction")
    inputs = request.get("input")
    response_format = request.get("response_format")
    response_schema = (
        response_format.get("schema") if isinstance(response_format, dict) else None
    )
    if model_id != MODEL_ID:
        raise ValueError("external projection source request used an unexpected model")
    if not isinstance(system_instruction, str) or not system_instruction:
        raise ValueError("external projection source request has no system instruction")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("external projection source request has no model input")
    if not isinstance(response_schema, dict):
        raise ValueError("external projection source request has no response schema")
    return {
        "source_request_sha256": sha256_file(request_path),
        "source_request_input_sha256": _sha256_json(inputs),
        "source_system_instruction_sha256": _sha256_text(system_instruction),
        "source_model_id": model_id,
        "source_model_id_sha256": _sha256_text(model_id),
        "source_response_schema_sha256": _sha256_json(response_schema),
    }


def _hashed_artifact(role: str, path: Path) -> dict[str, str]:
    if not role or not role.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"invalid external projection artifact role: {role!r}")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"role": role, "path": str(resolved), "sha256": sha256_file(resolved)}


def write_external_feature_plan_projection(
    *,
    plan_dir: Path,
    projection_contract_id: str,
    catalog_path: Path,
    brief_path: Path,
    feature_plan_path: Path,
    source_plan_path: Path,
    source_request_path: Path,
    source_artifacts: Mapping[str, Path] | None = None,
) -> Path:
    """Write immutable provenance for a deterministic external plan projection."""

    plan_dir = plan_dir.expanduser().resolve()
    catalog_path = catalog_path.expanduser().resolve()
    brief_path = brief_path.expanduser().resolve()
    feature_plan_path = feature_plan_path.expanduser().resolve()
    source_plan_path = source_plan_path.expanduser().resolve()
    source_request_path = source_request_path.expanduser().resolve()
    catalog = RushesCatalog.model_validate(read_json(catalog_path))
    brief = FeatureEditBrief.model_validate(read_json(brief_path))
    feature_plan = FeatureEditPlan.model_validate(read_json(feature_plan_path))
    if (
        feature_plan.catalog_id != catalog.catalog_id
        or feature_plan.project_id != brief.project_id
    ):
        raise ValueError("external projected feature plan does not match catalog/brief")
    source_plan = read_json(source_plan_path)
    if not isinstance(source_plan, dict):
        raise ValueError("external projection source plan must be an object")
    request_claims = _external_request_claims(source_request_path)
    contract = _EXTERNAL_PROJECTION_CONTRACTS[projection_contract_id]
    if contract.get("source_plan_has_model_provenance", True):
        source_provenance = source_plan.get("model_provenance")
        if (
            not isinstance(source_provenance, dict)
            or source_provenance.get("model_id")
            != request_claims["source_model_id"]
        ):
            raise ValueError(
                "external projection source plan provenance does not match its request"
            )
    artifacts = [
        _hashed_artifact(role, path)
        for role, path in sorted((source_artifacts or {}).items())
    ]
    artifact_paths = {
        item["role"]: Path(item["path"])
        for item in artifacts
    }
    _validate_external_projection_semantics(
        projection_contract_id=projection_contract_id,
        catalog_path=catalog_path,
        brief_path=brief_path,
        feature_plan_path=feature_plan_path,
        source_plan_path=source_plan_path,
        source_request_path=source_request_path,
        source_artifacts=artifact_paths,
    )
    core: dict[str, Any] = {
        "sidecar_version": _EXTERNAL_PROJECTION_SIDECAR_VERSION,
        "origin": "external_projection",
        "projection_contract_id": projection_contract_id,
        "projection_contract_sha256": _external_projection_contract_sha256(
            projection_contract_id
        ),
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "brief_path": str(brief_path),
        "brief_sha256": sha256_file(brief_path),
        "feature_plan_path": str(feature_plan_path),
        "feature_plan_sha256": sha256_file(feature_plan_path),
        "source_plan_path": str(source_plan_path),
        "source_plan_sha256": sha256_file(source_plan_path),
        "source_request_path": str(source_request_path),
        **request_claims,
        "source_artifacts": artifacts,
        "source_artifact_set_sha256": _sha256_json(
            [
                {"role": item["role"], "sha256": item["sha256"]}
                for item in artifacts
            ]
        ),
    }
    fingerprint = _sha256_json(core)
    record_dir = plan_dir / "feature-plan-projections"
    record_path = record_dir / f"projection-{fingerprint}.json"
    if record_path.exists():
        existing = read_json(record_path)
        if not isinstance(existing, dict) or any(
            existing.get(key) != value for key, value in core.items()
        ):
            raise ValueError("existing external projection record is inconsistent")
    else:
        write_json(record_path, {**core, "created_at": utc_now()})
    pointer_path = plan_dir / _EXTERNAL_PROJECTION_POINTER_NAME
    write_json(
        pointer_path,
        {
            "sidecar_version": _EXTERNAL_PROJECTION_SIDECAR_VERSION,
            "record_path": str(record_path.relative_to(plan_dir)),
            "record_sha256": sha256_file(record_path),
        },
    )
    return pointer_path


def load_external_feature_plan_projection(
    plan_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Load a contained immutable projection record through its small pointer."""

    plan_dir = plan_dir.expanduser().resolve()
    pointer_path = plan_dir / _EXTERNAL_PROJECTION_POINTER_NAME
    pointer = read_json(pointer_path)
    if not isinstance(pointer, dict):
        raise ValueError("external projection pointer must be an object")
    if pointer.get("sidecar_version") != _EXTERNAL_PROJECTION_SIDECAR_VERSION:
        raise ValueError("external projection pointer version is unsupported")
    relative_record = pointer.get("record_path")
    if not isinstance(relative_record, str) or not relative_record:
        raise ValueError("external projection pointer has no record path")
    record_root = (plan_dir / "feature-plan-projections").resolve()
    record_path = (plan_dir / relative_record).resolve()
    try:
        record_path.relative_to(record_root)
    except ValueError as error:
        raise ValueError("external projection record escapes its artifact root") from error
    if not record_path.is_file() or sha256_file(record_path) != pointer.get(
        "record_sha256"
    ):
        raise ValueError("external projection record hash does not match pointer")
    record = read_json(record_path)
    if not isinstance(record, dict):
        raise ValueError("external projection record must be an object")
    return pointer_path, record_path, record


def _current_external_projection_binding(
    *,
    plan_dir: Path,
    catalog_path: Path,
    catalog_reel_sha256: str,
    brief_path: Path,
    plan_path: Path,
    music_sha256: str | None,
    created_at: str,
) -> dict[str, Any]:
    """Verify every external projection artifact and derive a reusable binding."""

    if not re.fullmatch(r"[0-9a-f]{64}", catalog_reel_sha256):
        raise ValueError("catalog reel SHA-256 is required for feature plan binding")
    if music_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", music_sha256
    ):
        raise ValueError("music SHA-256 must be a lowercase hexadecimal digest")
    pointer_path, record_path, record = load_external_feature_plan_projection(
        plan_dir
    )
    if (
        record.get("sidecar_version") != _EXTERNAL_PROJECTION_SIDECAR_VERSION
        or record.get("origin") != "external_projection"
    ):
        raise ValueError("external projection record contract is unsupported")
    contract_id = record.get("projection_contract_id")
    if (
        not isinstance(contract_id, str)
        or record.get("projection_contract_sha256")
        != _external_projection_contract_sha256(contract_id)
    ):
        raise ValueError("external projection contract hash is invalid")
    current_files = {
        "catalog_sha256": sha256_file(catalog_path),
        "brief_sha256": sha256_file(brief_path),
        "feature_plan_sha256": sha256_file(plan_path),
    }
    for key, value in current_files.items():
        if record.get(key) != value:
            raise ValueError(f"external projection {key} differs from current input")
    for prefix in ("catalog", "brief", "feature_plan", "source_plan"):
        source_path = record.get(f"{prefix}_path")
        expected_hash = record.get(f"{prefix}_sha256")
        if not isinstance(source_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"external projection has incomplete {prefix} provenance")
        resolved = Path(source_path).expanduser().resolve()
        if not resolved.is_file() or sha256_file(resolved) != expected_hash:
            raise ValueError(f"external projection {prefix} source hash is invalid")
    request_path_value = record.get("source_request_path")
    if not isinstance(request_path_value, str):
        raise ValueError("external projection has no source request path")
    request_path = Path(request_path_value).expanduser().resolve()
    request_claims = _external_request_claims(request_path)
    for key, value in request_claims.items():
        if record.get(key) != value:
            raise ValueError(f"external projection request claim changed: {key}")
    source_plan = read_json(Path(str(record["source_plan_path"])))
    contract = _EXTERNAL_PROJECTION_CONTRACTS[contract_id]
    if contract.get("source_plan_has_model_provenance", True):
        source_provenance = (
            source_plan.get("model_provenance")
            if isinstance(source_plan, dict)
            else None
        )
        if (
            not isinstance(source_provenance, dict)
            or source_provenance.get("model_id")
            != request_claims["source_model_id"]
        ):
            raise ValueError(
                "external projection source plan provenance no longer "
                "matches its request"
            )
    artifact_claims: list[dict[str, str]] = []
    artifact_roles: set[str] = set()
    artifact_paths: dict[str, Path] = {}
    source_artifacts = record.get("source_artifacts")
    if not isinstance(source_artifacts, list):
        raise ValueError("external projection source artifacts must be a list")
    for artifact in source_artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("external projection source artifact must be an object")
        role = artifact.get("role")
        path_value = artifact.get("path")
        expected_hash = artifact.get("sha256")
        if not all(isinstance(value, str) for value in (role, path_value, expected_hash)):
            raise ValueError("external projection source artifact is incomplete")
        if role in artifact_roles:
            raise ValueError(f"external projection source artifact role is duplicated: {role}")
        artifact_roles.add(role)
        artifact_path = Path(path_value).expanduser().resolve()
        if not artifact_path.is_file() or sha256_file(artifact_path) != expected_hash:
            raise ValueError(f"external projection source artifact changed: {role}")
        artifact_claims.append({"role": role, "sha256": expected_hash})
        artifact_paths[role] = artifact_path
    artifact_set_hash = _sha256_json(artifact_claims)
    if record.get("source_artifact_set_sha256") != artifact_set_hash:
        raise ValueError("external projection source artifact set hash is invalid")
    _validate_external_projection_semantics(
        projection_contract_id=contract_id,
        catalog_path=catalog_path,
        brief_path=brief_path,
        feature_plan_path=plan_path,
        source_plan_path=Path(str(record["source_plan_path"])),
        source_request_path=request_path,
        source_artifacts=artifact_paths,
    )
    source_music_path = artifact_paths.get("source_music")
    source_music_sha256 = (
        sha256_file(source_music_path)
        if source_music_path is not None
        else None
    )
    source_request = read_json(request_path)
    source_inputs = (
        source_request.get("input")
        if isinstance(source_request, dict)
        else None
    )
    source_inputs = source_inputs if isinstance(source_inputs, list) else []
    request_has_audio = any(
        isinstance(item, dict) and item.get("type") == "audio"
        for item in source_inputs
    )
    if request_has_audio != (source_music_sha256 is not None):
        raise ValueError(
            "external projection source_music does not match the paid request "
            "audio presence"
        )
    source_prompt = "\n".join(
        str(item.get("text"))
        for item in source_inputs
        if isinstance(item, dict) and item.get("type") == "text"
    )
    if source_music_sha256 is not None and not _prompt_binds_sha256(
        source_prompt,
        "music_sha256",
        source_music_sha256,
    ):
        if contract_id != "direct-video-edit-plan-v1":
            raise ValueError(
                "external projection paid request does not bind the source music hash"
            )
        upload_path = artifact_paths.get("source_music_upload")
        if upload_path is None:
            raise ValueError(
                "direct-video projection requires its File API music binding"
            )
        upload = read_json(upload_path)
        upload_uri = str(upload.get("uri") or "")
        request_audio_uris = {
            str(item.get("uri") or "")
            for item in source_inputs
            if isinstance(item, dict) and item.get("type") == "audio"
        }
        encoded_hash = str(upload.get("sha256_hash") or "")
        try:
            decoded_hash = base64.b64decode(encoded_hash).decode("ascii")
        except Exception as error:
            raise ValueError("music upload has an invalid source hash") from error
        if (
            not upload_uri
            or upload_uri not in request_audio_uris
            or decoded_hash != source_music_sha256
        ):
            raise ValueError(
                "direct-video paid request music does not match its "
                "hash-bound File API object"
            )
    if music_sha256 != source_music_sha256:
        raise ValueError(
            "external projection music differs from the current render input; "
            "the Top-K selection must hear the same music, or both must omit it"
        )
    return {
        "binding_version": _FEATURE_PLAN_BINDING_VERSION,
        "origin": "external_projection",
        "external_projection_contract_id": contract_id,
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": current_files["catalog_sha256"],
        "catalog_reel_sha256": catalog_reel_sha256,
        "brief_path": str(brief_path.resolve()),
        "brief_sha256": current_files["brief_sha256"],
        "music_sha256": music_sha256,
        # For external projections, the actual source-model input is the prompt
        # contract; the renderer's unused direct-video plan prompt is irrelevant.
        "plan_prompt_sha256": request_claims["source_request_input_sha256"],
        "system_instruction_sha256": request_claims[
            "source_system_instruction_sha256"
        ],
        "model_id": request_claims["source_model_id"],
        "model_id_sha256": request_claims["source_model_id_sha256"],
        "response_schema_sha256": request_claims[
            "source_response_schema_sha256"
        ],
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": current_files["feature_plan_sha256"],
        "request_path": str(request_path),
        "request_sha256": request_claims["source_request_sha256"],
        "source_plan_sha256": record["source_plan_sha256"],
        "projection_contract_sha256": record["projection_contract_sha256"],
        "projection_pointer_sha256": sha256_file(pointer_path),
        "projection_record_sha256": sha256_file(record_path),
        "source_artifact_set_sha256": artifact_set_hash,
        "created_at": created_at,
    }


def validate_external_feature_plan_projection(plan_dir: Path) -> dict[str, Any]:
    """Validate an upstream projection in place and return its immutable record."""

    _, _, record = load_external_feature_plan_projection(plan_dir)
    required_paths = {
        key: record.get(key)
        for key in (
            "catalog_path",
            "brief_path",
            "feature_plan_path",
        )
    }
    if not all(isinstance(value, str) for value in required_paths.values()):
        raise ValueError("external projection record has incomplete primary paths")
    catalog_path = Path(required_paths["catalog_path"])  # type: ignore[arg-type]
    catalog = RushesCatalog.model_validate(read_json(catalog_path))
    reel_path = Path(catalog.analysis_reel_path).expanduser().resolve()
    if not reel_path.is_file():
        raise FileNotFoundError(reel_path)
    source_artifacts = record.get("source_artifacts")
    source_music_sha256 = next(
        (
            str(artifact["sha256"])
            for artifact in source_artifacts
            if isinstance(artifact, dict)
            and artifact.get("role") == "source_music"
        ),
        None,
    ) if isinstance(source_artifacts, list) else None
    _current_external_projection_binding(
        plan_dir=plan_dir,
        catalog_path=catalog_path,
        catalog_reel_sha256=sha256_file(reel_path),
        brief_path=Path(required_paths["brief_path"]),  # type: ignore[arg-type]
        plan_path=Path(required_paths["feature_plan_path"]),  # type: ignore[arg-type]
        music_sha256=source_music_sha256,
        created_at=utc_now(),
    )
    return record


def _current_feature_plan_binding(
    *,
    catalog_path: Path,
    catalog_reel_sha256: str,
    brief_path: Path,
    plan_path: Path,
    plan_prompt: str,
    music_sha256: str | None,
    request_path: Path | None,
    created_at: str,
    origin: Literal["generated", "migrated_legacy_reuse", "external_projection"],
) -> dict[str, Any]:
    """Build the immutable causal inputs for one saved editorial plan."""

    if not re.fullmatch(r"[0-9a-f]{64}", catalog_reel_sha256):
        raise ValueError("catalog reel SHA-256 is required for feature plan binding")
    if music_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", music_sha256
    ):
        raise ValueError("music SHA-256 must be a lowercase hexadecimal digest")
    binding: dict[str, Any] = {
        "binding_version": _FEATURE_PLAN_BINDING_VERSION,
        "origin": origin,
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": sha256_file(catalog_path),
        "catalog_reel_sha256": catalog_reel_sha256,
        "brief_path": str(brief_path.resolve()),
        "brief_sha256": sha256_file(brief_path),
        # Presence and absence are both causal.  ``None`` means that the paid
        # planner did not hear music, and therefore must not be reused for a
        # later run that supplies one (or vice versa).
        "music_sha256": music_sha256,
        "plan_prompt_sha256": _sha256_text(plan_prompt),
        "system_instruction_sha256": _sha256_text(
            EDITORIAL_SYSTEM_INSTRUCTION
        ),
        "model_id": MODEL_ID,
        "model_id_sha256": _sha256_text(MODEL_ID),
        "response_schema_sha256": _sha256_json(
            gemini_response_schema(FeatureEditPlan)
        ),
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "created_at": created_at,
    }
    if request_path is not None:
        binding.update(
            {
                "request_path": str(request_path.resolve()),
                "request_sha256": sha256_file(request_path),
            }
        )
    return binding


def _validate_feature_plan_binding(
    saved: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """Fail closed when any causal plan input differs from saved evidence."""

    required_hashes = (
        "catalog_sha256",
        "brief_sha256",
        "plan_prompt_sha256",
        "system_instruction_sha256",
        "model_id_sha256",
        "response_schema_sha256",
        "plan_sha256",
        "request_sha256",
    )
    required_exact_keys: tuple[str, ...] = ()
    if saved.get("origin") != REFRAME_POLICY_BINDING_ORIGIN:
        required_hashes += ("catalog_reel_sha256",)
        required_exact_keys += ("music_sha256",)
    if saved.get("origin") == "external_projection":
        required_hashes += (
            "source_plan_sha256",
            "projection_contract_sha256",
            "projection_pointer_sha256",
            "projection_record_sha256",
            "source_artifact_set_sha256",
        )
    missing = [key for key in required_hashes if not saved.get(key)]
    missing.extend(key for key in required_exact_keys if key not in saved)
    if saved.get("binding_version") != _FEATURE_PLAN_BINDING_VERSION:
        missing.insert(0, "binding_version")
    if missing:
        raise ValueError(
            "saved feature plan binding is incomplete or unsupported: "
            + ", ".join(missing)
        )
    mismatches = [
        key for key in required_hashes if saved[key] != current.get(key)
    ]
    mismatches.extend(
        key
        for key in required_exact_keys
        if saved.get(key) != current.get(key)
    )
    if saved.get("origin") != current.get("origin"):
        mismatches.append("origin")
    if saved.get("model_id") != current.get("model_id"):
        mismatches.append("model_id")
    if mismatches:
        raise ValueError(
            "saved feature plan causal binding differs from current inputs: "
            + ", ".join(sorted(set(mismatches)))
        )


def _migrate_legacy_feature_plan_binding(
    *,
    plan_dir: Path,
    catalog_path: Path,
    catalog_reel_sha256: str,
    brief_path: Path,
    plan_path: Path,
    plan_prompt: str,
    music_sha256: str | None,
) -> dict[str, Any]:
    """Validate old reuse evidence plus the original API request before migration.

    The legacy record used the wrong system-instruction hash.  It is accepted
    only when that value is one of the two known historical constants and the
    untouched API request independently proves the actual editorial system
    instruction, model, schema and prompt template.  The legacy file is never
    overwritten.
    """

    legacy_path = plan_dir / "feature-plan.reuse.json"
    request_path = plan_dir / "feature_edit_plan.request.json"
    if not legacy_path.exists() or not request_path.exists():
        raise ValueError(
            "saved feature plan has no immutable binding; legacy migration "
            "requires both feature-plan.reuse.json and the original request"
        )
    legacy = read_json(legacy_path)
    if not isinstance(legacy, dict):
        raise ValueError("legacy feature plan reuse record must be an object")
    expected_legacy = {
        "plan_sha256": sha256_file(plan_path),
        "current_catalog_sha256": sha256_file(catalog_path),
        "current_brief_sha256": sha256_file(brief_path),
        "current_plan_prompt_sha256": _sha256_text(plan_prompt),
        "model_id": MODEL_ID,
    }
    missing = [key for key in expected_legacy if key not in legacy]
    mismatches = [
        key
        for key, expected in expected_legacy.items()
        if key in legacy and legacy[key] != expected
    ]
    known_system_hashes = {
        _sha256_text(EDITORIAL_SYSTEM_INSTRUCTION),
        _sha256_text(VISUAL_EVIDENCE_SYSTEM_INSTRUCTION),
    }
    if legacy.get("system_instruction_sha256") not in known_system_hashes:
        mismatches.append("system_instruction_sha256")
    if missing or mismatches:
        details = sorted(set(missing + mismatches))
        raise ValueError(
            "legacy feature plan reuse evidence does not match current inputs: "
            + ", ".join(details)
        )

    request = read_json(request_path)
    if not isinstance(request, dict):
        raise ValueError("original feature plan request must be an object")
    response_format = request.get("response_format")
    inputs = request.get("input")
    text_inputs = (
        [item.get("text") for item in inputs if item.get("type") == "text"]
        if isinstance(inputs, list)
        and all(isinstance(item, dict) for item in inputs)
        else []
    )
    expected_prompt_prefix = plan_prompt + "\n\n## 本次不可變 metadata\n"
    request_schema = (
        response_format.get("schema") if isinstance(response_format, dict) else None
    )
    request_is_valid = (
        request.get("model") == MODEL_ID
        and request.get("system_instruction") == EDITORIAL_SYSTEM_INSTRUCTION
        and request_schema == gemini_response_schema(FeatureEditPlan)
        and len(text_inputs) == 1
        and isinstance(text_inputs[0], str)
        and text_inputs[0].startswith(expected_prompt_prefix)
    )
    if not request_is_valid:
        raise ValueError(
            "original feature plan request does not prove the current "
            "model/system/schema/prompt contract"
        )

    binding = _current_feature_plan_binding(
        catalog_path=catalog_path,
        catalog_reel_sha256=catalog_reel_sha256,
        brief_path=brief_path,
        plan_path=plan_path,
        plan_prompt=plan_prompt,
        music_sha256=music_sha256,
        request_path=request_path,
        created_at=utc_now(),
        origin="migrated_legacy_reuse",
    )
    binding["migration_source_path"] = str(legacy_path.resolve())
    binding["migration_source_sha256"] = sha256_file(legacy_path)
    return binding


def _write_incremental_pricing(
    *,
    output_dir: Path,
    prior_interaction_hashes: dict[str, str],
    prior_error_hashes: dict[str, str],
) -> dict[str, Any]:
    """Persist a best-effort cost delta without hiding the original failure."""

    try:
        incremental_interaction_paths = [
            path
            for path in output_dir.rglob("*raw_interaction.json")
            if prior_interaction_hashes.get(str(path.relative_to(output_dir)))
            != sha256_file(path)
        ]
        result = summarize_usage_files(
            incremental_interaction_paths,
            relative_to=output_dir,
        )
        changed_error_paths = [
            path
            for path in output_dir.rglob("errors.json")
            if prior_error_hashes.get(str(path.relative_to(output_dir)))
            != sha256_file(path)
        ]
        result.update(
            {
                "scope": "new_or_changed_raw_interactions_in_this_run",
                "historical_cache_excluded": True,
                "changed_error_artifact_count": len(changed_error_paths),
                "changed_error_artifact_paths": [
                    str(path.relative_to(output_dir))
                    for path in changed_error_paths
                ],
                "changed_error_artifacts_have_no_usage_metadata": True,
                "calculation_status": "ok",
            }
        )
    except Exception as error:  # preserve an earlier render/API exception
        result = {
            "scope": "new_or_changed_raw_interactions_in_this_run",
            "historical_cache_excluded": True,
            "calculation_status": "error",
            "calculation_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    try:
        write_json(output_dir / "pricing.incremental.json", result)
    except Exception as error:  # do not replace the render/API exception
        result["persistence_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return result


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: Any, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for character in text:
        candidate = current + character
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = character
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _render_text_layer(
    chapter: FeatureChapterBrief,
    output_path: Path,
    *,
    dimensions: tuple[int, int],
    missing_evidence: bool = False,
    opaque: bool = False,
) -> None:
    width, height = dimensions
    image = Image.new("RGBA", dimensions, (11, 14, 18, 255 if opaque else 0))
    draw = ImageDraw.Draw(image)
    title_font = _font(54 if width > height else 48)
    detail_font = _font(34 if width > height else 31)
    label_font = _font(23 if width > height else 24)
    panel_height = round(height * (0.35 if width < height else 0.30))
    top = height - panel_height
    draw.rectangle((0, top, width, height), fill=(8, 12, 16, 218 if not opaque else 255))
    draw.rectangle((0, top, 14 if width > height else 10, height), fill=(29, 196, 96, 255))
    margin = 64 if width > height else 48
    y = top + 36
    for line in _wrap_text(draw, chapter.title, title_font, width - margin * 2):
        draw.text((margin, y), line, font=title_font, fill="white")
        y += title_font.size + 9
    y += 5
    for detail in chapter.detail_lines:
        for line in _wrap_text(draw, detail, detail_font, width - margin * 2):
            draw.text((margin, y), line, font=detail_font, fill=(220, 231, 225, 255))
            y += detail_font.size + 6
    if missing_evidence:
        label = "CATALOG 中未找到直接功能示範畫面"
        box = draw.textbbox((0, 0), label, font=label_font)
        label_width = box[2] - box[0] + 28
        draw.rounded_rectangle(
            (margin, max(22, top - 58), margin + label_width, max(22, top - 58) + 42),
            radius=10,
            fill=(211, 70, 70, 235),
        )
        draw.text((margin + 14, max(22, top - 51)), label, font=label_font, fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _chapter_bounds(
    frame: RushFrame,
    clip: RushClip,
    duration_seconds: float,
    shot_cache: dict[str, ShotManifest],
    shots_dir: Path,
    scdet_threshold: float,
) -> tuple[int, int, str]:
    if clip.clip_id not in shot_cache:
        shot_cache[clip.clip_id] = detect_shots_ffmpeg(
            Path(clip.path),
            threshold=scdet_threshold,
            output_path=shots_dir / f"{clip.clip_id}.json",
        )
    shot = next(
        item
        for item in shot_cache[clip.clip_id].shots
        if item.start_time_ms <= frame.requested_time_ms < item.end_time_ms
    )
    start_ms, end_ms = _segment_bounds(
        center_ms=frame.requested_time_ms,
        requested_duration_ms=round(duration_seconds * 1000),
        clip_duration_ms=clip.duration_ms,
        shot=shot,
    )
    return start_ms, end_ms, shot.shot_id


def _load_trim_decisions(
    paths: Sequence[Path],
    *,
    allow_proposed_preview: bool = False,
) -> list[tuple[Path, TrimIntentDecision]]:
    accepted: list[tuple[Path, TrimIntentDecision]] = []
    for path in paths:
        decision = TrimIntentDecision.model_validate(read_json(path))
        is_approved = (
            decision.usable
            and decision.approval_status == "approved"
            and not decision.requires_human_review
            and decision.human_review is not None
            and decision.human_review.decision == "approved"
        )
        is_proposed_preview = (
            allow_proposed_preview
            and decision.usable
            and decision.approval_status == "proposed"
            and decision.requires_human_review
            and decision.human_review is None
        )
        if not is_approved and not is_proposed_preview:
            qualifier = (
                "human-approved or, with --allow-proposed-trim-preview, "
                "an unreviewed proposed"
            )
            raise ValueError(f"feature cut only accepts {qualifier} trim decision: {path}")
        if (
            not decision.usable
            or decision.approval_status == "rejected"
        ):
            raise ValueError(f"feature cut refuses unusable or rejected trim decision: {path}")
        accepted.append((path.resolve(), decision))
    return accepted


def _load_shot_quality_maps(
    paths: Sequence[Path],
) -> list[tuple[Path, ShotQualityMap]]:
    loaded: list[tuple[Path, ShotQualityMap]] = []
    identities: set[tuple[str, str]] = set()
    for path in paths:
        resolved, quality_map = load_shot_quality_map(path)
        identity = (quality_map.source_asset_id, quality_map.shot_id)
        if identity in identities:
            raise ValueError(
                "multiple ShotQualityMap artifacts describe the same source shot"
            )
        identities.add(identity)
        loaded.append((resolved, quality_map))
    return loaded


def _quality_map_for_shot(
    quality_maps: Sequence[tuple[Path, ShotQualityMap]],
    *,
    source_asset_id: str,
    shot_id: str,
) -> tuple[Path, ShotQualityMap] | None:
    matches = [
        (path, quality_map)
        for path, quality_map in quality_maps
        if quality_map.source_asset_id == source_asset_id
        and quality_map.shot_id == shot_id
    ]
    if len(matches) > 1:
        raise ValueError("quality-map source/shot lineage is ambiguous")
    return matches[0] if matches else None


def _matching_trim_decision(
    decisions: Sequence[tuple[Path, TrimIntentDecision]],
    *,
    source_asset_id: str,
    shot_id: str,
    event_id: str | None,
) -> tuple[Path, TrimIntentDecision] | None:
    matches = [
        (path, decision)
        for path, decision in decisions
        if decision.source_asset_id == source_asset_id
        and decision.shot_id == shot_id
        and (event_id is None or decision.event_id == event_id)
        and decision.source_in_ms is not None
        and decision.source_out_ms is not None
    ]
    if len(matches) > 1:
        raise ValueError(
            "multiple trim decisions match one selected source shot/event"
        )
    return matches[0] if matches else None


def _selected_event_id(
    selected: FeatureChapterSelect,
    *,
    frame_id: str,
    aspect: Literal["16:9", "9:16"],
) -> str | None:
    candidates = (
        selected.horizontal_candidates
        if aspect == "16:9"
        else selected.vertical_candidates
    )
    matches = [
        candidate.event_id
        for candidate in candidates
        if candidate.frame_id == frame_id
    ]
    if len(set(matches)) > 1:
        raise ValueError("rank-one candidate event mapping is ambiguous")
    return matches[0] if matches else None


def _chapter_bounds_with_approved_trim(
    frame: RushFrame,
    clip: RushClip,
    duration_seconds: float,
    shot_cache: dict[str, ShotManifest],
    shots_dir: Path,
    scdet_threshold: float,
    approved_decisions: Sequence[tuple[Path, TrimIntentDecision]],
    expected_event_id: str | None = None,
    quality_maps: Sequence[tuple[Path, ShotQualityMap]] = (),
) -> tuple[int, int, str, dict[str, Any]]:
    fallback_start, fallback_end, shot_id = _chapter_bounds(
        frame,
        clip,
        duration_seconds,
        shot_cache,
        shots_dir,
        scdet_threshold,
    )
    asset_id = f"sha256:{clip.sha256}"
    match = _matching_trim_decision(
        approved_decisions,
        source_asset_id=asset_id,
        shot_id=shot_id,
        event_id=expected_event_id,
    )
    quality_match = _quality_map_for_shot(
        quality_maps,
        source_asset_id=asset_id,
        shot_id=shot_id,
    )
    if quality_maps and quality_match is None:
        raise ValueError(
            "ShotQualityMap coverage is incomplete for the source shot selected "
            f"at runtime: {asset_id}/{shot_id}"
        )
    quality_path: Path | None = None
    quality_map: ShotQualityMap | None = None
    safe_intervals = []
    if quality_match is not None:
        quality_path, quality_map = quality_match
        if sha256_file(Path(quality_map.source_path)) != clip.sha256:
            raise ValueError("ShotQualityMap source hash differs from the selected clip")
        safe_intervals = build_quality_safe_intervals(
            quality_map,
            quality_map_sha256=sha256_file(quality_path),
        )
    if match is None:
        if quality_map is None:
            return fallback_start, fallback_end, shot_id, {
                "trim_method": "keyframe_centered_requested_duration",
                "trim_decision_path": None,
                "trim_event_id": None,
                "trim_tail_intent": None,
                "trim_human_review": None,
                "source_in_pts": None,
                "source_out_pts": None,
                "quality_map_path": None,
                "quality_safe_interval_id": None,
            }
        requested_ms = max(1, round(duration_seconds * 1000))
        containing = [
            interval
            for interval in safe_intervals
            if interval.start_ms
            <= frame.requested_time_ms
            < interval.end_ms
            and interval.end_ms - interval.start_ms >= requested_ms
        ]
        if not containing:
            raise ValueError(
                "selected evidence frame has no continuous QualitySafeInterval "
                "long enough for the resolved chapter duration"
            )
        interval = min(
            containing,
            key=lambda item: (
                item.requires_human_review,
                item.end_ms - item.start_ms,
                item.start_ms,
            ),
        )
        safe_start = max(
            interval.start_ms,
            min(
                frame.requested_time_ms - requested_ms // 2,
                interval.end_ms - requested_ms,
            ),
        )
        safe_end = safe_start + requested_ms
        return safe_start, safe_end, shot_id, {
            "trim_method": "keyframe_centered_requested_duration",
            "trim_decision_path": None,
            "trim_event_id": None,
            "trim_tail_intent": None,
            "trim_human_review": None,
            "source_in_pts": None,
            "source_out_pts": None,
            "quality_map_path": str(quality_path),
            "quality_map_sha256": sha256_file(quality_path),
            "quality_safe_interval_id": interval.interval_id,
            "quality_requires_human_review": interval.requires_human_review,
        }
    path, decision = match
    assert decision.source_in_ms is not None and decision.source_out_ms is not None
    shot = next(item for item in shot_cache[clip.clip_id].shots if item.shot_id == shot_id)
    if decision.shot_id != shot_id:
        raise ValueError(
            f"approved trim decision shot differs from current FFmpeg shot for {frame.frame_id}"
        )
    if not (
        shot.start_time_ms
        <= decision.source_in_ms
        < decision.source_out_ms
        <= shot.end_time_ms
    ):
        raise ValueError("approved trim decision crosses the selected shot boundary")
    quality_interval = None
    if quality_map is not None:
        quality_interval = next(
            (
                interval
                for interval in safe_intervals
                if interval.start_ms <= decision.source_in_ms
                and decision.source_out_ms <= interval.end_ms
            ),
            None,
        )
        if quality_interval is None:
            raise ValueError(
                "approved trim crosses an unresolved hard/trim quality risk; "
                "review the risk intent or revise the trim before rendering"
            )
    approved = decision.approval_status == "approved"
    return decision.source_in_ms, decision.source_out_ms, shot_id, {
        "trim_method": (
            "human_approved_frame_id_pts"
            if approved
            else "unreviewed_proposed_frame_id_pts"
        ),
        "trim_decision_path": str(path),
        "trim_event_id": decision.event_id,
        "trim_tail_intent": decision.tail_intent,
        "source_in_pts": decision.source_in_pts,
        "source_out_pts": decision.source_out_pts,
        "trim_requires_human_review": decision.requires_human_review,
        "trim_human_review": (
            decision.human_review.model_dump(mode="json")
            if decision.human_review is not None
            else None
        ),
        "quality_map_path": str(quality_path) if quality_path is not None else None,
        "quality_map_sha256": (
            sha256_file(quality_path) if quality_path is not None else None
        ),
        "quality_safe_interval_id": (
            quality_interval.interval_id if quality_interval is not None else None
        ),
        "quality_requires_human_review": (
            quality_interval.requires_human_review
            if quality_interval is not None
            else False
        ),
    }


def _run_ffmpeg(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _run_segment_encoder(command: list[str]) -> None:
    try:
        _run_ffmpeg(command)
    except subprocess.CalledProcessError:
        if "h264_videotoolbox" not in command:
            raise
        fallback = ["libx264" if value == "h264_videotoolbox" else value for value in command]
        _run_ffmpeg(fallback)


def _exact_render_source_interval(
    *,
    source_path: Path,
    source_sha256: str,
    start_ms: int,
    end_ms: int,
    trim: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Bind requested editorial bounds to exact decoded source PTS and frames."""

    source = source_path.expanduser().resolve(strict=True)
    if sha256_file(source) != source_sha256:
        raise ValueError("render source bytes differ from the catalog")
    media = probe_video(source)
    start_pts = trim.get("source_in_pts")
    end_pts = trim.get("source_out_pts")
    boundary_dir = output_dir / source_sha256[:16]
    if isinstance(start_pts, int):
        start_frame = extract_frame_at_pts(
            source,
            start_pts,
            boundary_dir / f"start-{start_pts}.png",
        )
    else:
        start_frame = extract_frame(
            source,
            start_ms,
            boundary_dir / f"start-ms-{start_ms}.png",
        )
        start_pts = start_frame.frame_pts
    if isinstance(end_pts, int):
        end_frame = extract_frame_at_pts(
            source,
            end_pts,
            boundary_dir / f"end-exclusive-{end_pts}.png",
        )
    else:
        try:
            end_frame = extract_frame(
                source,
                end_ms,
                boundary_dir / f"end-exclusive-ms-{end_ms}.png",
            )
            end_pts = end_frame.frame_pts
        except Exception:
            stream_start_pts = media.video.start_pts or 0
            if media.video.duration_ts is None:
                raise
            end_pts = stream_start_pts + media.video.duration_ts
            end_frame = None
    if end_pts <= start_pts:
        raise ValueError("exact render source interval must be non-empty")
    time_base = media.video.time_base
    stream_start_pts = media.video.start_pts or 0
    exact_start_ms = round(
        (start_pts - stream_start_pts)
        * time_base.numerator
        * 1000
        / time_base.denominator
    )
    exact_end_ms = round(
        (end_pts - stream_start_pts)
        * time_base.numerator
        * 1000
        / time_base.denominator
    )
    return {
        "contract_version": "source-pts-interval-v1",
        "source_path": str(source),
        "source_sha256": source_sha256,
        "video_stream_index": media.video.index,
        "source_start_pts": stream_start_pts,
        "source_time_base": time_base.model_dump(mode="json"),
        "start_pts": start_pts,
        "end_pts_exclusive": end_pts,
        "start_ms_display": exact_start_ms,
        "end_ms_display": exact_end_ms,
        "start_frame_sha256": start_frame.frame_hash,
        "end_exclusive_frame_sha256": (
            end_frame.frame_hash if end_frame is not None else None
        ),
        "start_frame_path": start_frame.path,
        "end_exclusive_frame_path": (
            end_frame.path if end_frame is not None else None
        ),
        "display_transform": {
            "rotation_degrees": media.video.rotation_degrees,
            "display_width": media.video.display_width,
            "display_height": media.video.display_height,
            "display_sample_aspect_ratio": (
                media.video.display_sample_aspect_ratio.model_dump(mode="json")
            ),
        },
    }


def _render_source_segment(
    *,
    source_path: Path,
    start_ms: int,
    end_ms: int,
    overlay_path: Path | None,
    base_filter: str,
    output_path: Path,
    source_has_audio: bool | None = None,
    source_interval: Mapping[str, Any] | None = None,
) -> str:
    if source_interval is None:
        duration = (end_ms - start_ms) / 1000
        source_input = [
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-i",
            str(source_path),
        ]
        video_filter = base_filter
        source_audio_trim = ""
    else:
        start_pts = int(source_interval["start_pts"])
        end_pts = int(source_interval["end_pts_exclusive"])
        time_base = source_interval["source_time_base"]
        source_start_pts = int(source_interval["source_start_pts"])
        numerator = int(time_base["numerator"])
        denominator = int(time_base["denominator"])
        duration = (end_pts - start_pts) * numerator / denominator
        relative_start = (
            (start_pts - source_start_pts) * numerator / denominator
        )
        relative_end = (
            (end_pts - source_start_pts) * numerator / denominator
        )
        source_input = ["-copyts", "-i", str(source_path)]
        trim_prefix = (
            "[0:v]select="
            f"'gte(pts\\,{start_pts})*lt(pts\\,{end_pts})',"
            "setpts=PTS-STARTPTS[src_exact];"
        )
        video_filter = trim_prefix + base_filter.replace(
            "[0:v]",
            "[src_exact]",
            1,
        )
        source_audio_trim = (
            f"atrim=start={relative_start:.9f}:end={relative_end:.9f},"
            "asetpts=N/SR/TB,"
        )
    audio_fade_out = max(0.0, duration - 0.12)
    if source_has_audio is None:
        source_has_audio = has_audio_stream(source_path)
    if overlay_path is None:
        filter_graph = video_filter + ";[base]null[v]"
        overlay_input: list[str] = []
    else:
        filter_graph = (
            video_filter
            + ";[1:v]format=rgba[card];"
            + "[base][card]overlay=0:0:shortest=1[v]"
        )
        overlay_input = ["-loop", "1", "-i", str(overlay_path)]
    if source_has_audio:
        audio_input: list[str] = []
        audio_map = "0:a:0"
        audio_origin = "source"
    else:
        silence_input_index = 2 if overlay_path is not None else 1
        audio_input = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        audio_map = f"{silence_input_index}:a:0"
        audio_origin = "synthetic_silence"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.partial.mp4")
    _run_segment_encoder(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *source_input,
            *overlay_input,
            *audio_input,
            "-t",
            f"{duration:.3f}",
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-map",
            audio_map,
            "-af",
            (
                source_audio_trim
                + "volume=0.58,afade=t=in:st=0:d=0.08,"
                f"afade=t=out:st={audio_fade_out:.3f}:d=0.12"
            ),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            "8M",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(temporary_path),
        ]
    )
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(temporary_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ]
    )
    temporary_path.replace(output_path)
    return audio_origin


def _write_render_boundary_lineage(
    *,
    segment_path: Path,
    source_interval: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Bind a rendered segment's decoded boundary frames to its source interval."""

    media = probe_video(segment_path)
    first = extract_frame(
        segment_path,
        0,
        output_path.parent / f"{output_path.stem}-output-first.png",
    )
    last_pts = last_decoded_video_frame_pts(segment_path)
    last = extract_frame_at_pts(
        segment_path,
        last_pts,
        output_path.parent / f"{output_path.stem}-output-last.png",
    )
    body = {
        "contract_version": "render-boundary-lineage-v1",
        "segment_path": str(segment_path.resolve()),
        "segment_sha256": media.sha256,
        "source_interval": dict(source_interval),
        "output_first_frame": first.model_dump(mode="json"),
        "output_last_frame": last.model_dump(mode="json"),
        "output_last_frame_pts_source": "ffprobe_decoded_frame_enumeration",
        "verified": True,
        "generated_at": utc_now(),
    }
    write_json(output_path, body)
    return {
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "verified": True,
    }


def _render_missing_segment(
    chapter: FeatureChapterBrief,
    output_path: Path,
    overlay_path: Path,
    dimensions: tuple[int, int],
    *,
    duration_seconds: float | None = None,
) -> None:
    _render_text_layer(
        chapter,
        overlay_path,
        dimensions=dimensions,
        missing_evidence=True,
        opaque=True,
    )
    duration = (
        duration_seconds
        if duration_seconds is not None
        else chapter.target_duration_seconds
    )
    _run_segment_encoder(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-i",
            str(overlay_path),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            f"{duration:.3f}",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            "8M",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            str(output_path),
        ]
    )


def _concat_segments(segment_paths: Sequence[Path], output_path: Path) -> None:
    if not segment_paths:
        raise ValueError("cannot concatenate an empty segment list")
    inputs: list[str] = []
    filter_inputs: list[str] = []
    for index, path in enumerate(segment_paths):
        inputs.extend(["-i", str(path.resolve())])
        filter_inputs.extend([f"[{index}:v:0]", f"[{index}:a:0]"])
    filter_graph = "".join(filter_inputs) + f"concat=n={len(segment_paths)}:v=1:a=1[v][a]"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.partial.mp4")
    _run_segment_encoder(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *inputs,
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            "8M",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(temporary_path),
        ]
    )
    expected_duration = sum(_probe_duration_seconds(path) for path in segment_paths)
    actual_duration = _probe_duration_seconds(temporary_path)
    if abs(actual_duration - expected_duration) > 0.25:
        raise RuntimeError(
            f"assembled duration mismatch: expected={expected_duration:.3f}s "
            f"actual={actual_duration:.3f}s"
        )
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(temporary_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ]
    )
    temporary_path.replace(output_path)


def _output_media_metadata(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_name,codec_type,width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    video = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
    audio = next(
        (stream for stream in payload["streams"] if stream["codec_type"] == "audio"),
        None,
    )
    return {
        "sha256": sha256_file(path),
        "duration_seconds": float(payload["format"]["duration"]),
        "size_bytes": int(payload["format"]["size"]),
        "video_codec": video["codec_name"],
        "width": int(video["width"]),
        "height": int(video["height"]),
        "frame_rate": video["r_frame_rate"],
        "video_frames": int(video["nb_frames"]),
        "has_audio": audio is not None,
        "audio_codec": audio["codec_name"] if audio is not None else None,
    }


def _probe_duration_seconds(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(completed.stdout)["format"]["duration"])


def _segment_is_valid(
    path: Path, *, expected_duration: float, dimensions: tuple[int, int]
) -> bool:
    if not path.exists():
        return False
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return False
    try:
        payload = json.loads(probe.stdout)
        stream = payload["streams"][0]
        duration = float(payload["format"]["duration"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if (stream["width"], stream["height"]) != dimensions:
        return False
    if abs(duration - expected_duration) > 0.15:
        return False
    decode = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    return decode.returncode == 0


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _feature_candidate_query_target_id(region: FramingRegionIntent) -> str:
    """Keep legacy regions addressable without claiming an upstream entity ID."""

    return region.entity_id or f"reframe_{region.region_id}"


def _feature_candidate_framing_intent(
    *,
    required_target_ids: Sequence[str],
    preferred_target_ids: Sequence[str],
    overlay_keepout_target_ids: Sequence[str],
    allow_controlled_required_clipping: bool,
) -> str:
    obligations: list[str] = []
    if required_target_ids:
        obligations.append(
            (
                "Keep required targets recognizable under the saved controlled "
                "clipping policy."
            )
            if allow_controlled_required_clipping
            else "Keep required targets fully visible and recognizable."
        )
    if preferred_target_ids:
        obligations.append("Retain preferred targets when the frame permits.")
    if overlay_keepout_target_ids:
        obligations.append("Keep overlay-protected targets unobscured.")
    return " ".join(obligations) or "Keep the selected target recognizable."


def feature_vertical_candidate_to_query_proposal_v2(
    candidate: FeatureVerticalCandidate,
    *,
    editorial_goal: str,
    created_at: str,
    created_by: str,
    source_reference: str | None = None,
    eligible_predicate: EvidencePredicateContractV2 | None = None,
    claim_source: EvidenceClaimSource = EvidenceClaimSource.MODEL_PROPOSAL,
    revision: int = 1,
) -> EvidenceQueryProposalV2:
    """Project a candidate into reviewable identity/predicate/framing layers.

    The adapter deliberately does not infer an event predicate from observed
    evidence or selection prose.  Callers may pass a separately established
    ``eligible_predicate``; normal v2 cross-reference validation then proves
    that its participants belong to the persistent identity contract.

    This function creates a proposal only.  It never creates approval
    provenance and never mutates the editorial candidate.
    """

    targets: list[EvidenceTargetIdentityV2] = []
    required_target_ids: list[str] = []
    preferred_target_ids: list[str] = []
    overlay_keepout_target_ids: list[str] = []
    visibility_constraints: list[EvidenceTargetVisibilityConstraintV2] = []

    for region in candidate.regions:
        target_description = region.target_description.strip()
        if not target_description:
            raise ValueError("candidate region target descriptions must be non-empty")
        target_id = _feature_candidate_query_target_id(region)
        targets.append(
            EvidenceTargetIdentityV2(
                target_id=target_id,
                target_description=target_description,
                identity_cues=(target_description,),
                context_cues=tuple(region.observable_relations),
                stable_exclusions=tuple(region.exclusions),
            )
        )
        if region.execution_role == "hard_core":
            required_target_ids.append(target_id)
        elif region.execution_role == "soft_extent":
            preferred_target_ids.append(target_id)
        else:
            overlay_keepout_target_ids.append(target_id)
        if region.execution_role != "overlay_keepout":
            visibility_constraints.append(
                EvidenceTargetVisibilityConstraintV2(
                    target_id=target_id,
                    minimum_visible_fraction=(
                        region.effective_minimum_visible_fraction
                    ),
                    atomic=region.atomic,
                )
            )

    if not required_target_ids and candidate.target_description is not None:
        target_description = candidate.target_description.strip()
        if not target_description:
            raise ValueError("candidate target_description must be non-empty")
        target_id = "reframe_subject"
        if any(target.target_id == target_id for target in targets):
            raise ValueError(
                "candidate target and region identities collide at reframe_subject"
            )
        targets.append(
            EvidenceTargetIdentityV2(
                target_id=target_id,
                target_description=target_description,
                identity_cues=(target_description,),
            )
        )
        required_target_ids.append(target_id)
        visibility_constraints.append(
            EvidenceTargetVisibilityConstraintV2(
                target_id=target_id,
                minimum_visible_fraction=1.0,
            )
        )

    if not targets:
        raise ValueError(
            "candidate has no persistent target identity; a fit-only candidate "
            "without a target does not require an evidence query"
        )
    target_ids = [target.target_id for target in targets]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("candidate regions resolve to duplicate query target IDs")

    identity = EvidenceIdentityContractV2(targets=tuple(targets))
    framing = EvidenceFramingObligationsV2(
        required_target_ids=tuple(required_target_ids),
        preferred_target_ids=tuple(preferred_target_ids),
        overlay_keepout_target_ids=tuple(overlay_keepout_target_ids),
        framing_intent=_feature_candidate_framing_intent(
            required_target_ids=required_target_ids,
            preferred_target_ids=preferred_target_ids,
            overlay_keepout_target_ids=overlay_keepout_target_ids,
            allow_controlled_required_clipping=(
                candidate.crop_mode == "primary_center"
            ),
        ),
        editing_uses=("portrait_reframe",),
        aspect_constraints=(
            EvidenceAspectConstraintV2(
                aspect_ratio="9:16",
                required_target_ids=tuple(required_target_ids),
                constraint=(
                    "Required targets must remain directly visible in the portrait frame."
                ),
                target_visibility_constraints=tuple(visibility_constraints),
                required_target_clipping_policy=(
                    "forbid"
                    if candidate.crop_mode == "strict"
                    else "allow_controlled"
                ),
            ),
        ),
    )
    resolved_claim_source = EvidenceClaimSource(claim_source)
    provenance = EvidenceQueryProvenanceV2(
        created_at=created_at,
        created_by=created_by,
        source_reference=(
            source_reference
            if source_reference is not None
            else f"feature-vertical-candidate:{candidate.candidate_id}"
        ),
    )
    proposal_fingerprint = _stable_fingerprint(
        {
            "contract_version": "feature-vertical-candidate-query-adapter-v1",
            "candidate": candidate.model_dump(mode="json", exclude_none=True),
            "revision": revision,
            "editorial_goal": editorial_goal,
            "identity_sha256": identity.definition_sha256(),
            "predicate_sha256": (
                eligible_predicate.definition_sha256()
                if eligible_predicate is not None
                else None
            ),
            "framing_sha256": framing.definition_sha256(),
            "claim_source": resolved_claim_source.value,
            "provenance": provenance.model_dump(mode="json", exclude_none=True),
        }
    )
    return EvidenceQueryProposalV2(
        proposal_id=f"feature-query-proposal:{proposal_fingerprint[:24]}",
        revision=revision,
        editorial_goal=editorial_goal,
        identity=identity,
        predicate=eligible_predicate,
        framing=framing,
        claim_source=resolved_claim_source,
        provenance=provenance,
    )


def _require_named_auto_policy_approval(
    approval: EvidenceQueryApprovalProvenance,
) -> None:
    if approval.approval_source != EvidenceApprovalSource.AUTO_POLICY:
        raise ValueError(
            "Full Auto QueryLock v2 runtime requires auto_policy approval provenance"
        )
    if approval.policy_reference is None or not approval.policy_reference.strip():
        raise ValueError(
            "Full Auto QueryLock v2 runtime requires a named policy_reference"
        )


def approve_feature_query_proposal_v2_for_auto(
    proposal: EvidenceQueryProposalV2,
    *,
    query_id: str,
    approval: EvidenceQueryApprovalProvenance,
) -> EvidenceQueryLockV2:
    """Approve a proposal only through explicit, named Full Auto policy provenance."""

    _require_named_auto_policy_approval(approval)
    return approve_evidence_query_proposal_v2(
        proposal,
        query_id=query_id,
        approval=approval,
    )


_FULL_AUTO_QUERYLOCK_POLICY_REFERENCE = (
    "policy:full-auto-topk-lazy-geometry-querylock-v2:v1"
)
_FULL_AUTO_QUERYLOCK_POLICY = {
    "contract_version": "full-auto-querylock-approval-policy-v1",
    "policy_reference": _FULL_AUTO_QUERYLOCK_POLICY_REFERENCE,
    "eligible_input": "validated FeatureVerticalCandidate from saved Top-K plan",
    "approval_timing": "only when lazy geometry evaluation is attempted",
    "maximum_topk_candidates": 4,
    "predicate_inference": "forbidden unless separately established",
    "semantic_effect": "authorize bounded geometry evaluation, not final edit approval",
}
_FULL_AUTO_QUERYLOCK_POLICY_SHA256 = _stable_fingerprint(
    _FULL_AUTO_QUERYLOCK_POLICY
)


def _load_or_create_feature_candidate_query_lock_v2(
    candidate: FeatureVerticalCandidate,
    *,
    feature_id: str,
    output_dir: Path,
) -> EvidenceQueryLockV2:
    """Persist one truthful auto-policy lock only when geometry is attempted."""

    candidate_sha256 = _stable_fingerprint(
        candidate.model_dump(mode="json", exclude_none=True)
    )
    query_scope_sha256 = _stable_fingerprint(
        {
            "contract_version": "feature-auto-querylock-v2-scope-v1",
            "feature_id": feature_id,
            "candidate_sha256": candidate_sha256,
        }
    )
    query_parent = output_dir / "query-lock-v2"
    query_dir = query_parent / f"variant-{query_scope_sha256[:16]}"
    proposal_path = query_dir / "proposal.json"
    lock_path = query_dir / "lock.json"
    manifest_path = query_dir / "manifest.json"
    policy_path = query_dir / "approval-policy.json"
    if (
        proposal_path.exists()
        or lock_path.exists()
        or manifest_path.exists()
        or policy_path.exists()
    ):
        if not all(
            path.exists()
            for path in (proposal_path, lock_path, manifest_path, policy_path)
        ):
            raise RuntimeError(f"incomplete automatic QueryLock v2 artifacts: {query_dir}")
        proposal = EvidenceQueryProposalV2.model_validate(read_json(proposal_path))
        lock = EvidenceQueryLockV2.model_validate(read_json(lock_path))
        manifest = read_json(manifest_path)
        expected_source_reference = (
            f"feature:{feature_id}:candidate:{candidate.candidate_id}"
        )
        expected = {
            "contract_version": "feature-auto-querylock-v2-manifest-v3",
            "feature_id": feature_id,
            "candidate_id": candidate.candidate_id,
            "candidate_sha256": candidate_sha256,
            "proposal_definition_sha256": _stable_fingerprint(
                proposal.model_dump(mode="json", exclude_none=True)
            ),
            "lock_definition_sha256": lock.definition_sha256(),
            "policy_reference": _FULL_AUTO_QUERYLOCK_POLICY_REFERENCE,
            "approval_policy_sha256": _FULL_AUTO_QUERYLOCK_POLICY_SHA256,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("cached automatic QueryLock v2 lineage does not match candidate")
        if (
            proposal.provenance.source_reference != expected_source_reference
            or lock.approval.source_reference != expected_source_reference
        ):
            raise ValueError(
                "cached automatic QueryLock v2 source provenance does not match candidate"
            )
        if read_json(policy_path) != _FULL_AUTO_QUERYLOCK_POLICY:
            raise ValueError("cached automatic QueryLock approval policy was modified")
        if (
            lock.approval.policy_reference
            != _FULL_AUTO_QUERYLOCK_POLICY_REFERENCE
        ):
            raise ValueError("cached automatic QueryLock used an unregistered policy")
        _require_named_auto_policy_approval(lock.approval)
        write_json(
            query_parent / "current.json",
            {
                "query_scope_sha256": query_scope_sha256,
                "variant": query_dir.name,
                "lock_definition_sha256": lock.definition_sha256(),
                "approval_policy_sha256": _FULL_AUTO_QUERYLOCK_POLICY_SHA256,
            },
        )
        return lock

    created_at = utc_now()
    proposal = feature_vertical_candidate_to_query_proposal_v2(
        candidate,
        editorial_goal=(
            "Preserve the selected evidence identities while evaluating the saved "
            "portrait framing candidate."
        ),
        created_at=created_at,
        created_by="feature-edit-plan.vertical-candidates",
        source_reference=f"feature:{feature_id}:candidate:{candidate.candidate_id}",
        claim_source=EvidenceClaimSource.MODEL_PROPOSAL,
    )
    lock = approve_feature_query_proposal_v2_for_auto(
        proposal,
        query_id=f"feature-query:{proposal.composite_sha256()[:24]}",
        approval=EvidenceQueryApprovalProvenance(
            approved_at=created_at,
            approved_by="full-auto-querylock-router",
            approval_source=EvidenceApprovalSource.AUTO_POLICY,
            source_reference=f"feature:{feature_id}:candidate:{candidate.candidate_id}",
            policy_reference=_FULL_AUTO_QUERYLOCK_POLICY_REFERENCE,
        ),
    )
    write_json(proposal_path, proposal)
    write_json(lock_path, lock)
    write_json(policy_path, _FULL_AUTO_QUERYLOCK_POLICY)
    write_json(
        manifest_path,
        {
            "contract_version": "feature-auto-querylock-v2-manifest-v3",
            "feature_id": feature_id,
            "candidate_id": candidate.candidate_id,
            "candidate_sha256": candidate_sha256,
            "query_scope_sha256": query_scope_sha256,
            "proposal_definition_sha256": _stable_fingerprint(
                proposal.model_dump(mode="json", exclude_none=True)
            ),
            "lock_definition_sha256": lock.definition_sha256(),
            "component_hashes": lock.component_hashes(),
            "policy_reference": _FULL_AUTO_QUERYLOCK_POLICY_REFERENCE,
            "approval_policy_sha256": _FULL_AUTO_QUERYLOCK_POLICY_SHA256,
            "created_at": created_at,
        },
    )
    write_json(
        query_parent / "current.json",
        {
            "query_scope_sha256": query_scope_sha256,
            "variant": query_dir.name,
            "lock_definition_sha256": lock.definition_sha256(),
            "approval_policy_sha256": _FULL_AUTO_QUERYLOCK_POLICY_SHA256,
        },
    )
    return lock


def evidence_query_lock_v2_lineage(
    lock: EvidenceQueryLockV2,
    *,
    target_id: str | None = None,
    target_description: str | None = None,
) -> dict[str, Any]:
    """Return the narrow, hash-bound lineage accepted by Full Auto geometry."""

    _require_named_auto_policy_approval(lock.approval)
    if target_id is not None:
        target = next(
            (item for item in lock.identity.targets if item.target_id == target_id),
            None,
        )
        if target is None:
            raise ValueError(
                f"QueryLock v2 does not define runtime target {target_id!r}"
            )
        if (
            target_description is not None
            and target.target_description.strip() != target_description.strip()
        ):
            raise ValueError(
                "runtime target description does not match its QueryLock v2 identity"
            )

    component_hashes = lock.component_hashes()
    lineage: dict[str, Any] = {
        "contract_version": "evidence-query-lock-v2-runtime-lineage-v1",
        "query_contract_version": lock.contract_version,
        "query_id": lock.query_id,
        "revision": lock.revision,
        **component_hashes,
        "composite_sha256": lock.composite_sha256(),
        "definition_sha256": lock.definition_sha256(),
        "claim_source": lock.claim_source.value,
        "approval": lock.approval.model_dump(mode="json", exclude_none=True),
    }
    if target_id is not None:
        lineage["target_id"] = target_id
    return lineage


def _bind_evidence_query_lock_v2_lineage(
    payload: Mapping[str, Any],
    *,
    lock: EvidenceQueryLockV2 | None,
    target_id: str,
    target_description: str,
) -> dict[str, Any]:
    """Copy a request/seed payload and optionally bind an approved v2 lock."""

    bound = dict(payload)
    if lock is not None:
        bound["evidence_query_v2"] = evidence_query_lock_v2_lineage(
            lock,
            target_id=target_id,
            target_description=target_description,
        )
    return bound


def _track_geometry_fingerprint(track: SegmentationTrack) -> str:
    """Fingerprint every consumed tracking sample and its model/source provenance."""
    return _stable_fingerprint(track.model_dump(mode="json"))


def _query_lock_v2_runtime_geometry_lineage(
    *,
    lock: EvidenceQueryLockV2,
    target_id: str,
    target_description: str,
    seed_fingerprint: str,
    seed_manifest_path: Path,
    track_path: Path,
    track: SegmentationTrack,
) -> dict[str, Any]:
    """Bind the approved definition to the exact SAM seed and resulting geometry."""

    return {
        "contract_version": "feature-query-lock-v2-runtime-geometry-lineage-v1",
        "evidence_query_v2": evidence_query_lock_v2_lineage(
            lock,
            target_id=target_id,
            target_description=target_description,
        ),
        "seed_fingerprint": seed_fingerprint,
        "seed_selection_sha256": sha256_file(seed_manifest_path),
        "track_sha256": sha256_file(track_path),
        "track_geometry_sha256": _track_geometry_fingerprint(track),
    }


def _segment_variant_fingerprint(
    *,
    source_sha256: str,
    start_ms: int,
    end_ms: int,
    filter_graph: str,
    geometry: dict[str, Any],
    track_fingerprint: str | None,
    source_interval: Mapping[str, Any] | None = None,
) -> str:
    return _stable_fingerprint(
        {
            "contract_version": "feature-segment-render-v2",
            "source_sha256": source_sha256,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "source_interval": source_interval,
            "filter_graph": filter_graph,
            "geometry": geometry,
            "track_fingerprint": track_fingerprint,
        }
    )


def _usable_track_centers(
    track: SegmentationTrack,
) -> tuple[list[float], list[float], list[list[int]]]:
    """Return only geometry eligible for unattended rendering.

    LOW_CONFIDENCE masks remain in the track artifact for review, but using
    them to drive a render would turn a warning into an implicit acceptance.
    """
    usable_states = {TrackingState.TRACKED}
    times: list[float] = []
    centers: list[float] = []
    boxes: list[list[int]] = []
    for sample in track.samples:
        if (
            sample.tracking_state in usable_states
            and sample.center_2d is not None
            and sample.derived_tracking_box is not None
        ):
            times.append(
                (sample.analysis_sample_time_ms - track.analysis_start_ms) / 1000
            )
            centers.append(float(sample.center_2d[0]))
            boxes.append([int(value) for value in sample.derived_tracking_box])
    return times, centers, boxes


def _track_confidence_diagnostics(track: SegmentationTrack) -> dict[str, Any]:
    """Derive the render gate from samples instead of trusting a summary."""

    total = len(track.samples)
    low_confidence_count = sum(
        sample.tracking_state == TrackingState.LOW_CONFIDENCE
        for sample in track.samples
    )
    return {
        "tracking_sample_count": total,
        "low_confidence_sample_count": low_confidence_count,
        "low_confidence_sample_ratio": round(
            low_confidence_count / total if total else 1.0, 6
        ),
        "tracking_confidence_gate_passed": low_confidence_count == 0,
    }


def _orientation_corrected_track_dimensions(
    tracks: Sequence[SegmentationTrack],
) -> tuple[int, int, dict[str, Any]]:
    """Resolve and validate the coordinate lineage used by Grounding and SAM."""

    if not tracks:
        raise ValueError("track_source_geometry_mismatch:no_tracks")
    source_dimensions = {
        (
            getattr(track, "seed_source_width", None),
            getattr(track, "seed_source_height", None),
        )
        for track in tracks
    }
    if None in {value for dimensions in source_dimensions for value in dimensions}:
        raise ValueError("track_source_geometry_mismatch:missing_seed_dimensions")
    if len(source_dimensions) != 1:
        raise ValueError("track_source_geometry_mismatch:required_tracks_disagree")
    source_width, source_height = next(iter(source_dimensions))
    if not isinstance(source_width, int) or not isinstance(source_height, int):
        raise ValueError("track_source_geometry_mismatch:invalid_seed_dimensions")
    source_aspect = source_width / source_height
    analysis_aspect_errors: list[float] = []
    for track in tracks:
        analysis_width = getattr(track, "analysis_width", None)
        analysis_height = getattr(track, "analysis_height", None)
        if not isinstance(analysis_width, int) or not isinstance(analysis_height, int):
            raise ValueError("track_source_geometry_mismatch:missing_analysis_dimensions")
        analysis_aspect = analysis_width / analysis_height
        relative_error = abs(analysis_aspect / source_aspect - 1)
        tolerance = max(0.01, 2 / min(analysis_width, analysis_height))
        if relative_error > tolerance:
            raise ValueError(
                "track_source_geometry_mismatch:analysis_aspect_disagrees"
            )
        analysis_aspect_errors.append(relative_error)
    return source_width, source_height, {
        "source_geometry_lineage_passed": True,
        "orientation_basis": "ffmpeg_autorotated_display",
        "source_display_width": source_width,
        "source_display_height": source_height,
        "max_analysis_aspect_relative_error": round(
            max(analysis_aspect_errors, default=0.0), 9
        ),
    }


def _horizontal_reframe_failure_geometry(
    zoom_intent: str,
    *,
    fallback_reason: str,
    risk_code: str,
    diagnostics: Mapping[str, Any] | None = None,
    geometry_safe_max_zoom: float | None = None,
) -> dict[str, Any]:
    """Describe a requested tracked reframe that was not safely applied."""

    requested = {"subtle": 1.12, "detail": 1.35}[zoom_intent]
    return {
        "requested_zoom": requested,
        "geometry_safe_max_zoom": geometry_safe_max_zoom,
        "applied_zoom": 1.0,
        "fallback_reason": fallback_reason,
        "risk_codes": list(
            dict.fromkeys([risk_code, "requested_tracked_reframe_not_applied"])
        ),
        "requires_gemini_review": True,
        **dict(diagnostics or {}),
    }


def _smooth(values: Sequence[float], alpha: float = 0.34) -> list[float]:
    if not values:
        return []
    forward = [float(values[0])]
    for value in values[1:]:
        forward.append(alpha * float(value) + (1 - alpha) * forward[-1])
    backward = [forward[-1]]
    for value in reversed(forward[:-1]):
        backward.append(alpha * value + (1 - alpha) * backward[-1])
    return list(reversed(backward))


def _piecewise_expression(
    times: Sequence[float],
    values: Sequence[float],
    *,
    cut_before_indexes: frozenset[int] = frozenset(),
) -> str:
    if not times or len(times) != len(values):
        raise ValueError("crop expression needs aligned non-empty times and values")
    if len(times) == 1:
        return f"{values[0]:.3f}"
    expression = f"{values[-1]:.3f}"
    for index in range(len(times) - 2, -1, -1):
        t0, t1 = times[index], times[index + 1]
        x0, x1 = values[index], values[index + 1]
        delta = max(0.001, t1 - t0)
        linear = (
            f"{x0:.3f}"
            if index + 1 in cut_before_indexes
            else f"{x0:.3f}+({x1 - x0:.3f})*(t-{t0:.3f})/{delta:.3f}"
        )
        expression = f"if(lt(t\\,{t1:.3f})\\,{linear}\\,{expression})"
    # Do not extrapolate before the first observed analysis frame.  FFmpeg
    # otherwise evaluates the first linear segment at negative relative time,
    # which can move the crop away from the seed before tracking evidence exists.
    return f"if(lt(t\\,{times[0]:.3f})\\,{values[0]:.3f}\\,{expression})"


def _projected_smooth(
    desired: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    iterations: int = 12,
) -> list[float]:
    """Smooth a crop path while keeping every point inside its legal interval."""
    if not desired or not (len(desired) == len(lower) == len(upper)):
        raise ValueError("projected smoothing needs aligned non-empty values")
    if any(low > high for low, high in zip(lower, upper, strict=True)):
        raise ValueError("projected smoothing received an empty legal interval")

    values = [
        max(low, min(high, float(value)))
        for value, low, high in zip(desired, lower, upper, strict=True)
    ]
    for _ in range(iterations):
        previous = list(values)
        for index, target in enumerate(desired):
            neighbors: list[float] = []
            if index:
                neighbors.append(previous[index - 1])
            if index + 1 < len(previous):
                neighbors.append(previous[index + 1])
            neighbor_mean = sum(neighbors) / len(neighbors) if neighbors else float(target)
            proposal = 0.58 * float(target) + 0.42 * neighbor_mean
            values[index] = max(lower[index], min(upper[index], proposal))
    return values


def _even_ceil(value: float) -> int:
    return max(2, int(math.ceil(value / 2) * 2))


def _cover_transform(
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    *,
    zoom: float = 1.0,
) -> dict[str, Any]:
    """Return one deterministic, aspect-preserving cover transform."""

    if min(source_width, source_height, output_width, output_height) <= 0:
        raise ValueError("cover transform dimensions must be positive")
    if zoom < 1.0:
        raise ValueError("cover transform zoom must be at least 1")
    source_aspect = source_width / source_height
    output_aspect = output_width / output_height
    if source_aspect >= output_aspect:
        scaled_height = _even_ceil(output_height * zoom)
        scaled_width = _even_ceil(scaled_height * source_aspect)
    else:
        scaled_width = _even_ceil(output_width * zoom)
        scaled_height = _even_ceil(scaled_width / source_aspect)
    if scaled_width < output_width or scaled_height < output_height:
        raise ValueError("cover transform failed to cover the output viewport")
    active_pan_axes = [
        axis
        for axis, active in (
            ("x", scaled_width > output_width),
            ("y", scaled_height > output_height),
        )
        if active
    ]
    return {
        "contract_version": "aspect-preserving-cover-v1",
        "orientation_basis": "ffmpeg_autorotated_display",
        "scale_policy": "aspect_preserving_cover",
        "source_display_width": source_width,
        "source_display_height": source_height,
        "source_aspect_ratio": round(source_aspect, 9),
        "output_aspect_ratio": round(output_aspect, 9),
        "zoom": round(zoom, 6),
        "scaled_width": scaled_width,
        "scaled_height": scaled_height,
        "crop_width": output_width,
        "crop_height": output_height,
        "origin": "top_left",
        "normalized_track_space": "orientation_corrected_source_0_1000",
        "normalized_box_order": "x_min_y_min_x_max_y_max",
        "active_pan_axes": active_pan_axes,
        "aspect_ratio_relative_error": round(
            abs((scaled_width / scaled_height) / source_aspect - 1), 9
        ),
    }


def _axis_crop_constraints(
    *,
    padded_min: float,
    padded_max: float,
    viewport_normalized: float,
    overflow_policy: Literal["preserve_all", "controlled_clip"],
    edge_priority: Literal["balanced", "preserve_start", "preserve_end"],
) -> tuple[float, float, bool, bool]:
    max_origin = max(0.0, 1000.0 - viewport_normalized)
    lower = max(0.0, padded_max - viewport_normalized)
    upper = min(max_origin, padded_min)
    if lower <= upper + 1e-6:
        return lower, upper, True, False
    if edge_priority == "preserve_start":
        aligned = max(0.0, min(max_origin, padded_min))
    elif edge_priority == "preserve_end":
        aligned = max(0.0, min(max_origin, padded_max - viewport_normalized))
    else:
        aligned = max(
            0.0,
            min(
                max_origin,
                (padded_min + padded_max) / 2 - viewport_normalized / 2,
            ),
        )
    return aligned, aligned, False, overflow_policy == "controlled_clip"


def _tracked_crop_geometry(
    times: Sequence[float],
    centers_x: Sequence[float],
    boxes: Sequence[Sequence[int]],
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    zoom: float = 1.0,
    safety_multiplier: float = 1.0,
    overflow_policy: Literal["preserve_all", "controlled_clip"] = "preserve_all",
    edge_priority: Literal["balanced", "preserve_start", "preserve_end"] = "balanced",
    desired_centers_y: Sequence[float] | None = None,
) -> tuple[list[float], list[float], dict[str, Any]]:
    """Return a 2D crop path projected into per-sample safety constraints.

    Boxes are required-region union boxes in the project's canonical
    ``[x_min, y_min, x_max, y_max]`` normalized coordinate system.  When a
    union can fit, every rendered keyframe contains it (plus the requested
    safety margin).  ``controlled_clip`` is explicit overflow behavior for a
    region that is wider than the portrait viewport; it never masquerades as
    full containment.
    """
    if not times or len(times) != len(centers_x) or len(times) != len(boxes):
        raise ValueError("tracked crop geometry needs aligned non-empty samples")
    if desired_centers_y is not None and len(desired_centers_y) != len(times):
        raise ValueError("desired y centers must align with tracked crop samples")
    if safety_multiplier < 1.0:
        raise ValueError("safety_multiplier must be at least 1")
    transform = _cover_transform(
        source_width,
        source_height,
        output_width,
        output_height,
        zoom=zoom,
    )
    scaled_width = int(transform["scaled_width"])
    scaled_height = int(transform["scaled_height"])
    crop_width = output_width
    crop_height = output_height
    crop_width_normalized = crop_width * 1000 / scaled_width
    crop_height_normalized = crop_height * 1000 / scaled_height
    source_crop_x_max_normalized = 1000 - crop_width_normalized
    source_crop_y_max_normalized = 1000 - crop_height_normalized
    centers_y: list[float] = []
    validated_boxes: list[list[float]] = []
    for box in boxes:
        if len(box) != 4:
            raise ValueError("tracked crop boxes must contain four coordinates")
        x_min, y_min, x_max, y_max = (float(value) for value in box)
        if not 0 <= x_min < x_max <= 1000 or not 0 <= y_min < y_max <= 1000:
            raise ValueError("tracked crop box coordinates are invalid")
        validated_boxes.append([x_min, y_min, x_max, y_max])
        centers_y.append((y_min + y_max) / 2)
    smooth_centers_x = _smooth(centers_x)
    composition_centers_y = (
        [float(value) for value in desired_centers_y]
        if desired_centers_y is not None
        else centers_y
    )
    smooth_centers_y = _smooth(composition_centers_y)
    desired_left = [
        max(
            0.0,
            min(
                source_crop_x_max_normalized,
                center - crop_width_normalized / 2,
            ),
        )
        for center in smooth_centers_x
    ]
    desired_top = [
        max(
            0.0,
            min(
                source_crop_y_max_normalized,
                center - crop_height_normalized / 2,
            ),
        )
        for center in smooth_centers_y
    ]
    legal_left_lower: list[float] = []
    legal_left_upper: list[float] = []
    legal_top_lower: list[float] = []
    legal_top_upper: list[float] = []
    full_containment_x: list[bool] = []
    full_containment_y: list[bool] = []
    controlled_clip_samples: list[bool] = []
    margins_x: list[float] = []
    margins_y: list[float] = []
    for x_min, y_min, x_max, y_max in validated_boxes:
        width = x_max - x_min
        height = y_max - y_min
        margin_x = width * (safety_multiplier - 1) / 2
        margin_y = height * (safety_multiplier - 1) / 2
        padded_x_min = max(0.0, x_min - margin_x)
        padded_x_max = min(1000.0, x_max + margin_x)
        padded_y_min = max(0.0, y_min - margin_y)
        padded_y_max = min(1000.0, y_max + margin_y)
        x_lower, x_upper, x_fits, x_controlled = _axis_crop_constraints(
            padded_min=padded_x_min,
            padded_max=padded_x_max,
            viewport_normalized=crop_width_normalized,
            overflow_policy=overflow_policy,
            edge_priority=edge_priority,
        )
        y_lower, y_upper, y_fits, y_controlled = _axis_crop_constraints(
            padded_min=padded_y_min,
            padded_max=padded_y_max,
            viewport_normalized=crop_height_normalized,
            overflow_policy=overflow_policy,
            edge_priority=edge_priority,
        )
        legal_left_lower.append(x_lower)
        legal_left_upper.append(x_upper)
        legal_top_lower.append(y_lower)
        legal_top_upper.append(y_upper)
        full_containment_x.append(x_fits)
        full_containment_y.append(y_fits)
        controlled_clip_samples.append(x_controlled or y_controlled)
        margins_x.append(margin_x)
        margins_y.append(margin_y)

    full_containment = [
        x_fits and y_fits
        for x_fits, y_fits in zip(
            full_containment_x, full_containment_y, strict=True
        )
    ]
    geometry_feasible = overflow_policy == "controlled_clip" or all(full_containment)
    if geometry_feasible:
        crop_left_normalized = _projected_smooth(
            desired_left,
            legal_left_lower,
            legal_left_upper,
        )
        crop_top_normalized = _projected_smooth(
            desired_top,
            legal_top_lower,
            legal_top_upper,
        )
    else:
        crop_left_normalized = [
            max(low, min(high, desired))
            for desired, low, high in zip(
                desired_left, legal_left_lower, legal_left_upper, strict=True
            )
        ]
        crop_top_normalized = [
            max(low, min(high, desired))
            for desired, low, high in zip(
                desired_top, legal_top_lower, legal_top_upper, strict=True
            )
        ]
    x_values = [value * scaled_width / 1000 for value in crop_left_normalized]
    y_values = [value * scaled_height / 1000 for value in crop_top_normalized]
    max_target_width = max(box[2] - box[0] for box in validated_boxes)
    max_target_height = max(box[3] - box[1] for box in validated_boxes)
    keyframes: list[dict[str, Any]] = []
    containment_failures = 0
    minimum_visible_width_fraction = 1.0
    minimum_visible_height_fraction = 1.0
    minimum_visible_area_fraction = 1.0
    for (
        time,
        center_x,
        center_y,
        smooth_center_x,
        smooth_center_y,
        crop_x,
        crop_y,
        crop_left,
        crop_top,
        box,
        left_low,
        left_high,
        top_low,
        top_high,
        margin_x,
        margin_y,
        contained_by_construction,
        controlled,
    ) in zip(
        times,
        centers_x,
        centers_y,
        smooth_centers_x,
        smooth_centers_y,
        x_values,
        y_values,
        crop_left_normalized,
        crop_top_normalized,
        validated_boxes,
        legal_left_lower,
        legal_left_upper,
        legal_top_lower,
        legal_top_upper,
        margins_x,
        margins_y,
        full_containment,
        controlled_clip_samples,
        strict=True,
    ):
        x_min, y_min, x_max, y_max = box
        padded_x_min = max(0.0, x_min - margin_x)
        padded_x_max = min(1000.0, x_max + margin_x)
        padded_y_min = max(0.0, y_min - margin_y)
        padded_y_max = min(1000.0, y_max + margin_y)
        visible_width = max(
            0.0,
            min(x_max, crop_left + crop_width_normalized) - max(x_min, crop_left),
        )
        visible_height = max(
            0.0,
            min(y_max, crop_top + crop_height_normalized) - max(y_min, crop_top),
        )
        visible_width_fraction = visible_width / (x_max - x_min)
        visible_height_fraction = visible_height / (y_max - y_min)
        visible_area_fraction = visible_width_fraction * visible_height_fraction
        minimum_visible_width_fraction = min(
            minimum_visible_width_fraction, visible_width_fraction
        )
        minimum_visible_height_fraction = min(
            minimum_visible_height_fraction, visible_height_fraction
        )
        minimum_visible_area_fraction = min(
            minimum_visible_area_fraction, visible_area_fraction
        )
        contained = (
            padded_x_min >= crop_left - 1e-6
            and padded_x_max <= crop_left + crop_width_normalized + 1e-6
            and padded_y_min >= crop_top - 1e-6
            and padded_y_max <= crop_top + crop_height_normalized + 1e-6
        )
        if not contained:
            containment_failures += 1
        keyframes.append(
            {
                "time_seconds": round(time, 6),
                "tracked_center_x_normalized": round(center_x, 4),
                "tracked_center_y_normalized": round(center_y, 4),
                "smoothed_center_x_normalized": round(smooth_center_x, 4),
                "smoothed_center_y_normalized": round(smooth_center_y, 4),
                "required_union_box": [int(value) for value in box],
                "legal_crop_left_min_normalized": round(left_low, 4),
                "legal_crop_left_max_normalized": round(left_high, 4),
                "legal_crop_top_min_normalized": round(top_low, 4),
                "legal_crop_top_max_normalized": round(top_high, 4),
                "effective_margin_x_normalized": round(margin_x, 4),
                "effective_margin_y_normalized": round(margin_y, 4),
                "effective_margin_normalized": round(margin_x, 4),
                "crop_x_pixels": round(crop_x, 3),
                "crop_y_pixels": round(crop_y, 3),
                "required_union_contained": contained,
                "full_containment_feasible": contained_by_construction,
                "controlled_clip_applied": controlled,
                "visible_required_width_fraction": round(
                    visible_width_fraction, 6
                ),
                "visible_required_height_fraction": round(
                    visible_height_fraction, 6
                ),
                "visible_required_area_fraction": round(visible_area_fraction, 6),
            }
        )

    x_velocities = []
    y_velocities = []
    combined_velocities = []
    for t0, t1, x0, x1, y0, y1 in zip(
        times[:-1],
        times[1:],
        x_values[:-1],
        x_values[1:],
        y_values[:-1],
        y_values[1:],
        strict=True,
    ):
        delta_seconds = max(0.001, t1 - t0)
        x_velocity = abs(x1 - x0) / delta_seconds
        y_velocity = abs(y1 - y0) / delta_seconds
        x_velocities.append(x_velocity)
        y_velocities.append(y_velocity)
        combined_velocities.append(math.hypot(x_velocity, y_velocity))
    accelerations = [
        abs(v1 - v0) / max(0.001, times[index + 2] - times[index + 1])
        for index, (v0, v1) in enumerate(
            zip(combined_velocities[:-1], combined_velocities[1:], strict=True)
        )
    ]
    jerks = [
        abs(a1 - a0) / max(0.001, times[index + 3] - times[index + 2])
        for index, (a0, a1) in enumerate(
            zip(accelerations[:-1], accelerations[1:], strict=True)
        )
    ]
    source_x_edge_contacts = sum(
        box[0] <= 5 or box[2] >= 995 for box in boxes
    )
    source_y_edge_contacts = sum(
        box[1] <= 5 or box[3] >= 995 for box in boxes
    )
    source_boundary_contacts = sum(
        box[0] <= 5 or box[1] <= 5 or box[2] >= 995 or box[3] >= 995
        for box in boxes
    )
    return x_values, y_values, {
        "crop_width_normalized": round(crop_width_normalized, 4),
        "crop_height_normalized": round(crop_height_normalized, 4),
        "max_target_width_normalized": max_target_width,
        "max_target_height_normalized": max_target_height,
        "overflow_policy": overflow_policy,
        "edge_priority": edge_priority,
        "geometry_feasible": geometry_feasible,
        "full_containment_feasible": all(full_containment),
        "controlled_clip_applied": any(controlled_clip_samples),
        "containment_failure_count": containment_failures,
        "minimum_visible_required_width_fraction": round(
            minimum_visible_width_fraction, 6
        ),
        "minimum_visible_required_height_fraction": round(
            minimum_visible_height_fraction, 6
        ),
        "minimum_visible_required_area_fraction": round(
            minimum_visible_area_fraction, 6
        ),
        "max_crop_x_speed_pixels_per_second": round(
            max(x_velocities, default=0.0), 4
        ),
        "max_crop_y_speed_pixels_per_second": round(
            max(y_velocities, default=0.0), 4
        ),
        "max_crop_speed_pixels_per_second": round(
            max(combined_velocities, default=0.0), 4
        ),
        "max_crop_acceleration_pixels_per_second_squared": round(
            max(accelerations, default=0.0), 4
        ),
        "max_crop_jerk_pixels_per_second_cubed": round(
            max(jerks, default=0.0), 4
        ),
        "source_x_edge_contact_count": source_x_edge_contacts,
        "source_y_edge_contact_count": source_y_edge_contacts,
        "source_boundary_contact_count": source_boundary_contacts,
        "source_boundary_contact_ratio": round(
            source_boundary_contacts / len(boxes), 6
        ),
        "crop_coordinate_space": transform,
        "crop_x_values_pixels": x_values,
        "crop_y_values_pixels": y_values,
        "crop_keyframes": keyframes,
    }


def _vertical_crop_geometry(
    times: Sequence[float],
    centers_x: Sequence[float],
    boxes: Sequence[Sequence[int]],
    *,
    source_width: int = 1920,
    source_height: int = 1080,
    safety_multiplier: float = 1.0,
    overflow_policy: Literal["preserve_all", "controlled_clip"] = "preserve_all",
    edge_priority: Literal["balanced", "preserve_start", "preserve_end"] = "balanced",
) -> tuple[list[float], dict[str, Any]]:
    """Compatibility wrapper for a 1080x1920 tracked crop."""

    x_values, _, audit = _tracked_crop_geometry(
        times,
        centers_x,
        boxes,
        source_width=source_width,
        source_height=source_height,
        output_width=1080,
        output_height=1920,
        safety_multiplier=safety_multiplier,
        overflow_policy=overflow_policy,
        edge_priority=edge_priority,
    )
    return x_values, audit


def _vertical_target_fits_crop(
    max_target_width_normalized: float,
    crop_width_normalized: float,
    *,
    primary_center: bool,
) -> tuple[bool, float]:
    """Primary-center may relax outer margin, never clip the selected target."""
    safety_multiplier = 1.0 if primary_center else 1.08
    return (
        max_target_width_normalized * safety_multiplier <= crop_width_normalized,
        safety_multiplier,
    )


def _required_track_union(
    tracks: Sequence[SegmentationTrack],
    *,
    region_ids: Sequence[str] | None = None,
) -> tuple[list[float], list[float], list[list[int]], dict[str, Any]]:
    """Build required-region union boxes and fail-closed coverage diagnostics."""
    if not tracks:
        raise ValueError("at least one required track is needed")
    starts = {track.analysis_start_ms for track in tracks}
    ends = {track.analysis_end_ms for track in tracks}
    rates = {float(track.analysis_fps) for track in tracks}
    if len(starts) != 1 or len(ends) != 1 or len(rates) != 1:
        raise ValueError("required tracks must share one analysis interval and rate")
    start_ms = starts.pop()
    end_ms = ends.pop()
    if end_ms is None:
        raise ValueError("required tracks must have an explicit analysis_end_ms")
    analysis_fps = rates.pop()
    labels = list(region_ids or [f"region_{index + 1}" for index in range(len(tracks))])
    if len(labels) != len(tracks):
        raise ValueError("region IDs must align with required tracks")

    # LOW_CONFIDENCE geometry remains evidence, but it cannot satisfy the
    # unattended render gate. A single required low-confidence sample forces
    # the caller onto a review-required fallback.
    usable_states = {TrackingState.TRACKED}
    all_times = sorted(
        {
            sample.analysis_sample_time_ms
            for track in tracks
            for sample in track.samples
        }
    )
    usable_by_region: dict[str, dict[int, list[int]]] = {}
    per_region: list[dict[str, Any]] = []
    low_confidence_required_sample_count = 0
    required_sample_count = 0
    low_confidence_region_ids: list[str] = []
    for label, track in zip(labels, tracks, strict=True):
        diagnostics = _track_confidence_diagnostics(track)
        low_confidence_required_sample_count += int(
            diagnostics["low_confidence_sample_count"]
        )
        required_sample_count += int(diagnostics["tracking_sample_count"])
        if not diagnostics["tracking_confidence_gate_passed"]:
            low_confidence_region_ids.append(label)
        usable = {
            sample.analysis_sample_time_ms: [
                int(value) for value in sample.derived_tracking_box
            ]
            for sample in track.samples
            if sample.tracking_state in usable_states
            and sample.derived_tracking_box is not None
        }
        usable_by_region[label] = usable
        per_region.append(
            {
                "region_id": label,
                "target_description": track.target_description,
                "state_counts": {
                    str(key): value for key, value in track.state_counts.items()
                },
                "usable_sample_count": len(usable),
                "total_sample_count": len(track.samples),
                **diagnostics,
            }
        )

    common_times = [
        time_ms
        for time_ms in all_times
        if all(time_ms in usable for usable in usable_by_region.values())
    ]
    boxes: list[list[int]] = []
    for time_ms in common_times:
        members = [usable_by_region[label][time_ms] for label in labels]
        boxes.append(
            [
                min(box[0] for box in members),
                min(box[1] for box in members),
                max(box[2] for box in members),
                max(box[3] for box in members),
            ]
        )
    centers = [(box[0] + box[2]) / 2 for box in boxes]
    times = [(time_ms - start_ms) / 1000 for time_ms in common_times]

    expected_interval_ms = 1000 / analysis_fps
    head_gap_ms = common_times[0] - start_ms if common_times else end_ms - start_ms
    tail_gap_ms = end_ms - common_times[-1] if common_times else end_ms - start_ms
    internal_gaps = [
        following - current
        for current, following in zip(common_times[:-1], common_times[1:], strict=True)
    ]
    max_internal_gap_ms = max(internal_gaps, default=0)
    unavailable_count = len(all_times) - len(common_times)
    unavailable_ratio = unavailable_count / len(all_times) if all_times else 1.0
    max_edge_gap_ms = expected_interval_ms * 1.35 + 35
    max_allowed_internal_gap_ms = expected_interval_ms * 2.25 + 35
    tracking_confidence_gate_passed = low_confidence_required_sample_count == 0
    coverage_passed = (
        tracking_confidence_gate_passed
        and len(common_times) >= 2
        and unavailable_ratio <= 0.20
        and head_gap_ms <= max_edge_gap_ms
        and tail_gap_ms <= max_edge_gap_ms
        and max_internal_gap_ms <= max_allowed_internal_gap_ms
    )
    coverage = {
        "required_region_count": len(tracks),
        "required_region_ids": labels,
        "expected_sample_count": len(all_times),
        "usable_union_sample_count": len(common_times),
        "unavailable_required_sample_count": unavailable_count,
        "unavailable_required_sample_ratio": round(unavailable_ratio, 6),
        "low_confidence_required_sample_count": low_confidence_required_sample_count,
        "low_confidence_required_sample_ratio": round(
            low_confidence_required_sample_count / required_sample_count
            if required_sample_count
            else 1.0,
            6,
        ),
        "low_confidence_region_ids": low_confidence_region_ids,
        "tracking_confidence_gate_passed": tracking_confidence_gate_passed,
        "analysis_head_gap_ms": round(head_gap_ms, 3),
        "analysis_tail_gap_ms": round(tail_gap_ms, 3),
        "max_internal_gap_ms": round(max_internal_gap_ms, 3),
        "expected_sample_interval_ms": round(expected_interval_ms, 3),
        "max_allowed_edge_gap_ms": round(max_edge_gap_ms, 3),
        "max_allowed_internal_gap_ms": round(max_allowed_internal_gap_ms, 3),
        "coverage_passed": coverage_passed,
        "per_region": per_region,
    }
    return times, centers, boxes, coverage


def _soft_extent_visibility_audit(
    *,
    tracks: Sequence[SegmentationTrack],
    regions: Sequence[FramingRegionIntent],
    crop_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure preferred context after the hard-core crop path is fixed.

    A preferred region may be clipped, but only up to its explicit/default
    visible-fraction floor. Missing or low-confidence geometry fails closed.
    """

    if len(tracks) != len(regions):
        raise ValueError("soft extent tracks and region contracts must align")
    keyframes = crop_audit.get("crop_keyframes")
    coordinate_space = crop_audit.get("crop_coordinate_space")
    if not isinstance(keyframes, list) or not isinstance(coordinate_space, Mapping):
        raise ValueError("crop audit is missing coordinate-space evidence")
    scaled_width = float(coordinate_space["scaled_width"])
    scaled_height = float(coordinate_space["scaled_height"])
    crop_width = float(crop_audit["crop_width_normalized"])
    crop_height = float(crop_audit["crop_height_normalized"])
    per_region: list[dict[str, Any]] = []
    all_passed = True
    for track, region in zip(tracks, regions, strict=True):
        boxes_by_relative_ms = {
            sample.analysis_sample_time_ms - track.analysis_start_ms: list(
                sample.derived_tracking_box
            )
            for sample in track.samples
            if sample.tracking_state == TrackingState.TRACKED
            and sample.derived_tracking_box is not None
        }
        fractions: list[float] = []
        clipped_edges: set[str] = set()
        missing_samples = 0
        for keyframe in keyframes:
            relative_ms = round(float(keyframe["time_seconds"]) * 1000)
            box = boxes_by_relative_ms.get(relative_ms)
            if box is None:
                missing_samples += 1
                fractions.append(0.0)
                continue
            crop_left = float(keyframe["crop_x_pixels"]) * 1000 / scaled_width
            crop_top = float(keyframe["crop_y_pixels"]) * 1000 / scaled_height
            x_min, y_min, x_max, y_max = (float(value) for value in box)
            if x_min < crop_left - 1e-6:
                clipped_edges.add("left")
            if x_max > crop_left + crop_width + 1e-6:
                clipped_edges.add("right")
            if y_min < crop_top - 1e-6:
                clipped_edges.add("top")
            if y_max > crop_top + crop_height + 1e-6:
                clipped_edges.add("bottom")
            visible_width = max(
                0.0,
                min(x_max, crop_left + crop_width) - max(x_min, crop_left),
            )
            visible_height = max(
                0.0,
                min(y_max, crop_top + crop_height) - max(y_min, crop_top),
            )
            fractions.append(
                visible_width
                / max(1e-6, x_max - x_min)
                * visible_height
                / max(1e-6, y_max - y_min)
            )
        minimum = min(fractions, default=0.0)
        required = region.effective_minimum_visible_fraction
        passed = missing_samples == 0 and minimum + 1e-6 >= required
        all_passed = all_passed and passed
        per_region.append(
            {
                "region_id": region.region_id,
                "entity_id": region.entity_id,
                "minimum_visible_area_fraction": round(minimum, 6),
                "required_visible_area_fraction": round(required, 6),
                "missing_sample_count": missing_samples,
                "clipped_edges": sorted(clipped_edges),
                "passed": passed,
            }
        )
    return {
        "soft_extent_count": len(regions),
        "soft_extent_visibility_passed": all_passed,
        "soft_extent_regions": per_region,
    }


def _virtual_camera_zoom_values(
    times: Sequence[float],
    *,
    intent: str,
    maximum_scale: float,
) -> tuple[list[float], str]:
    if not times:
        raise ValueError("virtual camera requires tracking times")
    start = times[0]
    duration = max(0.001, times[-1] - start)
    progress = [
        max(0.0, min(1.0, (time - start) / duration))
        for time in times
    ]
    smooth = [value * value * (3 - 2 * value) for value in progress]
    if intent in {"hold", "follow", "recenter", "pan_reveal"}:
        return [maximum_scale for _ in times], "hold"
    if intent == "push_in":
        return [
            1 + (maximum_scale - 1) * value for value in smooth
        ], "smoothstep"
    if intent == "pull_out":
        return [
            maximum_scale - (maximum_scale - 1) * value for value in smooth
        ], "smoothstep"
    if intent == "punch_in_cut":
        return [
            1.0 if value < 0.5 else maximum_scale for value in progress
        ], "cut"
    raise ValueError(f"unknown virtual-camera intent: {intent}")


def _motion_extrema(
    times: Sequence[float],
    x_values: Sequence[float],
    y_values: Sequence[float],
    scales: Sequence[float],
    *,
    cut_before_indexes: Collection[int] = (),
) -> tuple[float, float, float]:
    value_count = len(times)
    if not (
        len(x_values) == value_count
        and len(y_values) == value_count
        and len(scales) == value_count
    ):
        raise ValueError("motion samples must have equal lengths")
    cut_indexes = frozenset(cut_before_indexes)
    if any(index <= 0 or index >= value_count for index in cut_indexes):
        raise ValueError("hard-cut indexes must lie inside the sample sequence")

    run_starts = [0, *sorted(cut_indexes)]
    run_ends = [*sorted(cut_indexes), value_count]
    maximum_velocity = 0.0
    maximum_acceleration = 0.0
    maximum_jerk = 0.0
    for run_start, run_end in zip(run_starts, run_ends, strict=True):
        velocities: list[float] = []
        velocity_times: list[float] = []
        for index in range(run_start + 1, run_end):
            delta = max(0.001, times[index] - times[index - 1])
            velocities.append(
                math.sqrt(
                    ((x_values[index] - x_values[index - 1]) / delta) ** 2
                    + ((y_values[index] - y_values[index - 1]) / delta) ** 2
                    + (960 * (scales[index] - scales[index - 1]) / delta) ** 2
                )
            )
            velocity_times.append(times[index])
        accelerations = [
            abs(current - prior)
            / max(0.001, velocity_times[index] - velocity_times[index - 1])
            for index, (prior, current) in enumerate(
                zip(velocities[:-1], velocities[1:], strict=True),
                start=1,
            )
        ]
        acceleration_times = velocity_times[1:]
        jerks = [
            abs(current - prior)
            / max(
                0.001,
                acceleration_times[index] - acceleration_times[index - 1],
            )
            for index, (prior, current) in enumerate(
                zip(accelerations[:-1], accelerations[1:], strict=True),
                start=1,
            )
        ]
        maximum_velocity = max(maximum_velocity, max(velocities, default=0.0))
        maximum_acceleration = max(
            maximum_acceleration,
            max(accelerations, default=0.0),
        )
        maximum_jerk = max(maximum_jerk, max(jerks, default=0.0))
    return maximum_velocity, maximum_acceleration, maximum_jerk


def _virtual_camera_filter_from_track(
    track: SegmentationTrack,
    *,
    times: Sequence[float],
    boxes: Sequence[Sequence[int]],
    source_width: int,
    source_height: int,
    requested_intent: str,
    maximum_scale: float,
    geometry_safe_max_scale: float,
    source_resolution_native_scale_limit: float,
    source_track_fingerprint: str,
) -> tuple[str, VirtualCameraPlan, dict[str, Any]]:
    """Build one keyframed, containment-projected 16:9 virtual camera."""

    applied_intent = requested_intent
    execution_status: Literal["applied", "fallback", "blocked"] = "applied"
    fallback_reason: str | None = None
    if requested_intent == "pan_reveal":
        # The current horizontal contract has one identity anchor. Do not turn
        # one track into a fabricated two-anchor reveal.
        applied_intent = "follow"
        execution_status = "fallback"
        fallback_reason = "pan_reveal_requires_two_independently_locked_anchors"
    scales, easing = _virtual_camera_zoom_values(
        times,
        intent=applied_intent,
        maximum_scale=maximum_scale,
    )
    base = _cover_transform(source_width, source_height, 1920, 1080)
    base_width = int(base["scaled_width"])
    base_height = int(base["scaled_height"])
    base_crop_x = (base_width - 1920) / 2
    base_crop_y = (base_height - 1080) / 2
    centers_x: list[float] = []
    centers_y: list[float] = []
    mapped_boxes: list[tuple[float, float, float, float]] = []
    for box in boxes:
        x_min, y_min, x_max, y_max = (float(value) for value in box)
        mapped = (
            x_min / 1000 * base_width - base_crop_x,
            y_min / 1000 * base_height - base_crop_y,
            x_max / 1000 * base_width - base_crop_x,
            y_max / 1000 * base_height - base_crop_y,
        )
        mapped_boxes.append(mapped)
        centers_x.append((mapped[0] + mapped[2]) / 2)
        centers_y.append((mapped[1] + mapped[3]) / 2)
    if applied_intent == "recenter":
        progress = [
            (time - times[0]) / max(0.001, times[-1] - times[0])
            for time in times
        ]
        centers_x = [
            960 + (center - 960) * value
            for center, value in zip(centers_x, progress, strict=True)
        ]
        centers_y = [
            540 + (center - 540) * value
            for center, value in zip(centers_y, progress, strict=True)
        ]
    desired_x: list[float] = []
    desired_y: list[float] = []
    lower_x: list[float] = []
    upper_x: list[float] = []
    lower_y: list[float] = []
    upper_y: list[float] = []
    for center_x, center_y, box, scale in zip(
        centers_x,
        centers_y,
        mapped_boxes,
        scales,
        strict=True,
    ):
        scaled_width = 1920 * scale
        scaled_height = 1080 * scale
        x_min, y_min, x_max, y_max = box
        margin_x = max(8.0, (x_max - x_min) * 0.18)
        margin_y = max(8.0, (y_max - y_min) * 0.18)
        x_low = max(0.0, (x_max + margin_x) * scale - 1920)
        x_high = min(scaled_width - 1920, (x_min - margin_x) * scale)
        y_low = max(0.0, (y_max + margin_y) * scale - 1080)
        y_high = min(scaled_height - 1080, (y_min - margin_y) * scale)
        if x_low > x_high + 1e-6 or y_low > y_high + 1e-6:
            raise ValueError(
                "virtual-camera containment has no legal crop interval"
            )
        lower_x.append(x_low)
        upper_x.append(x_high)
        lower_y.append(y_low)
        upper_y.append(y_high)
        desired_x.append(
            max(0.0, min(scaled_width - 1920, center_x * scale - 960))
        )
        desired_y.append(
            max(0.0, min(scaled_height - 1080, center_y * scale - 540))
        )
    x_values = _projected_smooth(desired_x, lower_x, upper_x)
    y_values = _projected_smooth(desired_y, lower_y, upper_y)
    zoom_expression = _piecewise_expression(times, scales)
    width_expression = (
        f"max(1920\\,2*trunc((1920*({zoom_expression}))/2))"
    )
    height_expression = (
        f"max(1080\\,2*trunc((1080*({zoom_expression}))/2))"
    )
    x_expression = _piecewise_expression(times, x_values)
    y_expression = _piecewise_expression(times, y_values)
    usable_samples = [
        sample
        for sample in track.samples
        if (
            sample.tracking_state == TrackingState.TRACKED
            and sample.center_2d is not None
            and sample.derived_tracking_box is not None
        )
    ]
    keyframes = [
        VirtualCameraKeyframe(
            time_seconds=round(time, 6),
            source_pts=getattr(sample, "source_pts", None),
            scale=round(scale, 6),
            center_x_normalized=round(
                (crop_x + 960) / (1920 * scale) * 1000,
                6,
            ),
            center_y_normalized=round(
                (crop_y + 540) / (1080 * scale) * 1000,
                6,
            ),
        )
        for time, sample, scale, crop_x, crop_y in zip(
            times,
            usable_samples,
            scales,
            x_values,
            y_values,
            strict=True,
        )
    ]
    max_velocity, max_acceleration, max_jerk = _motion_extrema(
        times, x_values, y_values, scales
    )
    plan = VirtualCameraPlan(
        requested_intent=requested_intent,
        applied_intent=applied_intent,
        anchor_target_ids=[
            getattr(track, "target_id", None)
            or f"track:{source_track_fingerprint[:16]}"
        ],
        keyframes=keyframes,
        easing=easing,
        geometry_safe_max_scale=round(geometry_safe_max_scale, 6),
        source_resolution_native_scale_limit=round(
            source_resolution_native_scale_limit,
            6,
        ),
        source_resolution_upscale_required=(
            max(scales) > source_resolution_native_scale_limit + 0.001
        ),
        max_velocity=round(max_velocity, 6),
        max_acceleration=round(max_acceleration, 6),
        max_jerk=round(max_jerk, 6),
        execution_status=execution_status,
        fallback_reason=fallback_reason,
        editorial_reason=(
            "Apply the selected editorial camera intent only after target "
            "identity, tracking confidence, containment, and resolution gates."
        ),
        source_track_fingerprint=source_track_fingerprint,
    )
    filter_graph = (
        f"[0:v]fps=30,scale={base_width}:{base_height},"
        f"crop=1920:1080:x={base_crop_x:.3f}:y={base_crop_y:.3f},setsar=1,"
        f"scale=w='{width_expression}':h='{height_expression}':eval=frame,"
        f"crop=1920:1080:x='{x_expression}':y='{y_expression}',setsar=1[base]"
    )
    audit = {
        "virtual_camera_plan": plan.model_dump(mode="json"),
        "virtual_camera_containment_passed": True,
        "virtual_camera_scale_values": [round(value, 6) for value in scales],
        "virtual_camera_crop_x_values": [round(value, 3) for value in x_values],
        "virtual_camera_crop_y_values": [round(value, 3) for value in y_values],
    }
    return filter_graph, plan, audit


def _horizontal_filter_from_track(
    track: SegmentationTrack,
    zoom_intent: str,
    *,
    display_sample_aspect_ratio: float = 1.0,
    camera_intent: str = "hold",
) -> tuple[str, dict[str, Any]]:
    diagnostics = _track_confidence_diagnostics(track)
    if not math.isclose(display_sample_aspect_ratio, 1.0, rel_tol=0, abs_tol=1e-6):
        return _horizontal_original_filter(), _horizontal_reframe_failure_geometry(
            zoom_intent,
            fallback_reason="non_square_pixel_aspect_ratio_requires_static_reframe",
            risk_code="non_square_pixel_aspect_ratio_requires_static_reframe",
            diagnostics={
                **diagnostics,
                "source_display_sample_aspect_ratio": round(
                    display_sample_aspect_ratio, 9
                ),
                "sample_aspect_ratio_normalized_by_ffmpeg": True,
            },
        )
    try:
        source_width, source_height, lineage = (
            _orientation_corrected_track_dimensions([track])
        )
    except ValueError as error:
        return _horizontal_original_filter(), _horizontal_reframe_failure_geometry(
            zoom_intent,
            fallback_reason=str(error),
            risk_code="track_source_geometry_mismatch",
            diagnostics={
                **diagnostics,
                "source_geometry_lineage_passed": False,
            },
        )
    if not diagnostics["tracking_confidence_gate_passed"]:
        return _horizontal_original_filter(), _horizontal_reframe_failure_geometry(
            zoom_intent,
            fallback_reason="tracking_confidence_gate_failed",
            risk_code="tracking_low_confidence",
            diagnostics={**diagnostics, **lineage},
        )
    times, centers_x, boxes = _usable_track_centers(track)
    if len(times) < 2:
        return _horizontal_original_filter(), _horizontal_reframe_failure_geometry(
            zoom_intent,
            fallback_reason="fewer_than_two_usable_tracking_samples",
            risk_code="insufficient_high_confidence_tracking_samples",
            diagnostics={**diagnostics, **lineage},
        )
    requested = {"subtle": 1.12, "detail": 1.35}[zoom_intent]
    base_transform = _cover_transform(
        source_width,
        source_height,
        1920,
        1080,
    )
    max_width = max(box[2] - box[0] for box in boxes)
    max_height = max(box[3] - box[1] for box in boxes)
    safe_max = min(
        2.0,
        1920
        / (max_width / 1000 * int(base_transform["scaled_width"]) * 1.45),
        1080
        / (max_height / 1000 * int(base_transform["scaled_height"]) * 1.45),
    )
    resolution_safe_max = max(
        1.0,
        min(source_width / 1920, source_height / 1080),
    )
    applied = max(1.0, min(requested, safe_max))
    if applied < 1.035:
        return _horizontal_original_filter(), _horizontal_reframe_failure_geometry(
            zoom_intent,
            fallback_reason="mask_geometry_left_no_safe_zoom_margin",
            risk_code="tracked_reframe_no_safe_zoom_margin",
            diagnostics={**diagnostics, **lineage},
            geometry_safe_max_zoom=round(safe_max, 4),
        )
    track_fingerprint = (
        _track_geometry_fingerprint(track)
        if hasattr(track, "model_dump")
        else _stable_fingerprint(
            {
                "analysis_start_ms": track.analysis_start_ms,
                "seed_source_width": source_width,
                "seed_source_height": source_height,
                "samples": [
                    {
                        "time_ms": sample.analysis_sample_time_ms,
                        "source_pts": getattr(sample, "source_pts", None),
                        "center_2d": sample.center_2d,
                        "box": sample.derived_tracking_box,
                        "state": str(sample.tracking_state),
                    }
                    for sample in track.samples
                ],
            }
        )
    )
    if camera_intent not in {"hold", "follow"}:
        try:
            dynamic_filter, _, dynamic_audit = _virtual_camera_filter_from_track(
                track,
                times=times,
                boxes=boxes,
                source_width=source_width,
                source_height=source_height,
                requested_intent=camera_intent,
                maximum_scale=applied,
                geometry_safe_max_scale=safe_max,
                source_resolution_native_scale_limit=resolution_safe_max,
                source_track_fingerprint=track_fingerprint,
            )
        except ValueError as error:
            return _horizontal_original_filter(), _horizontal_reframe_failure_geometry(
                zoom_intent,
                fallback_reason=f"virtual_camera_preflight_failed:{error}",
                risk_code="virtual_camera_preflight_failed",
                diagnostics={
                    **diagnostics,
                    **lineage,
                    "requested_camera_intent": camera_intent,
                },
                geometry_safe_max_zoom=round(safe_max, 4),
            )
        return dynamic_filter, {
            "requested_zoom": requested,
            "geometry_safe_max_zoom": round(safe_max, 4),
            "resolution_safe_max_zoom": round(resolution_safe_max, 4),
            "applied_zoom": round(applied, 4),
            "requested_camera_intent": camera_intent,
            "fallback_reason": dynamic_audit[
                "virtual_camera_plan"
            ]["fallback_reason"],
            "risk_codes": (
                ["virtual_camera_intent_fallback"]
                if dynamic_audit["virtual_camera_plan"]["execution_status"]
                != "applied"
                else []
            ),
            "requires_gemini_review": (
                dynamic_audit["virtual_camera_plan"]["execution_status"]
                != "applied"
            ),
            **diagnostics,
            **lineage,
            **dynamic_audit,
        }
    x_values, y_values, crop_audit = _tracked_crop_geometry(
        times,
        centers_x,
        boxes,
        source_width=source_width,
        source_height=source_height,
        output_width=1920,
        output_height=1080,
        zoom=applied,
        safety_multiplier=1.45,
    )
    if (
        not crop_audit["full_containment_feasible"]
        or crop_audit["containment_failure_count"] != 0
    ):
        return _horizontal_original_filter(), _horizontal_reframe_failure_geometry(
            zoom_intent,
            fallback_reason="tracked_reframe_containment_gate_failed",
            risk_code="tracked_reframe_required_region_not_contained",
            diagnostics={**diagnostics, **lineage, **crop_audit},
            geometry_safe_max_zoom=round(safe_max, 4),
        )
    applied_camera_intent = camera_intent
    camera_execution_status: Literal["applied", "fallback", "blocked"] = "applied"
    camera_fallback_reason: str | None = None
    if camera_intent == "hold":
        keyframe_audit = crop_audit["crop_keyframes"]
        stable_left_low = max(
            float(item["legal_crop_left_min_normalized"])
            for item in keyframe_audit
        )
        stable_left_high = min(
            float(item["legal_crop_left_max_normalized"])
            for item in keyframe_audit
        )
        stable_top_low = max(
            float(item["legal_crop_top_min_normalized"])
            for item in keyframe_audit
        )
        stable_top_high = min(
            float(item["legal_crop_top_max_normalized"])
            for item in keyframe_audit
        )
        if (
            stable_left_low <= stable_left_high + 1e-6
            and stable_top_low <= stable_top_high + 1e-6
        ):
            coordinate_space = crop_audit["crop_coordinate_space"]
            scaled_width = int(coordinate_space["scaled_width"])
            scaled_height = int(coordinate_space["scaled_height"])
            preferred_left = sum(x_values) / len(x_values) * 1000 / scaled_width
            preferred_top = sum(y_values) / len(y_values) * 1000 / scaled_height
            stable_left = max(
                stable_left_low,
                min(stable_left_high, preferred_left),
            )
            stable_top = max(
                stable_top_low,
                min(stable_top_high, preferred_top),
            )
            x_values = [stable_left * scaled_width / 1000 for _ in times]
            y_values = [stable_top * scaled_height / 1000 for _ in times]
            crop_audit = {
                **crop_audit,
                "crop_x_values_pixels": x_values,
                "crop_y_values_pixels": y_values,
                "max_crop_x_speed_pixels_per_second": 0.0,
                "max_crop_y_speed_pixels_per_second": 0.0,
                "max_crop_speed_pixels_per_second": 0.0,
                "max_crop_acceleration_pixels_per_second_squared": 0.0,
                "max_crop_jerk_pixels_per_second_cubed": 0.0,
                "stable_hold_feasible": True,
                "crop_keyframes": [
                    {
                        **item,
                        "crop_x_pixels": round(x_values[index], 3),
                        "crop_y_pixels": round(y_values[index], 3),
                    }
                    for index, item in enumerate(keyframe_audit)
                ],
            }
        else:
            applied_camera_intent = "follow"
            camera_execution_status = "fallback"
            camera_fallback_reason = (
                "stable_hold_cannot_contain_locked_target_across_full_interval"
            )
            crop_audit = {
                **crop_audit,
                "stable_hold_feasible": False,
            }
    coordinate_space = crop_audit["crop_coordinate_space"]
    scaled_width = int(coordinate_space["scaled_width"])
    scaled_height = int(coordinate_space["scaled_height"])
    x_expression = _piecewise_expression(times, x_values)
    y_expression = _piecewise_expression(times, y_values)
    usable_samples = [
        sample
        for sample in track.samples
        if (
            sample.tracking_state == TrackingState.TRACKED
            and sample.center_2d is not None
            and sample.derived_tracking_box is not None
        )
    ]
    camera_keyframes = [
        VirtualCameraKeyframe(
            time_seconds=round(time, 6),
            source_pts=getattr(sample, "source_pts", None),
            scale=round(applied, 6),
            center_x_normalized=round(
                (crop_x + 960) / scaled_width * 1000,
                6,
            ),
            center_y_normalized=round(
                (crop_y + 540) / scaled_height * 1000,
                6,
            ),
        )
        for time, sample, crop_x, crop_y in zip(
            times,
            usable_samples,
            x_values,
            y_values,
            strict=True,
        )
    ]
    virtual_camera_plan = VirtualCameraPlan(
        requested_intent=camera_intent,
        applied_intent=applied_camera_intent,
        anchor_target_ids=[
            getattr(track, "target_id", None)
            or f"track:{track_fingerprint[:16]}"
        ],
        keyframes=camera_keyframes,
        easing="hold",
        geometry_safe_max_scale=round(safe_max, 6),
        source_resolution_native_scale_limit=round(resolution_safe_max, 6),
        source_resolution_upscale_required=(
            applied > resolution_safe_max + 0.001
        ),
        max_velocity=float(
            crop_audit["max_crop_speed_pixels_per_second"]
        ),
        max_acceleration=float(
            crop_audit["max_crop_acceleration_pixels_per_second_squared"]
        ),
        max_jerk=float(
            crop_audit["max_crop_jerk_pixels_per_second_cubed"]
        ),
        execution_status=camera_execution_status,
        fallback_reason=camera_fallback_reason,
        editorial_reason=(
            "Maintain stable scale while the safety path keeps the locked "
            "identity inside the composition."
        ),
        source_track_fingerprint=track_fingerprint,
    )
    return (
        f"[0:v]fps=30,scale={scaled_width}:{scaled_height},"
        f"crop=1920:1080:x='{x_expression}':y='{y_expression}',setsar=1[base]",
        {
            "requested_zoom": requested,
            "geometry_safe_max_zoom": round(safe_max, 4),
            "resolution_safe_max_zoom": round(resolution_safe_max, 4),
            "applied_zoom": round(applied, 4),
            "requested_camera_intent": camera_intent,
            "virtual_camera_plan": virtual_camera_plan.model_dump(mode="json"),
            "fallback_reason": camera_fallback_reason,
            "risk_codes": (
                ["virtual_camera_intent_fallback"]
                if camera_execution_status != "applied"
                else []
            ),
            "requires_gemini_review": camera_execution_status != "applied",
            **diagnostics,
            **lineage,
            **crop_audit,
        },
    )


def _horizontal_original_filter() -> str:
    return (
        "[0:v]fps=30,"
        "scale='max(2,trunc(iw*sar/2)*2)':ih,setsar=1,"
        "scale=1920:1080:force_original_aspect_ratio=increase,"
        "crop=1920:1080:x=(iw-ow)/2:y=(ih-oh)/2,setsar=1[base]"
    )


def _vertical_seed_anchor_fallback(
    tracks: Sequence[SegmentationTrack],
    *,
    source_width: int,
    source_height: int,
    coverage: dict[str, Any],
    allow_subject_clipping: bool,
    overflow_policy: Literal["preserve_all", "controlled_clip"],
    edge_priority: Literal["balanced", "preserve_start", "preserve_end"],
    fallback_strategy: Literal["fit_with_background", "center_crop"],
    failure_reason: str,
) -> tuple[str, dict[str, Any]] | None:
    """Use already-grounded seed geometry when propagation is incomplete.

    This is deliberately limited to the no-background fallback path.  It does
    not claim motion coverage: the static anchor is held for the shot and the
    result remains review-required.  The policy is domain-neutral and applies
    equally to subjects, text, UI, graphics, and other visible regions.
    """
    if fallback_strategy != "center_crop" or not tracks:
        return None
    seed_times = {track.seed_time_ms for track in tracks}
    if len(seed_times) != 1:
        return None
    anchor_boxes = [list(track.semantic_seed_box) for track in tracks]
    if any(len(box) != 4 for box in anchor_boxes):
        return None
    anchor_union = [
        min(box[0] for box in anchor_boxes),
        min(box[1] for box in anchor_boxes),
        max(box[2] for box in anchor_boxes),
        max(box[3] for box in anchor_boxes),
    ]
    start_ms = tracks[0].analysis_start_ms
    end_ms = tracks[0].analysis_end_ms
    if end_ms is None or end_ms <= start_ms:
        return None
    duration_seconds = (end_ms - start_ms) / 1000
    safety_multiplier = 1.0 if allow_subject_clipping else 1.08
    x_values, y_values, crop_audit = _tracked_crop_geometry(
        [0.0, duration_seconds],
        [(anchor_union[0] + anchor_union[2]) / 2] * 2,
        [anchor_union, anchor_union],
        source_width=source_width,
        source_height=source_height,
        output_width=1080,
        output_height=1920,
        safety_multiplier=safety_multiplier,
        overflow_policy=overflow_policy,
        edge_priority=edge_priority,
    )
    if overflow_policy == "preserve_all" and (
        not crop_audit["full_containment_feasible"]
        or crop_audit["containment_failure_count"] != 0
    ):
        return None
    controlled_clip_applied = bool(crop_audit["controlled_clip_applied"])
    risk_codes = [
        "required_region_tracking_coverage_failed",
        "seed_anchor_static_hold",
        "motion_outside_seed_unverified",
    ]
    if not coverage.get("tracking_confidence_gate_passed", True):
        risk_codes.append("required_region_low_confidence")
    if controlled_clip_applied:
        risk_codes.append("controlled_required_region_clip")
    if int(crop_audit["source_boundary_contact_count"]) > 0:
        risk_codes.extend(["source_boundary_contact", "not_recoverable_by_pan"])
    x_expression = _piecewise_expression([0.0, duration_seconds], x_values)
    y_expression = _piecewise_expression([0.0, duration_seconds], y_values)
    coordinate_space = crop_audit["crop_coordinate_space"]
    return (
        "[0:v]fps=30,"
        f"scale={coordinate_space['scaled_width']}:{coordinate_space['scaled_height']},"
        f"crop=1080:1920:x='{x_expression}':y='{y_expression}',setsar=1[base]",
        {
            "applied_strategy": "seed_anchor_crop",
            "fallback_reason": f"{failure_reason}_used_static_seed_anchor",
            "seed_anchor_time_ms": next(iter(seed_times)),
            "seed_anchor_union_box_2d": anchor_union,
            "subject_clipping_allowed": controlled_clip_applied,
            "secondary_context_clipping_allowed": allow_subject_clipping,
            "target_safety_multiplier": safety_multiplier,
            "risk_codes": list(dict.fromkeys(risk_codes)),
            "requires_gemini_review": True,
            "source_geometry_lineage_passed": True,
            "orientation_basis": "ffmpeg_autorotated_display",
            "source_display_width": source_width,
            "source_display_height": source_height,
            **coverage,
            **crop_audit,
        },
    )


def _vertical_filter_from_track(
    track: SegmentationTrack | Sequence[SegmentationTrack],
    *,
    allow_subject_clipping: bool = False,
    overflow_policy: Literal["preserve_all", "controlled_clip"] = "preserve_all",
    edge_priority: Literal["balanced", "preserve_start", "preserve_end"] = "balanced",
    region_ids: Sequence[str] | None = None,
    fallback_strategy: Literal["fit_with_background", "center_crop"] = (
        "fit_with_background"
    ),
    display_sample_aspect_ratio: float = 1.0,
    preferred_tracks: Sequence[SegmentationTrack] = (),
    preferred_regions: Sequence[FramingRegionIntent] = (),
) -> tuple[str, dict[str, Any]]:
    fallback_filter = (
        _vertical_center_crop_filter()
        if fallback_strategy == "center_crop"
        else _vertical_fit_filter()
    )
    tracks = [track] if isinstance(track, SegmentationTrack) else list(track)
    preferred_tracks = list(preferred_tracks)
    preferred_regions = list(preferred_regions)
    if len(preferred_tracks) != len(preferred_regions):
        raise ValueError("preferred tracks and region contracts must align")
    if not math.isclose(display_sample_aspect_ratio, 1.0, rel_tol=0, abs_tol=1e-6):
        return fallback_filter, {
            "applied_strategy": fallback_strategy,
            "fallback_reason": (
                "non_square_pixel_aspect_ratio_requires_static_reframe"
            ),
            "risk_codes": [
                "non_square_pixel_aspect_ratio_requires_static_reframe"
            ],
            "requires_gemini_review": True,
            "source_display_sample_aspect_ratio": round(
                display_sample_aspect_ratio, 9
            ),
            "sample_aspect_ratio_normalized_by_ffmpeg": True,
        }
    times, centers_x, boxes, coverage = _required_track_union(
        tracks,
        region_ids=region_ids,
    )
    try:
        source_width, source_height, lineage = (
            _orientation_corrected_track_dimensions(tracks)
        )
    except ValueError as error:
        return fallback_filter, {
            "applied_strategy": fallback_strategy,
            "fallback_reason": str(error),
            "risk_codes": ["track_source_geometry_mismatch"],
            "requires_gemini_review": True,
            "source_geometry_lineage_passed": False,
            **coverage,
        }
    confidence_gate_failed = not coverage["tracking_confidence_gate_passed"]
    coverage_risk_codes = ["required_region_unavailable"]
    if confidence_gate_failed:
        coverage_risk_codes.insert(0, "required_region_low_confidence")
    if len(times) < 2:
        failure_reason = (
            "required_region_tracking_confidence_failed"
            if confidence_gate_failed
            else "fewer_than_two_usable_tracking_samples"
        )
        anchor_fallback = _vertical_seed_anchor_fallback(
            tracks,
            source_width=source_width,
            source_height=source_height,
            coverage=coverage,
            allow_subject_clipping=allow_subject_clipping,
            overflow_policy=overflow_policy,
            edge_priority=edge_priority,
            fallback_strategy=fallback_strategy,
            failure_reason=failure_reason,
        )
        if anchor_fallback is not None:
            return anchor_fallback
        return fallback_filter, {
            "applied_strategy": fallback_strategy,
            "fallback_reason": failure_reason,
            "risk_codes": coverage_risk_codes,
            "requires_gemini_review": True,
            **lineage,
            **coverage,
        }
    if not coverage["coverage_passed"]:
        failure_reason = (
            "required_region_tracking_confidence_failed"
            if confidence_gate_failed
            else "required_region_tracking_coverage_failed"
        )
        anchor_fallback = _vertical_seed_anchor_fallback(
            tracks,
            source_width=source_width,
            source_height=source_height,
            coverage=coverage,
            allow_subject_clipping=allow_subject_clipping,
            overflow_policy=overflow_policy,
            edge_priority=edge_priority,
            fallback_strategy=fallback_strategy,
            failure_reason=failure_reason,
        )
        if anchor_fallback is not None:
            return anchor_fallback
        return fallback_filter, {
            "applied_strategy": fallback_strategy,
            "fallback_reason": failure_reason,
            "risk_codes": coverage_risk_codes,
            "requires_gemini_review": True,
            **lineage,
            **coverage,
        }
    composition_audit: dict[str, Any] = {
        "preferred_composition_used": False,
        "preferred_composition_reason": "no_preferred_regions",
    }
    desired_centers_y: list[float] | None = None
    if preferred_tracks:
        preferred_ids = [region.region_id for region in preferred_regions]
        try:
            (
                composition_times,
                composition_centers_x,
                composition_boxes,
                composition_coverage,
            ) = _required_track_union(
                [*tracks, *preferred_tracks],
                region_ids=[*(region_ids or []), *preferred_ids]
                if region_ids is not None
                else None,
            )
            if composition_times == times and composition_coverage["coverage_passed"]:
                centers_x = composition_centers_x
                desired_centers_y = [
                    (box[1] + box[3]) / 2 for box in composition_boxes
                ]
                composition_audit = {
                    "preferred_composition_used": True,
                    "preferred_composition_reason": "shared_coverage_available",
                    "preferred_region_ids": preferred_ids,
                }
            else:
                composition_audit = {
                    "preferred_composition_used": False,
                    "preferred_composition_reason": "preferred_coverage_not_aligned",
                    "preferred_region_ids": preferred_ids,
                    "preferred_composition_coverage": composition_coverage,
                }
        except ValueError as error:
            composition_audit = {
                "preferred_composition_used": False,
                "preferred_composition_reason": f"preferred_geometry_invalid:{error}",
                "preferred_region_ids": preferred_ids,
            }
    target_safety_multiplier = 1.0 if allow_subject_clipping else 1.08
    x_values, y_values, crop_audit = _tracked_crop_geometry(
        times,
        centers_x,
        boxes,
        source_width=source_width,
        source_height=source_height,
        output_width=1080,
        output_height=1920,
        safety_multiplier=target_safety_multiplier,
        overflow_policy=overflow_policy,
        edge_priority=edge_priority,
        desired_centers_y=desired_centers_y,
    )
    crop_width_normalized = float(crop_audit["crop_width_normalized"])
    crop_height_normalized = float(crop_audit["crop_height_normalized"])
    max_target_width = int(crop_audit["max_target_width_normalized"])
    max_target_height = int(crop_audit["max_target_height_normalized"])
    target_fits_legacy, _ = _vertical_target_fits_crop(
        max_target_width,
        crop_width_normalized,
        primary_center=allow_subject_clipping,
    )
    full_containment_feasible = bool(crop_audit["full_containment_feasible"])
    if overflow_policy == "preserve_all" and not full_containment_feasible:
        width_too_large = (
            max_target_width * target_safety_multiplier > crop_width_normalized
        )
        height_too_large = (
            max_target_height * target_safety_multiplier > crop_height_normalized
        )
        size_risk_codes = []
        if width_too_large:
            size_risk_codes.append("required_region_too_wide")
        if height_too_large:
            size_risk_codes.append("required_region_too_tall")
        if not size_risk_codes:
            size_risk_codes.append("required_region_not_containable")
        return fallback_filter, {
            "applied_strategy": fallback_strategy,
            "fallback_reason": "required_region_union_too_large_for_safe_9x16_crop",
            "subject_clipping_allowed": False,
            "secondary_context_clipping_allowed": allow_subject_clipping,
            "target_safety_multiplier": target_safety_multiplier,
            "legacy_max_width_gate_passed": target_fits_legacy,
            "risk_codes": size_risk_codes,
            "requires_gemini_review": True,
            **lineage,
            **coverage,
            **crop_audit,
        }
    if (
        overflow_policy == "preserve_all"
        and int(crop_audit["containment_failure_count"]) != 0
    ):
        return fallback_filter, {
            "applied_strategy": fallback_strategy,
            "fallback_reason": "required_region_containment_gate_failed",
            "subject_clipping_allowed": False,
            "secondary_context_clipping_allowed": allow_subject_clipping,
            "target_safety_multiplier": target_safety_multiplier,
            "legacy_max_width_gate_passed": target_fits_legacy,
            "risk_codes": ["required_region_not_contained"],
            "requires_gemini_review": True,
            **lineage,
            **coverage,
            **crop_audit,
        }
    x_expression = _piecewise_expression(times, x_values)
    y_expression = _piecewise_expression(times, y_values)
    controlled_clip_applied = bool(crop_audit["controlled_clip_applied"])
    risk_codes = ["controlled_required_region_clip"] if controlled_clip_applied else []
    soft_extent_audit: dict[str, Any] = {
        "soft_extent_count": 0,
        "soft_extent_visibility_passed": True,
        "soft_extent_regions": [],
    }
    if preferred_tracks:
        soft_extent_audit = _soft_extent_visibility_audit(
            tracks=preferred_tracks,
            regions=preferred_regions,
            crop_audit=crop_audit,
        )
        if not soft_extent_audit["soft_extent_visibility_passed"]:
            risk_codes.append("soft_extent_visibility_below_floor")
    edge_hold_warning_ms = float(coverage["expected_sample_interval_ms"]) * 0.55 + 35
    if (
        float(coverage["analysis_head_gap_ms"]) > edge_hold_warning_ms
        or float(coverage["analysis_tail_gap_ms"]) > edge_hold_warning_ms
    ):
        risk_codes.append("analysis_edge_hold_long")
    if int(crop_audit["source_boundary_contact_count"]) > 0:
        risk_codes.extend(["source_boundary_contact", "not_recoverable_by_pan"])
    if float(crop_audit["max_crop_speed_pixels_per_second"]) > 720:
        risk_codes.append("crop_motion_fast")
    if float(crop_audit["max_crop_acceleration_pixels_per_second_squared"]) > 1800:
        risk_codes.append("crop_motion_acceleration_high")
    coordinate_space = crop_audit["crop_coordinate_space"]
    return (
        "[0:v]fps=30,"
        f"scale={coordinate_space['scaled_width']}:{coordinate_space['scaled_height']},"
        f"crop=1080:1920:x='{x_expression}':y='{y_expression}',setsar=1[base]",
        {
            "applied_strategy": "tracked_crop",
            "fallback_reason": None,
            "subject_clipping_allowed": controlled_clip_applied,
            "secondary_context_clipping_allowed": allow_subject_clipping,
            "target_safety_multiplier": target_safety_multiplier,
            "legacy_max_width_gate_passed": target_fits_legacy,
            "risk_codes": risk_codes,
            "requires_gemini_review": bool(risk_codes),
            **lineage,
            **coverage,
            **composition_audit,
            **soft_extent_audit,
            **crop_audit,
        },
    )


def _visible_area_fraction(
    box: Sequence[float],
    *,
    crop_left: float,
    crop_top: float,
    crop_width: float,
    crop_height: float,
) -> float:
    x_min, y_min, x_max, y_max = (float(value) for value in box)
    visible_width = max(
        0.0,
        min(x_max, crop_left + crop_width) - max(x_min, crop_left),
    )
    visible_height = max(
        0.0,
        min(y_max, crop_top + crop_height) - max(y_min, crop_top),
    )
    return (
        visible_width
        / max(1e-6, x_max - x_min)
        * visible_height
        / max(1e-6, y_max - y_min)
    )


def _deadband_center(
    prior: tuple[float, float] | None,
    current: tuple[float, float],
    *,
    deadband_x: float,
    deadband_y: float,
) -> tuple[float, float]:
    """Move only enough to return a tracked target to the composition safe zone."""

    if prior is None:
        return current

    def resolve(previous: float, observed: float, radius: float) -> float:
        if observed < previous - radius:
            return observed + radius
        if observed > previous + radius:
            return observed - radius
        return previous

    return (
        resolve(prior[0], current[0], deadband_x),
        resolve(prior[1], current[1], deadband_y),
    )


def _minimum_smoothstep_transition_seconds(distance_pixels: float) -> float:
    """Return a conservative duration from smoothstep motion extrema."""

    if distance_pixels <= 1e-6:
        return 0.0
    return max(
        0.25,
        1.5 * distance_pixels / _PORTRAIT_MAX_SPEED_PX_S,
        math.sqrt(
            6.0 * distance_pixels / _PORTRAIT_MAX_ACCELERATION_PX_S2
        ),
        (
            12.0 * distance_pixels / _PORTRAIT_MAX_JERK_PX_S3
        )
        ** (1 / 3),
    )


def _vertical_virtual_camera_filter_from_tracks(
    *,
    tracks_by_region: Mapping[str, SegmentationTrack],
    phases: Sequence[VerticalVirtualCameraPhase],
    phase_origin: Literal["human_reviewed", "gemini_proposed"] = "human_reviewed",
    display_sample_aspect_ratio: float = 1.0,
) -> tuple[str, dict[str, Any]]:
    """Build a review-only phase-based 9:16 virtual camera.

    Each steady phase must fully contain its active, already-grounded anchors.
    A smooth transition may move between mutually exclusive anchors, but the
    audit measures both sides and requires at least one anchor to remain
    substantially visible. Exact source time still comes from SAM samples.
    """

    if not phases:
        raise ValueError("vertical virtual camera requires at least one phase")
    if not math.isclose(display_sample_aspect_ratio, 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError(
            "phase virtual camera requires square-pixel normalized tracking"
        )
    referenced_ids = list(
        dict.fromkeys(
            region_id
            for phase in phases
            for region_id in phase.anchor_region_ids
        )
    )
    missing = [
        region_id for region_id in referenced_ids if region_id not in tracks_by_region
    ]
    if missing:
        raise ValueError(
            "phase virtual camera has no track for: " + ", ".join(missing)
        )
    tracks = [tracks_by_region[region_id] for region_id in referenced_ids]
    source_width, source_height, lineage = _orientation_corrected_track_dimensions(
        tracks
    )
    starts = {track.analysis_start_ms for track in tracks}
    ends = {track.analysis_end_ms for track in tracks}
    rates = {float(track.analysis_fps) for track in tracks}
    if len(starts) != 1 or len(ends) != 1 or len(rates) != 1:
        raise ValueError("phase virtual-camera tracks must share one interval and rate")
    start_ms = starts.pop()
    end_ms = ends.pop()
    if end_ms is None or end_ms <= start_ms:
        raise ValueError("phase virtual-camera tracks require a positive interval")
    analysis_fps = rates.pop()
    boxes_by_region: dict[str, dict[int, list[int]]] = {}
    for region_id in referenced_ids:
        track = tracks_by_region[region_id]
        boxes_by_region[region_id] = {
            sample.analysis_sample_time_ms: [
                int(value) for value in sample.derived_tracking_box
            ]
            for sample in track.samples
            if sample.tracking_state == TrackingState.TRACKED
            and sample.derived_tracking_box is not None
        }
    exact_times_by_region = {
        region_id: sorted(boxes)
        for region_id, boxes in boxes_by_region.items()
    }
    imputed_anchor_samples: set[tuple[str, int]] = set()

    def resolved_box(region_id: str, time_ms: int) -> list[int] | None:
        exact = boxes_by_region[region_id].get(time_ms)
        if exact is not None:
            return exact
        known_times = exact_times_by_region[region_id]
        prior = max((value for value in known_times if value < time_ms), default=None)
        following = min(
            (value for value in known_times if value > time_ms),
            default=None,
        )
        maximum_gap_ms = (1000.0 / analysis_fps) * 1.6
        if (
            prior is not None
            and following is not None
            and time_ms - prior <= maximum_gap_ms
            and following - time_ms <= maximum_gap_ms
        ):
            alpha = (time_ms - prior) / (following - prior)
            before = boxes_by_region[region_id][prior]
            after = boxes_by_region[region_id][following]
            imputed_anchor_samples.add((region_id, time_ms))
            return [
                round(before[index] + (after[index] - before[index]) * alpha)
                for index in range(4)
            ]
        nearest = min(
            known_times,
            key=lambda value: abs(value - time_ms),
            default=None,
        )
        if nearest is not None and abs(nearest - time_ms) <= maximum_gap_ms:
            imputed_anchor_samples.add((region_id, time_ms))
            return boxes_by_region[region_id][nearest]
        return None
    all_times_ms = sorted(
        {
            time_ms
            for boxes in boxes_by_region.values()
            for time_ms in boxes
            if start_ms <= time_ms <= end_ms
        }
    )
    if len(all_times_ms) < 2:
        raise ValueError("phase virtual camera has fewer than two usable samples")
    duration_ms = end_ms - start_ms

    def phase_for_progress(progress: float) -> tuple[int, VerticalVirtualCameraPhase]:
        for index, phase in enumerate(phases):
            if progress < phase.end_progress - 1e-9 or index == len(phases) - 1:
                if progress + 1e-9 >= phase.start_progress:
                    return index, phase
        raise ValueError("phase virtual-camera progress is not covered")

    def union_box(
        region_ids: Sequence[str],
        time_ms: int,
    ) -> list[int] | None:
        members = [
            resolved_box(region_id, time_ms)
            for region_id in region_ids
        ]
        if any(member is None for member in members):
            return None
        resolved = [member for member in members if member is not None]
        return [
            min(box[0] for box in resolved),
            min(box[1] for box in resolved),
            max(box[2] for box in resolved),
            max(box[3] for box in resolved),
        ]

    transform = _cover_transform(
        source_width,
        source_height,
        1080,
        1920,
    )
    scaled_width = float(transform["scaled_width"])
    scaled_height = float(transform["scaled_height"])
    crop_width_normalized = 1080 / scaled_width * 1000
    crop_height_normalized = 1920 / scaled_height * 1000
    base_source_scale = max(
        scaled_width / source_width,
        scaled_height / source_height,
    )
    native_zoom_limit = max(1.0, 1.0 / max(1e-9, base_source_scale))
    maximum_zoom = min(_PORTRAIT_PHASE_MAX_ZOOM, native_zoom_limit)

    phase_centers: dict[str, tuple[float, float]] = {}
    for phase in phases:
        samples: list[tuple[int, list[int]]] = []
        for time_ms in all_times_ms:
            progress = max(0.0, min(1.0, (time_ms - start_ms) / duration_ms))
            _, active_phase = phase_for_progress(progress)
            if active_phase.phase_id != phase.phase_id:
                continue
            box = union_box(phase.anchor_region_ids, time_ms)
            if box is not None:
                samples.append((time_ms, box))
        if not samples:
            raise ValueError(
                f"phase {phase.phase_id} has no complete tracked anchor sample"
            )
        phase_centers[phase.phase_id] = (
            sum((box[0] + box[2]) / 2 for _, box in samples) / len(samples),
            sum((box[1] + box[3]) / 2 for _, box in samples) / len(samples),
        )

    effective_phases: list[VerticalVirtualCameraPhase] = []
    transition_audit: list[dict[str, Any]] = []
    for phase_index, phase in enumerate(phases):
        effective = phase
        if phase_index > 0 and phase.transition_in == "smoothstep":
            previous = phases[phase_index - 1]
            prior_center = phase_centers[previous.phase_id]
            current_center = phase_centers[phase.phase_id]
            distance_pixels = math.hypot(
                (current_center[0] - prior_center[0]) / 1000 * scaled_width,
                (current_center[1] - prior_center[1]) / 1000 * scaled_height,
            )
            phase_seconds = (
                (phase.end_progress - phase.start_progress)
                * duration_ms
                / 1000
            )
            requested_seconds = (
                phase_seconds * phase.transition_duration_fraction
            )
            minimum_seconds = _minimum_smoothstep_transition_seconds(
                distance_pixels
            )
            effective_seconds = max(requested_seconds, minimum_seconds)
            maximum_seconds = phase_seconds
            if effective_seconds > maximum_seconds + 1e-6:
                effective = phase.model_copy(
                    update={
                        "transition_in": "cut",
                        "transition_duration_fraction": 0.0,
                    }
                )
                effective_fraction = 0.0
                disposition = "converted_to_cut"
            else:
                effective_fraction = min(
                    1.0,
                    effective_seconds / max(0.001, phase_seconds),
                )
                effective = phase.model_copy(
                    update={
                        "transition_duration_fraction": effective_fraction,
                    }
                )
                disposition = "smoothstep"
            transition_audit.append(
                {
                    "phase_id": phase.phase_id,
                    "distance_pixels": round(distance_pixels, 6),
                    "requested_duration_seconds": round(requested_seconds, 6),
                    "minimum_safe_duration_seconds": round(minimum_seconds, 6),
                    "effective_duration_seconds": round(effective_seconds, 6),
                    "effective_duration_fraction": round(effective_fraction, 6),
                    "disposition": disposition,
                    "policy": "distance_aware_smoothstep_v1",
                }
            )
        effective_phases.append(effective)
    phases = effective_phases

    phase_scale_ranges: dict[str, tuple[float, float]] = {}
    carried_scale = 1.0
    for phase in phases:
        start_scale = carried_scale
        end_scale = carried_scale
        if phase.camera_behavior in {"push_in", "punch_in_cut"}:
            end_scale = maximum_zoom
        elif phase.camera_behavior == "pull_out":
            end_scale = 1.0
        phase_scale_ranges[phase.phase_id] = (start_scale, end_scale)
        carried_scale = end_scale

    times: list[float] = []
    x_values: list[float] = []
    y_values: list[float] = []
    scale_values: list[float] = []
    crop_keyframes: list[dict[str, Any]] = []
    steady_visible_fractions: list[float] = []
    transition_visible_fractions: list[float] = []
    transition_sample_count = 0
    missing_active_samples = 0
    active_anchor_samples: set[tuple[str, int]] = set()
    deadband_centers: dict[str, tuple[float, float]] = {}
    punch_cut_sample_indexes: set[int] = set()
    for time_ms in all_times_ms:
        progress = max(0.0, min(1.0, (time_ms - start_ms) / duration_ms))
        phase_index, phase = phase_for_progress(progress)
        active_anchor_samples.update(
            (region_id, time_ms) for region_id in phase.anchor_region_ids
        )
        current_box = union_box(phase.anchor_region_ids, time_ms)
        if current_box is None:
            missing_active_samples += 1
            continue
        current_center = (
            (current_box[0] + current_box[2]) / 2,
            (current_box[1] + current_box[3]) / 2,
        )
        if phase.camera_behavior in {
            "hold",
            "push_in",
            "pull_out",
            "punch_in_cut",
        }:
            current_center = phase_centers[phase.phase_id]
        elif phase.camera_behavior == "follow_deadband":
            current_center = _deadband_center(
                deadband_centers.get(phase.phase_id),
                current_center,
                deadband_x=(
                    crop_width_normalized
                    * _PORTRAIT_DEADBAND_VIEWPORT_FRACTION
                ),
                deadband_y=(
                    crop_height_normalized
                    * _PORTRAIT_DEADBAND_VIEWPORT_FRACTION
                ),
            )
            deadband_centers[phase.phase_id] = current_center

        phase_progress = (
            (progress - phase.start_progress)
            / max(1e-6, phase.end_progress - phase.start_progress)
        )
        phase_progress = max(0.0, min(1.0, phase_progress))
        phase_smoothstep = phase_progress * phase_progress * (
            3 - 2 * phase_progress
        )
        start_scale, end_scale = phase_scale_ranges[phase.phase_id]
        if phase.camera_behavior == "punch_in_cut":
            scale = start_scale if phase_progress < 0.5 else end_scale
        elif phase.camera_behavior in {"push_in", "pull_out"}:
            scale = start_scale + (end_scale - start_scale) * phase_smoothstep
        else:
            scale = start_scale
        transition = False
        previous_box: list[int] | None = None
        desired_center = current_center
        if phase_index > 0 and phase.transition_in == "smoothstep":
            phase_duration = phase.end_progress - phase.start_progress
            transition_end = phase.start_progress + (
                phase_duration * phase.transition_duration_fraction
            )
            if progress < transition_end - 1e-9:
                transition = True
                transition_sample_count += 1
                previous_phase = phases[phase_index - 1]
                previous_box = union_box(
                    previous_phase.anchor_region_ids,
                    time_ms,
                )
                previous_center = (
                    (
                        (previous_box[0] + previous_box[2]) / 2,
                        (previous_box[1] + previous_box[3]) / 2,
                    )
                    if previous_box is not None
                    else phase_centers[previous_phase.phase_id]
                )
                raw_alpha = (
                    (progress - phase.start_progress)
                    / max(1e-6, transition_end - phase.start_progress)
                )
                alpha = max(0.0, min(1.0, raw_alpha))
                alpha = alpha * alpha * (3 - 2 * alpha)
                desired_center = (
                    previous_center[0]
                    + (current_center[0] - previous_center[0]) * alpha,
                    previous_center[1]
                    + (current_center[1] - previous_center[1]) * alpha,
                )
        dynamic_scaled_width = scaled_width * scale
        dynamic_scaled_height = scaled_height * scale
        dynamic_crop_width_normalized = 1080 / dynamic_scaled_width * 1000
        dynamic_crop_height_normalized = 1920 / dynamic_scaled_height * 1000
        max_crop_left_pixels = max(0.0, dynamic_scaled_width - 1080)
        max_crop_top_pixels = max(0.0, dynamic_scaled_height - 1920)
        crop_left_pixels = max(
            0.0,
            min(
                max_crop_left_pixels,
                desired_center[0] / 1000 * dynamic_scaled_width - 540,
            ),
        )
        crop_top_pixels = max(
            0.0,
            min(
                max_crop_top_pixels,
                desired_center[1] / 1000 * dynamic_scaled_height - 960,
            ),
        )
        if not transition and phase.minimum_anchor_visible_fraction >= 1.0 - 1e-9:
            legal_left_low = max(
                0.0,
                current_box[2] / 1000 * dynamic_scaled_width - 1080,
            )
            legal_left_high = min(
                max_crop_left_pixels,
                current_box[0] / 1000 * dynamic_scaled_width,
            )
            legal_top_low = max(
                0.0,
                current_box[3] / 1000 * dynamic_scaled_height - 1920,
            )
            legal_top_high = min(
                max_crop_top_pixels,
                current_box[1] / 1000 * dynamic_scaled_height,
            )
            if (
                legal_left_low > legal_left_high + 1e-6
                or legal_top_low > legal_top_high + 1e-6
            ):
                raise ValueError(
                    f"phase {phase.phase_id} anchor union cannot fit 9:16"
                )
            crop_left_pixels = max(
                legal_left_low,
                min(legal_left_high, crop_left_pixels),
            )
            crop_top_pixels = max(
                legal_top_low,
                min(legal_top_high, crop_top_pixels),
            )
        crop_left = crop_left_pixels / dynamic_scaled_width * 1000
        crop_top = crop_top_pixels / dynamic_scaled_height * 1000
        current_visible = min(
            _visible_area_fraction(
                resolved_box(region_id, time_ms) or current_box,
                crop_left=crop_left,
                crop_top=crop_top,
                crop_width=dynamic_crop_width_normalized,
                crop_height=dynamic_crop_height_normalized,
            )
            for region_id in phase.anchor_region_ids
        )
        previous_visible = (
            _visible_area_fraction(
                previous_box,
                crop_left=crop_left,
                crop_top=crop_top,
                crop_width=dynamic_crop_width_normalized,
                crop_height=dynamic_crop_height_normalized,
            )
            if previous_box is not None
            else 0.0
        )
        if transition:
            transition_visible_fractions.append(
                max(current_visible, previous_visible)
            )
        else:
            if (
                current_visible + 1e-6
                < phase.minimum_anchor_visible_fraction
            ):
                raise ValueError(
                    f"phase {phase.phase_id} anchor visibility "
                    f"{current_visible:.3f} is below reviewed floor "
                    f"{phase.minimum_anchor_visible_fraction:.3f}"
                )
            steady_visible_fractions.append(current_visible)
        relative_seconds = (time_ms - start_ms) / 1000
        times.append(relative_seconds)
        x_values.append(crop_left_pixels)
        y_values.append(crop_top_pixels)
        scale_values.append(scale)
        if (
            phase.camera_behavior == "punch_in_cut"
            and len(scale_values) > 1
            and not math.isclose(
                scale_values[-1],
                scale_values[-2],
                rel_tol=0,
                abs_tol=1e-6,
            )
        ):
            punch_cut_sample_indexes.add(len(scale_values) - 1)
        crop_keyframes.append(
            {
                "time_seconds": round(relative_seconds, 6),
                "analysis_sample_time_ms": time_ms,
                "phase_id": phase.phase_id,
                "transition_sample": transition,
                "active_anchor_region_ids": list(phase.anchor_region_ids),
                "required_union_box": current_box,
                "previous_anchor_union_box": previous_box,
                "current_anchor_visible_fraction": round(current_visible, 6),
                "minimum_anchor_visible_fraction": (
                    phase.minimum_anchor_visible_fraction
                ),
                "previous_anchor_visible_fraction": round(previous_visible, 6),
                "scale": round(scale, 6),
                "crop_width_normalized": round(
                    dynamic_crop_width_normalized,
                    6,
                ),
                "crop_height_normalized": round(
                    dynamic_crop_height_normalized,
                    6,
                ),
                "crop_x_pixels": round(crop_left_pixels, 3),
                "crop_y_pixels": round(crop_top_pixels, 3),
            }
        )
    if len(times) < 2 or missing_active_samples:
        raise ValueError(
            "phase virtual-camera active anchors do not cover every analysis sample"
        )
    active_imputed_samples = imputed_anchor_samples & active_anchor_samples
    imputed_anchor_sample_ratio = len(active_imputed_samples) / max(
        1,
        len(active_anchor_samples),
    )
    if imputed_anchor_sample_ratio > 0.15:
        raise ValueError(
            "phase virtual-camera requires too many interpolated anchor samples"
        )
    steady_minimum = min(steady_visible_fractions, default=0.0)
    transition_minimum = min(transition_visible_fractions, default=1.0)
    if transition_minimum + 1e-6 < 0.10:
        raise ValueError(
            "phase virtual-camera transition loses both anchors"
        )
    phase_by_id = {phase.phase_id: phase for phase in phases}
    cut_before_indexes = frozenset(
        {
            index
            for index in range(1, len(crop_keyframes))
            if (
                crop_keyframes[index]["phase_id"]
                != crop_keyframes[index - 1]["phase_id"]
                and phase_by_id[crop_keyframes[index]["phase_id"]].transition_in
                == "cut"
            )
        }
        | punch_cut_sample_indexes
    )
    max_velocity, max_acceleration, max_jerk = _motion_extrema(
        times,
        x_values,
        y_values,
        scale_values,
        cut_before_indexes=cut_before_indexes,
    )
    keyframes = [
        VirtualCameraKeyframe(
            time_seconds=round(time, 6),
            source_pts=next(
                (
                    sample.source_pts
                    for sample in tracks[0].samples
                    if sample.analysis_sample_time_ms
                    == crop_keyframe["analysis_sample_time_ms"]
                ),
                None,
            ),
            scale=round(scale, 6),
            center_x_normalized=round(
                (crop_x + 540) / (scaled_width * scale) * 1000,
                6,
            ),
            center_y_normalized=round(
                (crop_y + 960) / (scaled_height * scale) * 1000,
                6,
            ),
        )
        for time, crop_x, crop_y, scale, crop_keyframe in zip(
            times,
            x_values,
            y_values,
            scale_values,
            crop_keyframes,
            strict=True,
        )
    ]
    plan = VerticalVirtualCameraPlan(
        phases=list(phases),
        anchor_region_ids=referenced_ids,
        keyframes=keyframes,
        steady_containment_passed=True,
        transition_minimum_anchor_visible_fraction=round(
            transition_minimum,
            6,
        ),
        max_velocity=round(max_velocity, 6),
        max_acceleration=round(max_acceleration, 6),
        max_jerk=round(max_jerk, 6),
        execution_status="applied",
        fallback_reason=None,
        editorial_reason=(
            "Apply geometry-validated phase-specific anchors instead of requiring "
            "every region to remain simultaneously visible for the whole segment."
        ),
        source_track_fingerprints={
            region_id: _track_geometry_fingerprint(tracks_by_region[region_id])
            for region_id in referenced_ids
        },
    )
    x_expression = _piecewise_expression(
        times,
        x_values,
        cut_before_indexes=cut_before_indexes,
    )
    y_expression = _piecewise_expression(
        times,
        y_values,
        cut_before_indexes=cut_before_indexes,
    )
    zoom_expression = _piecewise_expression(
        times,
        scale_values,
        cut_before_indexes=cut_before_indexes,
    )
    width_expression = (
        f"max(1080\\,2*trunc(({scaled_width:.3f}*"
        f"({zoom_expression}))/2))"
    )
    height_expression = (
        f"max(1920\\,2*trunc(({scaled_height:.3f}*"
        f"({zoom_expression}))/2))"
    )
    filter_graph = (
        "[0:v]fps=30,"
        f"scale=w='{width_expression}':h='{height_expression}':eval=frame,"
        f"crop=1080:1920:x='{x_expression}':y='{y_expression}',setsar=1[base]"
    )
    return filter_graph, {
        "applied_strategy": "phase_virtual_camera",
        "fallback_reason": None,
        "risk_codes": [
            (
                "human_reviewed_phase_virtual_camera"
                if phase_origin == "human_reviewed"
                else "gemini_proposed_geometry_validated_phase_virtual_camera"
            ),
            "phase_transition_containment_is_time_varying",
            *(
                ["phase_intentional_anchor_clipping"]
                if any(
                    phase.minimum_anchor_visible_fraction < 1.0
                    for phase in phases
                )
                else []
            ),
            *(
                ["phase_anchor_track_gap_interpolated"]
                if active_imputed_samples
                else []
            ),
        ],
        "requires_gemini_review": True,
        "full_containment_feasible": all(
            phase.minimum_anchor_visible_fraction >= 1.0
            for phase in phases
        ),
        "subject_clipping_allowed": any(
            phase.minimum_anchor_visible_fraction < 1.0
            for phase in phases
        ),
        "containment_failure_count": 0,
        "minimum_visible_required_area_fraction": round(steady_minimum, 6),
        "transition_minimum_anchor_visible_fraction": round(
            transition_minimum,
            6,
        ),
        "transition_sample_count": transition_sample_count,
        "distance_aware_transition_audit": transition_audit,
        "deadband_viewport_fraction": (
            _PORTRAIT_DEADBAND_VIEWPORT_FRACTION
        ),
        "native_zoom_limit": round(native_zoom_limit, 6),
        "maximum_applied_zoom": round(max(scale_values), 6),
        "minimum_applied_zoom": round(min(scale_values), 6),
        "scale_continuity_locked": math.isclose(
            min(scale_values),
            max(scale_values),
            rel_tol=0,
            abs_tol=1e-6,
        ),
        "zoom_limited_by_source_resolution": bool(
            maximum_zoom + 1e-6 < _PORTRAIT_PHASE_MAX_ZOOM
        ),
        "interpolated_anchor_sample_count": len(active_imputed_samples),
        "interpolated_anchor_sample_ratio": round(
            imputed_anchor_sample_ratio,
            6,
        ),
        "phase_virtual_camera_plan": plan.model_dump(mode="json"),
        "phase_virtual_camera_origin": phase_origin,
        "crop_coordinate_space": transform,
        "crop_width_normalized": round(crop_width_normalized, 6),
        "crop_height_normalized": round(crop_height_normalized, 6),
        "crop_x_values_pixels": x_values,
        "crop_y_values_pixels": y_values,
        "crop_scale_values": scale_values,
        "crop_keyframes": crop_keyframes,
        "max_crop_speed_pixels_per_second": round(max_velocity, 6),
        "max_crop_acceleration_pixels_per_second_squared": round(
            max_acceleration,
            6,
        ),
        "max_crop_jerk_pixels_per_second_cubed": round(max_jerk, 6),
        "tracking_confidence_gate_passed": True,
        "coverage_passed": True,
        "source_display_width": source_width,
        "source_display_height": source_height,
        **lineage,
    }


def _vertical_fit_filter() -> str:
    return (
        "[0:v]fps=30,"
        "scale='max(2,trunc(iw*sar/2)*2)':ih,setsar=1,"
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=0x0b0e12,setsar=1[base]"
    )


def _vertical_required_scope_fit_filter(
    geometry: Mapping[str, Any],
    *,
    margin_normalized: float = 45.0,
) -> tuple[str, dict[str, Any]] | None:
    """Build a static solid-matte fit around the tracked required envelope.

    A full-source fit preserves semantics but can make a horizontal subject
    occupy only a small strip inside 9:16.  When failed tracked-crop geometry
    already contains an auditable required-union box for every sampled frame,
    remove only source space that lies outside their all-frame envelope plus a
    fixed safety margin.  This never clips the required envelope and never
    invents a new target; it is a review fallback, not an approved crop.
    """

    keyframes = geometry.get("crop_keyframes")
    source_width = geometry.get("source_display_width")
    source_height = geometry.get("source_display_height")
    if (
        not isinstance(keyframes, list)
        or not keyframes
        or not isinstance(source_width, int)
        or not isinstance(source_height, int)
        or source_width <= 0
        or source_height <= 0
    ):
        return None
    boxes: list[list[float]] = []
    for keyframe in keyframes:
        box = (
            keyframe.get("required_union_box")
            if isinstance(keyframe, dict)
            else None
        )
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(isinstance(value, (int, float)) for value in box)
            or not (0 <= box[0] < box[2] <= 1000)
            or not (0 <= box[1] < box[3] <= 1000)
        ):
            return None
        boxes.append([float(value) for value in box])
    envelope = [
        max(0.0, min(box[0] for box in boxes) - margin_normalized),
        max(0.0, min(box[1] for box in boxes) - margin_normalized),
        min(1000.0, max(box[2] for box in boxes) + margin_normalized),
        min(1000.0, max(box[3] for box in boxes) + margin_normalized),
    ]
    if envelope[2] <= envelope[0] or envelope[3] <= envelope[1]:
        return None
    crop_x = max(0, round(source_width * envelope[0] / 1000))
    crop_y = max(0, round(source_height * envelope[1] / 1000))
    crop_width = max(
        2,
        int(source_width * (envelope[2] - envelope[0]) / 1000) // 2 * 2,
    )
    crop_height = max(
        2,
        int(source_height * (envelope[3] - envelope[1]) / 1000) // 2 * 2,
    )
    crop_width = min(crop_width, source_width - crop_x)
    crop_height = min(crop_height, source_height - crop_y)
    crop_width -= crop_width % 2
    crop_height -= crop_height % 2
    if crop_width < 2 or crop_height < 2:
        return None
    filter_graph = (
        "[0:v]fps=30,"
        "scale='max(2,trunc(iw*sar/2)*2)':ih,setsar=1,"
        f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},"
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=0x0b0e12,setsar=1[base]"
    )
    return filter_graph, {
        "applied_strategy": "required_scope_solid_fit",
        "fallback_reason": "required_region_union_too_large_for_safe_9x16_crop",
        "scope_fit_source": "tracked_required_union_all_frame_envelope",
        "scope_envelope_box_2d": [round(value, 3) for value in envelope],
        "scope_margin_normalized": margin_normalized,
        "scope_crop_pixels": {
            "x": crop_x,
            "y": crop_y,
            "width": crop_width,
            "height": crop_height,
        },
        "required_sample_count": len(boxes),
        "required_envelope_contained": True,
        "risk_codes": [
            "scope_preserving_solid_fit",
            "human_review_required",
        ],
        "requires_gemini_review": True,
        "source_geometry_lineage_passed": bool(
            geometry.get("source_geometry_lineage_passed")
        ),
        "tracking_confidence_gate_passed": bool(
            geometry.get("tracking_confidence_gate_passed")
        ),
    }


def _vertical_center_crop_filter() -> str:
    return (
        "[0:v]fps=30,"
        "scale='max(2,trunc(iw*sar/2)*2)':ih,setsar=1,"
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920:x=(iw-ow)/2:y=(ih-oh)/2,setsar=1[base]"
    )


def _vertical_delivery_fallback(
    strategy: Literal["fit_with_background", "center_crop"],
    *,
    reason: str,
) -> tuple[str, dict[str, Any]]:
    """Honor the explicit delivery preference while preserving review lineage."""

    if strategy == "center_crop":
        return _vertical_center_crop_filter(), {
            "applied_strategy": "full_bleed_center_crop_review",
            "fallback_reason": reason,
            "risk_codes": [
                "explicit_full_bleed_delivery_preference",
                "unverified_center_crop",
                "human_review_required",
            ],
            "requires_gemini_review": True,
            "full_bleed": True,
            "semantic_review_reasons": [
                "fallback_crop_requires_sequence_review",
            ],
        }
    return _vertical_fit_filter(), {
        "applied_strategy": "fit_with_solid_matte",
        "fallback_reason": reason,
        "risk_codes": [
            "scope_preserving_solid_fit",
            "human_review_required",
        ],
        "requires_gemini_review": True,
        "full_bleed": False,
        "semantic_review_reasons": [
            "scope_preserving_fit_is_review_only",
        ],
    }


def _tracking_seed_request_ms(
    frame: RushFrame,
    start_ms: int,
    end_ms: int,
) -> tuple[int, str]:
    if start_ms <= frame.requested_time_ms < end_ms:
        return frame.requested_time_ms, "catalog_anchor"
    return start_ms + (end_ms - start_ms) // 2, "trim_midpoint"


def _has_complete_cached_primary_track(output_dir: Path) -> bool:
    """A degraded composite fallback may only reuse complete local evidence."""

    return (
        any(output_dir.glob("grounding/bbox-*/grounding.json"))
        and any(output_dir.glob("sam21/bbox-*/segmentation-track.json"))
    )


def _is_non_retryable_spending_cap_error(error: Exception) -> bool:
    message = str(error).lower()
    return "spending cap" in message or "monthly spend" in message


def _is_exhausted_model_quota_error(error: Exception) -> bool:
    """Identify quota failures for which trying another candidate cannot help.

    Candidate switching is useful for evidence and geometry failures. It only
    repeats the same unavailable service call after the upstream API has
    returned a rate-limit, quota, or account spending failure.
    """

    message = str(error).lower()
    status_values = (
        getattr(error, "status_code", None),
        getattr(error, "code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    )
    explicit_429 = any(
        value == 429 or str(value).strip() == "429"
        for value in status_values
        if value is not None
    ) or re.match(r"^\s*(?:http(?:\s+status)?\s*)?429(?:\b|:)", message)
    return _is_non_retryable_spending_cap_error(error) or any(
        marker in message
        for marker in (
            "resource_exhausted",
            "resource exhausted",
            "quota exceeded",
            "rate limit exceeded",
            "too many requests",
        )
    ) or bool(explicit_429)


class GeometryModelQuotaError(RuntimeError):
    """Geometry processing stopped because another candidate cannot help."""


def _ground_tracking_seed(
    *,
    client: GeminiLabClient,
    clip: RushClip,
    frame: RushFrame,
    start_ms: int,
    end_ms: int,
    feature_id: str,
    event_description: str,
    entity_id: str,
    target_description: str,
    grounding_prompt: str,
    output_dir: Path,
    run_id: str,
    model_request_block_reason: str | None = None,
    query_lock_v2: EvidenceQueryLockV2 | None = None,
) -> tuple[GroundingProposal, Any, Any, Any, Path, int, str]:
    """Ground one immutable semantic region on one exact decoded source frame."""
    exact_frame_path = output_dir / "evidence-frame.png"
    seed_requested_time_ms, seed_anchor_source = _tracking_seed_request_ms(
        frame,
        start_ms,
        end_ms,
    )
    exact_frame = extract_frame(
        Path(clip.path),
        seed_requested_time_ms,
        exact_frame_path,
    )
    media = probe_video(Path(clip.path))
    query_lineage: dict[str, Any] | None = None
    grounding_target_description = target_description
    grounding_event_description = event_description
    grounding_key = {
            "contract_version": (
                "exact-frame-grounding-v2"
                if entity_id == "reframe_subject"
                else "exact-frame-grounding-v3-region-intent"
            ),
            "model_id": MODEL_ID,
            "source_asset_id": media.asset_id,
            "feature_id": feature_id,
            "frame_hash": exact_frame.frame_hash,
            "frame_pts": exact_frame.frame_pts,
            "frame_time_ms": exact_frame.frame_time_ms,
            "source_width": exact_frame.width,
            "source_height": exact_frame.height,
            "entity_id": entity_id,
            "event_description": event_description,
            "target_description": target_description,
            "prompt_sha256": hashlib.sha256(
                grounding_prompt.encode("utf-8")
            ).hexdigest(),
            "system_instruction_sha256": hashlib.sha256(
                VISUAL_EVIDENCE_SYSTEM_INSTRUCTION.encode("utf-8")
            ).hexdigest(),
            "response_schema_sha256": hashlib.sha256(
                json.dumps(
                    gemini_response_schema(GeminiNativeGroundingProposal),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "thinking_level": "low",
        }
    if query_lock_v2 is not None:
        query_lineage = evidence_query_lock_v2_lineage(
            query_lock_v2,
            target_id=entity_id,
            target_description=target_description,
        )
        identity = next(
            item for item in query_lock_v2.identity.targets if item.target_id == entity_id
        )
        identity_parts = [identity.target_description]
        if identity.identity_cues:
            identity_parts.append("identity cues: " + "; ".join(identity.identity_cues))
        if identity.context_cues:
            identity_parts.append(
                "context cues are auxiliary only: " + "; ".join(identity.context_cues)
            )
        if identity.stable_exclusions:
            identity_parts.append(
                "must exclude: " + "; ".join(identity.stable_exclusions)
            )
        grounding_target_description = "\n".join(identity_parts)
        grounding_event_description = (
            "Exact-frame identity Grounding only. Use only the supplied image pixels "
            "and locked identity cues; do not infer an event predicate or position "
            "from time, neighboring frames, or editorial prose."
        )
        grounding_key["evidence_query_v2_identity"] = {
            "contract_version": "evidence-query-v2-grounding-identity-v1",
            "identity_sha256": query_lock_v2.component_hashes()["identity_sha256"],
            "target_id": entity_id,
        }
        grounding_key["target_description"] = grounding_target_description
        grounding_key["event_description"] = grounding_event_description
    if seed_anchor_source != "catalog_anchor":
        grounding_key.update(
            {
                "catalog_frame_id": frame.frame_id,
                "catalog_requested_time_ms": frame.requested_time_ms,
                "seed_anchor_source": seed_anchor_source,
                "seed_requested_time_ms": seed_requested_time_ms,
            }
        )
    grounding_fingerprint = hashlib.sha256(
        json.dumps(
            grounding_key,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    grounding_dir = output_dir / "grounding" / f"bbox-{grounding_fingerprint[:16]}"
    write_json(
        grounding_dir / "request-key.json",
        {
            **grounding_key,
            "request_fingerprint": grounding_fingerprint,
        },
    )
    if query_lineage is not None:
        write_json(
            grounding_dir
            / f"query-lineage-{query_lineage['definition_sha256'][:16]}.json",
            query_lineage,
        )
    grounding_path = grounding_dir / "grounding.json"
    if grounding_path.exists():
        proposal = GroundingProposal.model_validate(read_json(grounding_path))
        require_grounding_request_match(
            proposal,
            asset_id=media.asset_id,
            event_id=feature_id,
            entity_id=entity_id,
            frame_pts=exact_frame.frame_pts,
            frame_time_ms=exact_frame.frame_time_ms,
            frame_hash=exact_frame.frame_hash,
            source_width=exact_frame.width,
            source_height=exact_frame.height,
            model_id=MODEL_ID,
        )
        frame_time_ms = proposal.frame_time_ms
    else:
        if model_request_block_reason:
            raise RuntimeError(
                "Gemini Grounding request skipped by run-level circuit breaker: "
                + model_request_block_reason
            )
        proposal = client.ground_frame(
            media=media,
            frame=exact_frame,
            event_id=feature_id,
            event_description=grounding_event_description,
            entity_id=entity_id,
            target_description=grounding_target_description,
            prompt_template=grounding_prompt,
            run_id=run_id,
            output_dir=grounding_dir,
        )
        frame_time_ms = exact_frame.frame_time_ms
    debug_path = grounding_dir / "debug.png"
    if not debug_path.exists():
        draw_grounding_overlay(exact_frame_path, proposal, debug_path)
    if debug_path.exists():
        shutil.copy2(debug_path, output_dir / "grounding-debug.png")
    if not proposal.visible or not proposal.candidates:
        raise ValueError(f"Gemini could not ground required region {entity_id} for {feature_id}")
    selected_seed = require_tracking_seed_candidate(proposal)
    return (
        proposal,
        selected_seed,
        exact_frame,
        media,
        grounding_path,
        frame_time_ms,
        seed_anchor_source,
    )


def _build_track(
    *,
    client: GeminiLabClient,
    clip: RushClip,
    frame: RushFrame,
    start_ms: int,
    end_ms: int,
    feature_id: str,
    event_description: str,
    target_description: str,
    checkpoint_path: Path,
    grounding_prompt: str,
    output_dir: Path,
    run_id: str,
    analysis_fps: float,
    scdet_threshold: float,
    entity_id: str = "reframe_subject",
    model_request_block_reason: str | None = None,
    query_lock_v2: EvidenceQueryLockV2 | None = None,
) -> tuple[GroundingProposal, SegmentationTrack]:
    track_root = output_dir / "sam21"
    (
        proposal,
        selected_seed,
        exact_frame,
        media,
        _,
        frame_time_ms,
        seed_anchor_source,
    ) = _ground_tracking_seed(
        client=client,
        clip=clip,
        frame=frame,
        start_ms=start_ms,
        end_ms=end_ms,
        feature_id=feature_id,
        event_description=event_description,
        entity_id=entity_id,
        target_description=target_description,
        grounding_prompt=grounding_prompt,
        output_dir=output_dir,
        run_id=run_id,
        model_request_block_reason=model_request_block_reason,
        query_lock_v2=query_lock_v2,
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    seed_manifest = {
            "contract_version": "bbox-seed-v2-exact-pts",
            "asset_id": proposal.asset_id,
            "event_id": proposal.event_id,
            "entity_id": proposal.entity_id,
            "target_description": target_description,
            "frame_hash": proposal.frame_hash,
            "frame_pts": proposal.frame_pts,
            "candidate_number": selected_seed.candidate_number,
            "candidate_index": selected_seed.candidate_index,
            "candidate_selection_source": selected_seed.selection_source,
            "box_2d": list(selected_seed.candidate.box_2d),
            "seed_type": "gemini_bbox",
            "source_start_ms": start_ms,
            "source_end_ms": end_ms,
            "normalized_seed_shot_start_ms": start_ms,
            "normalized_seed_shot_end_ms": end_ms,
            "analysis_fps": analysis_fps,
            "analysis_max_side": _TRACKING_MAX_SIDE,
            "ffmpeg_scdet_threshold": scdet_threshold,
            "seed_box_padding_ratio": _TRACKING_SEED_BOX_PADDING_RATIO,
            "device_request": _TRACKING_DEVICE,
            "sam_config": SAM21_CONFIG,
            "sam_implementation_revision": SAM21_IMPLEMENTATION_REVISION,
            "checkpoint_sha256": checkpoint_sha256,
        }
    query_lineage: dict[str, Any] | None = None
    if query_lock_v2 is not None:
        query_lineage = evidence_query_lock_v2_lineage(
            query_lock_v2,
            target_id=entity_id,
            target_description=target_description,
        )
        seed_manifest["evidence_query_v2_identity"] = {
            "contract_version": "evidence-query-v2-sam-seed-identity-v1",
            "identity_sha256": query_lock_v2.component_hashes()["identity_sha256"],
            "target_id": entity_id,
        }
    seed_fingerprint = hashlib.sha256(
        json.dumps(seed_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    track_dir = track_root / f"bbox-{seed_fingerprint[:16]}"
    seed_manifest_path = track_dir / "seed-selection.json"
    write_json(
        seed_manifest_path,
        {
            **seed_manifest,
            "seed_fingerprint": seed_fingerprint,
        },
    )
    if query_lineage is not None:
        write_json(
            track_dir
            / f"query-lineage-{query_lineage['definition_sha256'][:16]}.json",
            query_lineage,
        )
    track_path = track_dir / "segmentation-track.json"
    if track_path.exists():
        track = SegmentationTrack.model_validate(read_json(track_path))
    else:
        track = track_bbox_sam21(
            video_path=Path(clip.path),
            checkpoint_path=checkpoint_path,
            seed_time_ms=frame_time_ms,
            seed_box_2d=selected_seed.candidate.box_2d,
            target_description=target_description,
            output_dir=track_dir,
            seed_source=str(seed_manifest_path),
            asset_id=proposal.asset_id,
            seed_frame_pts=proposal.frame_pts,
            seed_frame_sha256=proposal.frame_hash,
            seed_source_width=proposal.source_width,
            seed_source_height=proposal.source_height,
            analysis_fps=analysis_fps,
            max_side=_TRACKING_MAX_SIDE,
            device=_TRACKING_DEVICE,
            ffmpeg_scdet_threshold=scdet_threshold,
            seed_box_padding_ratio=_TRACKING_SEED_BOX_PADDING_RATIO,
            allowed_start_ms=start_ms,
            allowed_end_ms=end_ms,
        )
    require_bbox_track_request_match(
        track,
        video_path=Path(clip.path),
        asset_id=proposal.asset_id,
        target_description=target_description,
        seed_time_ms=frame_time_ms,
        seed_box_2d=selected_seed.candidate.box_2d,
        seed_box_padding_ratio=_TRACKING_SEED_BOX_PADDING_RATIO,
        analysis_fps=analysis_fps,
        analysis_start_ms=start_ms,
        analysis_end_ms=end_ms,
        checkpoint_sha256=checkpoint_sha256,
        seed_frame_pts=proposal.frame_pts,
        seed_frame_sha256=proposal.frame_hash,
        seed_source_width=proposal.source_width,
        seed_source_height=proposal.source_height,
    )
    if query_lock_v2 is not None:
        runtime_lineage = _query_lock_v2_runtime_geometry_lineage(
            lock=query_lock_v2,
            target_id=entity_id,
            target_description=target_description,
            seed_fingerprint=seed_fingerprint,
            seed_manifest_path=seed_manifest_path,
            track_path=track_path,
            track=track,
        )
        runtime_lineage_path = (
            track_dir
            / "runtime-geometry-lineage-"
            f"{runtime_lineage['evidence_query_v2']['definition_sha256'][:16]}.json"
        )
        if (
            runtime_lineage_path.exists()
            and read_json(runtime_lineage_path) != runtime_lineage
        ):
            raise ValueError(
                "cached QueryLock v2 runtime geometry lineage does not match request"
            )
        write_json(runtime_lineage_path, runtime_lineage)
    return proposal, track


def _contained_shared_session_artifact(
    session_dir: Path,
    artifact_path: str,
    *,
    artifact_kind: str,
) -> Path:
    """Resolve a cached shared-session artifact without permitting path escape."""
    root = session_dir.expanduser().resolve(strict=True)
    resolved = (root / artifact_path).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"cached shared SAM {artifact_kind} escapes its session: {artifact_path}"
        ) from error
    if not resolved.is_file():
        raise ValueError(
            f"cached shared SAM {artifact_kind} is not a file: {artifact_path}"
        )
    return resolved


def _validate_shared_sam_session_cache(
    *,
    manifest: SharedSam21SessionManifest,
    session_dir: Path,
    video_path: Path,
    asset_id: str,
    start_ms: int,
    end_ms: int,
    analysis_fps: float,
    analysis_max_side: int,
    checkpoint_sha256: str,
    seeds: Sequence[SharedSam21BBoxSeed],
    seed_box_padding_ratio: float,
) -> list[SegmentationTrack]:
    """Validate immutable decode, model, seed, and track lineage before cache reuse."""
    resolved_video = video_path.expanduser().resolve(strict=True)
    manifest_video = Path(manifest.video_path).expanduser().resolve(strict=True)
    expected_ids = [seed.target_id for seed in seeds]
    actual_ids = [target.target_id for target in manifest.targets]
    mismatches: list[str] = []
    if manifest.asset_id != asset_id:
        mismatches.append("asset_id")
    if manifest_video != resolved_video:
        mismatches.append("video_path")
    if manifest.analysis_start_ms != start_ms:
        mismatches.append("analysis_start_ms")
    if manifest.analysis_end_ms != end_ms:
        mismatches.append("analysis_end_ms")
    if manifest.analysis_fps != analysis_fps:
        mismatches.append("analysis_fps")
    if max(manifest.analysis_width, manifest.analysis_height) > analysis_max_side:
        mismatches.append("analysis_dimensions")
    if actual_ids != expected_ids:
        mismatches.append("target_order")
    provenance = manifest.model_provenance
    if provenance.model_id != SAM21_TINY_MODEL_ID:
        mismatches.append("model_id")
    if provenance.implementation != "facebookresearch/sam2":
        mismatches.append("implementation")
    if provenance.implementation_revision != SAM21_IMPLEMENTATION_REVISION:
        mismatches.append("implementation_revision")
    if provenance.checkpoint_sha256 != checkpoint_sha256:
        mismatches.append("checkpoint_sha256")
    if mismatches:
        raise ValueError(
            "cached shared SAM session does not match request: "
            + ", ".join(mismatches)
        )

    frames_manifest_path = _contained_shared_session_artifact(
        session_dir,
        manifest.analysis_frames_path,
        artifact_kind="analysis frame manifest",
    )
    if sha256_file(frames_manifest_path) != manifest.analysis_frames_manifest_sha256:
        raise ValueError("cached shared SAM analysis frame manifest hash mismatch")
    frames_manifest = SharedSam21AnalysisFramesManifest.model_validate(
        read_json(frames_manifest_path)
    )
    if frames_manifest.frames != manifest.analysis_frames:
        raise ValueError(
            "cached shared SAM analysis frame manifest does not match session manifest"
        )
    for frame in frames_manifest.frames:
        frame_path = _contained_shared_session_artifact(
            session_dir,
            frame.path,
            artifact_kind="analysis frame",
        )
        if sha256_file(frame_path) != frame.sha256:
            raise ValueError(f"cached shared SAM analysis frame hash mismatch: {frame.path}")

    tracks: list[SegmentationTrack] = []
    for seed, member in zip(seeds, manifest.targets, strict=True):
        track_path = _contained_shared_session_artifact(
            session_dir,
            member.track_path,
            artifact_kind="track",
        )
        if sha256_file(track_path) != member.track_sha256:
            raise ValueError(f"cached shared SAM track hash mismatch: {member.target_id}")
        track = SegmentationTrack.model_validate(read_json(track_path))
        expected_prompt_box = pad_normalized_box(
            seed.seed_box_2d, seed_box_padding_ratio
        )
        track_mismatches: list[str] = []
        expected_values = {
            "asset_id": asset_id,
            "video_path": str(resolved_video),
            "target_id": seed.target_id,
            "target_description": seed.target_description,
            "seed_source": seed.seed_source,
            "seed_time_ms": seed.seed_time_ms,
            "seed_frame_pts": seed.seed_frame_pts,
            "seed_frame_sha256": seed.seed_frame_sha256,
            "seed_source_width": seed.seed_source_width,
            "seed_source_height": seed.seed_source_height,
            "semantic_seed_box": seed.seed_box_2d,
            "seed_prompt_type": "box",
            "sam_prompt_box": expected_prompt_box,
            "seed_box_padding_ratio": seed_box_padding_ratio,
            "analysis_fps": analysis_fps,
            "analysis_width": manifest.analysis_width,
            "analysis_height": manifest.analysis_height,
            "analysis_start_ms": start_ms,
            "analysis_end_ms": end_ms,
            "source_start_pts": manifest.source_start_pts,
            "source_time_base": manifest.source_time_base,
            "total_samples": len(manifest.analysis_frames),
            "state_counts": member.state_counts,
            "shared_session_id": manifest.session_id,
            "analysis_frames_manifest_sha256": (
                manifest.analysis_frames_manifest_sha256
            ),
        }
        for field, expected in expected_values.items():
            actual = getattr(track, field)
            if field == "video_path":
                actual = str(Path(actual).expanduser().resolve(strict=True))
            if actual != expected:
                track_mismatches.append(field)
        if track.model_provenance != provenance:
            track_mismatches.append("model_provenance")
        if member.target_description != seed.target_description:
            track_mismatches.append("member.target_description")
        if member.seed_time_ms != seed.seed_time_ms:
            track_mismatches.append("member.seed_time_ms")
        if member.seed_frame_pts != seed.seed_frame_pts:
            track_mismatches.append("member.seed_frame_pts")
        if member.seed_frame_sha256 != seed.seed_frame_sha256:
            track_mismatches.append("member.seed_frame_sha256")
        if member.seed_source_width != seed.seed_source_width:
            track_mismatches.append("member.seed_source_width")
        if member.seed_source_height != seed.seed_source_height:
            track_mismatches.append("member.seed_source_height")
        if track_mismatches:
            raise ValueError(
                f"cached shared SAM track {seed.target_id!r} does not match request: "
                + ", ".join(track_mismatches)
            )
        tracks.append(track)

    validate_segmentation_track_alignment(tracks)
    return tracks


def _build_framing_region_tracks(
    *,
    client: GeminiLabClient,
    clip: RushClip,
    frame: RushFrame,
    start_ms: int,
    end_ms: int,
    feature_id: str,
    event_description: str,
    regions: Sequence[FramingRegionIntent],
    checkpoint_path: Path,
    grounding_prompt: str,
    output_dir: Path,
    analysis_fps: float,
    scdet_threshold: float,
    include_execution_roles: frozenset[
        Literal["hard_core", "soft_extent", "overlay_keepout"]
    ] = frozenset({"hard_core"}),
    model_request_block_reason: str | None = None,
    query_lock_v2: EvidenceQueryLockV2 | None = None,
) -> tuple[list[GroundingProposal], list[SegmentationTrack], list[Path]]:
    """Ground named regions separately and share one SAM session when possible."""
    tracked_regions = [
        region
        for region in regions
        if region.execution_role in include_execution_roles
    ]
    if not tracked_regions:
        raise ValueError("a tracked portrait crop needs at least one selected region")
    if len(tracked_regions) == 1:
        region = tracked_regions[0]
        region_root = output_dir / "regions" / region.region_id
        proposal, track = _build_track(
            client=client,
            clip=clip,
            frame=frame,
            start_ms=start_ms,
            end_ms=end_ms,
            feature_id=feature_id,
            event_description=event_description,
            target_description=region.target_description,
            checkpoint_path=checkpoint_path,
            grounding_prompt=grounding_prompt,
            output_dir=region_root,
            run_id=f"feature-v-{region.region_id}-{uuid.uuid4().hex[:8]}",
            analysis_fps=analysis_fps,
            scdet_threshold=scdet_threshold,
            entity_id=region.entity_id or f"reframe_{region.region_id}",
            model_request_block_reason=model_request_block_reason,
            query_lock_v2=query_lock_v2,
        )
        return [proposal], [track], [region_root / "grounding-debug.png"]

    proposals: list[GroundingProposal] = []
    seeds: list[SharedSam21BBoxSeed] = []
    debug_paths: list[Path] = []
    for region in tracked_regions:
        region_root = output_dir / "regions" / region.region_id
        (
            proposal,
            selected_seed,
            _,
            _,
            grounding_path,
            frame_time_ms,
            _,
        ) = _ground_tracking_seed(
            client=client,
            clip=clip,
            frame=frame,
            start_ms=start_ms,
            end_ms=end_ms,
            feature_id=feature_id,
            event_description=event_description,
            entity_id=region.entity_id or f"reframe_{region.region_id}",
            target_description=region.target_description,
            grounding_prompt=grounding_prompt,
            output_dir=region_root,
            run_id=f"feature-v-{region.region_id}-{uuid.uuid4().hex[:8]}",
            model_request_block_reason=model_request_block_reason,
            query_lock_v2=query_lock_v2,
        )
        proposals.append(proposal)
        debug_paths.append(region_root / "grounding-debug.png")
        seeds.append(
            SharedSam21BBoxSeed(
                target_id=region.region_id,
                target_description=region.target_description,
                seed_source=str(grounding_path.resolve()),
                seed_time_ms=frame_time_ms,
                seed_frame_pts=proposal.frame_pts,
                seed_frame_sha256=proposal.frame_hash,
                seed_source_width=proposal.source_width,
                seed_source_height=proposal.source_height,
                seed_box_2d=list(selected_seed.candidate.box_2d),
            )
        )

    request_key = {
        "contract_version": "feature-cut-shared-framing-regions-v2",
        "asset_id": proposals[0].asset_id,
        "video_path": str(Path(clip.path).expanduser().resolve()),
        "feature_id": feature_id,
        "source_start_ms": start_ms,
        "source_end_ms": end_ms,
        "analysis_fps": analysis_fps,
        "analysis_max_side": _TRACKING_MAX_SIDE,
        "ffmpeg_scdet_threshold": scdet_threshold,
        "seed_box_padding_ratio": _TRACKING_SEED_BOX_PADDING_RATIO,
        "device_request": _TRACKING_DEVICE,
        "sam_config": SAM21_CONFIG,
        "sam_implementation_revision": SAM21_IMPLEMENTATION_REVISION,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "targets": [seed.model_dump(mode="json") for seed in seeds],
    }
    request_fingerprint = hashlib.sha256(
        json.dumps(request_key, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    session_parent = output_dir / "shared-sam21"
    session_dir = session_parent / f"session-{request_fingerprint[:16]}"
    manifest_path = session_dir / "shared-session.json"
    if manifest_path.exists():
        manifest = SharedSam21SessionManifest.model_validate(read_json(manifest_path))
    else:
        if session_dir.exists() and any(session_dir.iterdir()):
            raise RuntimeError(f"incomplete shared SAM session: {session_dir}")
        manifest = track_bboxes_shared_sam21(
            video_path=Path(clip.path),
            checkpoint_path=checkpoint_path,
            targets=seeds,
            output_dir=session_dir,
            asset_id=proposals[0].asset_id,
            analysis_fps=analysis_fps,
            max_side=_TRACKING_MAX_SIDE,
            device=_TRACKING_DEVICE,
            ffmpeg_scdet_threshold=scdet_threshold,
            seed_box_padding_ratio=_TRACKING_SEED_BOX_PADDING_RATIO,
            allowed_start_ms=start_ms,
            allowed_end_ms=end_ms,
        )
    session_parent.mkdir(parents=True, exist_ok=True)
    write_json(
        session_parent / f"session-{request_fingerprint[:16]}.request.json",
        {**request_key, "request_fingerprint": request_fingerprint},
    )
    tracks = _validate_shared_sam_session_cache(
        manifest=manifest,
        session_dir=session_dir,
        video_path=Path(clip.path),
        asset_id=proposals[0].asset_id,
        start_ms=start_ms,
        end_ms=end_ms,
        analysis_fps=analysis_fps,
        analysis_max_side=_TRACKING_MAX_SIDE,
        checkpoint_sha256=request_key["checkpoint_sha256"],
        seeds=seeds,
        seed_box_padding_ratio=_TRACKING_SEED_BOX_PADDING_RATIO,
    )
    if query_lock_v2 is not None:
        full_lineage = evidence_query_lock_v2_lineage(query_lock_v2)
        target_mappings = [
            {
                "region_id": region.region_id,
                "grounding_entity_id": (
                    region.entity_id or f"reframe_{region.region_id}"
                ),
                "query_target_id": _feature_candidate_query_target_id(region),
                "sam_target_id": seed.target_id,
            }
            for region, seed in zip(tracked_regions, seeds, strict=True)
        ]
        write_json(
            session_dir
            / f"runtime-query-lineage-{full_lineage['definition_sha256'][:16]}.json",
            {
                "contract_version": "feature-shared-sam-query-lineage-v1",
                "evidence_query_v2": full_lineage,
                "shared_session_manifest_sha256": sha256_file(manifest_path),
                "target_ids": [seed.target_id for seed in seeds],
                "target_namespace_mapping": target_mappings,
            },
        )
    return proposals, tracks, debug_paths


def _build_required_region_tracks(
    *,
    client: GeminiLabClient,
    clip: RushClip,
    frame: RushFrame,
    start_ms: int,
    end_ms: int,
    feature_id: str,
    event_description: str,
    regions: Sequence[FramingRegionIntent],
    checkpoint_path: Path,
    grounding_prompt: str,
    output_dir: Path,
    analysis_fps: float,
    scdet_threshold: float,
    model_request_block_reason: str | None = None,
    query_lock_v2: EvidenceQueryLockV2 | None = None,
) -> tuple[list[GroundingProposal], list[SegmentationTrack], list[Path]]:
    """Compatibility wrapper that tracks only hard-core framing regions."""

    return _build_framing_region_tracks(
        client=client,
        clip=clip,
        frame=frame,
        start_ms=start_ms,
        end_ms=end_ms,
        feature_id=feature_id,
        event_description=event_description,
        regions=regions,
        checkpoint_path=checkpoint_path,
        grounding_prompt=grounding_prompt,
        output_dir=output_dir,
        analysis_fps=analysis_fps,
        scdet_threshold=scdet_threshold,
        include_execution_roles=frozenset({"hard_core"}),
        model_request_block_reason=model_request_block_reason,
        query_lock_v2=query_lock_v2,
    )


def _identity_seed_reference(
    *,
    track: SegmentationTrack,
    target_id: str,
    target_description: str,
    output_dir: Path,
) -> GroundingIdentityReference:
    """Create a positive identity crop from the exact grounded seed PTS."""

    if track.seed_frame_pts is None:
        raise ValueError("tracked identity seed has no decoded source PTS")
    seed_frame = extract_frame_at_pts(
        Path(track.video_path),
        track.seed_frame_pts,
        output_dir / "seed-frame.png",
    )
    x_min, y_min, x_max, y_max = track.semantic_seed_box
    crop_path = output_dir / "seed-positive.png"
    with Image.open(seed_frame.path).convert("RGB") as image:
        crop_box = (
            max(0, round(x_min * image.width / 1000)),
            max(0, round(y_min * image.height / 1000)),
            min(image.width, round(x_max * image.width / 1000)),
            min(image.height, round(y_max * image.height / 1000)),
        )
        if crop_box[0] >= crop_box[2] or crop_box[1] >= crop_box[3]:
            raise ValueError("identity seed bbox produces an empty reference crop")
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop(crop_box).save(crop_path)
    return GroundingIdentityReference(
        reference_id=f"seed-positive:{target_id}",
        role="positive",
        target_id=target_id,
        description=(
            "Exact grounded seed crop for the locked target identity: "
            + target_description
        ),
        path=crop_path,
        sha256=sha256_file(crop_path),
    )


def _execute_feature_track_identity_checkpoints(
    *,
    client: GeminiLabClient,
    track: SegmentationTrack,
    target_id: str,
    target_description: str,
    output_dir: Path,
    identity_sha256: str,
    max_model_checks: int = 1,
) -> IdentityCheckpointExecution:
    """Run or reuse bounded exact-PTS identity checks for one SAM track."""

    track_fingerprint = _track_geometry_fingerprint(track)
    plan = plan_identity_checkpoints(
        track.samples,
        asset_id=track.asset_id,
        track_fingerprint=track_fingerprint,
        identity_sha256=identity_sha256,
        max_model_checks=max_model_checks,
        seed_sample_index=track.seed_sample_index,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "identity-checkpoint-plan.json", plan)
    checkpoint_frames = {}
    for candidate in plan.candidates:
        if not candidate.selected_for_verification:
            continue
        if candidate.source_pts is None:
            raise ValueError(
                "semantic identity checkpoint has no decoded source PTS"
            )
        checkpoint_frames[candidate.sample_index] = extract_frame_at_pts(
            Path(track.video_path),
            candidate.source_pts,
            output_dir
            / "exact-frames"
            / f"{candidate.sample_index:06d}-{candidate.source_pts}.png",
        )
    reference = _identity_seed_reference(
        track=track,
        target_id=target_id,
        target_description=target_description,
        output_dir=output_dir / "references",
    )
    verifier_id = f"{MODEL_ID}:interactions:semantic-identity-medium:v1"
    current_pointer = output_dir / "identity-checkpoint-execution.current.json"
    if current_pointer.is_file():
        pointer = read_json(current_pointer)
        cached_path = Path(str(pointer.get("path", "")))
        if cached_path.is_file():
            cached = IdentityCheckpointExecution.model_validate(
                read_json(cached_path)
            )
            cached_hashes = {
                result.sample_index: result.frame_hash
                for result in cached.results
            }
            current_hashes = {
                sample_index: frame.frame_hash
                for sample_index, frame in checkpoint_frames.items()
            }
            if (
                cached.track_fingerprint == track_fingerprint
                and cached.identity_sha256 == identity_sha256
                and cached.planning_request_sha256
                == plan.planning_request_sha256
                and cached.verifier_id == verifier_id
                and cached_hashes == current_hashes
            ):
                return cached

    def verifier(candidate, exact_frame):
        return client.verify_identity_checkpoint(
            frame=exact_frame,
            target_id=target_id,
            target_description=target_description,
            run_id=f"feature-identity-{uuid.uuid4().hex[:8]}",
            output_dir=(
                output_dir
                / "model-checks"
                / f"sample-{candidate.sample_index:06d}"
            ),
            identity_references=(reference,),
        )

    execution = execute_identity_checkpoints(
        plan,
        frames_by_sample_index=checkpoint_frames,
        verifier=verifier,
        verifier_id=verifier_id,
        abort_on_error=_is_exhausted_model_quota_error,
    )
    execution_path = (
        output_dir
        / "executions"
        / f"execution-{execution.execution_request_sha256[:16]}.json"
    )
    write_json(execution_path, execution)
    write_json(
        current_pointer,
        {
            "artifact_type": "identity_checkpoint_execution_pointer_v1",
            "path": str(execution_path.resolve()),
            "execution_request_sha256": execution.execution_request_sha256,
        },
    )
    return execution


def _combined_semantic_checkpoint_status(
    executions: Sequence[IdentityCheckpointExecution],
) -> SemanticCheckpointStatus:
    statuses = {
        execution.semantic_checkpoint_status for execution in executions
    }
    if SemanticCheckpointStatus.FAILED in statuses:
        return SemanticCheckpointStatus.FAILED
    if (
        SemanticCheckpointStatus.AMBIGUOUS in statuses
        or SemanticCheckpointStatus.REQUIRED_PENDING in statuses
    ):
        return SemanticCheckpointStatus.AMBIGUOUS
    if statuses == {SemanticCheckpointStatus.NOT_REQUIRED_BY_POLICY}:
        return SemanticCheckpointStatus.NOT_REQUIRED_BY_POLICY
    return SemanticCheckpointStatus.PASSED


def _vertical_candidate_geometry(
    *,
    client: GeminiLabClient,
    clip: RushClip,
    frame: RushFrame,
    start_ms: int,
    end_ms: int,
    feature_id: str,
    event_description: str,
    target_description: str | None,
    regions: Sequence[FramingRegionIntent],
    camera_phases: Sequence[VerticalVirtualCameraPhase],
    camera_phase_origin: Literal["human_reviewed", "gemini_proposed"],
    crop_mode: Literal["strict", "primary_center"],
    overflow_policy: Literal["preserve_all", "controlled_clip"],
    edge_priority: Literal["balanced", "preserve_start", "preserve_end"],
    fallback_strategy: Literal["fit_with_background", "center_crop"],
    checkpoint_path: Path,
    grounding_prompt: str,
    output_dir: Path,
    analysis_fps: float,
    scdet_threshold: float,
    display_sample_aspect_ratio: float,
    track_cache: dict[
        tuple[str, str, int, int],
        tuple[GroundingProposal, SegmentationTrack, Path],
    ],
    model_request_block_reason: str | None = None,
    query_lock_v2: EvidenceQueryLockV2 | None = None,
) -> tuple[str, dict[str, Any], list[Path], str | None]:
    """Evaluate one immutable vertical candidate without rendering a segment."""

    crop_regions = [
        region for region in regions if region.execution_role != "overlay_keepout"
    ]
    hard_regions = [
        region for region in crop_regions if region.execution_role == "hard_core"
    ]
    soft_regions = [
        region for region in crop_regions if region.execution_role == "soft_extent"
    ]
    debug_paths: list[Path] = []
    track_fingerprint: str | None = None
    optional_region_failures: list[dict[str, str]] = []
    identity_tracks: list[tuple[str, str, SegmentationTrack]] = []
    if crop_regions:
        if not hard_regions:
            raise ValueError("candidate region contract has no hard core")
        _, hard_tracks, hard_debug_paths = _build_framing_region_tracks(
            client=client,
            clip=clip,
            frame=frame,
            start_ms=start_ms,
            end_ms=end_ms,
            feature_id=feature_id,
            event_description=event_description,
            regions=hard_regions,
            checkpoint_path=checkpoint_path,
            grounding_prompt=grounding_prompt,
            output_dir=output_dir,
            analysis_fps=analysis_fps,
            scdet_threshold=scdet_threshold,
            include_execution_roles=frozenset({"hard_core"}),
            model_request_block_reason=model_request_block_reason,
            query_lock_v2=query_lock_v2,
        )
        debug_paths.extend(hard_debug_paths)
        tracks_by_region = {
            region.region_id: track
            for region, track in zip(hard_regions, hard_tracks, strict=True)
        }
        identity_tracks.extend(
            (
                region.entity_id or region.region_id,
                region.target_description,
                track,
            )
            for region, track in zip(hard_regions, hard_tracks, strict=True)
        )
        available_soft_regions: list[FramingRegionIntent] = []
        soft_tracks: list[SegmentationTrack] = []
        for soft_region in soft_regions:
            try:
                _, region_tracks, region_debug_paths = (
                    _build_framing_region_tracks(
                        client=client,
                        clip=clip,
                        frame=frame,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        feature_id=feature_id,
                        event_description=event_description,
                        regions=[soft_region],
                        checkpoint_path=checkpoint_path,
                        grounding_prompt=grounding_prompt,
                        output_dir=output_dir,
                        analysis_fps=analysis_fps,
                        scdet_threshold=scdet_threshold,
                        include_execution_roles=frozenset({"soft_extent"}),
                        model_request_block_reason=model_request_block_reason,
                        query_lock_v2=query_lock_v2,
                    )
                )
            except (RuntimeError, ValueError) as error:
                optional_region_failures.append(
                    {
                        "region_id": soft_region.region_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                continue
            available_soft_regions.append(soft_region)
            soft_tracks.append(region_tracks[0])
            tracks_by_region[soft_region.region_id] = region_tracks[0]
            debug_paths.extend(region_debug_paths)
        available_regions = [*hard_regions, *available_soft_regions]
        available_region_ids = set(tracks_by_region)
        effective_camera_phases: list[VerticalVirtualCameraPhase] = []
        for phase in camera_phases:
            available_anchors = [
                region_id
                for region_id in phase.anchor_region_ids
                if region_id in available_region_ids
            ]
            if not available_anchors:
                raise ValueError(
                    f"phase {phase.phase_id} has no available hard-core anchor"
                )
            effective_camera_phases.append(
                phase.model_copy(
                    update={"anchor_region_ids": available_anchors}
                )
            )
        track_fingerprint = _stable_fingerprint(
            {
                "contract_version": "feature-vertical-region-tracks-v3",
                "regions": [
                    region.model_dump(mode="json")
                    for region in available_regions
                ],
                "tracks": [
                    _track_geometry_fingerprint(tracks_by_region[region.region_id])
                    for region in available_regions
                ],
                "optional_region_failures": optional_region_failures,
            }
        )
        if effective_camera_phases:
            try:
                filter_graph, geometry = (
                    _vertical_virtual_camera_filter_from_tracks(
                        tracks_by_region=tracks_by_region,
                        phases=effective_camera_phases,
                        phase_origin=camera_phase_origin,
                        display_sample_aspect_ratio=display_sample_aspect_ratio,
                    )
                )
            except ValueError as error:
                filter_graph, geometry = _vertical_filter_from_track(
                    hard_tracks,
                    allow_subject_clipping=crop_mode == "primary_center",
                    overflow_policy=overflow_policy,
                    edge_priority=edge_priority,
                    region_ids=[region.region_id for region in hard_regions],
                    fallback_strategy=fallback_strategy,
                    display_sample_aspect_ratio=display_sample_aspect_ratio,
                    preferred_tracks=soft_tracks,
                    preferred_regions=available_soft_regions,
                )
                geometry.setdefault("risk_codes", []).append(
                    "phase_virtual_camera_preflight_failed"
                )
                geometry["phase_virtual_camera_fallback_reason"] = str(error)
                geometry["requires_gemini_review"] = True
        else:
            filter_graph, geometry = _vertical_filter_from_track(
                hard_tracks,
                allow_subject_clipping=crop_mode == "primary_center",
                overflow_policy=overflow_policy,
                edge_priority=edge_priority,
                region_ids=[region.region_id for region in hard_regions],
                fallback_strategy=fallback_strategy,
                display_sample_aspect_ratio=display_sample_aspect_ratio,
                preferred_tracks=soft_tracks,
                preferred_regions=available_soft_regions,
            )
    else:
        target = (target_description or "").strip()
        if not target:
            raise ValueError("tracked vertical candidate has no resolved target")
        identity_cache_target = target
        if query_lock_v2 is not None:
            identity_cache_target += (
                "#identity:" + query_lock_v2.component_hashes()["identity_sha256"]
            )
        cache_key = (frame.frame_id, identity_cache_target, start_ms, end_ms)
        if cache_key not in track_cache:
            proposal, track = _build_track(
                client=client,
                clip=clip,
                frame=frame,
                start_ms=start_ms,
                end_ms=end_ms,
                feature_id=feature_id,
                event_description=event_description,
                target_description=target,
                checkpoint_path=checkpoint_path,
                grounding_prompt=grounding_prompt,
                output_dir=output_dir,
                run_id=f"feature-v-{uuid.uuid4().hex[:8]}",
                analysis_fps=analysis_fps,
                scdet_threshold=scdet_threshold,
                model_request_block_reason=model_request_block_reason,
                query_lock_v2=query_lock_v2,
            )
            track_cache[cache_key] = (proposal, track, output_dir)
        _, track, cached_root = track_cache[cache_key]
        identity_tracks.append(
            (
                (
                    query_lock_v2.identity.targets[0].target_id
                    if query_lock_v2 is not None
                    and len(query_lock_v2.identity.targets) == 1
                    else "reframe_subject"
                ),
                target,
                track,
            )
        )
        if query_lock_v2 is not None:
            matching_targets = [
                locked_target
                for locked_target in query_lock_v2.identity.targets
                if locked_target.target_description.strip() == target
            ]
            if len(matching_targets) != 1:
                raise ValueError(
                    "cached track target does not resolve to exactly one QueryLock identity"
                )
            reused_lineage = evidence_query_lock_v2_lineage(
                query_lock_v2,
                target_id=matching_targets[0].target_id,
                target_description=target,
            )
            write_json(
                output_dir
                / "reused-track-query-lineage-"
                f"{reused_lineage['definition_sha256'][:16]}.json",
                {
                    "contract_version": "feature-reused-track-query-lineage-v1",
                    "evidence_query_v2": reused_lineage,
                    "cached_track_root": str(cached_root.resolve()),
                    "track_fingerprint": _track_geometry_fingerprint(track),
                },
            )
        debug_paths = [cached_root / "grounding-debug.png"]
        track_fingerprint = _track_geometry_fingerprint(track)
        filter_graph, geometry = _vertical_filter_from_track(
            track,
            allow_subject_clipping=crop_mode == "primary_center",
            overflow_policy=overflow_policy,
            edge_priority=edge_priority,
            fallback_strategy=fallback_strategy,
            display_sample_aspect_ratio=display_sample_aspect_ratio,
        )

    identity_executions: list[IdentityCheckpointExecution] = []
    if geometry.get("applied_strategy") == "tracked_crop":
        for target_id, locked_description, identity_track in identity_tracks:
            identity_sha256 = _stable_fingerprint(
                {
                    "query_identity_sha256": (
                        query_lock_v2.component_hashes()["identity_sha256"]
                        if query_lock_v2 is not None
                        else None
                    ),
                    "target_id": target_id,
                    "target_description": locked_description,
                    "seed_frame_pts": identity_track.seed_frame_pts,
                    "semantic_seed_box": identity_track.semantic_seed_box,
                }
            )
            identity_executions.append(
                _execute_feature_track_identity_checkpoints(
                    client=client,
                    track=identity_track,
                    target_id=target_id,
                    target_description=locked_description,
                    output_dir=output_dir
                    / "identity-checkpoints"
                    / target_id,
                    identity_sha256=identity_sha256,
                )
            )
        semantic_checkpoint_status = _combined_semantic_checkpoint_status(
            identity_executions
        )
    else:
        semantic_checkpoint_status = (
            SemanticCheckpointStatus.NOT_REQUIRED_BY_POLICY
        )
    geometry["semantic_checkpoint_status"] = semantic_checkpoint_status.value
    geometry["identity_checkpoint_executions"] = (
        [
            {
                "target_id": target_id,
                "target_description": target_description,
                "track_fingerprint": execution.track_fingerprint,
                "identity_sha256": execution.identity_sha256,
                "semantic_checkpoint_status": (
                    execution.semantic_checkpoint_status.value
                ),
                "execution_request_sha256": execution.execution_request_sha256,
                "model_calls_made": execution.model_calls_made,
            }
            for (target_id, target_description, _), execution in zip(
                identity_tracks,
                identity_executions,
                strict=True,
            )
        ]
        if identity_executions
        else []
    )

    semantic_review_reasons: list[str] = []
    if len(hard_regions) > 1:
        semantic_review_reasons.append("multiple_hard_core_regions")
    if camera_phases:
        semantic_review_reasons.append("phase_virtual_camera_requires_sequence_review")
    if any(
        region.kind in {"text_region", "ui_region"} or region.atomic
        for region in crop_regions
    ):
        semantic_review_reasons.append("atomic_text_or_ui_region")
    if any(region.execution_role == "overlay_keepout" for region in regions):
        semantic_review_reasons.append("overlay_keepout_requires_layout_gate")
    if geometry.get("fallback_reason"):
        semantic_review_reasons.append("fallback_applied")
    if optional_region_failures:
        semantic_review_reasons.append("optional_region_grounding_unavailable")
        geometry.setdefault("risk_codes", []).append(
            "optional_region_grounding_unavailable"
        )
        geometry["optional_region_failures"] = optional_region_failures
    if not geometry.get("soft_extent_visibility_passed", True):
        semantic_review_reasons.append("soft_extent_visibility_below_floor")
    geometry["semantic_review_reasons"] = list(
        dict.fromkeys(semantic_review_reasons)
    )
    if semantic_review_reasons:
        geometry["requires_gemini_review"] = True
    geometry["framing_regions"] = [
        {
            **region.model_dump(mode="json"),
            "execution_role": region.execution_role,
            "effective_minimum_visible_fraction": (
                region.effective_minimum_visible_fraction
            ),
        }
        for region in regions
    ]
    geometry["vertical_camera_phases"] = [
        phase.model_dump(mode="json") for phase in camera_phases
    ]
    return filter_graph, geometry, debug_paths, track_fingerprint


def _vertical_candidate_preflight(
    *,
    candidate_id: str,
    rank: int,
    confidence: float,
    source_sha256: str,
    filter_graph: str,
    geometry: Mapping[str, Any],
    regions: Sequence[FramingRegionIntent],
    track_fingerprint: str | None,
    titles_rendered: bool,
) -> tuple[CandidatePreflight, str]:
    """Translate renderer evidence into the versioned auto-reframe contract."""

    geometry_fingerprint = _stable_fingerprint(
        {
            "contract_version": "vertical-candidate-geometry-preflight-v1",
            "source_sha256": source_sha256,
            "filter_graph": filter_graph,
            "geometry": dict(geometry),
            "track_fingerprint": track_fingerprint,
        }
    )
    soft_by_id = {
        str(item.get("region_id")): item
        for item in geometry.get("soft_extent_regions", [])
        if isinstance(item, Mapping)
    }
    hard_minimum = float(
        geometry.get("minimum_visible_required_area_fraction", 1.0)
    )
    assessed_regions: list[RegionAssessment] = []
    for region in regions:
        if region.execution_role == "hard_core":
            assessed_regions.append(
                RegionAssessment(
                    region_id=region.region_id,
                    role="hard_core",
                    atomic=region.atomic,
                    assessed=True,
                    minimum_visible_fraction=hard_minimum,
                    required_visible_fraction=1.0,
                )
            )
        elif region.execution_role == "soft_extent":
            item = soft_by_id.get(region.region_id)
            assessed_regions.append(
                RegionAssessment(
                    region_id=region.region_id,
                    role="soft_extent",
                    atomic=False,
                    assessed=item is not None,
                    minimum_visible_fraction=(
                        float(item["minimum_visible_area_fraction"])
                        if item is not None
                        else 0.0
                    ),
                    required_visible_fraction=(
                        region.effective_minimum_visible_fraction
                    ),
                    clipped_edges=(
                        list(item.get("clipped_edges", []))
                        if item is not None
                        else []
                    ),
                )
            )
        else:
            # Until the title/layout solver emits exact overlay rectangles, a
            # keepout is safe only when this experiment renders no overlay.
            assessed_regions.append(
                RegionAssessment(
                    region_id=region.region_id,
                    role="overlay_keepout",
                    atomic=region.atomic,
                    assessed=not titles_rendered,
                    minimum_visible_fraction=1.0,
                    required_visible_fraction=0.0,
                    overlay_overlap_fraction=1.0 if titles_rendered else 0.0,
                )
            )
    preflight = CandidatePreflight(
        candidate_id=candidate_id,
        rank=rank,
        presentation=(
            "tracked_crop"
            if geometry.get("applied_strategy") == "tracked_crop"
            else "static_anchor"
            if geometry.get("applied_strategy") == "seed_anchor_crop"
            else "center_crop"
            if geometry.get("applied_strategy") == "center_crop"
            else "fit_with_background"
        ),
        source_lineage_valid=bool(
            geometry.get("source_geometry_lineage_passed", True)
        ),
        within_single_shot=True,
        evidence_confidence=confidence,
        semantic_status="matched",
        tracking_confidence_gate_passed=bool(
            geometry.get("tracking_confidence_gate_passed", True)
        ),
        tracking_coverage_passed=bool(geometry.get("coverage_passed", True)),
        semantic_checkpoint_status=(
            SemanticCheckpointStatus(
                geometry.get(
                    "semantic_checkpoint_status",
                    SemanticCheckpointStatus.REQUIRED_PENDING.value,
                )
            )
            if geometry.get("applied_strategy") == "tracked_crop"
            else SemanticCheckpointStatus.NOT_REQUIRED_BY_POLICY
        ),
        regions=assessed_regions,
        max_crop_speed_pixels_per_second=float(
            geometry.get("max_crop_speed_pixels_per_second", 0.0)
        ),
        max_crop_acceleration_pixels_per_second_squared=float(
            geometry.get(
                "max_crop_acceleration_pixels_per_second_squared", 0.0
            )
        ),
        max_crop_jerk_pixels_per_second_cubed=float(
            geometry.get("max_crop_jerk_pixels_per_second_cubed", 0.0)
        ),
        geometry_fingerprint=geometry_fingerprint,
        source_fingerprint=source_sha256,
        track_fingerprints=(
            [track_fingerprint] if track_fingerprint is not None else []
        ),
    )
    return preflight, geometry_fingerprint


def _controlled_primary_center_preview_allowed(
    *,
    crop_mode: str,
    geometry: Mapping[str, Any],
    regions: Sequence[FramingRegionIntent],
    failure_codes: Sequence[FailureCode],
    minimum_visible_fraction: float = 0.90,
) -> bool:
    """Gate a bounded, explicitly review-only portrait crop preview.

    Normal unattended delivery still requires complete hard-core containment.
    This path is only reachable behind ``--allow-unverified-geometry-preview``
    and an upstream ``primary_center`` framing choice. Eligibility is based on
    region semantics and measured geometry, never on a product or subject
    category.
    """

    if crop_mode != "primary_center":
        return False
    if geometry.get("applied_strategy") != "tracked_crop":
        return False
    if geometry.get("fallback_reason") is not None:
        return False
    if any(
        region.atomic or region.kind in {"text_region", "ui_region", "graphic"}
        for region in regions
        if region.execution_role == "hard_core"
    ):
        return False
    visible = float(
        geometry.get("minimum_visible_required_area_fraction", 0.0)
    )
    if visible + 1e-6 < minimum_visible_fraction:
        return False
    allowed_failures = {
        FailureCode.HARD_CORE_NOT_FULLY_RETAINED,
        FailureCode.IDENTITY_VERIFICATION_PENDING,
    }
    return bool(failure_codes) and set(failure_codes).issubset(allowed_failures)


def _failure_codes_from_geometry_error(error: Exception) -> list[FailureCode]:
    """Map existing Grounding/SAM errors to stable recovery categories."""

    message = f"{type(error).__name__}:{error}".casefold()
    if "ambiguous" in message:
        return [FailureCode.TARGET_AMBIGUITY_ABOVE_MAXIMUM]
    if any(
        marker in message
        for marker in (
            "not_visible",
            "target_mismatch",
            "insufficient_evidence",
            "request_match",
        )
    ):
        return [FailureCode.SEMANTIC_MATCH_BELOW_MINIMUM]
    if "confidence" in message:
        return [FailureCode.TRACK_CONFIDENCE_BELOW_MINIMUM]
    if any(marker in message for marker in ("coverage", "fewer_than_two")):
        return [FailureCode.TRACK_COVERAGE_BELOW_MINIMUM]
    if "shot" in message and "boundary" in message:
        return [FailureCode.SHOT_BOUNDARY_CROSSING]
    return [FailureCode.NO_FEASIBLE_PRESENTATION]


def _resolve_vertical_candidate_intent(
    *,
    option_regions: Sequence[FramingRegionIntent | Mapping[str, Any]],
    option_target_description: str | None,
    selected_target_description: str | None,
    brief_primary_target_description: str | None,
    brief_regions: Sequence[FramingRegionIntent],
    inherit_reviewed_brief_intent: bool,
) -> tuple[list[FramingRegionIntent], str | None]:
    """Resolve one take without allowing rank-1 prose to leak into rank-N."""

    regions = [
        region
        if isinstance(region, FramingRegionIntent)
        else FramingRegionIntent.model_validate(region)
        for region in option_regions
    ]
    if not regions and inherit_reviewed_brief_intent:
        regions = list(brief_regions)
    hard_regions = [
        region for region in regions if region.execution_role == "hard_core"
    ]
    if hard_regions:
        target = "; ".join(region.target_description for region in hard_regions)
    elif option_target_description:
        target = option_target_description
    elif inherit_reviewed_brief_intent:
        target = selected_target_description or brief_primary_target_description
    else:
        target = None
    return regions, target


def _validate_selected_framing_coverage_invariant(
    *,
    upstream_intent: str,
    upstream_target_descriptions: Sequence[str],
    proposal: SelectedVerticalFramingProposal,
) -> None:
    """Prevent selected-clip refinement from weakening planner obligations.

    Until every planner target description has a stable entity ID, the local
    contract conservatively preserves the semantic mode and the number of
    independently tracked hard-core participants.  It never relies on a
    paraphrased description to prove that a required participant survived.
    """

    stronger_or_equal: dict[str, set[str]] = {
        "single_primary": {
            "single_primary",
            "group_coverage",
            "sequential_attention",
            "simultaneous_relation",
        },
        "group_coverage": {"group_coverage", "simultaneous_relation"},
        "sequential_attention": {
            "sequential_attention",
            "simultaneous_relation",
        },
        "simultaneous_relation": {"simultaneous_relation"},
    }
    if upstream_intent not in stronger_or_equal:
        raise ValueError(f"unknown upstream coverage intent: {upstream_intent}")
    if proposal.recommended_action != "tracked_crop":
        # Fit/layout and try-next do not claim that a weaker crop is safe.
        return
    if proposal.semantic_requirement not in stronger_or_equal[upstream_intent]:
        raise ValueError(
            "selected framing weakens the upstream coverage obligation: "
            f"{upstream_intent} -> {proposal.semantic_requirement}"
        )
    required_participant_count = max(
        len(upstream_target_descriptions),
        2 if upstream_intent != "single_primary" else 1,
    )
    hard_regions = [
        region
        for region in proposal.regions
        if region.execution_role == "hard_core"
    ]
    if len(hard_regions) < required_participant_count:
        raise ValueError(
            "selected framing does not retain enough independently grounded "
            "hard-core participants for the upstream coverage obligation"
        )


def _refine_selected_vertical_candidate(
    *,
    client: GeminiLabClient,
    option_data: Mapping[str, Any],
    chapter: FeatureChapterBrief,
    clip: RushClip,
    frame: RushFrame,
    prompt_template: str,
    catalog_path: Path,
    output_dir: Path,
    vertical_fallback_strategy: Literal["fit_with_background", "center_crop"],
) -> tuple[dict[str, Any], SelectedVerticalFramingProposal, bool]:
    """Run or reuse the full-clip 9:16 presentation decision before geometry."""

    candidate_id = str(option_data["candidate_id"])
    source_asset_id = f"sha256:{clip.sha256}"
    event_id = str(option_data.get("event_id") or f"catalog-{chapter.feature_id}")
    candidate_context = {
        "candidate_id": candidate_id,
        "source_asset_id": source_asset_id,
        "event_id": event_id,
        "frame_id": frame.frame_id,
        "observed_visual_evidence": option_data.get("observed_visual_evidence"),
        "selection_reason": option_data.get("selection_reason"),
        "current_strategy": option_data.get("strategy"),
        "current_target_description": option_data.get("target_description"),
        "current_coverage_intent": option_data.get("coverage_intent"),
        "current_coverage_target_descriptions": option_data.get(
            "coverage_target_descriptions", []
        ),
        "current_regions": option_data.get("regions", []),
        "quality_risks": option_data.get("quality_risks", []),
        "delivery_preference": {
            "aspect": "9:16",
            "fallback_strategy": vertical_fallback_strategy,
            "full_bleed_preferred": vertical_fallback_strategy == "center_crop",
        },
    }
    chapter_context = chapter.model_dump(mode="json")
    fingerprint_payload = {
        "contract_version": "selected-vertical-framing-request-v2",
        "source_sha256": clip.sha256,
        "candidate_context": candidate_context,
        "chapter_context": chapter_context,
        "prompt_sha256": hashlib.sha256(prompt_template.encode("utf-8")).hexdigest(),
        "system_instruction_sha256": hashlib.sha256(
            EDITORIAL_SYSTEM_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "model_id": client.model_id,
        "response_schema": gemini_response_schema(
            SelectedVerticalFramingProposal
        ),
    }
    request_fingerprint = _stable_fingerprint(fingerprint_payload)
    run_dir = (
        output_dir
        / "selected-vertical-framing"
        / chapter.feature_id
        / candidate_id
        / request_fingerprint[:16]
    )
    proposal_path = run_dir / "selected_vertical_framing.json"
    binding_path = run_dir / "selected_vertical_framing.binding.json"
    raw_output_path = run_dir / "selected_vertical_framing.raw_output.json"
    raw_interaction_path = run_dir / "selected_vertical_framing.raw_interaction.json"
    reused = False
    if proposal_path.is_file() and binding_path.is_file():
        binding = read_json(binding_path)
        if binding.get("request_fingerprint") != request_fingerprint:
            raise ValueError("selected vertical framing cache binding mismatch")
        proposal = SelectedVerticalFramingProposal.model_validate(
            read_json(proposal_path)
        )
        reused = True
    elif raw_output_path.is_file() and raw_interaction_path.is_file():
        # Representation-only recovery after a stricter local validator
        # rejected an otherwise complete paid response.  Never send another
        # request merely because the consumer contract was repaired.
        raw_output = read_json(raw_output_path)
        canonical_text, normalization_changes = (
            canonicalize_selected_vertical_framing_output(
                str(raw_output["output_text"])
            )
        )
        proposal = SelectedVerticalFramingProposal.model_validate_json(
            canonical_text
        )
        immutable = {
            "candidate_id": candidate_id,
            "source_asset_id": source_asset_id,
            "event_id": event_id,
            "frame_id": frame.frame_id,
        }
        mismatches = {
            key: {"expected": value, "actual": getattr(proposal, key)}
            for key, value in immutable.items()
            if getattr(proposal, key) != value
        }
        if mismatches:
            raise ValueError(
                "saved selected framing response changed immutable selection: "
                f"{mismatches}"
            )
        raw_interaction = read_json(raw_interaction_path)
        proposal = proposal.model_copy(
            update={
                "model_provenance": proposal.model_provenance.model_copy(
                    update={
                        "interaction_id": raw_interaction.get("id")
                        or proposal.model_provenance.interaction_id
                    }
                )
            }
        )
        write_json(proposal_path, proposal)
        write_json(
            run_dir / "selected_vertical_framing.canonical_output.json",
            {"output_text": canonical_text},
        )
        write_json(
            run_dir / "selected_vertical_framing.normalization_audit.json",
            {
                "changes": normalization_changes,
                "editorial_selection_changed": False,
            },
        )
        write_json(
            run_dir / "selected_vertical_framing.raw_output_reuse.json",
            {
                "reused": True,
                "reason": "representation_only_local_contract_repair",
                "source_raw_output_sha256": sha256_file(raw_output_path),
                "source_raw_interaction_sha256": sha256_file(
                    raw_interaction_path
                ),
                "reused_at": utc_now(),
            },
        )
        write_json(
            binding_path,
            {
                **fingerprint_payload,
                "request_fingerprint": request_fingerprint,
                "file_api_reused": None,
                "raw_paid_response_reused": True,
                "proposal_sha256": sha256_file(proposal_path),
                "created_at": utc_now(),
            },
        )
        reused = True
    else:
        upload_dir = (
            catalog_path.parent
            / "file-cache"
            / clip.sha256
            / "selected-vertical-framing-upload"
        )
        uploaded, file_api_reused = client.ensure_video_upload(
            Path(clip.path), upload_dir
        )
        proposal = client.propose_selected_vertical_framing(
            uploaded=uploaded,
            candidate_id=candidate_id,
            source_asset_id=source_asset_id,
            event_id=event_id,
            frame_id=frame.frame_id,
            candidate_context=candidate_context,
            chapter_context=chapter_context,
            prompt_template=prompt_template,
            run_id=f"selected-framing-{uuid.uuid4().hex[:8]}",
            run_dir=run_dir,
        )
        write_json(
            binding_path,
            {
                **fingerprint_payload,
                "request_fingerprint": request_fingerprint,
                "file_api_reused": file_api_reused,
                "proposal_sha256": sha256_file(proposal_path),
                "created_at": utc_now(),
            },
        )

    _validate_selected_framing_coverage_invariant(
        upstream_intent=str(
            option_data.get("coverage_intent") or "single_primary"
        ),
        upstream_target_descriptions=[
            str(value)
            for value in option_data.get(
                "coverage_target_descriptions", []
            )
        ],
        proposal=proposal,
    )
    refined = dict(option_data)
    refined["source_asset_id"] = source_asset_id
    refined["event_id"] = event_id
    refined["framing_refinement"] = {
        "request_fingerprint": request_fingerprint,
        "proposal_path": str(proposal_path.resolve()),
        "proposal_sha256": sha256_file(proposal_path),
        "reused": reused,
        "recommended_action": proposal.recommended_action,
        "semantic_requirement": proposal.semantic_requirement,
        "relation_temporal_mode": proposal.relation_temporal_mode,
        "sequential_reconstruction": (
            proposal.sequential_reconstruction.model_dump(mode="json")
            if proposal.sequential_reconstruction is not None
            else None
        ),
        "presentation_options": [
            option.model_dump(mode="json")
            for option in proposal.presentation_options
        ],
        "decision_reason": proposal.decision_reason,
        "observed_evidence": proposal.observed_evidence,
        "uncertainties": proposal.uncertainties,
        "confidence": proposal.confidence,
        "non_executable_virtual_camera_ignored": bool(
            proposal.recommended_action != "tracked_crop"
            and proposal.virtual_camera_proposal is not None
        ),
    }
    if proposal.recommended_action == "tracked_crop":
        refined["strategy"] = "tracked_crop"
        refined["crop_mode"] = "strict"
        refined["regions"] = [
            region.model_dump(mode="python") for region in proposal.regions
        ]
        refined["virtual_camera_proposal"] = (
            proposal.virtual_camera_proposal.model_dump(mode="python")
            if proposal.virtual_camera_proposal is not None
            else None
        )
        hard_regions = [
            region
            for region in proposal.regions
            if region.execution_role == "hard_core"
        ]
        refined["target_description"] = (
            hard_regions[0].target_description
            if len(hard_regions) == 1
            else None
        )
        FeatureVerticalCandidate.model_validate(
            {
                key: refined[key]
                for key in (
                    "candidate_id",
                    "rank",
                    "source_asset_id",
                    "event_id",
                    "frame_id",
                    "observed_visual_evidence",
                    "selection_reason",
                    "strategy",
                    "crop_mode",
                    "target_description",
                    "regions",
                    "virtual_camera_proposal",
                    "quality_risks",
                    "confidence",
                )
            }
        )
    else:
        refined["strategy"] = "fit_with_background"
        refined["target_description"] = None
        refined["regions"] = []
        refined["virtual_camera_proposal"] = None
    return refined, proposal, reused


def _vertical_runtime_candidate_options(
    selected: FeatureChapterSelect,
    *,
    human_policy_binding_present: bool,
    max_candidates: int = 4,
) -> list[dict[str, Any]]:
    """Return immutable auto options or one legacy/human-reviewed selection."""

    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if selected.vertical_candidates and not human_policy_binding_present:
        options = []
        for candidate in sorted(
            selected.vertical_candidates, key=lambda item: item.rank
        )[:max_candidates]:
            option = candidate.model_dump(mode="python")
            option["coverage_intent"] = selected.vertical_coverage_intent
            option["coverage_target_descriptions"] = list(
                selected.vertical_coverage_target_descriptions
            )
            options.append(option)
        return options
    return [
        {
            "candidate_id": "legacy-primary",
            "rank": 1,
            "source_asset_id": None,
            "event_id": None,
            "frame_id": selected.vertical_frame_id,
            "observed_visual_evidence": selected.observed_visual_evidence,
            "selection_reason": selected.selection_reason,
            "strategy": selected.vertical_strategy,
            "crop_mode": "strict",
            "target_description": selected.vertical_target_description,
            "coverage_intent": selected.vertical_coverage_intent,
            "coverage_target_descriptions": list(
                selected.vertical_coverage_target_descriptions
            ),
            "regions": [],
            "quality_risks": selected.quality_risks,
            "confidence": selected.confidence,
        }
    ]


def _horizontal_runtime_candidate_options(
    selected: FeatureChapterSelect,
    *,
    max_candidates: int = 4,
) -> list[dict[str, Any]]:
    """Return ordered 16:9 candidates under the same runtime routing contract."""

    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if selected.horizontal_candidates:
        return [
            candidate.model_dump(mode="python")
            for candidate in sorted(
                selected.horizontal_candidates,
                key=lambda item: item.rank,
            )[:max_candidates]
        ]
    return [
        {
            "candidate_id": "legacy-primary",
            "rank": 1,
            "source_asset_id": None,
            "event_id": None,
            "frame_id": selected.horizontal_frame_id,
            "observed_visual_evidence": selected.observed_visual_evidence,
            "selection_reason": selected.selection_reason,
            "strategy": selected.horizontal_strategy,
            "zoom_intent": selected.horizontal_zoom_intent,
            "camera_intent": selected.horizontal_camera_intent,
            "target_description": selected.horizontal_target_description,
            "quality_risks": selected.quality_risks,
            "confidence": selected.confidence,
        }
    ]


def _should_refine_selected_vertical_candidate(
    *,
    auto_vertical_framing: bool,
    human_reframe_policy_requested: bool,
    feature_plan_origin: str,
    external_projection_contract_id: str | None,
    option_data: Mapping[str, Any],
) -> bool:
    """Return whether the legacy selected-clip semantic pass is still needed.

    The compact direct-video contract has already inspected the bounded
    candidate video and supplied the editorial framing/attention decision. Its
    downstream work is exact-frame Grounding, SAM propagation, and local
    geometry—not a second paid VLM reinterpretation of the same candidate.
    """

    if not auto_vertical_framing or human_reframe_policy_requested:
        return False
    if external_projection_contract_id == "direct-video-edit-plan-v1":
        return False
    return (
        feature_plan_origin != "external_projection"
        or option_data.get("virtual_camera_proposal") is None
    )


def _prepare_horizontal_runtime_candidate(
    *,
    option: Mapping[str, Any],
    selected: FeatureChapterSelect,
    brief_chapter: FeatureChapterBrief,
    chapter_duration_seconds: float,
    frames: Mapping[str, RushFrame],
    clips: Mapping[str, RushClip],
    shot_cache: dict[str, ShotManifest],
    shots_dir: Path,
    scdet_threshold: float,
    trim_decisions: Sequence[tuple[Path, TrimIntentDecision]],
    shot_quality_maps: Sequence[tuple[Path, ShotQualityMap]],
    source_audio_cache: dict[str, bool],
    source_media_cache: dict[str, MediaInfo],
    track_cache: dict[
        tuple[str, str, int, int],
        tuple[GroundingProposal, SegmentationTrack, Path],
    ],
    client: GeminiLabClient,
    checkpoint_path: Path,
    grounding_prompt: str,
    output_dir: Path,
    sam_analysis_fps: float,
    model_request_block_reason: str | None,
) -> dict[str, Any]:
    frame_id = option.get("frame_id")
    if not isinstance(frame_id, str) or frame_id not in frames:
        raise ValueError("horizontal candidate references an unknown frame")
    frame = frames[frame_id]
    clip = clips[frame.clip_id]
    if not _candidate_asset_reference_matches(
        option.get("source_asset_id"),
        clip,
    ):
        raise ValueError("horizontal candidate source asset differs from its frame")
    start_ms, end_ms, shot_id, trim = _chapter_bounds_with_approved_trim(
        frame,
        clip,
        chapter_duration_seconds,
        shot_cache,
        shots_dir,
        scdet_threshold,
        trim_decisions,
        expected_event_id=(
            str(option["event_id"]) if option.get("event_id") else None
        ),
        quality_maps=shot_quality_maps,
    )
    source_audio_cache.setdefault(
        clip.sha256,
        has_audio_stream(Path(clip.path)),
    )
    source_media_cache.setdefault(
        clip.sha256,
        probe_video(Path(clip.path)),
    )
    media = source_media_cache[clip.sha256]
    display_sar = (
        media.video.display_sample_aspect_ratio.numerator
        / media.video.display_sample_aspect_ratio.denominator
    )
    geometry: dict[str, Any] = {
        "requested_zoom": None,
        "geometry_safe_max_zoom": None,
        "applied_zoom": 1.0,
        "fallback_reason": None,
        "risk_codes": [],
        "requires_gemini_review": False,
    }
    filter_graph = _horizontal_original_filter()
    debug: Path | None = None
    track_fingerprint: str | None = None
    strategy = str(option.get("strategy") or "original")
    if strategy == "tracked_reframe":
        target = str(option.get("target_description") or "").strip()
        if not target:
            raise ValueError("tracked horizontal candidate has no target")
        cache_key = (frame.frame_id, target, start_ms, end_ms)
        track_root = (
            output_dir
            / "geometry"
            / selected.feature_id
            / "horizontal"
            / str(option.get("candidate_id") or "candidate")
        )
        if cache_key not in track_cache:
            proposal, track = _build_track(
                client=client,
                clip=clip,
                frame=frame,
                start_ms=start_ms,
                end_ms=end_ms,
                feature_id=selected.feature_id,
                event_description=(
                    brief_chapter.title
                    + "；"
                    + str(
                        option.get("observed_visual_evidence")
                        or selected.observed_visual_evidence
                    )
                ),
                target_description=target,
                checkpoint_path=checkpoint_path,
                grounding_prompt=grounding_prompt,
                output_dir=track_root,
                run_id=f"feature-h-{uuid.uuid4().hex[:8]}",
                analysis_fps=sam_analysis_fps,
                scdet_threshold=scdet_threshold,
                model_request_block_reason=model_request_block_reason,
            )
            track_cache[cache_key] = (proposal, track, track_root)
        _, track, track_root = track_cache[cache_key]
        track_fingerprint = _track_geometry_fingerprint(track)
        filter_graph, geometry = _horizontal_filter_from_track(
            track,
            str(option.get("zoom_intent") or "subtle"),
            display_sample_aspect_ratio=display_sar,
            camera_intent=str(option.get("camera_intent") or "hold"),
        )
        fallback_reason = geometry.get("fallback_reason")
        if fallback_reason == "mask_geometry_left_no_safe_zoom_margin":
            # A horizontal virtual-camera move is an editorial enhancement,
            # not an evidence-containment requirement. The geometry solver
            # already returns the original full-frame filter when no safe zoom
            # exists, so keep the valid evidence and expose the unapplied move
            # for review instead of exhausting Top-K.
            geometry.setdefault("risk_codes", []).append(
                "horizontal_virtual_camera_fallback_to_original"
            )
            geometry["requires_gemini_review"] = True
        elif fallback_reason is not None:
            raise ValueError(
                "horizontal candidate geometry could not execute: "
                + str(fallback_reason)
            )
        if "virtual_camera_plan" in geometry:
            write_json(
                track_root / "virtual-camera-plan.json",
                geometry["virtual_camera_plan"],
            )
        debug = track_root / "grounding-debug.png"
    return {
        "option": dict(option),
        "frame": frame,
        "clip": clip,
        "media": media,
        "source_has_audio": source_audio_cache[clip.sha256],
        "start_ms": start_ms,
        "end_ms": end_ms,
        "shot_id": shot_id,
        "trim": trim,
        "filter": filter_graph,
        "geometry": geometry,
        "debug": debug,
        "track_fingerprint": track_fingerprint,
    }


def _candidate_asset_reference_matches(
    expected_asset: str | None,
    clip: RushClip,
) -> bool:
    """Accept either catalog identity or its content-addressed projection."""

    if expected_asset is None:
        return True
    return expected_asset in {
        clip.clip_id,
        f"sha256:{clip.sha256}",
    }


def _feature_vertical_candidate_from_runtime_option(
    option_data: Mapping[str, Any],
) -> FeatureVerticalCandidate:
    """Validate the immutable candidate without transient runtime audit fields."""

    return FeatureVerticalCandidate.model_validate(
        {
            field_name: option_data[field_name]
            for field_name in FeatureVerticalCandidate.model_fields
            if field_name in option_data
        }
    )


def _resolve_vertical_camera_phases(
    *,
    option_data: Mapping[str, Any],
    reviewed_phases: Sequence[VerticalVirtualCameraPhase],
) -> tuple[list[VerticalVirtualCameraPhase], Literal["human_reviewed", "gemini_proposed"]]:
    """Convert an editorial proposal into fail-closed executable phase inputs.

    Human-reviewed phases take precedence.  Automatic proposals can only request
    full anchor visibility; they cannot authorize clipping, invent coordinates,
    or bypass the downstream Grounding/SAM and motion gates.
    """

    if reviewed_phases:
        return list(reviewed_phases), "human_reviewed"
    raw_proposal = option_data.get("virtual_camera_proposal")
    if raw_proposal is None:
        return [], "gemini_proposed"
    proposal = VerticalVirtualCameraProposal.model_validate(raw_proposal)
    phases = [
        VerticalVirtualCameraPhase(
            phase_id=phase.phase_id,
            start_progress=phase.start_progress,
            end_progress=phase.end_progress,
            anchor_region_ids=phase.anchor_region_ids,
            camera_behavior=phase.camera_behavior,
            transition_in=phase.transition_in,
            transition_duration_fraction=phase.transition_duration_fraction,
            minimum_anchor_visible_fraction=1.0,
            editorial_reason=(
                f"{phase.editorial_reason} Visible predicate: "
                f"{phase.observable_predicate} Transition condition: "
                f"{phase.transition_condition}"
            ),
        )
        for phase in proposal.phases
    ]
    return phases, "gemini_proposed"


def _audit_feature_plan_candidate_recall(
    plan: FeatureEditPlan,
    *,
    frame_source_assets: Mapping[str, str],
) -> dict[str, Any]:
    """Describe candidate depth and repeated rank-one sources without judging taste.

    Reusing a strong source can be editorially correct.  The audit therefore
    never rejects repetition by itself; it exposes whether the planner preserved
    alternatives and whether repeated rank-one choices carry an explicit reason.
    """

    rows: list[dict[str, Any]] = []
    selected_sources: dict[str, list[tuple[str, str | None]]] = {
        "16x9": [],
        "9x16": [],
    }
    for chapter in plan.chapters:
        horizontal_sources = {
            candidate.source_asset_id for candidate in chapter.horizontal_candidates
        }
        vertical_sources = {
            candidate.source_asset_id for candidate in chapter.vertical_candidates
        }
        applicable = chapter.evidence_status in {"supported", "partial"}
        horizontal_count = len(chapter.horizontal_candidates)
        vertical_count = len(chapter.vertical_candidates)
        top_k_complete = (
            not applicable or (horizontal_count >= 2 and vertical_count >= 2)
        )
        horizontal_selected_source = (
            frame_source_assets.get(chapter.horizontal_frame_id)
            if chapter.horizontal_frame_id is not None
            else None
        )
        vertical_selected_source = (
            frame_source_assets.get(chapter.vertical_frame_id)
            if chapter.vertical_frame_id is not None
            else None
        )
        if horizontal_selected_source is not None:
            selected_sources["16x9"].append(
                (chapter.feature_id, horizontal_selected_source)
            )
        if vertical_selected_source is not None:
            selected_sources["9x16"].append(
                (chapter.feature_id, vertical_selected_source)
            )
        rows.append(
            {
                "feature_id": chapter.feature_id,
                "evidence_status": chapter.evidence_status,
                "horizontal_candidate_count": horizontal_count,
                "vertical_candidate_count": vertical_count,
                "horizontal_distinct_source_count": len(horizontal_sources),
                "vertical_distinct_source_count": len(vertical_sources),
                "top_k_complete": top_k_complete,
                "rank_one_only": applicable and not (
                    chapter.horizontal_candidates or chapter.vertical_candidates
                ),
                "horizontal_selected_source_asset_id": horizontal_selected_source,
                "vertical_selected_source_asset_id": vertical_selected_source,
                "source_reuse_mode": chapter.source_reuse_mode,
                "source_reuse_justification": chapter.source_reuse_justification,
            }
        )

    reuse_groups: dict[str, list[dict[str, Any]]] = {}
    for aspect, selections in selected_sources.items():
        by_source: dict[str, list[str]] = {}
        for feature_id, source_asset_id in selections:
            if source_asset_id is None:
                continue
            by_source.setdefault(source_asset_id, []).append(feature_id)
        reuse_groups[aspect] = [
            {
                "source_asset_id": source_asset_id,
                "feature_ids": feature_ids,
                "chapter_count": len(feature_ids),
                "all_reuses_explained": all(
                    bool(
                        next(
                            (
                                chapter.source_reuse_mode != "none"
                                and chapter.source_reuse_justification
                            )
                            for chapter in plan.chapters
                            if chapter.feature_id == feature_id
                        )
                    )
                    for feature_id in feature_ids[1:]
                ),
            }
            for source_asset_id, feature_ids in sorted(by_source.items())
            if len(feature_ids) > 1
        ]

    applicable_rows = [
        row for row in rows if row["evidence_status"] in {"supported", "partial"}
    ]
    body = {
        "contract_version": "feature-plan-candidate-audit-v1",
        "applicable_chapter_count": len(applicable_rows),
        "top_k_complete_chapter_count": sum(
            bool(row["top_k_complete"]) for row in applicable_rows
        ),
        "candidate_recall_complete": all(
            bool(row["top_k_complete"]) for row in applicable_rows
        ),
        "rank_one_only_chapter_count": sum(
            bool(row["rank_one_only"]) for row in applicable_rows
        ),
        "selection_repetition_review_required": any(
            not bool(group["all_reuses_explained"])
            for groups in reuse_groups.values()
            for group in groups
        ),
        "reuse_groups": reuse_groups,
        "chapters": rows,
    }
    return {**body, "audit_sha256": _stable_fingerprint(body)}


def _audit_render_source_reuse(
    plan: FeatureEditPlan,
    chapters: Sequence[Mapping[str, Any]],
    *,
    aspect: Literal["16x9", "9x16"],
) -> dict[str, Any]:
    """Validate typed reuse authority against the rendered source intervals.

    A representative catalog frame cannot prove whether two final trims overlap.
    This audit therefore runs after exact source intervals and presentation
    fingerprints are known. Deliberate reprises remain reviewable; silent
    duration padding and falsely claimed distinct intervals fail closed.
    """

    plan_by_id = {chapter.feature_id: chapter for chapter in plan.chapters}
    prior_by_source: dict[str, list[Mapping[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    skipped_non_source_chapters: list[str] = []
    for chapter in chapters:
        feature_id = str(chapter["feature_id"])
        if (
            chapter.get("source_clip_id") is None
            or chapter.get("source_in_ms") is None
            or chapter.get("source_out_ms") is None
        ):
            skipped_non_source_chapters.append(feature_id)
            continue
        source_clip_id = str(chapter["source_clip_id"])
        selected = plan_by_id[feature_id]
        current_start = int(chapter["source_in_ms"])
        current_end = int(chapter["source_out_ms"])
        current_fingerprint = chapter.get("segment_render_fingerprint")
        for prior in prior_by_source.get(source_clip_id, []):
            prior_start = int(prior["source_in_ms"])
            prior_end = int(prior["source_out_ms"])
            overlap_ms = max(
                0,
                min(current_end, prior_end) - max(current_start, prior_start),
            )
            exact_interval = (
                current_start == prior_start and current_end == prior_end
            )
            same_presentation = (
                exact_interval
                and current_fingerprint
                == prior.get("segment_render_fingerprint")
            )
            row = {
                "aspect": aspect,
                "feature_id": feature_id,
                "prior_feature_id": prior["feature_id"],
                "source_clip_id": source_clip_id,
                "source_in_ms": current_start,
                "source_out_ms": current_end,
                "prior_source_in_ms": prior_start,
                "prior_source_out_ms": prior_end,
                "overlap_ms": overlap_ms,
                "exact_interval_repeat": exact_interval,
                "same_presentation": same_presentation,
                "reuse_mode": selected.source_reuse_mode,
                "justification": selected.source_reuse_justification,
                "requires_human_review": (
                    selected.source_reuse_mode
                    in {"alternate_presentation", "editorial_reprise"}
                    or overlap_ms > 0
                ),
            }
            row_violates = (
                selected.source_reuse_mode == "none"
                or not (
                    selected.source_reuse_justification
                    and selected.source_reuse_justification.strip()
                )
                or (
                    selected.source_reuse_mode == "distinct_interval"
                    and overlap_ms > 0
                )
                or (
                    selected.source_reuse_mode == "alternate_presentation"
                    and same_presentation
                )
            )
            row["status"] = "blocked" if row_violates else "authorized_review"
            rows.append(row)
            if row_violates:
                violations.append(row)
        prior_by_source.setdefault(source_clip_id, []).append(chapter)
    body = {
        "contract_version": "render-source-reuse-audit-v1",
        "aspect": aspect,
        "status": "blocked" if violations else "passed",
        "rows": rows,
        "violations": violations,
        "requires_human_review": any(
            bool(row["requires_human_review"]) for row in rows
        ),
        "unique_source_clip_count": len(prior_by_source),
        "rendered_chapter_count": len(chapters),
        "audited_source_chapter_count": sum(
            len(items) for items in prior_by_source.values()
        ),
        "skipped_non_source_chapters": skipped_non_source_chapters,
        "capacity_policy": (
            "authorized reuse may contribute output duration but never increases "
            "the reported unique source capacity"
        ),
    }
    return {**body, "audit_sha256": _stable_fingerprint(body)}


def _audit_requested_candidate_recall(
    plan: FeatureEditPlan,
    *,
    aspect: RenderAspect,
) -> dict[str, Any]:
    """Measure executable Top-K depth only for requested delivery aspects."""

    render_horizontal, render_vertical = _requested_render_aspects(aspect)
    rows: list[dict[str, Any]] = []
    for chapter in plan.chapters:
        if chapter.evidence_status == "not_found":
            continue
        aspect_counts: dict[str, int] = {}
        if render_horizontal:
            aspect_counts["16x9"] = (
                len(chapter.horizontal_candidates)
                if chapter.horizontal_candidates
                else int(chapter.horizontal_frame_id is not None)
            )
        if render_vertical:
            aspect_counts["9x16"] = (
                len(chapter.vertical_candidates)
                if chapter.vertical_candidates
                else int(chapter.vertical_frame_id is not None)
            )
        incomplete = [
            requested_aspect
            for requested_aspect, count in aspect_counts.items()
            if count < 2
        ]
        rows.append(
            {
                "feature_id": chapter.feature_id,
                "evidence_status": chapter.evidence_status,
                "candidate_counts": aspect_counts,
                "incomplete_aspects": incomplete,
                "complete": not incomplete,
            }
        )
    body = {
        "contract_version": "requested-candidate-recall-audit-v1",
        "requested_aspect": aspect,
        "complete": all(bool(row["complete"]) for row in rows),
        "rows": rows,
        "policy": (
            "production-review requires at least two evidence-bound candidates "
            "for every requested aspect until a typed only-evidence exception "
            "contract exists"
        ),
    }
    return {**body, "audit_sha256": _stable_fingerprint(body)}


def _audit_requested_quality_map_coverage(
    plan: FeatureEditPlan,
    *,
    aspect: RenderAspect,
    frames: Mapping[str, RushFrame],
    clips: Mapping[str, RushClip],
    shot_cache: dict[str, ShotManifest],
    shots_dir: Path,
    scdet_threshold: float,
    quality_maps: Sequence[tuple[Path, ShotQualityMap]],
    human_policy_binding_present: bool,
) -> dict[str, Any]:
    """Require quality evidence for every candidate the runtime may attempt."""

    render_horizontal, render_vertical = _requested_render_aspects(aspect)
    requested_frames: list[tuple[str, str, int, str]] = []
    for chapter in plan.chapters:
        if chapter.evidence_status == "not_found":
            continue
        if render_horizontal:
            horizontal_candidates = (
                list(chapter.horizontal_candidates)
                if chapter.horizontal_candidates
                else []
            )
            if horizontal_candidates:
                requested_frames.extend(
                    (
                        chapter.feature_id,
                        "16x9",
                        candidate.rank,
                        candidate.frame_id,
                    )
                    for candidate in horizontal_candidates
                )
            elif chapter.horizontal_frame_id is not None:
                requested_frames.append(
                    (
                        chapter.feature_id,
                        "16x9",
                        1,
                        chapter.horizontal_frame_id,
                    )
                )
        if render_vertical:
            vertical_candidates = (
                list(chapter.vertical_candidates)
                if chapter.vertical_candidates
                and not human_policy_binding_present
                else []
            )
            if vertical_candidates:
                requested_frames.extend(
                    (
                        chapter.feature_id,
                        "9x16",
                        candidate.rank,
                        candidate.frame_id,
                    )
                    for candidate in vertical_candidates
                )
            elif chapter.vertical_frame_id is not None:
                requested_frames.append(
                    (
                        chapter.feature_id,
                        "9x16",
                        1,
                        chapter.vertical_frame_id,
                    )
                )

    rows: list[dict[str, Any]] = []
    if not quality_maps:
        rows = [
            {
                "feature_id": feature_id,
                "aspect": requested_aspect,
                "candidate_rank": rank,
                "frame_id": frame_id,
                "source_asset_id": None,
                "shot_id": None,
                "covered": False,
                "quality_map_path": None,
            }
            for feature_id, requested_aspect, rank, frame_id in requested_frames
        ]
        body = {
            "contract_version": "requested-quality-map-coverage-v1",
            "requested_aspect": aspect,
            "complete": not rows,
            "quality_map_count": 0,
            "candidate_shot_count": len(rows),
            "missing": rows,
            "rows": rows,
            "resolution_status": "not_resolved_without_any_quality_maps",
        }
        return {**body, "audit_sha256": _stable_fingerprint(body)}
    seen: set[tuple[str, str, str]] = set()
    for feature_id, requested_aspect, rank, frame_id in requested_frames:
        frame = frames[frame_id]
        clip = clips[frame.clip_id]
        if clip.clip_id not in shot_cache:
            shot_cache[clip.clip_id] = detect_shots_ffmpeg(
                Path(clip.path),
                threshold=scdet_threshold,
                output_path=shots_dir / f"{clip.clip_id}.json",
            )
        shot = next(
            item
            for item in shot_cache[clip.clip_id].shots
            if item.start_time_ms
            <= frame.requested_time_ms
            < item.end_time_ms
        )
        source_asset_id = f"sha256:{clip.sha256}"
        identity = (requested_aspect, source_asset_id, shot.shot_id)
        if identity in seen:
            continue
        seen.add(identity)
        match = _quality_map_for_shot(
            quality_maps,
            source_asset_id=source_asset_id,
            shot_id=shot.shot_id,
        )
        quality_source_lineage_valid = False
        quality_source_lineage_error: str | None = None
        if match is not None:
            _, quality_map = match
            quality_source_path = Path(quality_map.source_path)
            if not quality_source_path.is_file():
                quality_source_lineage_error = "quality_map_source_missing"
            elif sha256_file(quality_source_path) != clip.sha256:
                quality_source_lineage_error = "quality_map_source_hash_mismatch"
            else:
                quality_source_lineage_valid = True
        rows.append(
            {
                "feature_id": feature_id,
                "aspect": requested_aspect,
                "candidate_rank": rank,
                "frame_id": frame_id,
                "source_asset_id": source_asset_id,
                "shot_id": shot.shot_id,
                "covered": match is not None and quality_source_lineage_valid,
                "quality_map_path": (
                    str(match[0].resolve()) if match is not None else None
                ),
                "quality_source_lineage_valid": (
                    quality_source_lineage_valid
                    if match is not None
                    else None
                ),
                "quality_source_lineage_error": quality_source_lineage_error,
            }
        )
    body = {
        "contract_version": "requested-quality-map-coverage-v1",
        "requested_aspect": aspect,
        "complete": all(bool(row["covered"]) for row in rows),
        "quality_map_count": len(quality_maps),
        "candidate_shot_count": len(rows),
        "missing": [row for row in rows if not bool(row["covered"])],
        "rows": rows,
    }
    return {**body, "audit_sha256": _stable_fingerprint(body)}


def _ensure_requested_quality_maps(
    plan: FeatureEditPlan,
    *,
    aspect: RenderAspect,
    frames: Mapping[str, RushFrame],
    clips: Mapping[str, RushClip],
    shot_cache: dict[str, ShotManifest],
    shots_dir: Path,
    scdet_threshold: float,
    quality_maps: Sequence[tuple[Path, ShotQualityMap]],
    human_policy_binding_present: bool,
    output_dir: Path,
) -> list[tuple[Path, ShotQualityMap]]:
    """Create missing deterministic maps for every production candidate shot."""

    render_horizontal, render_vertical = _requested_render_aspects(aspect)
    frame_ids: list[str] = []
    for chapter in plan.chapters:
        if chapter.evidence_status == "not_found":
            continue
        if render_horizontal:
            frame_ids.extend(
                candidate.frame_id for candidate in chapter.horizontal_candidates
            )
            if not chapter.horizontal_candidates and chapter.horizontal_frame_id:
                frame_ids.append(chapter.horizontal_frame_id)
        if render_vertical:
            vertical_candidates = (
                []
                if human_policy_binding_present
                else list(chapter.vertical_candidates)
            )
            frame_ids.extend(candidate.frame_id for candidate in vertical_candidates)
            if not vertical_candidates and chapter.vertical_frame_id:
                frame_ids.append(chapter.vertical_frame_id)

    resolved = list(quality_maps)
    generated_dir = output_dir / "auto-shot-quality"
    for frame_id in dict.fromkeys(frame_ids):
        frame = frames[frame_id]
        clip = clips[frame.clip_id]
        if clip.clip_id not in shot_cache:
            shot_cache[clip.clip_id] = detect_shots_ffmpeg(
                Path(clip.path),
                threshold=scdet_threshold,
                output_path=shots_dir / f"{clip.clip_id}.json",
            )
        shot_manifest = shot_cache[clip.clip_id]
        shot = next(
            item
            for item in shot_manifest.shots
            if item.start_time_ms
            <= frame.requested_time_ms
            < item.end_time_ms
        )
        source_asset_id = f"sha256:{clip.sha256}"
        if _quality_map_for_shot(
            resolved,
            source_asset_id=source_asset_id,
            shot_id=shot.shot_id,
        ) is not None:
            continue
        output_path = (
            generated_dir
            / clip.sha256[:16]
            / f"{shot.shot_id}.quality-map.json"
        )
        if output_path.is_file():
            quality_map = load_shot_quality_map(output_path)[1]
            if (
                quality_map.source_asset_id != source_asset_id
                or quality_map.shot_id != shot.shot_id
                or sha256_file(Path(quality_map.source_path)) != clip.sha256
            ):
                raise ValueError(
                    "cached automatic ShotQualityMap has stale source lineage"
                )
        else:
            quality_map = scan_shot_quality(
                Path(clip.path),
                shot_manifest=shot_manifest,
                shot_id=shot.shot_id,
                output_path=output_path,
            )
        resolved.append((output_path.resolve(), quality_map))
    return resolved


def _production_review_preflight_failures(
    candidate_recall_audit: Mapping[str, Any],
    quality_map_coverage_audit: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    if not bool(candidate_recall_audit.get("complete")):
        failures.append("candidate_recall_incomplete")
    if not bool(quality_map_coverage_audit.get("complete")):
        failures.append("quality_map_coverage_incomplete")
    return failures


def _build_feature_cut_eligibility_report(
    manifest: Mapping[str, Any],
    *,
    execution_profile: FeatureCutExecutionProfile,
) -> FeatureCutEligibilityReport:
    """Separate playable review media from semantic delivery eligibility."""

    requested_aspects = [
        aspect_name
        for aspect_name in ("horizontal", "vertical")
        if bool(manifest.get(aspect_name, {}).get("requested"))
    ]
    media_rendered = bool(requested_aspects) and all(
        manifest.get(aspect_name, {}).get("status") == "rendered"
        for aspect_name in requested_aspects
    )
    chapters = [
        chapter
        for aspect_name in requested_aspects
        for chapter in manifest.get(aspect_name, {}).get("chapters", [])
        if isinstance(chapter, Mapping)
    ]
    evidence_complete = not any(
        chapter.get("fallback_reason") == "catalog_evidence_not_found"
        or chapter.get("source_clip_id") is None
        for chapter in chapters
    )
    candidate_recall_audit = manifest.get("requested_candidate_recall_audit", {})
    candidate_recall_complete = bool(
        candidate_recall_audit.get("complete")
    )
    candidate_resolution_passed = not any(
        "automatic_candidate_exhaustion"
        in (chapter.get("risk_codes") or [])
        or chapter.get("fallback_reason") == "all_automatic_candidates_exhausted"
        or bool(
            (
                chapter.get("automatic_candidate_selection")
                if isinstance(
                    chapter.get("automatic_candidate_selection"), Mapping
                )
                else {}
            ).get("center_crop_used_as_unverified_fallback")
        )
        for chapter in chapters
    )
    source_reuse_contract_passed = bool(
        manifest.get("source_reuse_contract_passed", True)
    )
    quality_audit = manifest.get("quality_map_coverage_audit", {})
    quality_complete = bool(quality_audit.get("complete"))
    quality_status = (
        EligibilityGateStatus.PASSED
        if quality_complete
        else (
            EligibilityGateStatus.FAILED
            if execution_profile
            == FeatureCutExecutionProfile.PRODUCTION_REVIEW
            else EligibilityGateStatus.NOT_RUN
        )
    )
    geometry_execution_passed = not any(
        chapter.get("unverified_geometry_preview_override") is True
        or chapter.get("human_policy_execution_verified") is False
        or "human_policy_geometry_failed" in (chapter.get("risk_codes") or [])
        or "tracking_or_grounding_failed" in (chapter.get("risk_codes") or [])
        or (
            isinstance(chapter.get("auto_bounded_clip_audit"), Mapping)
            and not bool(
                chapter["auto_bounded_clip_audit"].get("approved")
            )
        )
        for chapter in chapters
    )
    human_policy_present = manifest.get("reframe_policy_binding") is not None
    human_execution_verified = (
        all(
            chapter.get("source_clip_id") is None
            or chapter.get("human_policy_execution_verified") is True
            for chapter in manifest.get("vertical", {}).get("chapters", [])
        )
        if human_policy_present
        else True
    )
    technical = manifest.get("post_render_quality_qc", {})
    technical_requested = bool(technical.get("requested"))
    technical_passed = bool(technical.get("technical_qc_passed"))
    technical_status = (
        EligibilityGateStatus.PASSED
        if technical_requested and technical_passed
        else (
            EligibilityGateStatus.FAILED
            if technical_requested
            or execution_profile
            == FeatureCutExecutionProfile.PRODUCTION_REVIEW
            else EligibilityGateStatus.NOT_RUN
        )
    )
    contract = FeatureCutEditorialContract(
        evidence_complete=(
            EligibilityGateStatus.PASSED
            if evidence_complete
            else EligibilityGateStatus.FAILED
        ),
        candidate_recall_complete=(
            EligibilityGateStatus.PASSED
            if candidate_recall_complete
            else EligibilityGateStatus.FAILED
        ),
        candidate_resolution_passed=(
            EligibilityGateStatus.PASSED
            if candidate_resolution_passed
            else EligibilityGateStatus.FAILED
        ),
        quality_coverage_complete=quality_status,
        geometry_execution_passed=(
            EligibilityGateStatus.PASSED
            if geometry_execution_passed
            else EligibilityGateStatus.FAILED
        ),
        human_intent_execution_verified=(
            (
                EligibilityGateStatus.PASSED
                if human_execution_verified
                else EligibilityGateStatus.FAILED
            )
            if human_policy_present
            else EligibilityGateStatus.NOT_REQUIRED
        ),
        technical_quality_passed=technical_status,
        final_sequence_qa_passed=EligibilityGateStatus.NOT_RUN,
        human_approval_passed=EligibilityGateStatus.NOT_RUN,
    )
    blocking_reasons: list[str] = []
    review_reasons = [
        "final_sequence_qa_not_run",
        "human_approval_not_run",
    ]
    if not evidence_complete:
        blocking_reasons.append("required_evidence_incomplete")
    if not candidate_resolution_passed:
        blocking_reasons.append("candidate_resolution_failed")
    if not source_reuse_contract_passed:
        blocking_reasons.append("source_reuse_contract_failed")
    if not geometry_execution_passed:
        blocking_reasons.append("geometry_execution_unverified")
    if human_policy_present and not human_execution_verified:
        blocking_reasons.append("human_intent_execution_not_verified")
    if not candidate_recall_complete:
        (
            blocking_reasons
            if execution_profile
            == FeatureCutExecutionProfile.PRODUCTION_REVIEW
            else review_reasons
        ).append("candidate_recall_incomplete")
    if quality_status != EligibilityGateStatus.PASSED:
        (
            blocking_reasons
            if execution_profile
            == FeatureCutExecutionProfile.PRODUCTION_REVIEW
            else review_reasons
        ).append("quality_map_coverage_incomplete")
    if technical_status != EligibilityGateStatus.PASSED:
        (
            blocking_reasons
            if execution_profile
            == FeatureCutExecutionProfile.PRODUCTION_REVIEW
            else review_reasons
        ).append("technical_quality_not_verified")

    if not media_rendered:
        run_state = FeatureCutRunState.FAILED
    elif not evidence_complete:
        run_state = FeatureCutRunState.PARTIAL
    elif (
        blocking_reasons
        or not candidate_recall_complete
        or quality_status != EligibilityGateStatus.PASSED
        or technical_status != EligibilityGateStatus.PASSED
    ):
        run_state = FeatureCutRunState.REVIEW_PREVIEW
    else:
        run_state = FeatureCutRunState.READY_FOR_HUMAN_REVIEW
    ready_for_human_review = (
        run_state == FeatureCutRunState.READY_FOR_HUMAN_REVIEW
    )
    return FeatureCutEligibilityReport(
        execution_profile=execution_profile,
        media_rendered=media_rendered,
        run_state=run_state,
        ready_for_human_review=ready_for_human_review,
        delivery_eligible=False,
        editorial_contract=contract,
        blocking_reasons=blocking_reasons,
        review_reasons=review_reasons,
        generated_at=utc_now(),
    )


def _summarize_automatic_reframe(
    vertical_chapters: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a compact, deterministic handoff gate from chapter audit trails."""

    chapters: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}
    total_attempts = 0
    for chapter in vertical_chapters:
        routing = chapter.get("automatic_candidate_selection")
        routing = routing if isinstance(routing, Mapping) else {}
        attempts_value = routing.get("attempts")
        attempts = attempts_value if isinstance(attempts_value, list) else []
        total_attempts += len(attempts)
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                continue
            codes = attempt.get("failure_codes")
            if not isinstance(codes, list):
                continue
            for code in codes:
                if isinstance(code, str) and code:
                    failure_counts[code] = failure_counts.get(code, 0) + 1
        selected_rank = routing.get("selected_candidate_rank")
        policy_blocked = chapter.get("applied_strategy") in {
            "policy_blocked_preview_fit",
            "policy_blocked_preview_solid_fit",
            "policy_blocked_preview_center_crop",
        }
        review_required = bool(chapter.get("requires_gemini_review"))
        applied_strategy = chapter.get("applied_strategy")
        scope_preserving_fit = applied_strategy in {
            "fit_with_solid_matte",
            "required_scope_solid_fit",
            "policy_blocked_preview_fit",
            "policy_blocked_preview_solid_fit",
        }
        portrait_crop = applied_strategy in {
            "tracked_crop",
            "seed_anchor_crop",
            "center_crop",
            "full_bleed_center_crop_review",
            "unverified_center_crop_preview",
            "policy_blocked_preview_center_crop",
        }
        chapters.append(
            {
                "feature_id": chapter.get("feature_id"),
                "automatic_routing_enabled": bool(routing.get("enabled")),
                "planned_candidate_count": routing.get(
                    "planned_candidate_count", 0
                ),
                "selected_candidate_id": routing.get("selected_candidate_id"),
                "selected_candidate_rank": selected_rank,
                "candidate_switch_applied": bool(
                    isinstance(selected_rank, int) and selected_rank > 1
                ),
                "candidate_attempt_count": len(attempts),
                "applied_strategy": applied_strategy,
                "portrait_crop_applied": portrait_crop,
                "scope_preserving_fit_applied": scope_preserving_fit,
                "policy_blocked": policy_blocked,
                "review_required": review_required,
            }
        )
    body = {
        "contract_version": "full-auto-reframe-summary-v1",
        "chapter_count": len(chapters),
        "automatic_routing_chapter_count": sum(
            bool(item["automatic_routing_enabled"]) for item in chapters
        ),
        "candidate_attempt_count": total_attempts,
        "candidate_switch_count": sum(
            bool(item["candidate_switch_applied"]) for item in chapters
        ),
        "candidate_recall_incomplete_chapter_count": sum(
            int(item["planned_candidate_count"]) < 2 for item in chapters
        ),
        "portrait_crop_chapter_count": sum(
            bool(item["portrait_crop_applied"]) for item in chapters
        ),
        "scope_preserving_fit_chapter_count": sum(
            bool(item["scope_preserving_fit_applied"]) for item in chapters
        ),
        "policy_blocked_chapter_count": sum(
            bool(item["policy_blocked"]) for item in chapters
        ),
        "review_required_chapter_count": sum(
            bool(item["review_required"]) for item in chapters
        ),
        "failure_code_counts": dict(sorted(failure_counts.items())),
        "chapters": chapters,
    }
    return {**body, "summary_sha256": _stable_fingerprint(body)}


def _requested_render_aspects(aspect: str) -> tuple[bool, bool]:
    """Return horizontal/vertical render gates and reject unknown API values."""

    if aspect not in {"both", "9x16", "16x9"}:
        raise ValueError("aspect must be one of: both, 9x16, 16x9")
    return aspect in {"both", "16x9"}, aspect in {"both", "9x16"}


def _horizontal_manifest_entry(
    *,
    selected: FeatureChapterSelect,
    brief_chapter: FeatureChapterBrief,
    frame: RushFrame,
    clip: RushClip,
    start_ms: int,
    end_ms: int,
    shot_id: str,
    segment_fingerprint: str,
    track_fingerprint: str | None,
    segment: Path,
    source_has_audio: bool,
    source_media: MediaInfo,
    grounding_debug: Path | None,
    trim: Mapping[str, Any],
    geometry: Mapping[str, Any],
    source_interval: Mapping[str, Any],
    render_boundary_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "feature_id": selected.feature_id,
        "semantic_intent": (
            brief_chapter.title
            + (
                " — " + "; ".join(brief_chapter.detail_lines)
                if brief_chapter.detail_lines
                else ""
            )
        ),
        "observed_visual_evidence": selected.observed_visual_evidence,
        "selection_reason": selected.selection_reason,
        "source_reuse_mode": selected.source_reuse_mode,
        "source_reuse_justification": selected.source_reuse_justification,
        "source_frame_id": frame.frame_id,
        "source_clip_id": clip.clip_id,
        "source_in_ms": start_ms,
        "source_out_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "source_shot_id": shot_id,
        "source_interval": dict(source_interval),
        "render_boundary_lineage": dict(render_boundary_lineage),
        "segment_render_fingerprint": segment_fingerprint,
        "track_geometry_fingerprint": track_fingerprint,
        "segment_path": str(segment.resolve()),
        "audio_origin": "source" if source_has_audio else "synthetic_silence",
        "source_sample_aspect_ratio": (
            source_media.video.sample_aspect_ratio.model_dump(mode="json")
        ),
        "source_display_sample_aspect_ratio": (
            source_media.video.display_sample_aspect_ratio.model_dump(mode="json")
        ),
        "grounding_debug": (
            str(grounding_debug.resolve()) if grounding_debug else None
        ),
        **trim,
        **geometry,
    }


def _render_review_html(
    output_dir: Path,
    brief: FeatureEditBrief,
    plan: FeatureEditPlan,
    manifest: dict[str, Any],
) -> None:
    overlay_note = (
        "成片不燒錄實驗字卡；使用者 brief 只作審查 metadata。"
        if not brief.render_title_overlays
        else "成片字卡來自使用者 editorial brief。"
    )
    render_horizontal = bool(manifest["horizontal"].get("requested", True))
    render_vertical = bool(manifest["vertical"].get("requested", True))
    rows: list[str] = []
    by_id = {chapter.feature_id: chapter for chapter in plan.chapters}
    for brief_chapter in brief.chapters:
        selected = by_id[brief_chapter.feature_id]
        vertical = next(
            (
                item
                for item in manifest["vertical"]["chapters"]
                if item["feature_id"] == brief_chapter.feature_id
            ),
            {},
        )
        horizontal = next(
            (
                item
                for item in manifest["horizontal"]["chapters"]
                if item["feature_id"] == brief_chapter.feature_id
            ),
            {},
        )
        debug_paths = list(vertical.get("grounding_debugs") or [])
        if not debug_paths and vertical.get("grounding_debug"):
            debug_paths = [vertical["grounding_debug"]]
        debug_links: list[str] = []
        for debug_index, debug_path in enumerate(debug_paths, start=1):
            relative_debug = Path(debug_path).relative_to(output_dir.resolve())
            debug_links.append(
                f'<a href="{html.escape(str(relative_debug))}">bbox {debug_index}</a>'
            )
        debug_link = " · ".join(debug_links) or "—"
        rows.append(
            "<tr>"
            f"<td>{html.escape(brief_chapter.title)}</td>"
            f"<td>{html.escape(selected.evidence_status)}</td>"
            f"<td>{html.escape(str(selected.horizontal_frame_id) if render_horizontal else 'not requested')}</td>"
            f"<td>{html.escape(str(horizontal.get('applied_zoom', 1.0)) if render_horizontal else 'not requested')}</td>"
            f"<td>{html.escape(str(horizontal.get('trim_method', 'not_applicable')) if render_horizontal else 'not requested')}</td>"
            f"<td>{html.escape(str(selected.vertical_frame_id) if render_vertical else 'not requested')}</td>"
            f"<td>{html.escape(str(vertical.get('applied_strategy', 'not requested')))}</td>"
            f"<td>{html.escape(str(vertical.get('trim_method', 'not_applicable')) if render_vertical else 'not requested')}</td>"
            f"<td>{debug_link}</td>"
            f"<td>{html.escape(selected.observed_visual_evidence)}</td>"
            f"<td>{html.escape('; '.join(selected.quality_risks) or 'none')}</td>"
            "</tr>"
        )
    video_sections: list[str] = []
    if render_horizontal:
        horizontal_path = Path(manifest["horizontal"]["output_path"]).relative_to(
            output_dir.resolve()
        )
        video_sections.append(
            f'<section><h2>16:9</h2><video controls src="{html.escape(str(horizontal_path))}"></video></section>'
        )
    if render_vertical:
        vertical_path = Path(manifest["vertical"]["output_path"]).relative_to(
            output_dir.resolve()
        )
        video_sections.append(
            f'<section><h2>9:16</h2><video controls src="{html.escape(str(vertical_path))}"></video></section>'
        )
    (output_dir / "index.html").write_text(
        """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Feature cut review</title>
<style>body{font:15px system-ui;background:#101214;color:#eee;max-width:1500px;margin:24px auto;padding:0 20px}section{background:#1b1f24;padding:20px;margin:20px 0;border-radius:12px}video{width:min(100%,960px);max-height:76vh;background:#000}table{border-collapse:collapse;width:100%}th,td{border:1px solid #3b424a;padding:8px;text-align:left;vertical-align:top}a{color:#71e59c}</style>
<h1>Feature cut review</h1><p>"""
        + html.escape(overlay_note)
        + " 畫面證據、frame ID、Gemini bbox、SAM tracking 與 fallback 分開保存。</p>"
        + "".join(video_sections)
        + "<table><thead><tr><th>chapter</th><th>evidence</th><th>16:9 frame</th><th>zoom</th><th>16:9 trim</th><th>9:16 frame</th><th>vertical</th><th>9:16 trim</th><th>debug</th><th>observed evidence</th><th>risks</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></html>",
        encoding="utf-8",
    )


def _selected_source_capacity_seconds(
    plan: FeatureEditPlan,
    *,
    aspect: RenderAspect,
    frames: Mapping[str, RushFrame],
    clips: Mapping[str, RushClip],
    shot_cache: dict[str, ShotManifest],
    shots_dir: Path,
    scdet_threshold: float,
    approved_decisions: Sequence[tuple[Path, TrimIntentDecision]] = (),
    quality_maps: Sequence[tuple[Path, ShotQualityMap]] = (),
) -> dict[str, float]:
    """Return the largest executable capacity among requested-aspect candidates.

    Every candidate is measured against the interval containing its own evidence
    anchor.  Planning may use the largest candidate capacity because the runtime
    router rejects shorter candidates for the assigned dwell and tries the next
    option.  It must never collapse the chapter to the shortest fallback.
    """

    render_horizontal, render_vertical = _requested_render_aspects(aspect)
    capacities: dict[str, float] = {}
    for selected in plan.chapters:
        if selected.evidence_status == "not_found":
            continue
        aspect_frames: list[tuple[str, str | None, str | None]] = []
        if render_horizontal:
            if selected.horizontal_candidates:
                aspect_frames.extend(
                    ("16:9", candidate.frame_id, candidate.event_id)
                    for candidate in selected.horizontal_candidates
                )
            else:
                aspect_frames.append(
                    ("16:9", selected.horizontal_frame_id, None)
                )
        if render_vertical:
            if selected.vertical_candidates:
                aspect_frames.extend(
                    ("9:16", candidate.frame_id, candidate.event_id)
                    for candidate in selected.vertical_candidates
                )
            else:
                aspect_frames.append(
                    ("9:16", selected.vertical_frame_id, None)
                )
        chapter_capacities: dict[str, list[int]] = {
            "16:9": [],
            "9:16": [],
        }
        seen: set[tuple[str, str, str | None]] = set()
        for selected_aspect, frame_id, candidate_event_id in aspect_frames:
            if frame_id is None:
                continue
            frame = frames[frame_id]
            clip = clips[frame.clip_id]
            if clip.clip_id not in shot_cache:
                shot_cache[clip.clip_id] = detect_shots_ffmpeg(
                    Path(clip.path),
                    threshold=scdet_threshold,
                    output_path=shots_dir / f"{clip.clip_id}.json",
                )
            shot = next(
                item
                for item in shot_cache[clip.clip_id].shots
                if item.start_time_ms
                <= frame.requested_time_ms
                < item.end_time_ms
            )
            event_id = candidate_event_id or _selected_event_id(
                selected,
                frame_id=frame_id,
                aspect=selected_aspect,
            )
            source_asset_id = f"sha256:{clip.sha256}"
            identity = (source_asset_id, shot.shot_id, event_id)
            if identity in seen:
                continue
            seen.add(identity)
            interval_start = max(shot.start_time_ms, 0)
            interval_end = min(shot.end_time_ms, clip.duration_ms)
            trim_match = _matching_trim_decision(
                approved_decisions,
                source_asset_id=source_asset_id,
                shot_id=shot.shot_id,
                event_id=event_id,
            )
            if trim_match is not None:
                _, decision = trim_match
                assert decision.source_in_ms is not None
                assert decision.source_out_ms is not None
                interval_start = decision.source_in_ms
                interval_end = decision.source_out_ms
            quality_match = _quality_map_for_shot(
                quality_maps,
                source_asset_id=source_asset_id,
                shot_id=shot.shot_id,
            )
            if quality_maps and quality_match is None:
                raise ValueError(
                    "ShotQualityMap coverage is incomplete for a selected source "
                    f"shot: {source_asset_id}/{shot.shot_id}"
                )
            if quality_match is not None:
                quality_path, quality_map = quality_match
                if sha256_file(Path(quality_map.source_path)) != clip.sha256:
                    raise ValueError(
                        "ShotQualityMap source hash differs from the selected clip"
                    )
                safe_intervals = build_quality_safe_intervals(
                    quality_map,
                    quality_map_sha256=sha256_file(quality_path),
                    allowed_start_ms=interval_start,
                    allowed_end_ms=interval_end,
                )
                capacity_ms = max(
                    (
                        interval.end_ms - interval.start_ms
                        for interval in safe_intervals
                        if interval.start_ms
                        <= frame.requested_time_ms
                        < interval.end_ms
                    ),
                    default=0,
                )
            else:
                capacity_ms = interval_end - interval_start
            chapter_capacities[selected_aspect].append(max(0, capacity_ms))
        requested_maxima = [
            max(chapter_capacities[selected_aspect])
            for selected_aspect, requested in (
                ("16:9", render_horizontal),
                ("9:16", render_vertical),
            )
            if requested and chapter_capacities[selected_aspect]
        ]
        if requested_maxima:
            # One shared chapter duration must be executable in every requested
            # aspect, while each aspect may independently choose its own best
            # Top-K candidate. A global max would let a long horizontal take
            # over-allocate a chapter that no vertical candidate can sustain.
            capacities[selected.feature_id] = min(requested_maxima) / 1000
    return capacities


def _selected_fixed_trim_durations_seconds(
    plan: FeatureEditPlan,
    *,
    aspect: RenderAspect,
    frames: Mapping[str, RushFrame],
    clips: Mapping[str, RushClip],
    shot_cache: dict[str, ShotManifest],
    shots_dir: Path,
    scdet_threshold: float,
    approved_decisions: Sequence[tuple[Path, TrimIntentDecision]],
) -> dict[str, float]:
    """Return exact human-approved trim durations used by the renderer."""

    if not approved_decisions:
        return {}
    render_horizontal, render_vertical = _requested_render_aspects(aspect)
    fixed: dict[str, float] = {}
    for selected in plan.chapters:
        if selected.evidence_status == "not_found":
            continue
        durations_ms: list[int] = []
        seen: set[tuple[str, str, str | None]] = set()
        aspect_frames: list[tuple[str, str | None]] = []
        if render_horizontal:
            aspect_frames.append(("16:9", selected.horizontal_frame_id))
        if render_vertical:
            aspect_frames.append(("9:16", selected.vertical_frame_id))
        for selected_aspect, frame_id in aspect_frames:
            if frame_id is None:
                continue
            frame = frames[frame_id]
            clip = clips[frame.clip_id]
            if clip.clip_id not in shot_cache:
                shot_cache[clip.clip_id] = detect_shots_ffmpeg(
                    Path(clip.path),
                    threshold=scdet_threshold,
                    output_path=shots_dir / f"{clip.clip_id}.json",
                )
            shot = next(
                item
                for item in shot_cache[clip.clip_id].shots
                if item.start_time_ms
                <= frame.requested_time_ms
                < item.end_time_ms
            )
            event_id = _selected_event_id(
                selected,
                frame_id=frame_id,
                aspect=selected_aspect,
            )
            identity = (clip.sha256, shot.shot_id, event_id)
            if identity in seen:
                continue
            seen.add(identity)
            match = _matching_trim_decision(
                approved_decisions,
                source_asset_id=f"sha256:{clip.sha256}",
                shot_id=shot.shot_id,
                event_id=event_id,
            )
            if match is None:
                continue
            _, decision = match
            assert decision.source_in_ms is not None
            assert decision.source_out_ms is not None
            durations_ms.append(decision.source_out_ms - decision.source_in_ms)
        if durations_ms:
            if max(durations_ms) - min(durations_ms) > 1:
                # Horizontal and vertical are independent deliverables unless a
                # later synchronization contract explicitly binds them. Their
                # approved evidence intervals may therefore have different
                # editorial lengths. Do not invent one shared fixed duration;
                # each renderer consumes its own approved half-open interval,
                # and the delivery pipeline assembles music against each final
                # picture duration separately.
                continue
            fixed[selected.feature_id] = durations_ms[0] / 1000
    return fixed


def _feature_edit_user_duration_range_seconds() -> tuple[float, float]:
    """Read the public user-duration bounds from the brief contract itself."""

    duration_schema = FeatureEditBrief.model_json_schema()["properties"][
        "target_duration_seconds"
    ]
    minimum = duration_schema.get("minimum")
    maximum = duration_schema.get("maximum")
    if not isinstance(minimum, (int, float)) or not isinstance(
        maximum, (int, float)
    ):
        raise ValueError("FeatureEditBrief duration range is not machine-readable")
    return float(minimum), float(maximum)


def _build_duration_capacity_shortfall_audit(
    *,
    brief: FeatureEditBrief,
    plan: FeatureEditPlan,
    weighted: Sequence[tuple[FeatureChapterSelect, float, str]],
    capacity_ms: Mapping[str, int],
    preferred_total_ms: int,
    feasible_total_ms: int,
) -> dict[str, Any]:
    minimum_seconds, maximum_seconds = (
        _feature_edit_user_duration_range_seconds()
    )
    shortfall_ms = preferred_total_ms - feasible_total_ms
    if shortfall_ms <= 0:
        raise ValueError("duration shortfall audit requires a positive shortfall")
    rows = [
        {
            "feature_id": selected.feature_id,
            "preferred_weight_seconds": round(weight, 3),
            "preferred_weight_authority": authority,
            "feasible_capacity_seconds": (
                round(capacity_ms[selected.feature_id] / 1000, 3)
                if selected.feature_id in capacity_ms
                else None
            ),
            "capacity_evidence": (
                "selected_shot_boundary"
                if selected.feature_id in capacity_ms
                else "not_finitely_bounded_in_this_audit"
            ),
        }
        for selected, weight, authority in weighted
    ]
    shorter_within_range = feasible_total_ms >= round(minimum_seconds * 1000)
    return {
        "contract_version": "editorial-duration-capacity-shortfall-v1",
        "status": "blocked",
        "failure_policy": "fail_closed_before_render",
        "reason_code": "selected_source_capacity_below_preferred_total",
        "project_id": brief.project_id,
        "catalog_id": plan.catalog_id,
        "brief_sha256": _sha256_json(brief.model_dump(mode="json")),
        "feature_plan_sha256": _sha256_json(plan.model_dump(mode="json")),
        "preferred_total_ms": preferred_total_ms,
        "preferred_total_seconds": round(preferred_total_ms / 1000, 3),
        "relative_dwell_weight_total_seconds": round(
            sum(weight for _, weight, _ in weighted),
            3,
        ),
        "feasible_total_ms": feasible_total_ms,
        "feasible_total_seconds": round(feasible_total_ms / 1000, 3),
        "shortfall_ms": shortfall_ms,
        "shortfall_seconds": round(shortfall_ms / 1000, 3),
        "user_duration_range": {
            "minimum_seconds": minimum_seconds,
            "preferred_seconds": brief.target_duration_seconds,
            "maximum_seconds": maximum_seconds,
            "source": "FeatureEditBrief.target_duration_seconds contract and user brief",
        },
        "chapter_capacities": rows,
        "next_actions": [
            {
                "action_id": "select_alternate_candidates",
                "description": (
                    "Choose evidence-bound source candidates whose legal shot "
                    "intervals provide enough additional capacity."
                ),
                "requires_user_approval": True,
                "available": True,
            },
            {
                "action_id": "provide_additional_source",
                "description": (
                    "Add usable source material that supports the same brief "
                    "without repeating or freezing existing footage."
                ),
                "requires_user_approval": True,
                "available": True,
            },
            {
                "action_id": "approve_shorter_project_duration",
                "description": (
                    "Approve a shorter total only when it remains inside the "
                    "declared user-duration range."
                ),
                "requires_user_approval": True,
                "available": shorter_within_range,
                "maximum_feasible_seconds": round(feasible_total_ms / 1000, 3),
            },
            {
                "action_id": "revise_required_scope",
                "description": (
                    "Explicitly revise required chapters or claims before "
                    "replanning; the renderer cannot silently drop them."
                ),
                "requires_user_approval": True,
                "available": True,
            },
        ],
        "prohibited_automatic_actions": [
            "repeat_selected_footage",
            "freeze_last_frame_to_fill_duration",
            "extend_across_shot_boundary",
            "reduce_duration_outside_user_range",
            "drop_required_chapter",
        ],
        "generated_at": utc_now(),
    }


def _resolve_editorial_chapter_durations(
    brief: FeatureEditBrief,
    plan: FeatureEditPlan,
    *,
    music_lock: MusicMapLock | None = None,
    source_capacity_seconds: Mapping[str, float] | None = None,
    fixed_duration_seconds: Mapping[str, float] | None = None,
    rhythm_plan: RhythmPlan | None = None,
    shortfall_audit_path: Path | None = None,
    project_duration_seconds: float | None = None,
    allow_music_lock_prefix: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Reconcile Gemini's relative dwell judgment to one legal project total.

    The model never supplies source cut points. It only ranks the relative
    amount of viewing time justified by the observed information/action and,
    when present, the soundtrack. Local code preserves those proportions while
    making the chapter durations sum exactly to the user-approved project total.
    Legacy plans without recommendations retain their brief values.
    """

    brief_by_id = {chapter.feature_id: chapter for chapter in brief.chapters}
    rhythm_by_id = (
        {chapter.feature_id: chapter for chapter in rhythm_plan.chapters}
        if rhythm_plan is not None
        else {}
    )
    if rhythm_plan is not None and set(rhythm_by_id) != {
        chapter.feature_id for chapter in plan.chapters
    }:
        raise ValueError("RhythmPlan chapters differ from the feature plan")
    weighted: list[tuple[FeatureChapterSelect, float, str]] = []
    for selected in plan.chapters:
        fallback = brief_by_id[selected.feature_id].target_duration_seconds
        if selected.feature_id in rhythm_by_id:
            weighted.append(
                (
                    selected,
                    rhythm_by_id[selected.feature_id].preferred_duration_seconds,
                    "attention_rhythm_plan",
                )
            )
        elif selected.recommended_duration_seconds is None:
            weighted.append((selected, fallback, "brief_fallback"))
        else:
            weighted.append(
                (selected, selected.recommended_duration_seconds, "gemini_relative_dwell")
            )
    weight_total = sum(weight for _, weight, _ in weighted)
    if weight_total <= 0:
        raise ValueError("editorial duration weights must have a positive sum")

    resolved_project_duration_seconds = (
        brief.target_duration_seconds
        if project_duration_seconds is None
        else project_duration_seconds
    )
    target_duration_ms = round(resolved_project_duration_seconds * 1000)
    capacity_ms = {
        feature_id: max(1, round(seconds * 1000))
        for feature_id, seconds in (source_capacity_seconds or {}).items()
    }
    if rhythm_plan is not None:
        for feature_id, chapter in rhythm_by_id.items():
            rhythm_maximum = max(
                1, round(chapter.maximum_duration_seconds * 1000)
            )
            capacity_ms[feature_id] = min(
                capacity_ms.get(feature_id, rhythm_maximum),
                rhythm_maximum,
            )
    minimum_ms = {
        feature_id: max(1, round(chapter.minimum_duration_seconds * 1000))
        for feature_id, chapter in rhythm_by_id.items()
    }
    known_feature_ids = {selected.feature_id for selected, _, _ in weighted}
    fixed_ms = {
        feature_id: max(1, round(seconds * 1000))
        for feature_id, seconds in (fixed_duration_seconds or {}).items()
    }
    unknown_fixed_ids = set(fixed_ms) - known_feature_ids
    if unknown_fixed_ids:
        raise ValueError(
            "fixed editorial durations reference unknown chapters: "
            + ", ".join(sorted(unknown_fixed_ids))
        )
    for feature_id, duration_ms in fixed_ms.items():
        feature_capacity = capacity_ms.get(feature_id)
        if feature_capacity is not None and duration_ms > feature_capacity:
            raise ValueError(
                f"approved trim for {feature_id} exceeds its QualitySafeInterval "
                "capacity"
            )
    for selected, _, _ in weighted:
        feature_id = selected.feature_id
        if feature_id in fixed_ms:
            continue
        minimum = minimum_ms.get(feature_id, 0)
        maximum = capacity_ms.get(feature_id, target_duration_ms)
        if minimum > maximum:
            raise ValueError(
                f"minimum attention dwell for {feature_id} exceeds its "
                "QualitySafeInterval capacity"
            )
    flexible_weighted = [
        item for item in weighted if item[0].feature_id not in fixed_ms
    ]
    finite_capacity_total_ms = sum(fixed_ms.values()) + sum(
        capacity_ms.get(selected.feature_id, target_duration_ms)
        for selected, _, _ in flexible_weighted
    )
    if finite_capacity_total_ms < target_duration_ms:
        shortfall_audit = _build_duration_capacity_shortfall_audit(
            brief=brief,
            plan=plan,
            weighted=weighted,
            capacity_ms=capacity_ms,
            preferred_total_ms=target_duration_ms,
            feasible_total_ms=finite_capacity_total_ms,
        )
        if shortfall_audit_path is not None:
            write_json(shortfall_audit_path, shortfall_audit)
        raise ValueError(
            "selected source shots cannot satisfy the requested project duration; "
            "choose another candidate or approve a shorter project duration"
            + (
                f"; audit saved to {shortfall_audit_path}"
                if shortfall_audit_path is not None
                else ""
            )
        )

    # Allocate in integer milliseconds. When a selected source shot is shorter
    # than Gemini's relative dwell recommendation, cap that chapter and
    # redistribute only the remaining duration across the other Gemini weights.
    # This preserves the model's editorial ordering without inventing repeated
    # frames, synthetic holds, or source time outside a legal shot.
    flexible_minimum_total_ms = sum(
        minimum_ms.get(selected.feature_id, 0)
        for selected, _, _ in flexible_weighted
    )
    remaining_ms = (
        target_duration_ms
        - sum(fixed_ms.values())
        - flexible_minimum_total_ms
    )
    if remaining_ms < 0:
        raise ValueError(
            "human-approved trims plus minimum attention dwell exceed the "
            "requested project duration"
        )
    active = list(flexible_weighted)
    allocated_ms: dict[str, int] = {
        **fixed_ms,
        **{
            selected.feature_id: minimum_ms.get(selected.feature_id, 0)
            for selected, _, _ in flexible_weighted
        },
    }
    if not active and remaining_ms != 0:
        raise ValueError(
            "fixed approved trims do not sum to the requested project duration "
            "and no flexible chapter remains"
        )
    while active:
        active_weight = sum(weight for _, weight, _ in active)
        if active_weight <= 0:
            raise ValueError("active editorial duration weights must be positive")
        newly_capped: list[tuple[FeatureChapterSelect, float, str]] = []
        for selected, weight, authority in active:
            feature_capacity = capacity_ms.get(
                selected.feature_id,
                target_duration_ms,
            )
            available_headroom = (
                feature_capacity - allocated_ms[selected.feature_id]
            )
            proportional_ms = remaining_ms * weight / active_weight
            if proportional_ms > available_headroom:
                allocated_ms[selected.feature_id] = feature_capacity
                remaining_ms -= available_headroom
                newly_capped.append((selected, weight, authority))
        if newly_capped:
            capped_ids = {selected.feature_id for selected, _, _ in newly_capped}
            active = [
                item for item in active if item[0].feature_id not in capped_ids
            ]
            continue

        raw_allocations = [
            (
                selected,
                weight,
                authority,
                remaining_ms * weight / active_weight,
            )
            for selected, weight, authority in active
        ]
        floor_allocations = {
            selected.feature_id: math.floor(raw_ms)
            for selected, _, _, raw_ms in raw_allocations
        }
        remainder = remaining_ms - sum(floor_allocations.values())
        for selected, _, _, raw_ms in sorted(
            raw_allocations,
            key=lambda item: (
                -(item[3] - math.floor(item[3])),
                item[0].feature_id,
            ),
        ):
            value = floor_allocations[selected.feature_id]
            if remainder > 0:
                value += 1
                remainder -= 1
            allocated_ms[selected.feature_id] += value
        remaining_ms = 0
        break
    if remaining_ms != 0 or sum(allocated_ms.values()) != target_duration_ms:
        raise ValueError("editorial duration capacity reconciliation did not close")

    scale = resolved_project_duration_seconds / weight_total
    unsnapped: list[tuple[FeatureChapterSelect, float, str, float]] = []
    for selected, weight, authority in weighted:
        seconds = allocated_ms[selected.feature_id] / 1000
        if seconds <= 0:
            raise ValueError(
                f"resolved duration is not positive for {selected.feature_id}"
            )
        unsnapped.append((selected, weight, authority, seconds))

    boundary_audit: list[dict[str, Any]] = []
    snapped_boundaries_ms: list[int] | None = None
    capped_feature_ids = {
        feature_id
        for feature_id, allocated in allocated_ms.items()
        if feature_id in capacity_ms and allocated >= capacity_ms[feature_id]
    } | set(fixed_ms)
    if music_lock is not None and len(unsnapped) > 1:
        music_lock_prefix_used = (
            allow_music_lock_prefix
            and music_lock.duration_ms >= target_duration_ms
        )
        if (
            abs(music_lock.duration_ms - target_duration_ms) > 80
            and not music_lock_prefix_used
        ):
            raise ValueError(
                "music lock duration differs from the requested project duration"
            )
        cue_candidates = [
            cue
            for cue in music_lock.cues
            if cue.kind in {"section_boundary", "downbeat", "accent", "ending_hit"}
        ]
        proposed_boundaries: list[int] = []
        running_ms = 0
        for _, _, _, seconds in unsnapped[:-1]:
            running_ms += round(seconds * 1000)
            proposed_boundaries.append(running_ms)
        snapped_boundaries_ms = []
        previous = 0
        for boundary_index, proposed_ms in enumerate(proposed_boundaries):
            remaining_minimum_ms = sum(
                minimum_ms.get(item[0].feature_id, 750)
                for item in unsnapped[boundary_index + 1 :]
            )
            latest_ms = target_duration_ms - remaining_minimum_ms
            current_feature_id = unsnapped[boundary_index][0].feature_id
            next_feature_id = unsnapped[boundary_index + 1][0].feature_id
            current_capacity_ms = capacity_ms.get(
                current_feature_id,
                target_duration_ms,
            )
            remaining_capacity_ms = sum(
                capacity_ms.get(item[0].feature_id, target_duration_ms)
                for item in unsnapped[boundary_index + 1 :]
            )
            boundary_priority = (
                rhythm_by_id[current_feature_id].boundary_priority
                if current_feature_id in rhythm_by_id
                else "normal"
            )
            snap_window_ms = {
                "low": 250,
                "normal": 450,
                "high": 650,
            }[boundary_priority]
            allowed_cue_kinds = (
                {"section_boundary", "downbeat"}
                if boundary_priority == "low"
                else {"section_boundary", "downbeat", "accent", "ending_hit"}
            )
            eligible = [
                cue
                for cue in cue_candidates
                if (
                    current_feature_id not in capped_feature_ids
                    and next_feature_id not in capped_feature_ids
                    and previous
                    + minimum_ms.get(current_feature_id, 750)
                    <= cue.time_ms
                    <= latest_ms
                    and cue.kind in allowed_cue_kinds
                    and abs(cue.time_ms - proposed_ms) <= snap_window_ms
                    and cue.time_ms - previous <= current_capacity_ms
                    and target_duration_ms - cue.time_ms
                    <= remaining_capacity_ms
                )
            ]
            if eligible:
                chosen = min(
                    eligible,
                    key=lambda cue: (
                        0 if cue.kind == "section_boundary" else 1,
                        0 if cue.kind == "downbeat" else 1,
                        abs(cue.time_ms - proposed_ms),
                        -cue.strength,
                        cue.time_ms,
                    ),
                )
                applied_ms = chosen.time_ms
                cue_id = chosen.cue_id
                cue_kind = chosen.kind
            else:
                applied_ms = proposed_ms
                cue_id = None
                cue_kind = None
            snapped_boundaries_ms.append(applied_ms)
            boundary_audit.append(
                {
                    "boundary_after_feature_id": unsnapped[boundary_index][
                        0
                    ].feature_id,
                    "gemini_relative_boundary_ms": proposed_ms,
                    "resolved_boundary_ms": applied_ms,
                    "snap_delta_ms": applied_ms - proposed_ms,
                    "music_cue_id": cue_id,
                    "music_cue_kind": cue_kind,
                    "rhythm_boundary_priority": boundary_priority,
                    "rhythm_snap_window_ms": snap_window_ms,
                }
            )
            previous = applied_ms

        candidate_boundaries = [
            0,
            *snapped_boundaries_ms,
            target_duration_ms,
        ]
        capacity_violations: list[dict[str, Any]] = []
        for index, (start_ms, end_ms) in enumerate(
            zip(
                candidate_boundaries[:-1],
                candidate_boundaries[1:],
                strict=True,
            )
        ):
            feature_id = unsnapped[index][0].feature_id
            chapter_ms = end_ms - start_ms
            feature_capacity_ms = capacity_ms.get(feature_id, target_duration_ms)
            feature_minimum_ms = minimum_ms.get(feature_id, 1)
            if (
                chapter_ms < feature_minimum_ms
                or chapter_ms > feature_capacity_ms
            ):
                capacity_violations.append(
                    {
                        "feature_id": feature_id,
                        "candidate_duration_ms": chapter_ms,
                        "attention_minimum_ms": feature_minimum_ms,
                        "source_capacity_ms": feature_capacity_ms,
                    }
                )
        if capacity_violations:
            snapped_boundaries_ms = None
            for row in boundary_audit:
                row["music_snap_applied"] = False
                row["music_snap_rejected_reason"] = (
                    "global_source_capacity_validation_failed"
                )
                row["candidate_resolved_boundary_ms"] = row[
                    "resolved_boundary_ms"
                ]
                row["resolved_boundary_ms"] = row[
                    "gemini_relative_boundary_ms"
                ]
                row["snap_delta_ms"] = 0
                row["source_capacity_violations"] = capacity_violations
        else:
            for row in boundary_audit:
                row["music_snap_applied"] = row["music_cue_id"] is not None
                row["music_snap_rejected_reason"] = None

    resolved: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    prior_boundary_ms = 0
    final_boundary_ms = target_duration_ms
    for index, (selected, weight, authority, unsnapped_seconds) in enumerate(
        unsnapped
    ):
        next_boundary_ms = (
            snapped_boundaries_ms[index]
            if snapped_boundaries_ms is not None
            and index < len(snapped_boundaries_ms)
            else (
                final_boundary_ms
                if index == len(unsnapped) - 1
                else prior_boundary_ms + round(unsnapped_seconds * 1000)
            )
        )
        seconds = round((next_boundary_ms - prior_boundary_ms) / 1000, 3)
        prior_boundary_ms = next_boundary_ms
        resolved[selected.feature_id] = seconds
        rows.append(
            {
                "feature_id": selected.feature_id,
                "input_weight_seconds": weight,
                "input_authority": authority,
                "fixed_duration_authority": (
                    "human_approved_trim_exact_pts"
                    if selected.feature_id in fixed_ms
                    else None
                ),
                "duration_rationale": selected.duration_rationale,
                "unsnapped_duration_seconds": unsnapped_seconds,
                "resolved_duration_seconds": seconds,
                "source_capacity_seconds": (
                    round(capacity_ms[selected.feature_id] / 1000, 3)
                    if selected.feature_id in capacity_ms
                    else None
                ),
                "minimum_attention_dwell_seconds": (
                    round(minimum_ms[selected.feature_id] / 1000, 3)
                    if selected.feature_id in minimum_ms
                    else None
                ),
                "rhythm_boundary_priority": (
                    rhythm_by_id[selected.feature_id].boundary_priority
                    if selected.feature_id in rhythm_by_id
                    else None
                ),
                "source_capacity_applied": (
                    selected.feature_id in capacity_ms
                    and allocated_ms[selected.feature_id]
                    >= capacity_ms[selected.feature_id]
                ),
            }
        )
    audit = {
        "contract_version": "editorial-duration-plan-v2",
        "interpretation": (
            "Gemini proposes relative dwell; local code only reconciles the "
            "approved total and later maps it to legal shot-local source PTS."
        ),
        "brief_preferred_duration_seconds": brief.target_duration_seconds,
        "project_target_duration_seconds": resolved_project_duration_seconds,
        "input_weight_total_seconds": round(weight_total, 3),
        "reconciliation_scale": scale,
        "capacity_reconciliation_applied": bool(capacity_ms),
        "fixed_approved_trim_duration_seconds": {
            feature_id: round(duration_ms / 1000, 3)
            for feature_id, duration_ms in sorted(fixed_ms.items())
        },
        "attention_profile_sha256": (
            rhythm_plan.attention_profile_sha256
            if rhythm_plan is not None
            else None
        ),
        "source_capacity_total_seconds": (
            round(finite_capacity_total_ms / 1000, 3)
            if capacity_ms
            else None
        ),
        "resolved_total_seconds": round(sum(resolved.values()), 3),
        "music_lock_definition_sha256": (
            music_lock.definition_sha256 if music_lock is not None else None
        ),
        "music_lock_prefix_used": (
            music_lock is not None
            and allow_music_lock_prefix
            and music_lock.duration_ms > target_duration_ms + 80
        ),
        "music_lock_duration_ms": (
            music_lock.duration_ms if music_lock is not None else None
        ),
        "project_timeline_end_ms": target_duration_ms,
        "music_boundary_refinements": boundary_audit,
        "chapters": rows,
    }
    return resolved, audit


def _run_feature_cut_experiment_impl(
    *,
    catalog_path: Path,
    brief_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    plan_prompt: str,
    grounding_prompt: str,
    vertical_framing_prompt: str | None = None,
    scdet_threshold: float = 4.0,
    sam_analysis_fps: float = 2.0,
    trim_decision_paths: Sequence[Path] = (),
    shot_quality_map_paths: Sequence[Path] = (),
    allow_proposed_trim_preview: bool = False,
    reuse_feature_plan: bool = False,
    reuse_feature_plan_raw_output: bool = False,
    allow_unverified_geometry_preview: bool = False,
    aspect: RenderAspect = "both",
    music_path: Path | None = None,
    music_lock_path: Path | None = None,
    post_render_quality_qc: bool = True,
    rhythm_style: Literal["calm", "standard", "energetic"] = "standard",
    allow_shorter_within_delivery_range: bool = False,
    auto_vertical_framing: bool = True,
    execution_profile: FeatureCutExecutionProfile = (
        FeatureCutExecutionProfile.REVIEW_PREVIEW
    ),
) -> dict[str, Any]:
    resolved_vertical_framing_prompt = vertical_framing_prompt or (
        "Inspect the complete selected clip and propose a generic evidence-only "
        "9:16 framing decision. Preserve simultaneous semantic relations; use "
        "ordered virtual-camera phases only for directly observed sequential "
        "attention. Do not output timestamps or coordinates."
    )
    render_horizontal, render_vertical = _requested_render_aspects(aspect)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_interaction_hashes = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in output_dir.rglob("*raw_interaction.json")
    }
    prior_error_hashes = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in output_dir.rglob("errors.json")
    }
    catalog = RushesCatalog.model_validate(read_json(catalog_path))
    catalog_source_validation = validate_rushes_catalog_sources(catalog)
    write_json(
        output_dir / "catalog-source-validation.json",
        catalog_source_validation,
    )
    brief = FeatureEditBrief.model_validate(read_json(brief_path))
    controlled_reframe_requested = render_vertical and any(
        chapter.vertical_overflow_policy == "controlled_clip"
        for chapter in brief.chapters
    )
    human_reframe_policy_requested = (
        render_vertical and brief.reframe_policy_binding is not None
    )
    if controlled_reframe_requested and brief.reframe_policy_binding is None:
        raise ValueError(
            "controlled_clip requires an immutable human reframe policy sidecar"
        )
    if human_reframe_policy_requested and not reuse_feature_plan:
        raise ValueError(
            "a human reframe policy can only reuse its bound feature plan; "
            "pass --reuse-feature-plan"
        )
    plan_dir = output_dir / "gemini-plan"
    plan_path = plan_dir / "feature_edit_plan.json"
    plan_binding_path = plan_dir / "feature-plan.binding.json"
    if human_reframe_policy_requested:
        if not plan_path.is_file() or not plan_binding_path.is_file():
            raise ValueError(
                "human reframe policy bundle requires its bound feature plan and binding"
            )
        saved_human_binding = read_json(plan_binding_path)
        if (
            not isinstance(saved_human_binding, dict)
            or saved_human_binding.get("origin") != REFRAME_POLICY_BINDING_ORIGIN
        ):
            raise ValueError(
                "human reframe policy requires a human_reframe_policy plan binding"
            )
        # Validate the complete sidecar chain before probing media or creating
        # a Gemini client. A binding-shaped object is not authorization.
        validate_reframe_policy_bundle(
            catalog_path=catalog_path,
            brief_path=brief_path,
            feature_plan_path=plan_path,
            saved_plan_binding=saved_human_binding,
        )
    frames = {frame.frame_id: frame for frame in catalog.frames}
    clips = {clip.clip_id: clip for clip in catalog.clips}
    trim_decisions = _load_trim_decisions(
        trim_decision_paths,
        allow_proposed_preview=allow_proposed_trim_preview,
    )
    shot_quality_maps = _load_shot_quality_maps(shot_quality_map_paths)
    brief_by_id = {chapter.feature_id: chapter for chapter in brief.chapters}
    timings: dict[str, float] = {}
    incremental_pricing: dict[str, Any] = {}
    started = monotonic()
    reel_path = Path(catalog.analysis_reel_path)
    reel_media = probe_video(reel_path)
    upload_dir = catalog_path.parent / "file-cache" / reel_media.sha256 / "upload"
    resolved_music_path = (
        music_path.expanduser().resolve(strict=True)
        if music_path is not None
        else None
    )
    music_sha256 = (
        sha256_file(resolved_music_path)
        if resolved_music_path is not None
        else None
    )
    resolved_music_lock_path = (
        music_lock_path.expanduser().resolve(strict=True)
        if music_lock_path is not None
        else None
    )
    music_lock = (
        MusicMapLock.model_validate(read_json(resolved_music_lock_path))
        if resolved_music_lock_path is not None
        else None
    )
    if (
        music_lock is not None
        and music_sha256 is not None
        and music_lock.music_id != f"sha256:{music_sha256}"
    ):
        raise ValueError("music file does not match the supplied MusicMap lock")
    client = GeminiLabClient()
    feature_plan_origin = "generated"
    external_projection_contract_id: str | None = None
    plan_reuse_record_path: Path | None = None
    gemini_geometry_block_reason: str | None = None
    geometry_circuit_run_id = uuid.uuid4().hex
    circuit_started = {
        "run_id": geometry_circuit_run_id,
        "blocked": False,
        "reason": None,
        "interpretation": "no_geometry_quota_error_seen_in_this_run",
        "started_at": utc_now(),
    }
    write_json(
        output_dir / "geometry-model-circuit-breaker.json",
        circuit_started,
    )
    write_json(
        output_dir
        / "geometry-model-circuit-breaker-events"
        / f"{geometry_circuit_run_id}.started.json",
        circuit_started,
    )

    def latch_geometry_quota_error(error: Exception) -> bool:
        nonlocal gemini_geometry_block_reason
        if not _is_exhausted_model_quota_error(error):
            return False
        if gemini_geometry_block_reason is None:
            gemini_geometry_block_reason = f"{type(error).__name__}:{error}"
            spending_cap = _is_non_retryable_spending_cap_error(error)
            blocked_record = {
                    "run_id": geometry_circuit_run_id,
                    "blocked": True,
                    "reason": gemini_geometry_block_reason,
                    "retryable_later": not spending_cap,
                    "action": "abort_render_before_trying_another_candidate",
                    "interpretation": (
                        "spending_cap_requires_account_action"
                        if spending_cap
                        else "upstream_quota_rejected_retry_the_run_later"
                    ),
                    "latched_at": utc_now(),
                }
            write_json(
                output_dir / "geometry-model-circuit-breaker.json",
                blocked_record,
            )
            write_json(
                output_dir
                / "geometry-model-circuit-breaker-events"
                / f"{geometry_circuit_run_id}.blocked.json",
                blocked_record,
            )
        return True

    def abort_for_geometry_quota(error: Exception) -> None:
        if not latch_geometry_quota_error(error):
            return
        raise GeometryModelQuotaError(
            "Gemini geometry is unavailable; stopped before trying another "
            "candidate because candidate switching cannot resolve a 429/quota "
            "failure. See geometry-model-circuit-breaker.json."
        ) from error
    try:
        if controlled_reframe_requested and not plan_path.exists():
            raise ValueError(
                "controlled_clip policy bundle has no bound saved feature plan"
            )
        if plan_path.exists():
            if not reuse_feature_plan:
                raise ValueError(
                    "saved feature plan exists; pass --reuse-feature-plan to reuse "
                    "that editorial decision explicitly, or choose a fresh output directory"
                )
            plan = FeatureEditPlan.model_validate(read_json(plan_path))
            expected_ids = [chapter.feature_id for chapter in brief.chapters]
            actual_ids = [chapter.feature_id for chapter in plan.chapters]
            if (
                plan.project_id != brief.project_id
                or plan.catalog_id != catalog.catalog_id
                or actual_ids != expected_ids
            ):
                raise ValueError("saved feature plan does not match the current brief/catalog")
            if plan_binding_path.exists():
                saved_binding = read_json(plan_binding_path)
                if not isinstance(saved_binding, dict):
                    raise ValueError("saved feature plan binding must be an object")
                if saved_binding.get("origin") == REFRAME_POLICY_BINDING_ORIGIN:
                    current_binding = validate_reframe_policy_bundle(
                        catalog_path=catalog_path,
                        brief_path=brief_path,
                        feature_plan_path=plan_path,
                        saved_plan_binding=saved_binding,
                    )
                elif saved_binding.get("origin") == "external_projection":
                    current_binding = _current_external_projection_binding(
                        plan_dir=plan_dir,
                        catalog_path=catalog_path,
                        catalog_reel_sha256=reel_media.sha256,
                        brief_path=brief_path,
                        plan_path=plan_path,
                        music_sha256=music_sha256,
                        created_at=utc_now(),
                    )
                else:
                    saved_origin = saved_binding.get("origin")
                    if saved_origin not in {"generated", "migrated_legacy_reuse"}:
                        raise ValueError("saved feature plan binding origin is unsupported")
                    current_binding = _current_feature_plan_binding(
                        catalog_path=catalog_path,
                        catalog_reel_sha256=reel_media.sha256,
                        brief_path=brief_path,
                        plan_path=plan_path,
                        plan_prompt=plan_prompt,
                        music_sha256=music_sha256,
                        request_path=(
                            plan_dir / "feature_edit_plan.request.json"
                            if (plan_dir / "feature_edit_plan.request.json").exists()
                            else None
                        ),
                        created_at=utc_now(),
                        origin=saved_origin,
                    )
            elif (plan_dir / _EXTERNAL_PROJECTION_POINTER_NAME).exists():
                current_binding = _current_external_projection_binding(
                    plan_dir=plan_dir,
                    catalog_path=catalog_path,
                    catalog_reel_sha256=reel_media.sha256,
                    brief_path=brief_path,
                    plan_path=plan_path,
                    music_sha256=music_sha256,
                    created_at=utc_now(),
                )
                saved_binding = current_binding
                write_json(plan_binding_path, saved_binding)
            else:
                current_binding = _current_feature_plan_binding(
                    catalog_path=catalog_path,
                    catalog_reel_sha256=reel_media.sha256,
                    brief_path=brief_path,
                    plan_path=plan_path,
                    plan_prompt=plan_prompt,
                    music_sha256=music_sha256,
                    request_path=(
                        plan_dir / "feature_edit_plan.request.json"
                        if (plan_dir / "feature_edit_plan.request.json").exists()
                        else None
                    ),
                    created_at=utc_now(),
                    origin="generated",
                )
                saved_binding = _migrate_legacy_feature_plan_binding(
                    plan_dir=plan_dir,
                    catalog_path=catalog_path,
                    catalog_reel_sha256=reel_media.sha256,
                    brief_path=brief_path,
                    plan_path=plan_path,
                    plan_prompt=plan_prompt,
                    music_sha256=music_sha256,
                )
                write_json(plan_binding_path, saved_binding)
                current_binding["origin"] = saved_binding["origin"]
            _validate_feature_plan_binding(saved_binding, current_binding)
            feature_plan_origin = str(current_binding["origin"])
            contract_id = current_binding.get("external_projection_contract_id")
            external_projection_contract_id = (
                str(contract_id) if isinstance(contract_id, str) else None
            )
            reuse_event_dir = plan_dir / "feature-plan-reuse-events"
            plan_reuse_record_path = (
                reuse_event_dir / f"reuse-{uuid.uuid4().hex}.json"
            )
            write_json(
                plan_reuse_record_path,
                {
                    "interpretation": (
                        "explicit_editorial_plan_reuse_geometry_is_recomputed"
                    ),
                    "binding_path": str(plan_binding_path.resolve()),
                    "binding_sha256": sha256_file(plan_binding_path),
                    "binding_origin": current_binding["origin"],
                    "validated_causal_hashes": {
                        key: current_binding[key]
                        for key in (
                            "catalog_sha256",
                            "catalog_reel_sha256",
                            "brief_sha256",
                            "music_sha256",
                            "plan_prompt_sha256",
                            "system_instruction_sha256",
                            "model_id_sha256",
                            "response_schema_sha256",
                            "plan_sha256",
                            "request_sha256",
                            "source_plan_sha256",
                            "projection_contract_sha256",
                            "projection_pointer_sha256",
                            "projection_record_sha256",
                            "source_artifact_set_sha256",
                            "reframe_policy_sidecar_sha256",
                            "source_plan_binding_sha256",
                            "selection_fingerprint",
                        )
                        if key in current_binding
                    },
                    "reused_at": utc_now(),
                },
            )
            timings["file_api_seconds"] = 0.0
            file_api_reused: bool | None = None
            timings["gemini_plan_seconds"] = 0.0
            plan_reused = True
        else:
            uploaded = None
            uploaded_audio = None
            music_file_api_reused: bool | None = None
            if reuse_feature_plan_raw_output:
                file_api_reused = True
                timings["file_api_seconds"] = 0.0
                timings["music_file_api_seconds"] = 0.0
            else:
                stage = monotonic()
                uploaded, file_api_reused = client.ensure_video_upload(
                    reel_path, upload_dir
                )
                timings["file_api_seconds"] = round(monotonic() - stage, 3)
                if resolved_music_path is not None and music_sha256 is not None:
                    music_upload_dir = (
                        catalog_path.parent
                        / "file-cache"
                        / music_sha256
                        / "music-upload"
                    )
                    music_upload_stage = monotonic()
                    uploaded_audio, music_file_api_reused = client.ensure_video_upload(
                        resolved_music_path,
                        music_upload_dir,
                    )
                    timings["music_file_api_seconds"] = round(
                        monotonic() - music_upload_stage, 3
                    )
            stage = monotonic()
            plan = client.plan_feature_edit(
                catalog=catalog,
                brief=brief,
                uploaded=uploaded,
                uploaded_audio=uploaded_audio,
                music_sha256=music_sha256,
                prompt_template=plan_prompt,
                run_id=f"feature-plan-{uuid.uuid4().hex[:8]}",
                run_dir=plan_dir,
                reuse_raw_output=reuse_feature_plan_raw_output,
            )
            timings["gemini_plan_seconds"] = round(monotonic() - stage, 3)
            request_path = plan_dir / "feature_edit_plan.request.json"
            binding = _current_feature_plan_binding(
                catalog_path=catalog_path,
                catalog_reel_sha256=reel_media.sha256,
                brief_path=brief_path,
                plan_path=plan_path,
                plan_prompt=plan_prompt,
                music_sha256=music_sha256,
                request_path=request_path,
                created_at=utc_now(),
                origin="generated",
            )
            write_json(plan_binding_path, binding)
            plan_reused = False
        shot_cache: dict[str, ShotManifest] = {}
        shots_dir = output_dir / "shots"
        requested_candidate_recall_audit = (
            _audit_requested_candidate_recall(
                plan,
                aspect=aspect,
            )
        )
        write_json(
            output_dir / "requested-candidate-recall-audit.json",
            requested_candidate_recall_audit,
        )
        if execution_profile == FeatureCutExecutionProfile.PRODUCTION_REVIEW:
            quality_stage = monotonic()
            shot_quality_maps = _ensure_requested_quality_maps(
                plan,
                aspect=aspect,
                frames=frames,
                clips=clips,
                shot_cache=shot_cache,
                shots_dir=shots_dir,
                scdet_threshold=scdet_threshold,
                quality_maps=shot_quality_maps,
                human_policy_binding_present=human_reframe_policy_requested,
                output_dir=output_dir,
            )
            timings["automatic_shot_quality_seconds"] = round(
                monotonic() - quality_stage,
                3,
            )
        quality_map_coverage_audit = (
            _audit_requested_quality_map_coverage(
                plan,
                aspect=aspect,
                frames=frames,
                clips=clips,
                shot_cache=shot_cache,
                shots_dir=shots_dir,
                scdet_threshold=scdet_threshold,
                quality_maps=shot_quality_maps,
                human_policy_binding_present=human_reframe_policy_requested,
            )
        )
        write_json(
            output_dir / "quality-map-coverage-audit.json",
            quality_map_coverage_audit,
        )
        if execution_profile == FeatureCutExecutionProfile.PRODUCTION_REVIEW:
            preflight_failures = _production_review_preflight_failures(
                requested_candidate_recall_audit,
                quality_map_coverage_audit,
            )
            if preflight_failures:
                raise ValueError(
                    "production_review preflight failed: "
                    + ", ".join(preflight_failures)
                    + "; use review_preview only when an auditable preview is "
                    "intended"
                )
        source_capacity_seconds = _selected_source_capacity_seconds(
            plan,
            aspect=aspect,
            frames=frames,
            clips=clips,
            shot_cache=shot_cache,
            shots_dir=shots_dir,
            scdet_threshold=scdet_threshold,
            approved_decisions=trim_decisions,
            quality_maps=shot_quality_maps,
        )
        fixed_duration_seconds = _selected_fixed_trim_durations_seconds(
            plan,
            aspect=aspect,
            frames=frames,
            clips=clips,
            shot_cache=shot_cache,
            shots_dir=shots_dir,
            scdet_threshold=scdet_threshold,
            approved_decisions=trim_decisions,
        )
        editorial_dir = output_dir / "editorial-planning"
        attention_path = editorial_dir / "attention-profile.json"
        attention_profile = build_attention_profile(
            brief,
            plan,
            source_brief_sha256=sha256_file(brief_path),
            source_feature_plan_sha256=sha256_file(plan_path),
            quality_safe_capacity_seconds=source_capacity_seconds,
        )
        project_duration_seconds = brief.target_duration_seconds
        duration_resolution_authority = "brief_preferred_duration"
        attention_maximum_seconds = round(
            sum(
                chapter.maximum_dwell_seconds
                for chapter in attention_profile.chapters
            ),
            3,
        )
        if (
            attention_maximum_seconds + 0.001
            < project_duration_seconds
            and allow_shorter_within_delivery_range
            and not fixed_duration_seconds
        ):
            if attention_maximum_seconds < 60.0:
                attention_profile, floor_reconciliation = (
                    reconcile_attention_delivery_floor(
                        attention_profile,
                        delivery_floor_seconds=60.0,
                        maximum_shortfall_tolerance_seconds=1.0,
                    )
                )
                write_json(
                    editorial_dir
                    / "attention-delivery-floor-reconciliation.json",
                    floor_reconciliation,
                )
                attention_maximum_seconds = round(
                    sum(
                        chapter.maximum_dwell_seconds
                        for chapter in attention_profile.chapters
                    ),
                    3,
                )
            project_duration_seconds = attention_maximum_seconds
            duration_resolution_authority = (
                "operator_authorized_shorter_attention_maximum"
            )
        write_json(attention_path, attention_profile)
        write_json(
            editorial_dir / "project-duration-resolution.json",
            {
                "brief_preferred_duration_seconds": brief.target_duration_seconds,
                "attention_maximum_seconds": attention_maximum_seconds,
                "resolved_project_duration_seconds": project_duration_seconds,
                "authority": duration_resolution_authority,
                "allow_shorter_within_delivery_range": (
                    allow_shorter_within_delivery_range
                ),
                "fixed_trim_duration_count": len(fixed_duration_seconds),
                "generated_at": utc_now(),
            },
        )
        rhythm_path = editorial_dir / "rhythm-plan.json"
        rhythm_plan = build_rhythm_plan(
            attention_profile,
            target_duration_seconds=project_duration_seconds,
            attention_profile_sha256=sha256_file(attention_path),
            style_profile=rhythm_style,
        )
        write_json(rhythm_path, rhythm_plan)
        chapter_durations, duration_audit = _resolve_editorial_chapter_durations(
            brief,
            plan,
            music_lock=music_lock,
            source_capacity_seconds=source_capacity_seconds,
            fixed_duration_seconds=fixed_duration_seconds,
            rhythm_plan=rhythm_plan,
            project_duration_seconds=project_duration_seconds,
            allow_music_lock_prefix=allow_shorter_within_delivery_range,
            shortfall_audit_path=(
                output_dir / "editorial-duration-capacity-shortfall.json"
            ),
        )
        write_json(output_dir / "editorial-duration-plan.json", duration_audit)
        horizontal_segments: list[Path] = []
        vertical_segments: list[Path] = []
        render_config = {
            "pipeline_version": _RENDER_PIPELINE_VERSION,
            "aspect": aspect,
            "brief": brief.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "editorial_duration_plan": duration_audit,
            "music_sha256": music_sha256,
            "sam_analysis_fps": sam_analysis_fps,
            "scdet_threshold": scdet_threshold,
            "trim_decisions": [
                {
                    "path": str(path),
                    "decision": decision.model_dump(mode="json"),
                }
                for path, decision in trim_decisions
            ],
            "shot_quality_maps": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "source_asset_id": quality_map.source_asset_id,
                    "shot_id": quality_map.shot_id,
                    "request_sha256": quality_map.request_sha256,
                }
                for path, quality_map in shot_quality_maps
            ],
            "post_render_quality_qc": post_render_quality_qc,
            "rhythm_style": rhythm_style,
            "allow_shorter_within_delivery_range": (
                allow_shorter_within_delivery_range
            ),
            "resolved_project_duration_seconds": project_duration_seconds,
            "attention_profile_path": str(attention_path.resolve()),
            "attention_profile_sha256": sha256_file(attention_path),
            "rhythm_plan_path": str(rhythm_path.resolve()),
            "rhythm_plan_sha256": sha256_file(rhythm_path),
            "allow_proposed_trim_preview": allow_proposed_trim_preview,
            "allow_unverified_geometry_preview": allow_unverified_geometry_preview,
        }
        render_key = hashlib.sha256(
            json.dumps(render_config, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        render_variant = (
            f"with-titles-{render_key}"
            if brief.render_title_overlays
            else f"clean-{render_key}"
        )
        manifest: dict[str, Any] = {
            "project_id": brief.project_id,
            "catalog_id": catalog.catalog_id,
            "render_title_overlays": brief.render_title_overlays,
            "render_pipeline_version": _RENDER_PIPELINE_VERSION,
            "render_cache_key": render_key,
            "requested_aspect": aspect,
            "execution_profile": execution_profile.value,
            "feature_plan_reused": plan_reused,
            "music_supplied_to_feature_planner": resolved_music_path is not None,
            "music_sha256": music_sha256,
            "feature_plan_candidate_audit": _audit_feature_plan_candidate_recall(
                plan,
                frame_source_assets={
                    frame.frame_id: f"sha256:{clips[frame.clip_id].sha256}"
                    for frame in catalog.frames
                },
            ),
            "requested_candidate_recall_audit": (
                requested_candidate_recall_audit
            ),
            "quality_map_coverage_audit": quality_map_coverage_audit,
            "editorial_duration_plan": duration_audit,
            "project_duration_resolution": {
                "brief_preferred_duration_seconds": brief.target_duration_seconds,
                "resolved_project_duration_seconds": project_duration_seconds,
                "authority": duration_resolution_authority,
            },
            "feature_plan_binding": str(plan_binding_path.resolve()),
            "feature_plan_reuse_record": (
                str(plan_reuse_record_path.resolve())
                if plan_reuse_record_path is not None
                else None
            ),
            "reframe_policy_binding": (
                brief.reframe_policy_binding.model_dump(mode="json")
                if brief.reframe_policy_binding is not None
                else None
            ),
            "approved_trim_decision_count": sum(
                decision.approval_status == "approved" for _, decision in trim_decisions
            ),
            "unreviewed_trim_proposal_count": sum(
                decision.approval_status == "proposed" for _, decision in trim_decisions
            ),
            "contains_unreviewed_trim_proposals": any(
                decision.approval_status == "proposed" for _, decision in trim_decisions
            ),
            "allow_unverified_geometry_preview": allow_unverified_geometry_preview,
            "trim_decisions": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "approval_status": decision.approval_status,
                    "requires_human_review": decision.requires_human_review,
                    "event_id": decision.event_id,
                    "source_asset_id": decision.source_asset_id,
                }
                for path, decision in trim_decisions
            ],
            "shot_quality_maps": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "source_asset_id": quality_map.source_asset_id,
                    "shot_id": quality_map.shot_id,
                    "request_sha256": quality_map.request_sha256,
                }
                for path, quality_map in shot_quality_maps
            ],
            "post_render_quality_qc_requested": post_render_quality_qc,
            "editorial_planning": {
                "attention_profile_path": str(attention_path.resolve()),
                "attention_profile_sha256": sha256_file(attention_path),
                "rhythm_plan_path": str(rhythm_path.resolve()),
                "rhythm_plan_sha256": sha256_file(rhythm_path),
                "rhythm_style": rhythm_style,
            },
            "horizontal": {
                "requested": render_horizontal,
                "status": "pending" if render_horizontal else "not_requested",
                "chapters": [],
            },
            "vertical": {
                "requested": render_vertical,
                "status": "pending" if render_vertical else "not_requested",
                "chapters": [],
            },
        }
        track_cache: dict[tuple[str, str, int, int], tuple[GroundingProposal, SegmentationTrack, Path]] = {}
        source_audio_cache: dict[str, bool] = {}
        source_media_cache: dict[str, MediaInfo] = {}
        stage = monotonic()
        for index, selected in enumerate(plan.chapters):
            brief_chapter = brief_by_id[selected.feature_id]
            chapter_duration_seconds = chapter_durations[selected.feature_id]
            horizontal_overlay = output_dir / "overlays" / "16x9" / f"{index:02d}.png"
            vertical_overlay = output_dir / "overlays" / "9x16" / f"{index:02d}.png"
            horizontal_segment = (
                output_dir / "segments" / render_variant / "16x9" / f"{index:02d}.mp4"
            )
            vertical_segment = (
                output_dir / "segments" / render_variant / "9x16" / f"{index:02d}.mp4"
            )
            if selected.evidence_status == "not_found":
                if render_horizontal:
                    if not _segment_is_valid(
                        horizontal_segment,
                        expected_duration=chapter_duration_seconds,
                        dimensions=(1920, 1080),
                    ):
                        _render_missing_segment(
                            brief_chapter.model_copy(
                                update={
                                    "target_duration_seconds": chapter_duration_seconds
                                }
                            ),
                            horizontal_segment,
                            horizontal_overlay,
                            (1920, 1080),
                        )
                    horizontal_entry = {
                        "feature_id": selected.feature_id,
                        "source_frame_id": None,
                        "semantic_intent": brief_chapter.title,
                        "observed_visual_evidence": selected.observed_visual_evidence,
                        "selection_reason": selected.selection_reason,
                        "duration_ms": round(
                            chapter_duration_seconds * 1000
                        ),
                        "source_clip_id": None,
                        "source_in_ms": None,
                        "source_out_ms": None,
                        "segment_render_fingerprint": sha256_file(horizontal_segment),
                        "segment_path": str(horizontal_segment.resolve()),
                        "applied_zoom": 1.0,
                        "fallback_reason": "catalog_evidence_not_found",
                        "audio_origin": "synthetic_silence",
                    }
                if render_vertical:
                    if not _segment_is_valid(
                        vertical_segment,
                        expected_duration=chapter_duration_seconds,
                        dimensions=(1080, 1920),
                    ):
                        _render_missing_segment(
                            brief_chapter.model_copy(
                                update={
                                    "target_duration_seconds": chapter_duration_seconds
                                }
                            ),
                            vertical_segment,
                            vertical_overlay,
                            (1080, 1920),
                        )
                    vertical_entry = {
                        "feature_id": selected.feature_id,
                        "source_frame_id": None,
                        "semantic_intent": brief_chapter.title,
                        "observed_visual_evidence": selected.observed_visual_evidence,
                        "selection_reason": selected.selection_reason,
                        "duration_ms": round(
                            chapter_duration_seconds * 1000
                        ),
                        "source_clip_id": None,
                        "source_in_ms": None,
                        "source_out_ms": None,
                        "segment_render_fingerprint": sha256_file(vertical_segment),
                        "segment_path": str(vertical_segment.resolve()),
                        "applied_strategy": "graphic_missing_evidence_card",
                        "fallback_reason": "catalog_evidence_not_found",
                        "audio_origin": "synthetic_silence",
                    }
            else:
                if brief.render_title_overlays:
                    if render_horizontal:
                        _render_text_layer(
                            brief_chapter, horizontal_overlay, dimensions=(1920, 1080)
                        )
                    if render_vertical:
                        _render_text_layer(
                            brief_chapter, vertical_overlay, dimensions=(1080, 1920)
                        )
                horizontal_candidate_attempts: list[dict[str, Any]] = []
                horizontal_selected_option: dict[str, Any] | None = None
                if render_horizontal:
                    prepared_horizontal: dict[str, Any] | None = None
                    for horizontal_option in _horizontal_runtime_candidate_options(
                        selected
                    ):
                        try:
                            prepared_horizontal = (
                                _prepare_horizontal_runtime_candidate(
                                    option=horizontal_option,
                                    selected=selected,
                                    brief_chapter=brief_chapter,
                                    chapter_duration_seconds=(
                                        chapter_duration_seconds
                                    ),
                                    frames=frames,
                                    clips=clips,
                                    shot_cache=shot_cache,
                                    shots_dir=shots_dir,
                                    scdet_threshold=scdet_threshold,
                                    trim_decisions=trim_decisions,
                                    shot_quality_maps=shot_quality_maps,
                                    source_audio_cache=source_audio_cache,
                                    source_media_cache=source_media_cache,
                                    track_cache=track_cache,
                                    client=client,
                                    checkpoint_path=checkpoint_path,
                                    grounding_prompt=grounding_prompt,
                                    output_dir=output_dir,
                                    sam_analysis_fps=sam_analysis_fps,
                                    model_request_block_reason=(
                                        gemini_geometry_block_reason
                                    ),
                                )
                            )
                        except Exception as error:
                            abort_for_geometry_quota(error)
                            horizontal_candidate_attempts.append(
                                {
                                    "candidate_id": horizontal_option.get(
                                        "candidate_id"
                                    ),
                                    "rank": horizontal_option.get("rank"),
                                    "status": "rejected",
                                    "reason": (
                                        f"{type(error).__name__}:{error}"
                                    ),
                                }
                            )
                            continue
                        horizontal_selected_option = dict(horizontal_option)
                        horizontal_candidate_attempts.append(
                            {
                                "candidate_id": horizontal_option.get(
                                    "candidate_id"
                                ),
                                "rank": horizontal_option.get("rank"),
                                "status": "selected",
                                "reason": None,
                            }
                        )
                        break
                    if (
                        prepared_horizontal is None
                        or horizontal_selected_option is None
                    ):
                        raise ValueError(
                            "all 16:9 candidates failed quality, capacity, "
                            "geometry or lineage preflight"
                        )
                    # The immutable plan remains unchanged on disk. This local
                    # projection lets the established renderer consume the
                    # runtime-selected candidate while the manifest records
                    # every attempt.
                    selected = selected.model_copy(
                        update={
                            "horizontal_frame_id": prepared_horizontal[
                                "frame"
                            ].frame_id,
                            "horizontal_strategy": (
                                horizontal_selected_option["strategy"]
                            ),
                            "horizontal_zoom_intent": (
                                horizontal_selected_option["zoom_intent"]
                            ),
                            "horizontal_camera_intent": (
                                horizontal_selected_option.get(
                                    "camera_intent",
                                    "hold",
                                )
                            ),
                            "horizontal_target_description": (
                                horizontal_selected_option.get(
                                    "target_description"
                                )
                            ),
                        }
                    )
                if render_horizontal:
                    if prepared_horizontal is None:
                        raise AssertionError(
                            "horizontal renderer requires a preflighted candidate"
                        )
                    horizontal_frame = prepared_horizontal["frame"]
                    horizontal_clip = prepared_horizontal["clip"]
                    h_start = int(prepared_horizontal["start_ms"])
                    h_end = int(prepared_horizontal["end_ms"])
                    h_shot = str(prepared_horizontal["shot_id"])
                    horizontal_trim = prepared_horizontal["trim"]
                    horizontal_source_has_audio = bool(
                        prepared_horizontal["source_has_audio"]
                    )
                    horizontal_source_media = prepared_horizontal["media"]
                    horizontal_filter = str(prepared_horizontal["filter"])
                    horizontal_geometry = dict(
                        prepared_horizontal["geometry"]
                    )
                    horizontal_debug = prepared_horizontal["debug"]
                    horizontal_track_fingerprint = prepared_horizontal[
                        "track_fingerprint"
                    ]
                    horizontal_geometry["automatic_candidate_selection"] = {
                        "contract_version": "full-auto-candidate-routing-v2",
                        "enabled": bool(selected.horizontal_candidates),
                        "planned_candidate_count": len(
                            selected.horizontal_candidates
                        ),
                        "selected_candidate_id": (
                            horizontal_selected_option or {}
                        ).get("candidate_id"),
                        "selected_candidate_rank": (
                            horizontal_selected_option or {}
                        ).get("rank"),
                        "attempts": horizontal_candidate_attempts,
                    }
                    horizontal_source_interval = _exact_render_source_interval(
                        source_path=Path(horizontal_clip.path),
                        source_sha256=horizontal_clip.sha256,
                        start_ms=h_start,
                        end_ms=h_end,
                        trim=horizontal_trim,
                        output_dir=(
                            output_dir
                            / "render-boundary-evidence"
                            / selected.feature_id
                            / "16x9"
                        ),
                    )
                    horizontal_segment_fingerprint = _segment_variant_fingerprint(
                        source_sha256=horizontal_clip.sha256,
                        start_ms=h_start,
                        end_ms=h_end,
                        filter_graph=horizontal_filter,
                        geometry=horizontal_geometry,
                        track_fingerprint=horizontal_track_fingerprint,
                        source_interval=horizontal_source_interval,
                    )
                    horizontal_segment = (
                        output_dir
                        / "segments"
                        / render_variant
                        / "16x9"
                        / f"{index:02d}-{horizontal_segment_fingerprint[:12]}.mp4"
                    )
                    if not _segment_is_valid(
                        horizontal_segment,
                        expected_duration=(h_end - h_start) / 1000,
                        dimensions=(1920, 1080),
                    ):
                        _render_source_segment(
                            source_path=Path(horizontal_clip.path),
                            start_ms=h_start,
                            end_ms=h_end,
                            overlay_path=(
                                horizontal_overlay
                                if brief.render_title_overlays
                                else None
                            ),
                            base_filter=horizontal_filter,
                            output_path=horizontal_segment,
                            source_has_audio=horizontal_source_has_audio,
                            source_interval=horizontal_source_interval,
                        )
                    horizontal_boundary_lineage = _write_render_boundary_lineage(
                        segment_path=horizontal_segment,
                        source_interval=horizontal_source_interval,
                        output_path=(
                            output_dir
                            / "render-boundary-lineage"
                            / selected.feature_id
                            / "16x9.json"
                        ),
                    )
                if not render_vertical:
                    horizontal_entry = _horizontal_manifest_entry(
                        selected=selected,
                        brief_chapter=brief_chapter,
                        frame=horizontal_frame,
                        clip=horizontal_clip,
                        start_ms=h_start,
                        end_ms=h_end,
                        shot_id=h_shot,
                        segment_fingerprint=horizontal_segment_fingerprint,
                        track_fingerprint=horizontal_track_fingerprint,
                        segment=horizontal_segment,
                        source_has_audio=horizontal_source_has_audio,
                        source_media=horizontal_source_media,
                        grounding_debug=horizontal_debug,
                        trim=horizontal_trim,
                        geometry=horizontal_geometry,
                        source_interval=horizontal_source_interval,
                        render_boundary_lineage=horizontal_boundary_lineage,
                    )
                    horizontal_segments.append(horizontal_segment)
                    manifest["horizontal"]["chapters"].append(horizontal_entry)
                    continue
                # Candidate alternatives are already hash-bound inside the saved
                # FeatureEditPlan. Geometry may select among them at runtime, but
                # never rewrites the editorial plan. A human policy binding disables
                # this automatic switching path entirely.
                vertical_primary_override = (
                    brief_chapter.vertical_primary_target_description
                )
                # feature-cut produces an auditable review render, not an
                # unattended production approval. Missing independent
                # identity checkpoints and optional-context shortfalls remain
                # advisories; hard-core, atomic, lineage, tracking, and motion
                # failures still block the candidate.
                auto_reframe_policy = AutoReframePolicy(
                    require_semantic_checkpoints_for_tracked_crop=False,
                    soft_extent_below_minimum_is_failure=False,
                )
                vertical_options = _vertical_runtime_candidate_options(
                    selected,
                    human_policy_binding_present=human_reframe_policy_requested,
                    max_candidates=auto_reframe_policy.max_candidates,
                )
                candidate_attempts: list[dict[str, Any]] = []
                deferred_fit: dict[str, Any] | None = None
                deferred_required_scope_fit: dict[str, Any] | None = None
                selected_vertical: dict[str, Any] | None = None
                for option_index, option in enumerate(vertical_options):
                    option_data = option
                    candidate_id = str(option_data["candidate_id"])
                    candidate_rank = int(option_data["rank"])
                    frame_id = str(option_data["frame_id"])
                    try:
                        candidate_frame = frames[frame_id]
                        candidate_clip = clips[candidate_frame.clip_id]
                        expected_asset = option_data.get("source_asset_id")
                        if not _candidate_asset_reference_matches(
                            expected_asset,
                            candidate_clip,
                        ):
                            raise ValueError(
                                "vertical candidate source asset differs from "
                                f"its frame: {frame_id}"
                            )
                        if candidate_clip.sha256 not in source_audio_cache:
                            source_audio_cache[candidate_clip.sha256] = (
                                has_audio_stream(Path(candidate_clip.path))
                            )
                        if candidate_clip.sha256 not in source_media_cache:
                            source_media_cache[candidate_clip.sha256] = probe_video(
                                Path(candidate_clip.path)
                            )
                        candidate_media = source_media_cache[
                            candidate_clip.sha256
                        ]
                        candidate_display_sar = (
                            candidate_media.video.display_sample_aspect_ratio.numerator
                            / candidate_media.video.display_sample_aspect_ratio.denominator
                        )
                        (
                            candidate_start,
                            candidate_end,
                            candidate_shot,
                            candidate_trim,
                        ) = _chapter_bounds_with_approved_trim(
                            candidate_frame,
                            candidate_clip,
                            chapter_duration_seconds,
                            shot_cache,
                            shots_dir,
                            scdet_threshold,
                            trim_decisions,
                            expected_event_id=option_data.get("event_id"),
                            quality_maps=shot_quality_maps,
                        )
                    except Exception as error:
                        abort_for_geometry_quota(error)
                        candidate_attempts.append(
                            {
                                "candidate_id": candidate_id,
                                "rank": candidate_rank,
                                "frame_id": frame_id,
                                "source_asset_id": option_data.get(
                                    "source_asset_id"
                                ),
                                "event_id": option_data.get("event_id"),
                                "strategy": option_data.get("strategy"),
                                "decision": "try_next",
                                "reason_code": (
                                    "candidate_lineage_quality_or_capacity_failed"
                                ),
                                "failure_codes": [
                                    FailureCode.NO_FEASIBLE_PRESENTATION.value
                                ],
                                "recovery_action": (
                                    "try_next_candidate"
                                    if option_index + 1 < len(vertical_options)
                                    else "fallback_requires_review"
                                ),
                                "error": f"{type(error).__name__}:{error}",
                            }
                        )
                        continue
                    framing_refinement: dict[str, Any] | None = None
                    framing_recommended_action: str | None = None
                    if _should_refine_selected_vertical_candidate(
                        auto_vertical_framing=auto_vertical_framing,
                        human_reframe_policy_requested=(
                            human_reframe_policy_requested
                        ),
                        feature_plan_origin=feature_plan_origin,
                        external_projection_contract_id=(
                            external_projection_contract_id
                        ),
                        option_data=option_data,
                    ):
                        try:
                            (
                                option_data,
                                framing_proposal,
                                framing_reused,
                            ) = _refine_selected_vertical_candidate(
                                client=client,
                                option_data=option_data,
                                chapter=brief_chapter,
                                clip=candidate_clip,
                                frame=candidate_frame,
                                prompt_template=resolved_vertical_framing_prompt,
                                catalog_path=catalog_path,
                                output_dir=output_dir,
                                vertical_fallback_strategy=(
                                    brief.vertical_fallback_strategy
                                ),
                            )
                        except Exception as error:
                            abort_for_geometry_quota(error)
                            failure_codes = _failure_codes_from_geometry_error(
                                error
                            )
                            recovery = choose_recovery(
                                failure_codes,
                                candidates_remaining=(
                                    option_index + 1 < len(vertical_options)
                                ),
                            )
                            candidate_attempts.append(
                                {
                                    "candidate_id": candidate_id,
                                    "rank": candidate_rank,
                                    "frame_id": frame_id,
                                    "source_asset_id": (
                                        f"sha256:{candidate_clip.sha256}"
                                    ),
                                    "event_id": option_data.get("event_id"),
                                    "strategy": option_data["strategy"],
                                    "decision": "try_next",
                                    "reason_code": (
                                        "selected_framing_contract_failed"
                                    ),
                                    "failure_codes": [
                                        failure.value
                                        for failure in failure_codes
                                    ],
                                    "recovery_action": recovery.value,
                                    "error": str(error),
                                }
                            )
                            continue
                        framing_recommended_action = (
                            framing_proposal.recommended_action
                        )
                        framing_refinement = {
                            **dict(option_data["framing_refinement"]),
                            "reused": framing_reused,
                        }
                    candidate_regions, candidate_target = (
                        _resolve_vertical_candidate_intent(
                            option_regions=option_data.get("regions", []),
                            option_target_description=option_data.get(
                                "target_description"
                            ),
                            selected_target_description=(
                                selected.vertical_target_description
                            ),
                            brief_primary_target_description=(
                                brief_chapter.vertical_primary_target_description
                            ),
                            brief_regions=brief_chapter.vertical_regions,
                            inherit_reviewed_brief_intent=(
                                not selected.vertical_candidates
                                or human_reframe_policy_requested
                            ),
                        )
                    )
                    candidate_camera_phases, candidate_camera_phase_origin = (
                        _resolve_vertical_camera_phases(
                            option_data=option_data,
                            reviewed_phases=brief_chapter.vertical_camera_phases,
                        )
                    )
                    attempt: dict[str, Any] = {
                        "candidate_id": candidate_id,
                        "rank": candidate_rank,
                        "frame_id": frame_id,
                        "source_asset_id": f"sha256:{candidate_clip.sha256}",
                        "event_id": option_data.get("event_id"),
                        "strategy": option_data["strategy"],
                        "target_description": candidate_target,
                        "regions": [
                            region.model_dump(mode="json")
                            for region in candidate_regions
                        ],
                        "virtual_camera_proposal": option_data.get(
                            "virtual_camera_proposal"
                        ),
                        "camera_phase_origin": candidate_camera_phase_origin,
                        "framing_refinement": framing_refinement,
                    }
                    if framing_recommended_action == "try_next_candidate":
                        attempt.update(
                            {
                                "decision": "try_next",
                                "reason_code": (
                                    "full_clip_framing_recommends_alternate_candidate"
                                ),
                                "failure_codes": [
                                    FailureCode.NO_FEASIBLE_PRESENTATION.value
                                ],
                                "recovery_action": (
                                    "try_next_candidate"
                                    if option_index + 1 < len(vertical_options)
                                    else "fallback_requires_review"
                                ),
                            }
                        )
                        candidate_attempts.append(attempt)
                        continue
                    if (
                        option_data["strategy"] == "fit_with_background"
                        and not candidate_camera_phases
                    ):
                        fallback_filter, fallback_geometry = (
                            _vertical_delivery_fallback(
                                brief.vertical_fallback_strategy,
                                reason=(
                                    "gemini_fit_or_layout_after_full_bleed_attempts"
                                ),
                            )
                        )
                        attempt.update(
                            {
                                "decision": "deferred_fallback",
                                "reason_code": (
                                    "planner_requested_fit_or_layout;"
                                    f"delivery_fallback={brief.vertical_fallback_strategy}"
                                ),
                            }
                        )
                        candidate_attempts.append(attempt)
                        if deferred_fit is None:
                            deferred_fit = {
                                "option": option_data,
                                "frame": candidate_frame,
                                "clip": candidate_clip,
                                "media": candidate_media,
                                "start_ms": candidate_start,
                                "end_ms": candidate_end,
                                "shot_id": candidate_shot,
                                "trim": candidate_trim,
                                "regions": candidate_regions,
                                "target": candidate_target,
                                "filter": fallback_filter,
                                "geometry": fallback_geometry,
                                "debugs": [],
                                "track_fingerprint": None,
                            }
                        continue

                    candidate_root = (
                        output_dir
                        / "geometry"
                        / selected.feature_id
                        / "vertical"
                        / f"candidate-{candidate_rank:02d}-{candidate_id}"
                    )
                    try:
                        candidate_crop_mode = str(
                            option_data.get(
                                "crop_mode", brief_chapter.vertical_crop_mode
                            )
                        )
                        controlled_preview_requested = (
                            allow_unverified_geometry_preview
                            and candidate_crop_mode == "primary_center"
                            and not human_reframe_policy_requested
                        )
                        candidate_query_lock: EvidenceQueryLockV2 | None = None
                        if (
                            selected.vertical_candidates
                            and not human_reframe_policy_requested
                        ):
                            candidate_query_lock = (
                                _load_or_create_feature_candidate_query_lock_v2(
                                    _feature_vertical_candidate_from_runtime_option(
                                        option_data
                                    ),
                                    feature_id=selected.feature_id,
                                    output_dir=candidate_root,
                                )
                            )
                            attempt["evidence_query_v2"] = {
                                "query_id": candidate_query_lock.query_id,
                                "definition_sha256": (
                                    candidate_query_lock.definition_sha256()
                                ),
                                "component_hashes": (
                                    candidate_query_lock.component_hashes()
                                ),
                                "approval_source": (
                                    candidate_query_lock.approval.approval_source.value
                                ),
                                "policy_reference": (
                                    candidate_query_lock.approval.policy_reference
                                ),
                            }
                        (
                            candidate_filter,
                            candidate_geometry,
                            candidate_debugs,
                            candidate_track_fingerprint,
                        ) = _vertical_candidate_geometry(
                            client=client,
                            clip=candidate_clip,
                            frame=candidate_frame,
                            start_ms=candidate_start,
                            end_ms=candidate_end,
                            feature_id=selected.feature_id,
                            event_description=(
                                brief_chapter.title
                                + "；"
                                + str(option_data["observed_visual_evidence"])
                            ),
                            target_description=(
                                str(candidate_target) if candidate_target else None
                            ),
                            regions=candidate_regions,
                            camera_phases=candidate_camera_phases,
                            camera_phase_origin=candidate_camera_phase_origin,
                            crop_mode=candidate_crop_mode,
                            overflow_policy=(
                                brief_chapter.vertical_overflow_policy
                                if human_reframe_policy_requested
                                else (
                                    "controlled_clip"
                                    if controlled_preview_requested
                                    else "preserve_all"
                                )
                            ),
                            edge_priority=(
                                brief_chapter.vertical_edge_priority
                                if human_reframe_policy_requested
                                else "balanced"
                            ),
                            fallback_strategy=(
                                brief.vertical_fallback_strategy
                                if human_reframe_policy_requested
                                else "fit_with_background"
                            ),
                            checkpoint_path=checkpoint_path,
                            grounding_prompt=grounding_prompt,
                            output_dir=candidate_root,
                            analysis_fps=sam_analysis_fps,
                            scdet_threshold=scdet_threshold,
                            display_sample_aspect_ratio=candidate_display_sar,
                            track_cache=track_cache,
                            model_request_block_reason=gemini_geometry_block_reason,
                            query_lock_v2=candidate_query_lock,
                        )
                        auto_audit = None
                        failure_codes: list[FailureCode] = []
                        candidate_execution_verified = False
                        if candidate_geometry.get("fallback_reason") is not None:
                            hard_gate_passed = False
                            failure_codes = _failure_codes_from_geometry_error(
                                ValueError(str(candidate_geometry["fallback_reason"]))
                            )
                        else:
                            preflight, expected_geometry_fingerprint = (
                                _vertical_candidate_preflight(
                                    candidate_id=candidate_id,
                                    rank=candidate_rank,
                                    confidence=float(option_data["confidence"]),
                                    source_sha256=candidate_clip.sha256,
                                    filter_graph=candidate_filter,
                                    geometry=candidate_geometry,
                                    regions=candidate_regions,
                                    track_fingerprint=(
                                        candidate_track_fingerprint
                                    ),
                                    titles_rendered=brief.render_title_overlays,
                                )
                            )
                            auto_audit = audit_auto_bounded_clip(
                                preflight,
                                auto_reframe_policy,
                                expected_geometry_fingerprint=(
                                    expected_geometry_fingerprint
                                ),
                            )
                            hard_gate_passed = auto_audit.approved
                            candidate_execution_verified = auto_audit.approved
                            failure_codes = list(auto_audit.failure_codes)
                            preview_allowed_failures = {
                                FailureCode.IDENTITY_VERIFICATION_PENDING,
                            }
                            standard_preview_allowed = (
                                allow_unverified_geometry_preview
                                and candidate_geometry.get("fallback_reason") is None
                                and bool(failure_codes)
                                and set(failure_codes).issubset(
                                    preview_allowed_failures
                                )
                            )
                            controlled_primary_center_allowed = (
                                allow_unverified_geometry_preview
                                and _controlled_primary_center_preview_allowed(
                                    crop_mode=candidate_crop_mode,
                                    geometry=candidate_geometry,
                                    regions=candidate_regions,
                                    failure_codes=failure_codes,
                                )
                            )
                            if (
                                standard_preview_allowed
                                or controlled_primary_center_allowed
                            ):
                                hard_gate_passed = True
                                candidate_geometry[
                                    "unverified_geometry_preview_override"
                                ] = True
                                if controlled_primary_center_allowed:
                                    candidate_geometry[
                                        "controlled_primary_center_preview"
                                    ] = True
                                    candidate_geometry.setdefault(
                                        "risk_codes", []
                                    ).append(
                                        "review_only_controlled_primary_center_clip"
                                    )
                                candidate_geometry["requires_gemini_review"] = True
                                candidate_geometry.setdefault("risk_codes", []).append(
                                    "explicit_unverified_geometry_preview"
                                )
                            candidate_geometry["auto_bounded_clip_audit"] = (
                                auto_audit.model_dump(mode="json")
                            )
                            if auto_audit.advisory_codes:
                                candidate_geometry["requires_gemini_review"] = True
                                candidate_geometry.setdefault(
                                    "risk_codes", []
                                ).extend(
                                    code.value
                                    for code in auto_audit.advisory_codes
                                )
                            candidate_geometry["auto_bounded_clip_applied"] = (
                                auto_audit.auto_bounded_clip_applied
                            )
                            if auto_audit.auto_bounded_clip_applied:
                                candidate_geometry["automatic_policy_label"] = (
                                    "auto_bounded_clip_v1"
                                )
                        if human_reframe_policy_requested:
                            candidate_geometry[
                                "human_policy_execution_verified"
                            ] = candidate_execution_verified
                        recovery = (
                            None
                            if hard_gate_passed
                            else choose_recovery(
                                failure_codes,
                                candidates_remaining=(
                                    option_index + 1 < len(vertical_options)
                                ),
                            )
                        )
                        if (
                            not hard_gate_passed
                            and deferred_required_scope_fit is None
                            and FailureCode.NO_FEASIBLE_PRESENTATION
                            in failure_codes
                        ):
                            # A project preference for full bleed cannot turn a
                            # known semantic failure into a safe center crop.
                            # Preserve the tracked required-scope envelope as a
                            # deliberate solid-matte review layout when no
                            # full-bleed candidate can contain the evidence.
                            scoped_fit = _vertical_required_scope_fit_filter(
                                candidate_geometry
                            )
                            if scoped_fit is not None:
                                scoped_filter, scoped_geometry = scoped_fit
                                deferred_required_scope_fit = {
                                    "option": option_data,
                                    "frame": candidate_frame,
                                    "clip": candidate_clip,
                                    "media": candidate_media,
                                    "start_ms": candidate_start,
                                    "end_ms": candidate_end,
                                    "shot_id": candidate_shot,
                                    "trim": candidate_trim,
                                    "regions": candidate_regions,
                                    "target": candidate_target,
                                    "filter": scoped_filter,
                                    "geometry": {
                                        **scoped_geometry,
                                        "source_failed_geometry": {
                                            "applied_strategy": (
                                                candidate_geometry.get(
                                                    "applied_strategy"
                                                )
                                            ),
                                            "fallback_reason": (
                                                candidate_geometry.get(
                                                    "fallback_reason"
                                                )
                                            ),
                                            "track_geometry_fingerprint": (
                                                candidate_track_fingerprint
                                            ),
                                        },
                                    },
                                    "debugs": candidate_debugs,
                                    "track_fingerprint": (
                                        candidate_track_fingerprint
                                    ),
                                }
                        attempt.update(
                            {
                                "decision": (
                                    "accepted" if hard_gate_passed else "try_next"
                                ),
                                "reason_code": (
                                    "all_hard_gates_passed"
                                    if hard_gate_passed
                                    else ",".join(
                                        failure.value for failure in failure_codes
                                    )
                                    or candidate_geometry.get("fallback_reason")
                                    or "geometry_quality_gate_failed"
                                ),
                                "failure_codes": [
                                    failure.value for failure in failure_codes
                                ],
                                "recovery_action": (
                                    recovery.value if recovery is not None else None
                                ),
                                # Preserve the attempt as a value snapshot. The
                                # selected geometry later receives the complete
                                # attempts list; retaining this same dict here
                                # would make that audit payload self-referential.
                                "geometry": dict(candidate_geometry),
                                "track_fingerprint": candidate_track_fingerprint,
                            }
                        )
                        candidate_attempts.append(attempt)
                        if hard_gate_passed:
                            selected_vertical = {
                                "option": option_data,
                                "frame": candidate_frame,
                                "clip": candidate_clip,
                                "media": candidate_media,
                                "start_ms": candidate_start,
                                "end_ms": candidate_end,
                                "shot_id": candidate_shot,
                                "trim": candidate_trim,
                                "regions": candidate_regions,
                                "target": candidate_target,
                                "filter": candidate_filter,
                                "geometry": candidate_geometry,
                                "debugs": candidate_debugs,
                                "track_fingerprint": candidate_track_fingerprint,
                            }
                            break
                    except Exception as error:
                        abort_for_geometry_quota(error)
                        failure_codes = _failure_codes_from_geometry_error(error)
                        recovery = choose_recovery(
                            failure_codes,
                            candidates_remaining=(
                                option_index + 1 < len(vertical_options)
                            ),
                        )
                        attempt.update(
                            {
                                "decision": "try_next",
                                "reason_code": ",".join(
                                    failure.value for failure in failure_codes
                                ),
                                "failure_codes": [
                                    failure.value for failure in failure_codes
                                ],
                                "recovery_action": recovery.value,
                                "error": str(error),
                            }
                        )
                        candidate_attempts.append(attempt)
                        if human_reframe_policy_requested:
                            selected_vertical = {
                                "option": option_data,
                                "frame": candidate_frame,
                                "clip": candidate_clip,
                                "media": candidate_media,
                                "start_ms": candidate_start,
                                "end_ms": candidate_end,
                                "shot_id": candidate_shot,
                                "trim": candidate_trim,
                                "regions": candidate_regions,
                                "target": candidate_target,
                                "filter": (
                                    _vertical_center_crop_filter()
                                    if brief.vertical_fallback_strategy == "center_crop"
                                    else _vertical_fit_filter()
                                ),
                                "geometry": {
                                    "applied_strategy": brief.vertical_fallback_strategy,
                                    "fallback_reason": (
                                        "tracking_or_grounding_failed:"
                                        f"{type(error).__name__}:{error}"
                                    ),
                                    "risk_codes": [
                                        "tracking_or_grounding_failed",
                                        "human_policy_geometry_failed",
                                    ],
                                    "human_policy_execution_verified": False,
                                    "requires_gemini_review": True,
                                },
                                "debugs": [],
                                "track_fingerprint": None,
                            }
                            break

                if selected_vertical is None:
                    selected_vertical = (
                        deferred_required_scope_fit or deferred_fit
                    )
                if selected_vertical is None:
                    # Do not retry the first candidate outside the audited
                    # candidate loop. That previously masked the real geometry
                    # or schema failure with a second, misleading capacity
                    # exception. A caller may still produce a review-only
                    # fallback in a separate typed execution profile, but the
                    # production path must fail with the complete attempt
                    # ledger intact.
                    write_json(
                        output_dir
                        / "candidate-attempts"
                        / selected.feature_id
                        / "9x16.exhausted.json",
                        {
                            "contract_version": (
                                "automatic-candidate-exhaustion-v1"
                            ),
                            "feature_id": selected.feature_id,
                            "attempts": candidate_attempts,
                            "delivery_eligible": False,
                        },
                    )
                    raise ValueError(
                        "all 9:16 evidence-bound candidates failed quality, "
                        "capacity, geometry, identity or lineage preflight"
                    )
                vertical_frame = selected_vertical["frame"]
                vertical_clip = selected_vertical["clip"]
                vertical_source_media = selected_vertical["media"]
                v_start = selected_vertical["start_ms"]
                v_end = selected_vertical["end_ms"]
                v_shot = selected_vertical["shot_id"]
                vertical_trim = selected_vertical["trim"]
                vertical_regions = selected_vertical["regions"]
                vertical_target_description = selected_vertical["target"]
                vertical_filter = selected_vertical["filter"]
                vertical_geometry = selected_vertical["geometry"]
                vertical_debugs = selected_vertical["debugs"]
                vertical_track_fingerprint = selected_vertical[
                    "track_fingerprint"
                ]
                selected_candidate_id = str(
                    selected_vertical["option"]["candidate_id"]
                )
                selected_candidate_rank = int(
                    selected_vertical["option"]["rank"]
                )
                vertical_source_has_audio = source_audio_cache[vertical_clip.sha256]
                vertical_debug = next(
                    (path for path in vertical_debugs if path.exists()), None
                )
                vertical_geometry["automatic_candidate_selection"] = {
                    "contract_version": "full-auto-candidate-routing-v2",
                    "policy_id": auto_reframe_policy.policy_id,
                    "policy_sha256": auto_reframe_policy.definition_sha256(),
                    "enabled": bool(
                        selected.vertical_candidates
                        and not human_reframe_policy_requested
                    ),
                    "planned_candidate_count": len(
                        selected.vertical_candidates
                    ),
                    "selected_candidate_id": selected_candidate_id,
                    "selected_candidate_rank": selected_candidate_rank,
                    "attempts": candidate_attempts,
                    "human_policy_binding_present": human_reframe_policy_requested,
                    "center_crop_used_as_unverified_fallback": (
                        vertical_geometry.get("applied_strategy")
                        in {
                            "full_bleed_center_crop_review",
                            "unverified_center_crop_preview",
                            "policy_blocked_preview_center_crop",
                        }
                    ),
                }
                vertical_source_interval = _exact_render_source_interval(
                    source_path=Path(vertical_clip.path),
                    source_sha256=vertical_clip.sha256,
                    start_ms=v_start,
                    end_ms=v_end,
                    trim=vertical_trim,
                    output_dir=(
                        output_dir
                        / "render-boundary-evidence"
                        / selected.feature_id
                        / "9x16"
                    ),
                )
                vertical_segment_fingerprint = _segment_variant_fingerprint(
                    source_sha256=vertical_clip.sha256,
                    start_ms=v_start,
                    end_ms=v_end,
                    filter_graph=vertical_filter,
                    geometry=vertical_geometry,
                    track_fingerprint=vertical_track_fingerprint,
                    source_interval=vertical_source_interval,
                )
                vertical_segment = (
                    output_dir
                    / "segments"
                    / render_variant
                    / "9x16"
                    / f"{index:02d}-{vertical_segment_fingerprint[:12]}.mp4"
                )
                if not _segment_is_valid(
                    vertical_segment,
                    expected_duration=(v_end - v_start) / 1000,
                    dimensions=(1080, 1920),
                ):
                    _render_source_segment(
                        source_path=Path(vertical_clip.path),
                        start_ms=v_start,
                        end_ms=v_end,
                        overlay_path=(vertical_overlay if brief.render_title_overlays else None),
                        base_filter=vertical_filter,
                        output_path=vertical_segment,
                        source_has_audio=vertical_source_has_audio,
                        source_interval=vertical_source_interval,
                    )
                vertical_boundary_lineage = _write_render_boundary_lineage(
                    segment_path=vertical_segment,
                    source_interval=vertical_source_interval,
                    output_path=(
                        output_dir
                        / "render-boundary-lineage"
                        / selected.feature_id
                        / "9x16.json"
                    ),
                )
                if render_horizontal:
                    horizontal_entry = _horizontal_manifest_entry(
                        selected=selected,
                        brief_chapter=brief_chapter,
                        frame=horizontal_frame,
                        clip=horizontal_clip,
                        start_ms=h_start,
                        end_ms=h_end,
                        shot_id=h_shot,
                        segment_fingerprint=horizontal_segment_fingerprint,
                        track_fingerprint=horizontal_track_fingerprint,
                        segment=horizontal_segment,
                        source_has_audio=horizontal_source_has_audio,
                        source_media=horizontal_source_media,
                        grounding_debug=horizontal_debug,
                        trim=horizontal_trim,
                        geometry=horizontal_geometry,
                        source_interval=horizontal_source_interval,
                        render_boundary_lineage=horizontal_boundary_lineage,
                    )
                vertical_entry = {
                    "feature_id": selected.feature_id,
                    "semantic_intent": (
                        brief_chapter.title
                        + (" — " + "; ".join(brief_chapter.detail_lines) if brief_chapter.detail_lines else "")
                    ),
                    "observed_visual_evidence": selected.observed_visual_evidence,
                    "selection_reason": selected.selection_reason,
                    "source_reuse_mode": selected.source_reuse_mode,
                    "source_reuse_justification": (
                        selected.source_reuse_justification
                    ),
                    "source_frame_id": vertical_frame.frame_id,
                    "source_clip_id": vertical_clip.clip_id,
                    "source_in_ms": v_start,
                    "source_out_ms": v_end,
                    "duration_ms": v_end - v_start,
                    "source_shot_id": v_shot,
                    "source_interval": vertical_source_interval,
                    "render_boundary_lineage": vertical_boundary_lineage,
                    "segment_render_fingerprint": vertical_segment_fingerprint,
                    "track_geometry_fingerprint": vertical_track_fingerprint,
                    "segment_path": str(vertical_segment.resolve()),
                    "audio_origin": (
                        "source" if vertical_source_has_audio else "synthetic_silence"
                    ),
                    "source_sample_aspect_ratio": (
                        vertical_source_media.video.sample_aspect_ratio.model_dump(
                            mode="json"
                        )
                    ),
                    "source_display_sample_aspect_ratio": (
                        vertical_source_media.video.display_sample_aspect_ratio.model_dump(
                            mode="json"
                        )
                    ),
                    "target_description": vertical_target_description,
                    "primary_target_override": vertical_primary_override is not None,
                    "vertical_regions": [
                        region.model_dump(mode="json") for region in vertical_regions
                    ],
                    "vertical_overflow_policy": brief_chapter.vertical_overflow_policy,
                    "vertical_edge_priority": brief_chapter.vertical_edge_priority,
                    "vertical_crop_mode": brief_chapter.vertical_crop_mode,
                    "grounding_debug": str(vertical_debug.resolve()) if vertical_debug else None,
                    "grounding_debugs": [
                        str(path.resolve()) for path in vertical_debugs if path.exists()
                    ],
                    **vertical_trim,
                    **vertical_geometry,
                }
            if render_horizontal:
                horizontal_segments.append(horizontal_segment)
                manifest["horizontal"]["chapters"].append(horizontal_entry)
            if render_vertical:
                vertical_segments.append(vertical_segment)
                manifest["vertical"]["chapters"].append(vertical_entry)
        source_reuse_audits: dict[str, Any] = {}
        if render_horizontal:
            source_reuse_audits["16x9"] = _audit_render_source_reuse(
                plan,
                manifest["horizontal"]["chapters"],
                aspect="16x9",
            )
        if render_vertical:
            source_reuse_audits["9x16"] = _audit_render_source_reuse(
                plan,
                manifest["vertical"]["chapters"],
                aspect="9x16",
            )
        write_json(
            output_dir / "render-source-reuse-audit.json",
            {
                "contract_version": "render-source-reuse-audit-bundle-v1",
                "aspects": source_reuse_audits,
            },
        )
        blocked_reuse = [
            violation
            for audit in source_reuse_audits.values()
            for violation in audit["violations"]
        ]
        manifest["source_reuse_audit"] = source_reuse_audits
        manifest["source_reuse_contract_passed"] = not blocked_reuse
        # Source reuse is an editorial contract, not a decoder failure. Keep
        # the deterministic media available for the required human review, but
        # never let an unauthorised overlap become delivery eligible. This also
        # preserves the full rendered evidence needed to decide whether the
        # recurrence is a legitimate callback/recap or an accidental repeat.
        if blocked_reuse:
            manifest.setdefault("review_flags", []).append(
                "source_reuse_contract_failed"
            )
        timings["geometry_and_segment_render_seconds"] = round(monotonic() - stage, 3)
    finally:
        try:
            client.close()
        finally:
            incremental_pricing = _write_incremental_pricing(
                output_dir=output_dir,
                prior_interaction_hashes=prior_interaction_hashes,
                prior_error_hashes=prior_error_hashes,
            )
    try:
        output_suffix = "" if brief.render_title_overlays else "-clean"
        renders_dir = output_dir / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)
        horizontal_output = (
            renders_dir / f"feature-cut-16x9{output_suffix}.mp4"
            if render_horizontal
            else None
        )
        vertical_output = (
            renders_dir / f"feature-cut-9x16{output_suffix}.mp4"
            if render_vertical
            else None
        )
        stage = monotonic()
        if horizontal_output is not None:
            _concat_segments(horizontal_segments, horizontal_output)
        if vertical_output is not None:
            _concat_segments(vertical_segments, vertical_output)
        timings["concat_seconds"] = round(monotonic() - stage, 3)
        timings["total_seconds"] = round(monotonic() - started, 3)
        if horizontal_output is not None:
            manifest["horizontal"].update(
                {
                    "status": "rendered",
                    "output_path": str(horizontal_output.resolve()),
                    "media": _output_media_metadata(horizontal_output),
                }
            )
        if vertical_output is not None:
            manifest["vertical"].update(
                {
                    "status": "rendered",
                    "output_path": str(vertical_output.resolve()),
                    "media": _output_media_metadata(vertical_output),
                }
            )
            manifest["automatic_reframe_summary"] = _summarize_automatic_reframe(
                manifest["vertical"]["chapters"]
            )
        else:
            manifest["automatic_reframe_summary"] = {
                "status": "not_requested",
                "chapter_count": 0,
            }
        post_render_reports: dict[str, Any] = {}
        if post_render_quality_qc:
            quality_stage = monotonic()
            for aspect_name, output_path in (
                ("16x9", horizontal_output),
                ("9x16", vertical_output),
            ):
                if output_path is None:
                    continue
                report = build_render_quality_report(
                    output_path,
                    scdet_threshold=scdet_threshold,
                    output_dir=output_dir / "post-render-quality" / aspect_name,
                )
                post_render_reports[aspect_name] = report
            timings["post_render_quality_qc_seconds"] = round(
                monotonic() - quality_stage,
                3,
            )
        timings["total_seconds"] = round(monotonic() - started, 3)
        manifest["post_render_quality_qc"] = {
            "requested": post_render_quality_qc,
            "reports": post_render_reports,
            "technical_qc_passed": all(
                bool(report["technical_qc_passed"])
                for report in post_render_reports.values()
            ),
            "requires_human_review": any(
                bool(report["requires_human_review"])
                for report in post_render_reports.values()
            ),
        }
        eligibility = _build_feature_cut_eligibility_report(
            manifest,
            execution_profile=execution_profile,
        )
        manifest["media_rendered"] = eligibility.media_rendered
        manifest["run_state"] = eligibility.run_state.value
        manifest["delivery_eligible"] = eligibility.delivery_eligible
        manifest["delivery_eligibility"] = eligibility.model_dump(mode="json")
        manifest["generated_at"] = utc_now()
        write_json(output_dir / "delivery-eligibility.json", eligibility)
        write_json(output_dir / "render-manifest.json", manifest)
        if post_render_quality_qc and not manifest["post_render_quality_qc"][
            "technical_qc_passed"
        ]:
            raise ValueError(
                "post-render technical quality QC found hard decoder/PTS defects; "
                "inspect post-render-quality before delivery"
            )
        pricing = summarize_usage_and_list_price(output_dir)
        write_json(output_dir / "pricing.json", pricing)
        incremental_pricing = _write_incremental_pricing(
            output_dir=output_dir,
            prior_interaction_hashes=prior_interaction_hashes,
            prior_error_hashes=prior_error_hashes,
        )
        write_json(
            output_dir / "timing.json",
            {
                **timings,
                "file_api_reused": file_api_reused,
                "feature_plan_reuse_explicit": reuse_feature_plan and plan_reused,
                "feature_plan_reused": plan_reused,
                "generated_at": utc_now(),
            },
        )
        _render_review_html(output_dir, brief, plan, manifest)
        result = {
            "requested_aspect": aspect,
            "media_rendered": eligibility.media_rendered,
            "run_state": eligibility.run_state.value,
            "ready_for_human_review": eligibility.ready_for_human_review,
            "delivery_eligible": eligibility.delivery_eligible,
            "delivery_eligibility_path": str(
                (output_dir / "delivery-eligibility.json").resolve()
            ),
            "horizontal_output": (
                str(horizontal_output.resolve()) if horizontal_output is not None else None
            ),
            "vertical_output": (
                str(vertical_output.resolve()) if vertical_output is not None else None
            ),
            "review_path": str((output_dir / "index.html").resolve()),
            "plan_path": str((plan_dir / "feature_edit_plan.json").resolve()),
            "manifest_path": str((output_dir / "render-manifest.json").resolve()),
            "timing": timings,
            "pricing": pricing,
            "incremental_pricing": incremental_pricing,
        }
        write_json(output_dir / "result.json", result)
        return result
    finally:
        _write_incremental_pricing(
            output_dir=output_dir,
            prior_interaction_hashes=prior_interaction_hashes,
            prior_error_hashes=prior_error_hashes,
        )


def run_feature_cut_experiment(
    *,
    catalog_path: Path,
    brief_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    plan_prompt: str,
    grounding_prompt: str,
    vertical_framing_prompt: str | None = None,
    scdet_threshold: float = 4.0,
    sam_analysis_fps: float = 2.0,
    trim_decision_paths: Sequence[Path] = (),
    shot_quality_map_paths: Sequence[Path] = (),
    allow_proposed_trim_preview: bool = False,
    reuse_feature_plan: bool = False,
    reuse_feature_plan_raw_output: bool = False,
    allow_unverified_geometry_preview: bool = False,
    aspect: RenderAspect = "both",
    music_path: Path | None = None,
    music_lock_path: Path | None = None,
    post_render_quality_qc: bool = True,
    rhythm_style: Literal["calm", "standard", "energetic"] = "standard",
    allow_shorter_within_delivery_range: bool = False,
    auto_vertical_framing: bool = True,
    execution_profile: FeatureCutExecutionProfile | str = (
        FeatureCutExecutionProfile.REVIEW_PREVIEW
    ),
) -> dict[str, Any]:
    """Run feature-cut while atomically preserving terminal editorial state."""

    profile = FeatureCutExecutionProfile(execution_profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "run-status.json"
    if status_path.is_file():
        previous_status = read_json(status_path)
        if (
            isinstance(previous_status, dict)
            and previous_status.get("stage") == "running"
            and not bool(previous_status.get("terminal"))
        ):
            abandoned = {
                **previous_status,
                "stage": "abandoned",
                "terminal": True,
                "run_state": FeatureCutRunState.FAILED.value,
                "delivery_eligible": False,
                "error": {
                    "type": "AbandonedRun",
                    "message": (
                        "a prior process ended without writing terminal state; "
                        "this new explicit run reconciled it before resuming"
                    ),
                },
                "reconciled_at": utc_now(),
            }
            write_json(
                output_dir
                / "run-status-events"
                / f"abandoned-{uuid.uuid4().hex}.json",
                abandoned,
            )
    started_at = utc_now()
    run_instance_id = uuid.uuid4().hex
    write_json(
        status_path,
        {
            "contract_version": "feature-cut-run-status-v1",
            "run_instance_id": run_instance_id,
            "pid": os.getpid(),
            "execution_profile": profile.value,
            "stage": "running",
            "terminal": False,
            "media_rendered": False,
            "run_state": None,
            "delivery_eligible": False,
            "error": None,
            "started_at": started_at,
            "updated_at": started_at,
        },
    )
    try:
        result = _run_feature_cut_experiment_impl(
            catalog_path=catalog_path,
            brief_path=brief_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            plan_prompt=plan_prompt,
            grounding_prompt=grounding_prompt,
            vertical_framing_prompt=vertical_framing_prompt,
            scdet_threshold=scdet_threshold,
            sam_analysis_fps=sam_analysis_fps,
            trim_decision_paths=trim_decision_paths,
            shot_quality_map_paths=shot_quality_map_paths,
            allow_proposed_trim_preview=allow_proposed_trim_preview,
            reuse_feature_plan=reuse_feature_plan,
            reuse_feature_plan_raw_output=reuse_feature_plan_raw_output,
            allow_unverified_geometry_preview=allow_unverified_geometry_preview,
            aspect=aspect,
            music_path=music_path,
            music_lock_path=music_lock_path,
            post_render_quality_qc=post_render_quality_qc,
            rhythm_style=rhythm_style,
            allow_shorter_within_delivery_range=(
                allow_shorter_within_delivery_range
            ),
            auto_vertical_framing=auto_vertical_framing,
            execution_profile=profile,
        )
    except Exception as error:
        saved_eligibility_path = output_dir / "delivery-eligibility.json"
        saved_eligibility = (
            read_json(saved_eligibility_path)
            if saved_eligibility_path.is_file()
            else {}
        )
        write_json(
            status_path,
            {
                "contract_version": "feature-cut-run-status-v1",
                "run_instance_id": run_instance_id,
                "pid": os.getpid(),
                "execution_profile": profile.value,
                "stage": "failed",
                "terminal": True,
                "media_rendered": bool(
                    saved_eligibility.get("media_rendered", False)
                ),
                "run_state": FeatureCutRunState.FAILED.value,
                "delivery_eligible": False,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "started_at": started_at,
                "updated_at": utc_now(),
            },
        )
        raise
    write_json(
        status_path,
        {
            "contract_version": "feature-cut-run-status-v1",
            "run_instance_id": run_instance_id,
            "pid": os.getpid(),
            "execution_profile": profile.value,
            "stage": "completed",
            "terminal": True,
            "media_rendered": bool(result["media_rendered"]),
            "run_state": result["run_state"],
            "delivery_eligible": bool(result["delivery_eligible"]),
            "error": None,
            "started_at": started_at,
            "updated_at": utc_now(),
        },
    )
    return result
