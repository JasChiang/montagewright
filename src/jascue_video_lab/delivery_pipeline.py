from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .feature_cut import run_feature_cut_experiment
from .final_delivery import assemble_music_only_delivery
from .final_edit_qa import execute_final_edit_qa, prepare_final_edit_qa
from .gemini import GeminiLabClient, MODEL_ID
from .media import probe_video, sha256_file
from .music import MusicMapLock
from .music_assembly import (
    plan_single_interval_music_assembly,
    render_single_interval_music_assembly,
    write_music_assembly_artifacts,
)
from .storage import read_json, utc_now, write_json


class DeliveryPipelineBlocked(RuntimeError):
    """The pipeline preserved review artifacts but cannot continue safely."""


def _write_status(
    path: Path,
    *,
    stage: str,
    terminal: bool,
    state: str,
    error: BaseException | None = None,
    outputs: Mapping[str, Any] | None = None,
) -> None:
    write_json(
        path,
        {
            "contract_version": "feature-delivery-run-status-v1",
            "stage": stage,
            "terminal": terminal,
            "state": state,
            "delivery_eligible": False,
            "error": (
                None
                if error is None
                else {"type": type(error).__name__, "message": str(error)}
            ),
            "outputs": dict(outputs or {}),
            "updated_at": utc_now(),
        },
    )


def _qa_disposition(execution: Any) -> str:
    review = execution.result.global_review
    disposition = getattr(review, "disposition", None)
    if not isinstance(disposition, str):
        raise ValueError("FinalEditQA omitted its typed global disposition")
    return disposition


def run_feature_delivery_pipeline(
    *,
    feature_cut_kwargs: Mapping[str, Any],
    brief_path: Path,
    music_path: Path,
    music_lock_path: Path,
    output_dir: Path,
    model_id: str = MODEL_ID,
    execution_profile: str = "production_review",
) -> dict[str, Any]:
    """Run picture → continuous music → final mux → final QA as one chain.

    This function deliberately never grants final delivery eligibility. A
    successful run means the immutable final muxes and their QA packages are
    ready for a high-value human review. Human approval must be a later,
    separately hash-bound artifact.
    """

    resolved_output = output_dir.expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    status_path = resolved_output / "run-status.json"
    started_at = utc_now()
    _write_status(
        status_path,
        stage="feature_cut",
        terminal=False,
        state="running",
    )
    client: GeminiLabClient | None = None
    outputs: dict[str, Any] = {}
    try:
        kwargs = dict(feature_cut_kwargs)
        kwargs["output_dir"] = resolved_output / "picture"
        kwargs["brief_path"] = brief_path
        kwargs["music_path"] = music_path
        kwargs["music_lock_path"] = music_lock_path
        kwargs["execution_profile"] = execution_profile
        feature_result = run_feature_cut_experiment(**kwargs)
        outputs["feature_cut"] = feature_result
        picture_media_rendered = bool(feature_result.get("media_rendered"))
        picture_outputs_present = any(
            feature_result.get(f"{aspect_key}_output") is not None
            for aspect_key in ("horizontal", "vertical")
        )
        if not picture_media_rendered or not picture_outputs_present:
            raise DeliveryPipelineBlocked(
                "feature-cut did not produce reviewable picture media"
            )
        picture_ready_for_review = bool(
            feature_result.get("ready_for_human_review")
        )

        resolved_music = music_path.expanduser().resolve(strict=True)
        resolved_lock_path = music_lock_path.expanduser().resolve(strict=True)
        music_lock = MusicMapLock.model_validate(read_json(resolved_lock_path))
        if music_lock.music_id != f"sha256:{sha256_file(resolved_music)}":
            raise DeliveryPipelineBlocked(
                "reviewed MusicMap lock does not bind the supplied soundtrack"
            )

        render_manifest_path = Path(feature_result["manifest_path"]).resolve(
            strict=True
        )
        final_results: dict[str, Any] = {}
        client = GeminiLabClient(model_id=model_id)
        for aspect_key, aspect_ratio, qa_mode in (
            ("horizontal", "16:9", "canonical_16x9"),
            ("vertical", "9:16", "crop_only_9x16"),
        ):
            picture_value = feature_result.get(f"{aspect_key}_output")
            if picture_value is None:
                continue
            picture = Path(str(picture_value)).resolve(strict=True)
            picture_duration_ms = probe_video(picture).duration_ms
            aspect_dir = resolved_output / "aspects" / aspect_key
            assembly_dir = aspect_dir / "music-assembly"
            plan = plan_single_interval_music_assembly(
                music_lock,
                music_lock_path=resolved_lock_path,
                target_duration_ms=picture_duration_ms,
                minimum_duration_ms=max(1, picture_duration_ms - 100),
                maximum_duration_ms=picture_duration_ms + 100,
            )
            write_music_assembly_artifacts(plan, output_dir=assembly_dir)
            rendered_music = render_single_interval_music_assembly(
                resolved_music,
                plan,
                assembly_dir / "music.wav",
                assembly_dir,
            )
            delivery = assemble_music_only_delivery(
                picture_path=picture,
                music_path=rendered_music.output_audio_path,
                output_path=aspect_dir / f"final-{aspect_key}.mp4",
                manifest_path=aspect_dir / "final-delivery.json",
                music_assembly_artifact_dir=assembly_dir,
                aspect_ratio=aspect_ratio,
                artifact_bindings={
                    "feature_render_manifest_sha256": sha256_file(
                        render_manifest_path
                    ),
                    "music_map_lock_sha256": sha256_file(resolved_lock_path),
                },
            )
            qa_dir = aspect_dir / "final-qa"
            prepared = prepare_final_edit_qa(
                mode=qa_mode,
                render_path=delivery.output_path,
                manifest_path=render_manifest_path,
                output_dir=qa_dir,
                model_id=model_id,
                brief_path=brief_path if qa_mode == "canonical_16x9" else None,
                crop_include_audio=qa_mode == "canonical_16x9",
            )
            uploaded, file_reused = client.ensure_video_upload(
                prepared.proxy_path,
                qa_dir / "file-api" / prepared.input_hashes["proxy_sha256"],
            )
            qa = execute_final_edit_qa(
                prepared=prepared,
                client=client.client,
                uploaded_video=uploaded,
                output_dir=qa_dir,
            )
            final_results[aspect_key] = {
                "final_output": str(delivery.output_path),
                "final_output_sha256": sha256_file(delivery.output_path),
                "delivery_manifest": str(delivery.manifest_path),
                "music_assembly_manifest": str(rendered_music.manifest_path),
                "qa_run_dir": str(qa.run_dir),
                "qa_disposition": _qa_disposition(qa),
                "qa_cache_hit": qa.cache_hit,
                "file_api_reused": file_reused,
            }

        if not final_results:
            raise DeliveryPipelineBlocked(
                "feature-cut did not produce any requested picture output"
            )
        dispositions = {
            row["qa_disposition"] for row in final_results.values()
        }
        state = (
            "ready_for_human_review"
            if (
                picture_ready_for_review
                and dispositions == {"ready_for_human_review"}
            )
            else "review_required"
        )
        result = {
            "contract_version": "feature-delivery-result-v1",
            "started_at": started_at,
            "completed_at": utc_now(),
            "state": state,
            "media_rendered": True,
            "final_sequence_qa_completed": True,
            "human_approval_status": "not_run",
            "delivery_eligible": False,
            "picture_ready_for_human_review": picture_ready_for_review,
            "picture_run_state": feature_result.get("run_state"),
            "feature_cut": feature_result,
            "aspects": final_results,
        }
        write_json(resolved_output / "result.json", result)
        _write_status(
            status_path,
            stage="completed",
            terminal=True,
            state=state,
            outputs=final_results,
        )
        return result
    except Exception as error:
        _write_status(
            status_path,
            stage="blocked",
            terminal=True,
            state="blocked",
            error=error,
            outputs=outputs,
        )
        raise
    finally:
        if client is not None:
            client.close()
