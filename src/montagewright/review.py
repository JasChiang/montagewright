"""Look at the finished cut and say whether it is done.

The reviewer may approve. That is the loop's main way of converging: a system
whose critic can only ever ask for another round will keep asking until
something else stops it, and the something else is usually money.

It sees the cut, the brief, and the direction the cut set for itself. It does
not see manifests, ledgers, or degradation tables -- those are for the local
gates that already ran, and handing them over invites a reviewer to audit
paperwork instead of watching the film.

Rounds stop on the first of: approval, the round cap, no progress, or the
budget. Whichever fires, there is a finished cut in hand, because every round
renders before it reviews.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from montagewright.capabilities import describe_limits_for_prompt
from montagewright.gemini import structured_json, video_content
from montagewright.cost import BudgetSpent, Ledger
from montagewright.planner import ask
from montagewright.schema import looks_of, move_of_shot, must_be_whole_of, DegradationStep, Issue, ReviewVerdict

from montagewright.planner import MAX_OUTPUT_TOKENS, MODEL_ID

from montagewright.uploads import upload_now

PROMPTS = Path(__file__).resolve().parent / "prompts"
MAX_ROUNDS = 3


def _verdict_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdict", "overall", "issues"],
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["approve", "revise"],
                "description": (
                    "revise needs at least one issue at major or blocking. "
                    "Minor notes alone are recorded and do not start another "
                    "render, so a verdict of revise carrying only minor "
                    "issues is rejected -- say approve and leave the minor "
                    "notes, or name what actually has to change."
                ),
            },
            "overall": {"type": "string"},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["severity", "description", "fix"],
                    "properties": {
                        "clip_id": {"type": "string"},
                        "at_seconds": {
                            "type": "string",
                            "description": (
                                "片子裡大概哪個位置，寫成 MM:SS（`0:12`）。"
                                "大概就好，本機會對到最近的一顆上。"
                                "不要寫成秒數——一分鐘以上的片子，`1:12` "
                                "會變成 112 或 1.12，兩種讀法都落在片長內。"
                            ),
                        },
                        "issue_type": {
                            "type": "string",
                            "enum": [
                                "pacing", "framing", "music_sync", "coverage",
                                "named_fact", "continuity", "audio_content", "other",
                            ],
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["minor", "major", "blocking"],
                        },
                        "description": {"type": "string"},
                        "fix": {
                            "type": "string",
                            "description": (
                                "What to change. A note nobody can act on "
                                "does not start another round."
                            ),
                        },
                    },
                },
            },
        },
    }


@dataclass
class Round:
    index: int
    verdict: ReviewVerdict
    actionable: tuple[str, ...]


@dataclass
class Outcome:
    """Why the loop stopped, and what it left."""

    stopped_because: str
    rounds: list[Round] = field(default_factory=list)
    unadjudicated: list[DegradationStep] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return bool(self.rounds) and self.rounds[-1].verdict.verdict == "approve"


def _what_happened_already(rounds: "list[Round] | None") -> str:
    """What earlier rounds asked for, so this one is not asked blind."""

    if not rounds:
        return ""
    lines = ["\n## 這支片已經被改過，前面幾輪說了什麼\n"]
    for one in rounds:
        said = one.verdict
        lines.append(
            f"第 {one.index} 輪：{said.verdict}"
            + (
                "（"
                + "；".join(
                    f"{one.clip_id or '整體'} {one.severity}: {one.description}"
                    for one in said.issues[:4]
                )
                + "）"
                if said.issues else ""
            )
        )
        if one.actionable:
            lines.append(
                "  依此重規劃了：" + "、".join(str(x) for x in one.actionable[:6])
            )
    lines.append(
        "\n已經處理過的問題若已改善就不要重提；仍未改善要說明是同一個問題。\n"
    )
    return "\n".join(lines)


def _identity_review_parts(grounding_spec: Any, client: Any, cache: Any) -> list[dict[str, Any]]:
    if grounding_spec is None:
        return []
    from montagewright.reference_grounding import reference_prompt_parts

    return [{"type": "text", "text": (
        "以下是本片共用的身分證據與 scope。產品型號判斷必須比對這些參考圖，"
        "不可用記憶中的外觀替影片改名。看不清或證據矛盾請標示未確認，"
        "說明缺少哪個可見特徵；身分相符仍須獨立檢查裁切、可讀性與運鏡。"
    )}] + reference_prompt_parts(
        grounding_spec, client=client, cache=cache, resolution="high",
    )


def review_cut(
    preview: Path,
    *,
    brief: str,
    direction: str,
    client: Any,
    wanted_seconds: float = 0.0,
    delivered_seconds: float = 0.0,
    already: "list[Round] | None" = None,
    cache: Any = None,
    ledger: Ledger | None = None,
    model_id: str = MODEL_ID,
    grounding_spec: Any = None,
    complete_pass: bool = False,
    editorial_decisions: list[dict] | None = None,
    duration_contract: dict | None = None,
) -> ReviewVerdict:
    """One pass over a finished cut.

    The length is given because nothing else was checking it. Three planning
    layers each made a defensible call and delivered 17.9 seconds against 30,
    and the only trace was a number in the report that no stage read. The
    reviewer already judges whether this works as a film; whether it is the
    film that was asked for is the same question.
    """

    if ledger is not None:
        pass  # ask() checks budget after durable response lookup.

    instruction = (PROMPTS / "review_zh-TW.txt").read_text(
        encoding="utf-8"
    ).replace("{limits}", describe_limits_for_prompt())
    if editorial_decisions:
        instruction += ("\n目前剪輯決策如下，用來理解本版意圖，不是完成證據。初始創意定調可以經工具驗證後修訂，"
                        "使用者 brief 的硬條件始終優先。fit 代表刻意保留原畫面並留白，9:16 比例本身不等於必須裁切填滿。"
                        "仍須根據實際影片檢查主體、可讀性、節奏與聲音，不能因計畫聲稱做到就放行。\n"
                        + json.dumps(editorial_decisions, ensure_ascii=False))
    # Verify near-digital silence locally before asking for editorial audio
    # judgments. This evidence prevents invented speech from driving recuts.
    from montagewright.renderer import _peak
    if preview.exists() and _peak(preview) <= -90:
        instruction += "\n本機已量測此成片音軌為數位靜音（峰值低於 -90 dBFS）。不可聲稱聽到人聲或音樂；若需求需要聲音，仍應指出缺少聲音。"
    if duration_contract:
        instruction += (
            "\n\n## 本機交付片長規格\n\n"
            + json.dumps(duration_contract, ensure_ascii=False)
            + f"\n實際成片 {delivered_seconds:.3f} 秒。range 在上下限內即可，不必硬湊中心；"
            "低於下限只能作短版草稿並說明缺口，不能宣稱已符合片長。"
            "不得以重複、拖慢或停格補秒數；素材不足時保留自然短版，不要求無限重看重剪。"
            "首尾呼應或動作慢放必須有實際敘事功能，不能只相信計畫的理由。\n"
        )
    elif wanted_seconds > 0 and delivered_seconds > 0:
        off = delivered_seconds - wanted_seconds
        instruction += (
            f"\n\n## 長度\n\n這支片要 {wanted_seconds:.0f} 秒，"
            f"交出來是 {delivered_seconds:.1f} 秒"
            + (
                f"，差 {off:+.1f} 秒。差得多就是一個要修的問題——"
                "不是把每顆按比例縮放，是有顆不該在裡面，或有顆給得不夠。"
                if abs(off) > max(1.5, wanted_seconds * 0.12)
                else "，在範圍內。"
            )
            + "\n"
        )
    approval_path = None
    if ledger is not None and ledger.journal_path and preview.exists() and grounding_spec is None:
        from montagewright.checkpoints import key_for, read_json
        from montagewright.editor_workspace import decoded_digest
        identity = {"version": "approved-cut-v1", "media": decoded_digest(preview),
                    "instruction": instruction, "brief": brief, "direction": direction,
                    "model": model_id, "complete_pass": complete_pass,
                    "wanted_seconds": wanted_seconds, "delivered_seconds": delivered_seconds}
        approval_path = Path(ledger.journal_path).parent / "work" / "approved-cuts" / (key_for(identity) + ".json")
        saved = read_json(approval_path)
        if saved:
            print("review: reused approval for unchanged picture, sound and brief", flush=True)
            return ReviewVerdict.model_validate(saved)

    if cache is None:
        uri = upload_now(preview, client).uri
    else:
        uri, _ = cache.uri_for(preview, client, mime_type="video/mp4")

    interaction = ask(
        client,
        upload_cache=cache,
        model=model_id,
        store=False,
        input=[
        # The video first, the question after it: Google's guidance for a
        # single video is to put the text last.
            video_content(uri, resolution="high" if complete_pass else "low",
                          processing={"type": "static", "fps": 2.0} if complete_pass else "agentic"),
            *_identity_review_parts(grounding_spec, client, cache),
            {
                "type": "text",
                "text": (
                    f"{instruction}\n\n## 剪輯 brief\n\n{brief}\n\n"
                    f"## 這支片的創意定調\n\n{direction}\n"
                    # A second look at a cut that was changed because of the
                    # first one is a different question from a first look,
                    # and it was being asked as though it were the same. So
                    # the reviewer could repeat a complaint that had just
                    # been acted on, or approve a change without knowing one
                    # had been made.
                    + _what_happened_already(already)
                ),
            },
        ],
        generation_config={"thinking_level": "high", "max_output_tokens": MAX_OUTPUT_TOKENS},
        response_format=structured_json(_verdict_schema()),
        ledger=ledger,
        budget_stage="review",
    )
    from montagewright.planner import Usage, _parse

    payload = _parse(interaction, what="review")
    # A point in the finished film, so it comes back as a clock reading and
    # is resolved here rather than by the model that has to write it.
    from montagewright.spans import seconds_of

    for issue in payload.get("issues") or []:
        if isinstance(issue, dict) and "at_seconds" in issue:
            issue["at_seconds"] = seconds_of(issue["at_seconds"])
    verdict = ReviewVerdict.model_validate(payload)
    if approval_path is not None and verdict.verdict == "approve":
        from montagewright.checkpoints import write_json
        write_json(approval_path, verdict.model_dump(mode="json"))
    return verdict


def _shot_schema(clip_ids: list[str]) -> dict[str, Any]:
    """One verdict per shot, bound to the ids that were sent.

    One array of small flat objects. The reasoning that would nest lives in
    `note`, written once per shot, because a clause asked for inside a
    repeated item is what overran two output ceilings before.
    """

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["shots"],
        "properties": {
            "shots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "clip_id", "delivered", "note", "degradation_verdict"
                    ],
                    "properties": {
                        "clip_id": {"type": "string", "enum": clip_ids},
                        "delivered": {
                            "type": "boolean",
                            "description": (
                                "Whether the shot did what its plan said it "
                                "would."
                            ),
                        },
                        "note": {
                            "type": "string",
                            "description": (
                                "What is on screen, in one sentence. "
                                "'the push stops at the whole handset, never "
                                "reaching the camera module' beats 'poorly "
                                "executed'."
                            ),
                        },
                        "degradation_verdict": {
                            "type": "string",
                            "enum": ["none_recorded", "acceptable", "replan"],
                        },
                    },
                },
            }
        },
    }


def _describe_plan(shot: dict[str, Any], seconds: float, steps: list) -> str:
    """What this shot promised, so the promise can be checked."""

    # The looks themselves, not a move name and one subject. A reviewer
    # checking whether the shot did what it promised needs to know it
    # promised three stops, not that it promised "pan".
    looks = looks_of(shot)
    lines = [
        f"運鏡：{move_of_shot(shot)}",
        "畫面停在：" + "　→　".join(
            f"「{one.at}」（{one.framing}"
            + (f"，停 {one.seconds:g}s" if one.seconds else "")
            + ("，必須完整" if one.must_be_whole else "")
            + "）"
            for one in looks
        ) if looks else "畫面停在：（沒有指定主體）",
        f"長度：{seconds:.2f}s（選片估要 "
        f"{float(shot.get('seconds_needed') or 0):.1f}s）",
        f"為什麼挑這顆：{shot.get('why', '')}",
    ]
    if must_be_whole_of(shot):
        lines.append("這顆的主體被裁掉一部分就失去意義，字要讀得完。")
    for step in steps:
        measured = "、".join(f"{k}={v}" for k, v in step.measured.items())
        lines.append(
            f"降級紀錄：{step.ladder_other or step.ladder} —— "
            f"{step.trigger}（{measured}）"
        )
    return "\n".join(lines)


def review_shots(
    segments: dict[str, Path],
    shots: list[dict[str, Any]],
    *,
    seconds: dict[str, float],
    degradations: list[DegradationStep],
    client: Any,
    brief: str = "",
    cache: Any = None,
    ledger: Ledger | None = None,
    model_id: str = MODEL_ID,
    grounding_spec: Any = None,
    _batch: bool = True,
) -> dict[str, dict[str, Any]]:
    """Check each rendered shot against the plan that asked for it.

    A finished cut answers whether this is a film. It does not answer whether
    any one shot came out as planned, and it turns out it cannot: a wordmark
    cropped to "Galaxy Unpac", a pan ending on background wall, a push whose
    first frame is empty -- six in a row survived review of the whole thing,
    every one of them found by opening a single shot. At thirty seconds and
    low resolution the next shot arrives before the fault registers.

    It is also given the brief, which the whole-cut pass had and this one did
    not. A rule the brief states -- the folding phone's screen must be lit
    when it opens, no black screens -- is a per-shot fact, and this is the
    pass that watches one shot at a time at delivery resolution, where a dark
    screen the card read as lit at 1fps is plain. Without the brief it could
    only ask whether a shot met its own plan, never whether the plan met the
    film's requirements.
    """

    if not segments:
        return {}
    if _batch and len(segments) > 3:
        # Stable small groups give ask() independent paid checkpoints. A
        # replacement in k09 must not invalidate the verdicts for k00-k08.
        combined = {}
        keys = sorted(segments)
        for offset in range(0, len(keys), 3):
            combined.update(review_shots(
                {key: segments[key] for key in keys[offset:offset+3]}, shots,
                seconds=seconds, degradations=degradations, client=client,
                brief=brief, cache=cache, ledger=ledger, model_id=model_id,
                grounding_spec=grounding_spec, _batch=False))
        return combined
    if ledger is not None:
        pass  # ask() checks budget after durable response lookup.

    by_clip: dict[str, list[DegradationStep]] = {}
    for step in degradations:
        by_clip.setdefault(step.clip_id, []).append(step)

    instruction = (PROMPTS / "shotreview_zh-TW.txt").read_text(encoding="utf-8")
    if brief.strip():
        instruction += (
            f"\n\n## 剪輯 brief\n\n{brief}\n\n"
            "brief 裡對品質、內容或素材的要求，每一顆都適用。一顆鏡頭做到了"
            "它自己的規劃，卻違反 brief（例如 brief 要求螢幕要亮，這顆的"
            "螢幕是暗的），仍然算沒交出——在 `note` 裡指出是違反 brief 的"
            "哪一條。"
        )
    sent = [
        (f"k{index:02d}", shot)
        for index, shot in enumerate(shots)
        if f"k{index:02d}" in segments
    ]
    body: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    body.extend(_identity_review_parts(grounding_spec, client, cache))
    for clip_id, shot in sent:
        plan = _describe_plan(
            shot, seconds.get(clip_id, 0.0), by_clip.get(clip_id, [])
        )
        body.append({"type": "text", "text": f"\n## {clip_id}\n\n{plan}\n"})
        path = segments[clip_id]
        if cache is None:
            uri = upload_now(path, client).uri
        else:
            uri, _ = cache.uri_for(path, client, mime_type="video/mp4")
        body.append(
            video_content(uri, resolution="low", processing="agentic")
        )

    interaction = ask(
        client,
        upload_cache=cache,
        model=model_id,
        store=False,
        input=body,
        generation_config={
            "thinking_level": "high",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        response_format=structured_json(
            _shot_schema([clip_id for clip_id, _ in sent])
        ),
        ledger=ledger,
        budget_stage="shot_review",
    )
    from montagewright.planner import Usage, _parse

    payload = _parse(interaction, what="shot review")
    return {entry["clip_id"]: entry for entry in payload.get("shots", [])}


def actionable_keys(verdict: ReviewVerdict) -> tuple[str, ...]:
    """What this round is actually asking to change.

    Keyed so two rounds can be compared. A round that asks for the same things
    as the last one has made no progress, and another render will not change
    that -- which is a stopping condition, not a reason to try harder.
    """

    return tuple(
        sorted(
            f"{issue.clip_id or issue.at_seconds}:{issue.issue_type}"
            for issue in verdict.issues
            if issue.severity in {"major", "blocking"}
        )
    )


def should_continue(
    rounds: list[Round], *, ledger: Ledger | None = None, undelivered: int = 0
) -> tuple[bool, str]:
    """Decide whether another round is worth rendering.

    Two reviewers report here and they answer different questions. The shot
    reviewer watches one shot against its own plan; the film reviewer watches
    the whole cut. An approval from the second used to end the loop before
    the first was consulted, so a run finished with three shots that did not
    do what they said they would, forty-five degradations and five seconds
    missing off the length -- and a verdict of "approve (0 issues)".

    That is not the film reviewer being wrong. It is watching a finished cut
    and cannot see a promise it was never told about; a shot that was meant
    to read a wordmark and instead held on half of one looks like a
    considered composition from the outside. The two verdicts are both
    correct and only one of them was being read.
    """

    if not rounds:
        return True, "not started"
    latest = rounds[-1]

    # The limits come first, because everything below them is a reason to go
    # round again and a run has to be able to stop. Ordering the shot
    # reviewer above the round cap would let a shot nobody can fix spend the
    # budget one replan at a time.
    if len(rounds) >= MAX_ROUNDS:
        return False, f"reached the {MAX_ROUNDS}-round cap"
    # The paid dispatcher enforces the cap after checking saved responses.
    # Cached review/replan work must remain possible at zero balance.
    if len(rounds) >= 2:
        previous = set(rounds[-2].actionable)
        current = set(latest.actionable)
        if current and current.issubset(previous):
            # Including the case where they are equal: the same complaint
            # twice means the change did not land, and a third render will
            # not make it land either.
            return False, "no progress between rounds"

    if undelivered:
        return True, f"{undelivered} shots did not do what they planned"
    if latest.verdict.verdict == "approve":
        return False, "approved"
    if not latest.actionable:
        return False, "nothing actionable was raised"
    return True, "continuing"


def adjudicate(
    degradations: list[DegradationStep],
    verdict: ReviewVerdict,
    shot_verdicts: dict[str, dict[str, Any]] | None = None,
) -> list[DegradationStep]:
    """Settle each degradation against whoever actually saw the shot.

    This used to rest entirely on the whole-cut reviewer, who never saw the
    shot in question -- they had a thirty-second film and a line of numbers.
    "The subject is 0.88 of frame wide and can show 36% of itself" is not
    judgeable from that; it is judgeable from the shot. So the shot reviewer
    settles it when there is one, and silence from the whole-cut reviewer
    remains the fallback.

    A degradation the whole-cut reviewer raised still goes back for a replan,
    because the way to answer a bad fallback is a different plan rather than
    a better fallback. Anything left unsettled is labelled as such rather
    than quietly shipped.
    """

    shot_verdicts = shot_verdicts or {}
    touched = {issue.clip_id for issue in verdict.issues if issue.clip_id}
    settled: list[DegradationStep] = []
    for step in degradations:
        seen = shot_verdicts.get(step.clip_id, {}).get("degradation_verdict")
        if seen in {"acceptable", "replan"}:
            settled.append(
                step.model_copy(
                    update={
                        "adjudication": (
                            "accept" if seen == "acceptable" else "replan"
                        ),
                        "adjudication_reason": (
                            "the shot reviewer watched this shot: "
                            + shot_verdicts[step.clip_id].get("note", "")
                        )[:300],
                    }
                )
            )
            continue
        if step.clip_id in touched:
            settled.append(
                step.model_copy(
                    update={
                        "adjudication": "replan",
                        "adjudication_reason": "the reviewer raised this shot",
                    }
                )
            )
        else:
            settled.append(
                step.model_copy(
                    update={
                        "adjudication": "accept",
                        "adjudication_reason": (
                            "the reviewer watched the cut and did not raise it"
                        ),
                    }
                )
            )
    return settled
