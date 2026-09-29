"""Visible completion estimates and protected downstream inference reserves."""
from pathlib import Path
from montagewright.checkpoints import write_json
from montagewright.cost import pricing_for, BudgetSpent


def prepare(ledger, proxies, library: Path, output: Path, *, mode, subtitles, review, speech="auto"):
    from montagewright.clipcard import load_card, clip_seconds
    from montagewright.uploads import content_hash
    from montagewright.planner import MAX_OUTPUT_TOKENS
    rates = pricing_for(ledger.model_id)
    missing = [p for p in proxies.values()
               if load_card(library / "cards" / (content_hash(p)[:20]+".json")) is None]
    total = sum(clip_seconds(p) for p in proxies.values())
    fresh_seconds = sum(clip_seconds(p) for p in missing)
    # Planning estimates, not provider guarantees: actual video sampling,
    # thinking and agentic tool use vary. Never advertise exact completion cost.
    planning = (min(total, 1800)*350*rates["input"] + MAX_OUTPUT_TOKENS*rates["output"])/1e6
    reserves = {"editorial_plan": planning}
    if mode == "edit" and review:
        reserves["review"] = (90*350*rates["input"] + MAX_OUTPUT_TOKENS*rates["output"])/1e6
    if mode == "edit" and subtitles != "none":
        reserves["subtitle_layout"] = 16384*rates["output"]/1e6
    if mode == "edit" and review:
        reserves["editor_tools"] = (3*180*4*350*rates["input"] + 3*8192*rates["output"])/1e6
    speech_seconds = 0.0
    speech_sources = 0
    if speech != "never":
        import json
        import subprocess
        from montagewright.transcript import load as load_transcript
        for proxy in proxies.values():
            digest = content_hash(proxy)[:20]
            if load_transcript(library / "transcripts" / (digest+".json")):
                continue
            card = load_card(library / "cards" / (digest+".json"))
            if card and (card.get("speech") == "none" or
                         (card.get("speech") != "content" and subtitles == "none")):
                continue
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                                    "-show_entries", "stream=index", "-of", "json", str(proxy)],
                                   capture_output=True, text=True, check=True)
            if json.loads(probe.stdout).get("streams"):
                speech_seconds += clip_seconds(proxy)
                speech_sources += 1
        if speech_sources:
            reserves["transcript"] = (2*speech_seconds*350*rates["input"] +
                max(speech_sources*4096, speech_seconds*80)*rates["output"])/1e6
    for stage in list(reserves):
        if (output / "work" / f"{stage}.json").exists():
            reserves.pop(stage)
    # One saved call cannot prove that a multi-call stage is finished. Exact
    # request replay releases its own reserve at ask(), not by stage-name scan.
    ledger.completion_reserve = reserves
    expected_cards = (fresh_seconds*350*rates["input"] + len(missing)*4096*rates["output"])/1e6
    payload = {"version": 1, "target_usd": ledger.target_usd,
               "authorized_total_usd": ledger.cap_usd,
               "spent_usd": ledger.prior_spend_usd+ledger.spent_usd,
               "uncached_sources": len(missing), "source_seconds": total,
               "estimated_speech_seconds": speech_seconds,
               "estimated_cards_usd": expected_cards, "protected_stages_usd": reserves,
               "estimated_base_usd": expected_cards+sum(reserves.values()),
               "excludes": ["identity escalation", "additional repair rounds beyond the first tool budget"],
               "notice": "Estimates are uncertain; dynamic tool use and unknown failed-request billing can exceed local accounting."}
    write_json(output / "work" / "budget-plan.json", payload)
    if expected_cards+sum(reserves.values()) > ledger.remaining_usd and missing:
        raise BudgetSpent("estimated base workflow exceeds remaining authorization before new material calls; see work/budget-plan.json")
    print(f"completion budget: {len(missing)} uncached sources; ${sum(reserves.values()):.3f} protected for planning/delivery", flush=True)
