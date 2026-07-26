from __future__ import annotations

import argparse
import json
from pathlib import Path

from jascue_video_lab.final_delivery import assemble_music_only_delivery


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mux one already-approved continuous soundtrack with an editorial "
            "picture cut. Refuses mismatched durations instead of hard-cutting."
        )
    )
    parser.add_argument("picture", type=Path)
    parser.add_argument("music", type=Path)
    parser.add_argument(
        "--music-assembly-artifacts",
        type=Path,
        required=True,
        help=(
            "Directory containing the immutable MusicAssembly plan, binding, "
            "and render manifest for the supplied music file."
        ),
    )
    parser.add_argument("--aspect", choices=("16:9", "9:16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--duration-tolerance-ms", type=int, default=100)
    args = parser.parse_args()
    result = assemble_music_only_delivery(
        picture_path=args.picture,
        music_path=args.music,
        output_path=args.output,
        manifest_path=args.manifest,
        music_assembly_artifact_dir=args.music_assembly_artifacts,
        aspect_ratio=args.aspect,
        duration_tolerance_ms=args.duration_tolerance_ms,
    )
    print(
        json.dumps(
            {
                "output": str(result.output_path),
                "manifest": str(result.manifest_path),
                "duration_ms": result.manifest["output"]["duration_ms"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
