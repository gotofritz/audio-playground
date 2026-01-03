#!/usr/bin/env python3
"""Check segment durations to find where padding occurs."""
import sys
from pathlib import Path


def check_durations(tmp_dir: str) -> None:
    import torchaudio

    tmp_path = Path(tmp_dir)

    # Check original segments (before model processing)
    print("=== Original Segments (before model) ===")
    segment_files = sorted(tmp_path.glob("segment-[0-9]*.wav"))
    segment_files = [
        f
        for f in segment_files
        if "-target" not in f.name and "-residual" not in f.name
    ]

    total_original = 0.0
    for f in segment_files:
        audio, sr = torchaudio.load(f)
        duration = audio.shape[1] / sr
        total_original += duration
        print(f"{f.name}: {duration:.6f}s ({audio.shape[1]} samples @ {sr} Hz)")

    print(f"\nTotal original segments: {total_original:.6f}s")

    # Check target segments (after model processing)
    print("\n=== Target Segments (after model) ===")
    target_files = sorted(tmp_path.glob("segment-*-target-*.wav"))

    if target_files:
        total_target = 0.0
        prompt = target_files[0].stem.split("-target-")[1]
        prompt_files = [f for f in target_files if f"-target-{prompt}.wav" in f.name]

        for f in sorted(prompt_files):
            audio, sr = torchaudio.load(f)
            duration = audio.shape[1] / sr
            total_target += duration
            print(f"{f.name}: {duration:.6f}s ({audio.shape[1]} samples @ {sr} Hz)")

        print(f"\nTotal target segments: {total_target:.6f}s")
        print(f"Difference: {total_target - total_original:.6f}s")

        # Show padding per segment
        if len(segment_files) == len(prompt_files):
            print("\n=== Per-Segment Padding ===")
            for orig_f, target_f in zip(sorted(segment_files), sorted(prompt_files)):
                orig_audio, orig_sr = torchaudio.load(orig_f)
                target_audio, target_sr = torchaudio.load(target_f)
                orig_dur = orig_audio.shape[1] / orig_sr
                target_dur = target_audio.shape[1] / target_sr
                padding = target_dur - orig_dur
                print(
                    f"{orig_f.name} -> {target_f.name}: +{padding:.6f}s "
                    f"({orig_audio.shape[1]} -> {target_audio.shape[1]} samples)"
                )
    else:
        print("No target files found")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_durations.py <tmp_dir>")
        sys.exit(1)
    check_durations(sys.argv[1])
