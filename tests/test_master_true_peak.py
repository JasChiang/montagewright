import re
import subprocess

from montagewright import renderer


def test_a_hot_music_bed_is_delivered_under_the_true_peak_ceiling(tmp_path):
    picture = tmp_path / "picture.mp4"
    music = tmp_path / "music.mp3"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "color=c=black:s=320x180:r=30:d=8", "-c:v", "libx264",
         str(picture)],
        check=True,
    )
    # Square-ish waves at full scale: dense, clipped material is where the
    # 192 kHz loudnorm output and AAC overshoot pushed masters past 0 dBFS.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "aevalsrc='0.98*sgn(sin(2*PI*110*t))+0.02*sin(2*PI*3000*t)':"
               "s=44100:d=12",
         "-c:a", "libmp3lame", "-b:a", "192k", str(music)],
        check=True,
    )
    out = tmp_path / "deliverable.mp4"
    renderer._mux_music(picture, music, out, video_encoder="libx264")
    measured = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(out),
         "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    ).stderr
    peak = float(re.findall(r"Peak:\s*(-?[0-9.]+) dBFS", measured)[-1])
    assert peak <= -1.0, peak
    rate = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert rate == "48000"
