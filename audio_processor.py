"""
VinylFlow - Audio Processing Module

Handles silence detection, track splitting, and audio format conversion.
Uses FFmpeg for audio analysis and format conversion.
Supports FLAC, MP3, and AIFF output formats.
"""

import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

# Track Boundary calculation lives in its own pure module.  Re-export Track
# so existing callers that do ``from audio_processor import Track`` keep
# working — the dependency arrow stays one-way (audio_processor ->
# track_boundaries, no import cycle).
from track_boundaries import (
    Track,
    boundaries_for_track_count,
    boundaries_from_duration_guide,
    boundaries_from_durations,
    boundaries_from_gaps,
    parse_silence_log,
    refine_rescan_boundaries,
)


def _ffmpeg() -> str:
    """Return the ffmpeg executable to use.

    In a bundled (PyInstaller) build the launcher sets VINYLFLOW_FFMPEG_PATH
    to the absolute path of the bundled binary.  Fall back to bare 'ffmpeg'
    so plain source-code runs still work.
    """
    return os.environ.get("VINYLFLOW_FFMPEG_PATH") or "ffmpeg"


def run_ffmpeg(args: list, *, text: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run ffmpeg with the bundled binary path + correct text-mode defaults.

    All ffmpeg invocations across the project must go through this seam,
    not call ``subprocess.run(["ffmpeg", ...])`` directly — bundled
    builds depend on ``VINYLFLOW_FFMPEG_PATH`` and Windows requires
    utf-8/replace decoding so non-ASCII filenames don't crash the
    subprocess module.

    ``text=True`` (default) captures stdout/stderr as utf-8 strings with
    error replacement — correct for parsing ffmpeg's stderr (silence
    detection, duration probing).  Pass ``text=False`` when you need
    raw bytes (e.g. piping PCM through stdout for waveform peaks).

    All other ``subprocess.run`` kwargs flow through unchanged.
    """
    cmd = [_ffmpeg(), *args]
    if text:
        kwargs.setdefault("encoding", "utf-8")
        kwargs.setdefault("errors", "replace")
    return subprocess.run(cmd, **kwargs)


def parse_loudnorm_json(ffmpeg_stderr: str) -> Optional[dict]:
    """The JSON block ``loudnorm=print_format=json`` prints at the end of stderr."""
    text = ffmpeg_stderr or ""
    start, end = text.rfind("{"), text.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None


_SAMPLE_RATE_RE = re.compile(r"Audio:.*?(\d+) Hz")


def parse_sample_rate(ffmpeg_stderr: str) -> Optional[int]:
    """Sample rate of the first audio stream from ffmpeg's stderr header, or None."""
    match = _SAMPLE_RATE_RE.search(ffmpeg_stderr or "")
    return int(match.group(1)) if match else None


_DURATION_HEADER_RE = re.compile(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})")


def parse_duration_header(ffmpeg_stderr: str) -> Optional[float]:
    """Parse total duration in seconds from ffmpeg's stderr header, or None."""
    match = _DURATION_HEADER_RE.search(ffmpeg_stderr or "")
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


# Supported input formats
SUPPORTED_INPUT_EXTENSIONS = {".wav", ".aiff", ".aif"}

# Output format configurations: codec flags for FFmpeg + file extension
OUTPUT_FORMATS = {
    "flac": {
        "extension": ".flac",
        "codec_args": ["-c:a", "flac"],
        "label": "FLAC (Lossless)",
    },
    "mp3": {
        "extension": ".mp3",
        "codec_args": ["-c:a", "libmp3lame", "-b:a", "320k"],
        "label": "MP3 (320kbps)",
    },
    "aiff": {
        "extension": ".aiff",
        "codec_args": ["-c:a", "pcm_s16be"],
        "label": "AIFF (Lossless)",
    },
}


class AudioProcessor:
    """Handles audio processing operations."""

    def __init__(
        self,
        silence_threshold=-40,
        min_silence_duration=1.5,
        min_track_length=30,
        flac_compression=8,
    ):
        """
        Initialize audio processor.

        Args:
            silence_threshold: Silence threshold in dB (negative)
            min_silence_duration: Minimum silence duration in seconds
            min_track_length: Minimum track length in seconds
            flac_compression: FLAC compression level (0-8)
        """
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.min_track_length = min_track_length
        self.flac_compression = flac_compression

    def get_audio_duration(self, file_path: Path) -> Optional[float]:
        """
        Get total duration of audio file in seconds.

        Args:
            file_path: Path to audio file

        Returns:
            Duration in seconds, or None if error
        """
        try:
            # No output target: ffmpeg prints the Duration header and exits
            # nonzero without decoding the file, so this is O(1) in file size.
            result = run_ffmpeg(
                ["-i", str(file_path)],
                capture_output=True,
                timeout=30,
            )

            return parse_duration_header(result.stderr)
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return None

    def get_sample_rate(self, file_path: Path) -> Optional[int]:
        """Sample rate of ``file_path`` in Hz, or None if it can't be read."""
        try:
            result = run_ffmpeg(["-i", str(file_path)], capture_output=True, timeout=30)
            return parse_sample_rate(result.stderr)
        except Exception as e:
            print(f"Error getting sample rate: {e}")
            return None

    def detect_silence(self, file_path: Path, verbose=False) -> List[Track]:
        """
        Detect silence in audio file and return track boundaries.

        Args:
            file_path: Path to audio file
            verbose: Print detailed output

        Returns:
            List of Track objects
        """
        if verbose:
            print(f"Detecting silence in: {file_path.name}")
            print(
                f"Threshold: {self.silence_threshold}dB, Min duration: {self.min_silence_duration}s"
            )

        try:
            gaps, total_duration = self._scan_gaps(
                file_path, self.silence_threshold, self.min_silence_duration
            )

            tracks = boundaries_from_gaps(gaps, total_duration, self.min_track_length)

            if verbose:
                print(f"\nDetected {len(tracks)} tracks:")
                for track in tracks:
                    print(f"  {track}")

            return tracks

        except subprocess.TimeoutExpired:
            raise RuntimeError("Silence detection timed out (>5 minutes)")
        except Exception as e:
            raise RuntimeError(f"Silence detection failed: {e}")

    def _scan_gaps(self, file_path: Path, threshold: float, min_duration: float):
        """Run one ffmpeg silencedetect pass and return (gaps, total_duration).

        Owns the ffmpeg subprocess; parsing Gaps and computing Track
        Boundaries stay in the pure ``track_boundaries`` module.
        """
        args = [
            "-i",
            str(file_path),
            "-af",
            f"silencedetect=noise={threshold}dB:duration={min_duration}",
            "-f",
            "null",
            "-",
        ]

        result = run_ffmpeg(args, capture_output=True, timeout=300)

        gaps = parse_silence_log(result.stderr)

        # The silencedetect run's stderr already contains the Duration
        # header, so avoid a second ffmpeg probe of the same file.
        total_duration = parse_duration_header(result.stderr)
        if total_duration is None:
            total_duration = self.get_audio_duration(file_path)
        if total_duration is None:
            raise ValueError("Could not determine audio duration")

        return gaps, total_duration

    # Smart Rescan scan ladder.  Each pass is a full-file ffmpeg decode, so
    # the first pass is permissive enough (-30dB, 0.3s) to surface nearly
    # every real inter-track gap as a candidate in one decode — picking the
    # right ones happens in pure Python (boundaries_for_track_count), not in
    # more ffmpeg runs.  Later passes escalate for very noisy records whose
    # surface noise sits above the previous threshold.
    SMART_RESCAN_LADDER = [(-30.0, 0.3), (-25.0, 0.25), (-20.0, 0.2)]

    def detect_tracks_for_count(self, file_path: Path, target_count: int):
        """Find the split that yields exactly ``target_count`` tracks.

        Args:
            file_path: Path to audio file
            target_count: How many tracks the recording should contain

        Returns:
            (tracks, scan_info) where scan_info holds the settings of the
            pass that produced the result and ``matched`` (True when the
            track count equals ``target_count``).  When no pass matches
            exactly, the closest result across all passes is returned.
        """
        best_tracks = None
        best_info = None

        try:
            for threshold, min_duration in self.SMART_RESCAN_LADDER:
                gaps, total_duration = self._scan_gaps(file_path, threshold, min_duration)
                tracks = boundaries_for_track_count(
                    gaps, total_duration, self.min_track_length, target_count
                )
                info = {
                    "silence_threshold": threshold,
                    "min_silence_duration": min_duration,
                    "matched": len(tracks) == target_count,
                }
                if info["matched"]:
                    return self._refine_rescan(file_path, tracks, total_duration), info
                if best_tracks is None or abs(len(tracks) - target_count) < abs(
                    len(best_tracks) - target_count
                ):
                    best_tracks, best_info = tracks, info

            if best_tracks:
                best_tracks = self._refine_rescan(file_path, best_tracks, total_duration)
            return best_tracks, best_info

        except subprocess.TimeoutExpired:
            raise RuntimeError("Silence detection timed out (>5 minutes)")
        except Exception as e:
            raise RuntimeError(f"Smart rescan failed: {e}")

    def _refine_rescan(self, file_path: Path, tracks, total_duration):
        """Put Smart Rescan's side edges where a normal (strict) scan finds the music."""
        # Quiet threshold, but short silences count: the run-in groove before
        # the first note can be well under the usual minimum gap length
        strict_gaps, _ = self._scan_gaps(file_path, self.silence_threshold, min(self.min_silence_duration, 0.5))
        return refine_rescan_boundaries(tracks, strict_gaps, total_duration, self.min_track_length)

    ENVELOPE_FRAME = 0.1  # seconds per loudness frame

    def loudness_envelope(self, file_path: Path):
        """Mono loudness in dB per ``ENVELOPE_FRAME`` seconds (one ffmpeg decode)."""
        import numpy as np

        rate = 8000
        result = run_ffmpeg(
            ["-i", str(file_path), "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"],
            text=False, capture_output=True, timeout=300,
        )
        samples = np.frombuffer(result.stdout, dtype="<i2").astype(np.float64) / 32768
        hop = int(rate * self.ENVELOPE_FRAME)
        frames = samples[: samples.size // hop * hop].reshape(-1, hop)
        rms = np.sqrt((frames ** 2).mean(axis=1))
        return 20 * np.log10(np.maximum(rms, 10 ** (-90 / 20)))

    def detect_tracks_from_durations(self, file_path: Path, durations: List[float]):
        """Split using the side's Discogs durations to choose the track changes.

        Returns ``(tracks, info)`` like ``detect_tracks_for_count``.
        """
        try:
            strict_gaps, total_duration = self._scan_gaps(
                file_path, self.silence_threshold, min(self.min_silence_duration, 0.5)
            )
            env = self.loudness_envelope(file_path)
            tracks, info = boundaries_from_duration_guide(
                env, self.ENVELOPE_FRAME, strict_gaps, total_duration,
                durations, self.min_track_length,
            )
            info.update(matched=len(tracks) == len(durations), guided=True)
            return tracks, info

        except subprocess.TimeoutExpired:
            raise RuntimeError("Silence detection timed out (>5 minutes)")
        except Exception as e:
            raise RuntimeError(f"Duration-guided split failed: {e}")

    def split_tracks_duration_based(
        self, file_path: Path, durations: List[float], verbose=False
    ) -> List[Track]:
        """
        Create track splits based on provided durations (for when silence detection fails).

        Args:
            file_path: Path to audio file (unused; kept for caller compatibility)
            durations: List of track durations from Discogs
            verbose: Print detailed output

        Returns:
            List of Track objects
        """
        tracks = boundaries_from_durations(durations)

        if verbose:
            print(f"\nCreated {len(tracks)} duration-based tracks:")
            for track in tracks:
                print(f"  {track}")

        return tracks

    def _build_restoration_filters(self, restoration_level: int, hum_freq: int = 0) -> Optional[str]:
        """Build an ffmpeg -af filter chain for audio restoration.

        Args:
            restoration_level: 0=disabled, 1=enabled
            hum_freq: Optional hum notch frequency in Hz (0=off, 50=EU, 60=US)

        Returns:
            Filter chain string, or None if restoration is disabled
        """
        if restoration_level != 1:
            return None

        # highpass=f=15: removes inaudible turntable motor rumble (below 15 Hz) without touching bass
        # adeclick=t=3:b=3: conservative click/pop removal (threshold 3 vs default 2, less likely to catch drum transients)
        # Normalisation is a separate, linear gain stage (see _normalise_gain):
        # single-pass loudnorm compressed dynamics and lifted surface noise.
        filters = ["highpass=f=15", "adeclick=t=3:b=3"]
        return ", ".join(filters)

    # Linear normalisation targets: loudness goal, but never past this true peak.
    NORMALISE_TARGET_LUFS = -14.0
    NORMALISE_CEILING_DBTP = -1.0

    def _normalise_gain(self, input_file: Path, track: Track, af_chain: str) -> Optional[float]:
        """dB of plain gain to apply after ``af_chain``: towards the loudness
        target, capped so true peak stays under the ceiling.  One measuring
        pass; None if it can't be measured (the track is then left as is)."""
        result = run_ffmpeg(
            ["-ss", str(track.start), "-t", str(track.duration), "-i", str(input_file),
             "-af", f"{af_chain}, loudnorm=print_format=json", "-f", "null", "-"],
            capture_output=True, timeout=600,
        )
        stats = parse_loudnorm_json(result.stderr)
        if not stats:
            return None
        try:
            measured_i, measured_tp = float(stats["input_i"]), float(stats["input_tp"])
        except (KeyError, ValueError):
            return None
        if not (math.isfinite(measured_i) and math.isfinite(measured_tp)):
            return None  # silent track ("-inf"): leave it alone
        return min(self.NORMALISE_TARGET_LUFS - measured_i, self.NORMALISE_CEILING_DBTP - measured_tp)

    def extract_track(
        self,
        input_file: Path,
        track: Track,
        output_file: Path,
        output_format: str = "flac",
        verbose: bool = False,
        restoration_level: int = 0,
        hum_freq: int = 50,
    ) -> bool:
        """
        Extract a single track and convert to the specified format.

        Args:
            input_file: Source audio file
            track: Track object with start/end times
            output_file: Output file path
            output_format: One of 'flac', 'mp3', 'aiff'
            verbose: Print detailed output
            restoration_level: 0=none, 1=light clean, 2=full restore
            hum_freq: Electrical hum frequency in Hz (50 or 60) for full restore

        Returns:
            True if successful
        """
        if verbose:
            print(f"Extracting {track.vinyl_number or f'Track {track.number}'}: {output_file.name}")

        format_config = OUTPUT_FORMATS.get(output_format, OUTPUT_FORMATS["flac"])

        # -ss/-t before -i: input-side seek, so ffmpeg jumps straight to the
        # track instead of decoding everything before it. Sample-accurate for
        # PCM WAV/AIFF input.
        args = [
            "-ss",
            str(track.start),
            "-t",
            str(track.duration),
            "-i",
            str(input_file),
        ]

        # Add codec-specific args
        args.extend(format_config["codec_args"])

        # Add FLAC compression level if applicable
        if output_format == "flac":
            args.extend(["-compression_level", str(self.flac_compression)])

        # Add audio restoration filter chain if requested
        af_chain = self._build_restoration_filters(restoration_level, hum_freq)
        if af_chain:
            gain = self._normalise_gain(input_file, track, af_chain)
            if gain is not None:
                af_chain += f", volume={gain:.2f}dB"
            args.extend(["-af", af_chain])
            # Keep the source sample rate whatever the filters do
            source_rate = self.get_sample_rate(input_file)
            if source_rate:
                args.extend(["-ar", str(source_rate)])

        args.extend([
            "-y",  # Overwrite output file
            str(output_file),
        ])

        try:
            result = run_ffmpeg(args, capture_output=True, timeout=600)

            if result.returncode != 0:
                print(f"Error: ffmpeg failed: {result.stderr}")
                return False

            # Verify output file exists and has reasonable size
            if not output_file.exists():
                print(f"Error: Output file not created: {output_file}")
                return False

            if output_file.stat().st_size < 1000:
                print(f"Error: Output file suspiciously small: {output_file}")
                return False

            # Verify duration is close to expected
            actual_duration = self.get_audio_duration(output_file)
            if actual_duration is None:
                print(f"Warning: Could not verify duration of {output_file}")
            elif abs(actual_duration - track.duration) > 2.0:
                print(
                    f"Warning: Duration mismatch for {output_file}: "
                    f"expected {track.duration:.1f}s, got {actual_duration:.1f}s"
                )

            return True

        except subprocess.TimeoutExpired:
            print(f"Error: Track extraction timed out")
            return False
        except Exception as e:
            print(f"Error extracting track: {e}")
            return False

    def validate_audio_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        Validate that file is a valid audio file.

        Args:
            file_path: Path to file

        Returns:
            (is_valid, message)
        """
        if not file_path.exists():
            return False, f"File not found: {file_path}"

        if not file_path.is_file():
            return False, f"Not a file: {file_path}"

        if file_path.stat().st_size == 0:
            return False, f"File is empty: {file_path}"

        # Check extension
        if file_path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
            return False, (
                f"Unsupported format: {file_path.suffix}. "
                f"Supported: {', '.join(SUPPORTED_INPUT_EXTENSIONS)}"
            )

        # Try to get duration (validates it's a readable audio file)
        duration = self.get_audio_duration(file_path)
        if duration is None:
            return False, f"Not a valid audio file: {file_path}"

        if duration < 60:
            return False, f"Audio file too short ({duration:.1f}s): {file_path}"

        return True, f"Valid audio file ({duration / 60:.1f} minutes)"
