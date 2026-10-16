"""VinylFlow — Track Boundary calculation.

Owns the Track Boundary concept end-to-end, independent of ffmpeg.  Three
pure functions over plain data turn audio analysis into Track Boundaries:

- ``parse_silence_log``       ffmpeg silencedetect stderr -> list[Gap]
- ``boundaries_from_gaps``    Gaps + total duration       -> list[Track]
- ``boundaries_from_durations``  Discogs track durations  -> list[Track]
- ``boundaries_for_track_count``  permissive-scan Gaps + a target Track
  count -> list[Track] (Smart Rescan: pick the likeliest separator Gaps)

A **Gap** is a ``(start, end)`` span of detected silence.  A Track occupies
the audio *between* consecutive Gaps.  The final Gap may be **open**
(``end=None``) when the side fades to silence at EOF — an open Gap yields no
trailing Track.

The ffmpeg subprocess sits outside this module, in
``AudioProcessor.detect_silence``; everything here is pure and testable
without a subprocess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Gap:
    """A span of detected silence within a Source Audio file.

    ``end is None`` marks an *open* Gap: the silence runs to end-of-file
    (the side fades out).  Only the final Gap in a sequence may be open.
    """

    start: float
    end: Optional[float] = None


class Track:
    """Represents a detected or split track — i.e. one Track Boundary."""

    def __init__(self, number: int, start: float, end: float):
        """
        Initialize track.

        Args:
            number: Track number (1-indexed)
            start: Start time in seconds
            end: End time in seconds
        """
        self.number = number
        self.start = start
        self.end = end
        self.duration = end - start
        self.vinyl_number = None  # Will be set during mapping (e.g., "A1", "B2")
        self.title = None  # Will be set from Discogs

    def format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    def __repr__(self):
        duration_str = self.format_time(self.duration)
        time_range = f"{self.format_time(self.start)} - {self.format_time(self.end)}"
        vinyl = f" [{self.vinyl_number}]" if self.vinyl_number else ""
        title = f" - {self.title}" if self.title else ""
        return f"Track {self.number}{vinyl}: {time_range} ({duration_str}){title}"


# ffmpeg can report a slightly negative silence_start for leading silence
# (e.g. "silence_start: -0.00768"), so accept a sign and clamp to 0 below.
_SILENCE_START_RE = re.compile(r"silence_start: (-?[\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end: (-?[\d.]+)")


def parse_silence_log(stderr: str) -> List[Gap]:
    """Parse ffmpeg ``silencedetect`` stderr into a list of Gaps.

    Each ``silence_start`` is paired with the next ``silence_end`` in line
    order.  A trailing ``silence_start`` with no matching ``silence_end``
    becomes an open Gap (``end=None``) — the side faded to silence at EOF.
    Lines that aren't silence markers, and any stray ``silence_end`` with no
    pending start, are ignored.
    """
    gaps: List[Gap] = []
    pending_start: Optional[float] = None

    for line in stderr.split("\n"):
        if "silence_start" in line:
            match = _SILENCE_START_RE.search(line)
            if match:
                pending_start = max(0.0, float(match.group(1)))
        elif "silence_end" in line:
            match = _SILENCE_END_RE.search(line)
            if match and pending_start is not None:
                gaps.append(Gap(pending_start, float(match.group(1))))
                pending_start = None

    if pending_start is not None:
        gaps.append(Gap(pending_start, None))

    return gaps


def boundaries_from_gaps(
    gaps: List[Gap], total_duration: float, min_track_length: float
) -> List[Track]:
    """Compute Track Boundaries from the Gaps between audio.

    A Track occupies the audio between consecutive Gaps:
    lead-in before the first Gap, the spans between Gaps, and the lead-out
    after the last Gap.  An open final Gap (``end is None``) yields no
    trailing Track.  Any span shorter than ``min_track_length`` is dropped;
    track numbers stay sequential across dropped spans.
    """
    tracks: List[Track] = []
    num = 1

    # No silence detected: the whole file is one Track if it's long enough.
    if not gaps:
        if total_duration >= min_track_length:
            tracks.append(Track(num, 0.0, total_duration))
        return tracks

    # Lead-in: the Track before the first Gap (skipped if the side opens
    # with silence, i.e. the first Gap starts within the lead-in window).
    first = gaps[0]
    if first.start >= min_track_length:
        tracks.append(Track(num, 0.0, first.start))
        num += 1

    # Between Gaps: a Track runs from one Gap's end to the next Gap's start.
    # Only the final Gap may be open, so gaps[i].end is never None here.
    for i in range(len(gaps) - 1):
        start = gaps[i].end
        end = gaps[i + 1].start
        if end - start >= min_track_length:
            tracks.append(Track(num, start, end))
            num += 1

    # Lead-out: the Track after the last Gap — only if that Gap is closed.
    last = gaps[-1]
    if last.end is not None and total_duration - last.end >= min_track_length:
        tracks.append(Track(num, last.end, total_duration))

    return tracks


def boundaries_for_track_count(
    gaps: List[Gap],
    total_duration: float,
    min_track_length: float,
    target_count: int,
) -> List[Track]:
    """Pick the Gaps most likely to be real inter-track gaps so the result
    has ``target_count`` Tracks (best effort).

    Intended for a *permissive* silence scan whose Gaps are a superset of
    the real inter-track gaps (quiet musical passages show up too).  Real
    vinyl track gaps are the longest silences on a side, so candidates are
    ranked by duration and exactly ``target_count - 1`` separators are
    chosen greedily, skipping any that would leave a span shorter than
    ``min_track_length`` next to an already-chosen Gap.

    Edge Gaps — silence at the very start of the side, silence running to
    (or near) EOF, and open Gaps — are always kept: they trim lead-in and
    lead-out rather than separate two Tracks, so they cost nothing toward
    the separator budget.

    Returns whatever ``boundaries_from_gaps`` yields for the chosen Gaps,
    which may hold fewer than ``target_count`` Tracks when not enough
    feasible separators exist; it never holds more.
    """
    if target_count < 1:
        return []

    def _is_edge(gap: Gap) -> bool:
        if gap.start < min_track_length:
            return True  # leading silence: trims the lead-in
        if gap.end is None or total_duration - gap.end < min_track_length:
            return True  # trailing / open silence: trims the lead-out
        return False

    chosen = [g for g in gaps if _is_edge(g)]
    candidates = [g for g in gaps if not _is_edge(g)]
    # Longest silences first; start time breaks ties deterministically.
    candidates.sort(key=lambda g: (-(g.end - g.start), g.start))

    separators = 0
    for gap in candidates:
        if separators == target_count - 1:
            break
        # The chosen Gap nearest on each side bounds the spans this Gap
        # would create; both spans must be long enough to survive as Tracks.
        left = max((g.end for g in chosen if g.start < gap.start and g.end is not None), default=0.0)
        right = min((g.start for g in chosen if g.start > gap.start), default=total_duration)
        if gap.start - left >= min_track_length and right - gap.end >= min_track_length:
            chosen.append(gap)
            separators += 1

    chosen.sort(key=lambda g: g.start)
    return boundaries_from_gaps(chosen, total_duration, min_track_length)


def boundaries_from_durations(durations: List[float]) -> List[Track]:
    """Lay Tracks end-to-end from Discogs durations (the fallback path).

    Used when silence detection fails: each duration becomes a Track placed
    immediately after the previous one, starting at t=0, numbered
    sequentially.
    """
    tracks: List[Track] = []
    current_time = 0.0

    for num, duration in enumerate(durations, start=1):
        start = current_time
        end = current_time + duration
        tracks.append(Track(num, start, end))
        current_time = end

    return tracks


def refine_rescan_boundaries(
    tracks: List[Track],
    strict_gaps: List[Gap],
    total_duration: float,
    min_track_length: float,
    max_pad: float = 1.0,
) -> List[Track]:
    """Tidy Smart Rescan's Tracks using a strict (quiet-threshold) scan.

    Smart Rescan's permissive thresholds treat a quiet intro or fade-out as
    silence, so the side's first Track starts late and its last ends early.
    A strict scan's leading/trailing silence marks where the music really
    starts and stops; edges only ever move outward to it.  Separators
    between Tracks move up to ``max_pad`` seconds into each gap (at most its
    middle), so soft attacks and decays aren't clipped.
    """
    if not tracks:
        return tracks
    bounds = [[t.start, t.end] for t in tracks]

    # The music starts after the *last* strict silence before the first
    # Track (earlier ones are needle-drop/run-in), and ends at the *first*
    # strict silence after the last Track.
    lead = [g for g in strict_gaps
            if g.end is not None and g.start < min_track_length and g.end <= bounds[0][0]]
    if lead:
        bounds[0][0] = lead[-1].end

    tail = [g for g in strict_gaps
            if g.start >= bounds[-1][1]
            and (g.end is None or total_duration - g.end < min_track_length)]
    if tail:
        bounds[-1][1] = tail[0].start

    for prev, nxt in zip(bounds, bounds[1:]):
        gap = nxt[0] - prev[1]
        if gap > 0:
            pad = min(max_pad, gap / 2)
            prev[1] += pad
            nxt[0] -= pad

    refined = []
    for t, (start, end) in zip(tracks, bounds):
        r = Track(t.number, start, end)
        r.vinyl_number, r.title = t.vinyl_number, t.title
        refined.append(r)
    return refined



def quiet_points(env_db, frame_s: float, start: float, end: float):
    """Candidate track changes from a loudness envelope (dB per frame).

    A candidate is a dip: the quietest point within 3 s either side, at
    least 4 dB below the surrounding 20 s.  True silence makes a deep dip;
    tracks that fade into each other still leave a shallow one.  Returns
    ``(position, cut_in, cut_out, depth_db, level_db)`` tuples, where the cut
    spans the frames within 3 dB of the dip's floor (at most 6 s each way)
    and the position is its middle.
    """
    import numpy as np

    env = np.asarray(env_db, dtype=float)
    if env.size < 3:
        return []
    power = 10 ** (env / 10)

    def mean_db(width):
        k = max(1, int(round(width / frame_s)))
        kernel = np.ones(k) / k
        return 10 * np.log10(np.convolve(power, kernel, mode="same") + 1e-12)

    smooth, context = mean_db(0.5), mean_db(20.0)
    w = max(1, int(round(3.0 / frame_s)))
    padded = np.pad(smooth, w, mode="edge")
    local_min = np.lib.stride_tricks.sliding_window_view(padded, 2 * w + 1).min(axis=1)
    reach = int(round(6.0 / frame_s))

    points, last_out = [], -1.0
    for i in np.flatnonzero((smooth <= local_min) & (context - smooth >= 4.0)):
        t = (i + 0.5) * frame_s
        if not start < t < end or t <= last_out:
            continue
        floor = smooth[i] + 3.0
        lo = i
        while lo > 0 and i - lo < reach and smooth[lo - 1] <= floor:
            lo -= 1
        hi = i
        while hi < smooth.size - 1 and hi - i < reach and smooth[hi + 1] <= floor:
            hi += 1
        cut_in, cut_out = float(lo * frame_s), float((hi + 1) * frame_s)
        points.append(((cut_in + cut_out) / 2, cut_in, cut_out,
                       float(context[i] - smooth[i]), float(smooth[i])))
        last_out = cut_out
    return points


def boundaries_from_duration_guide(
    env_db,
    frame_s: float,
    strict_gaps: List[Gap],
    total_duration: float,
    durations: List[float],
    min_track_length: float,
):
    """Split a side using its Discogs track lengths to pick the track changes.

    The music runs from the end of the strict scan's lead-in silence to the
    start of its lead-out silence.  Between those, one change is chosen per
    track from the strict scan's silences and the envelope's quiet points, by dynamic programming over two
    things: how closely each resulting Track's length matches its Discogs
    duration (scaled so the durations fill the music, and loosely — Discogs
    lengths are often tens of seconds out), and how deep the dip is.  Where
    nothing fits, a cut goes at the expected time.

    Returns ``(tracks, info)``; ``info["separators"]`` holds one entry per
    track change: ``{"time", "kind", "offset"}`` where kind is "gap" (silence
    by the strict scan), "quiet" (a dip, e.g. a fade between tracks — worth checking)
    or "cut" (no dip found), and ``offset`` is seconds from where the
    durations alone put it.
    """
    n = len(durations)
    if n < 1 or any(d <= 0 for d in durations):
        raise ValueError("every track needs a duration")

    start, end = 0.0, total_duration
    lead = [g for g in strict_gaps if g.end is not None and g.start < min_track_length]
    if lead:
        start = lead[-1].end
    tail = [g for g in strict_gaps
            if (g.end is None or total_duration - g.end < min_track_length) and g.start > start]
    if tail:
        end = tail[0].start

    if n == 1:
        return [Track(1, start, end)], {"separators": [], "start": start, "end": end, "cost": 0.0}

    # Durations rarely sum exactly to the music (gaps, Discogs rounding);
    # scale them to fit, unless the span looks wrong (noise run to EOF).
    scale = min(1.08, max(0.92, (end - start) / sum(durations)))
    cum, expected = 0.0, []
    for d in durations[:-1]:
        cum += d
        expected.append(start + cum * scale)

    # Candidates: (position, cut_in, cut_out, bonus, kind).  Real silence
    # (the strict scan's gaps) is cut exactly and scores best, longer more
    # so; envelope dips elsewhere score by depth; a plain cut at the
    # expected time costs a fixed penalty.
    silences = [g for g in strict_gaps if g.end is not None and start < g.start and g.end < end]
    cands = [((g.start + g.end) / 2, g.start, g.end, 4.5 + 0.5 * min(g.end - g.start, 3.0), "gap")
             for g in silences]
    # A dip isn't silence, so it's a single cut point: nothing between the
    # tracks is dropped.
    for pos, _, _, depth, _ in quiet_points(env_db, frame_s, start, end):
        if not any(g.start - 1 <= pos <= g.end + 1 for g in silences):
            cands.append((pos, pos, pos, 0.15 * min(depth, 30.0), "quiet"))
    cands += [(t, t, t, -2.0, "cut") for t in expected]
    cands.sort(key=lambda c: c[0])

    def seg_cost(length, k):
        want = durations[k] * scale
        if length < min(min_track_length, want / 2):
            return None
        sigma = max(5.0, 0.08 * want)
        return min(((length - want) / sigma) ** 2, 9.0)

    INF = float("inf")
    m = len(cands)
    # cost[k][j]: best cost with the k-th change (0-based) at candidate j
    cost = [[INF] * m for _ in range(n - 1)]
    back = [[-1] * m for _ in range(n - 1)]
    for j, c in enumerate(cands):
        sc = seg_cost(c[0] - start, 0)
        if sc is not None:
            cost[0][j] = sc - c[3]
    for k in range(1, n - 1):
        for j, c in enumerate(cands):
            best, arg = INF, -1
            for i in range(j):
                if cost[k - 1][i] == INF:
                    continue
                sc = seg_cost(c[0] - cands[i][0], k)
                if sc is not None and cost[k - 1][i] + sc < best:
                    best, arg = cost[k - 1][i] + sc, i
            if arg >= 0:
                cost[k][j], back[k][j] = best - c[3], arg

    best, arg = INF, -1
    for j, c in enumerate(cands):
        if cost[n - 2][j] == INF:
            continue
        sc = seg_cost(end - c[0], n - 1)
        if sc is not None and cost[n - 2][j] + sc < best:
            best, arg = cost[n - 2][j] + sc, j
    if arg < 0:
        raise ValueError("the durations don't fit this recording")

    picks = []
    for k in range(n - 2, -1, -1):
        picks.append(arg)
        arg = back[k][arg]
    picks.reverse()

    tracks, separators, t0 = [], [], start
    for k, j in enumerate(picks):
        pos, cut_in, cut_out, _, kind = cands[j]
        tracks.append(Track(k + 1, t0, cut_in))
        separators.append({"time": round(pos, 2), "kind": kind,
                           "offset": round(pos - expected[k], 2)})
        t0 = cut_out
    tracks.append(Track(n, t0, end))

    tracks = refine_rescan_boundaries(tracks, [], total_duration, min_track_length)
    return tracks, {"separators": separators, "start": start, "end": end, "cost": best}
