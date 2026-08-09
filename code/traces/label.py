"""
Mechanical trace labelling — reasoning structure without interpreting anyone.
=============================================================================

The identifiability objection says you cannot recover the logic behind a trace,
because many internal processes emit the same keystrokes. That objection is
correct and this module does not argue with it. It sidesteps it.

The move: never ask what the person was thinking. Ask only questions the trace
answers by itself.

    Did this span survive into the final answer?
    Was it written, deleted, and never returned to?
    How long did they pause before writing it?

Those are mechanical facts about the document, not attributions of intent. No
annotator, no interpreting model, nothing to confabulate. And they are enough
to define a supervised task with automatic labels:

    Given steps 1..k, will the content of step k+1 survive to the solution?

A model that predicts that has learned something about search -- when a branch
is going nowhere -- and it learned it from labels no one assigned by hand.
That is a process reward signal derived from human problem-solving, with the
annotation bottleneck removed. It is also why this scales where Newell &
Simon's protocol analysis did not: they hand-built a symbolic model per
protocol; nothing here is hand-built.

What this does NOT claim: that surviving spans are "correct reasoning" or that
abandoned ones are "errors". A student may delete a correct line for a better
one. The label is survival, not virtue, and the paper must say so.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, asdict, field


# A pause longer than this ends a step. 1.2 s is above typing rhythm and below
# deliberate thought; it is a parameter and the sweep in main() shows what it
# costs to move it.
DEFAULT_PAUSE_MS = 1200

# A span counts as surviving if this much of it is still in the final text.
SURVIVAL_THRESHOLD = 0.80


@dataclass
class Event:
    """One document state, timestamped from the first keystroke."""
    t: int
    text: str


@dataclass
class Step:
    index: int
    t_start: int
    t_end: int
    pause_before_ms: int          # thinking time -- the only cognitive proxy here
    added: str                    # text this step introduced
    removed: str                  # text this step deleted
    text_after: str = field(repr=False, default='')
    survives: bool | None = None  # filled by label_survival
    deleted_later: bool = False
    returned_to: bool = False     # deleted then substantially re-derived


def segment(events: list[Event], pause_ms: int = DEFAULT_PAUSE_MS) -> list[Step]:
    """Split a keystroke stream into steps at pauses.

    Pause length is the one signal here that plausibly tracks cognition, and
    it is observable rather than inferred: the clock does not need to know why
    the person stopped.
    """
    if not events:
        return []

    steps: list[Step] = []
    burst_start = 0
    prev_end_text = ''

    for i, ev in enumerate(events):
        is_last = i == len(events) - 1
        gap_next = (events[i + 1].t - ev.t) if not is_last else None
        if is_last or (gap_next is not None and gap_next >= pause_ms):
            start_ev = events[burst_start]
            pause_before = (start_ev.t - events[burst_start - 1].t) if burst_start > 0 else 0
            added, removed = _diff(prev_end_text, ev.text)
            steps.append(Step(
                index=len(steps),
                t_start=start_ev.t,
                t_end=ev.t,
                pause_before_ms=pause_before,
                added=added,
                removed=removed,
                text_after=ev.text,
            ))
            prev_end_text = ev.text
            burst_start = i + 1

    return steps


def _diff(before: str, after: str) -> tuple[str, str]:
    """Text introduced and text removed between two document states."""
    sm = difflib.SequenceMatcher(None, before, after, autojunk=False)
    added, removed = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ('insert', 'replace'):
            added.append(after[j1:j2])
        if tag in ('delete', 'replace'):
            removed.append(before[i1:i2])
    return ''.join(added), ''.join(removed)


def _norm_lines(text: str) -> list[str]:
    """Whitespace-normalised non-empty lines."""
    return [' '.join(ln.split()) for ln in text.splitlines() if ln.strip()]


def _contained(needle: str, haystack: str, threshold: float) -> bool:
    """Did the content of `needle` reach `haystack`?

    Compared line by line, and a line must match exactly after whitespace
    normalisation. Character-level fuzzy matching is actively wrong here: in
    mathematical text "x + 6 = 10" and "2x + 6 = 20" overlap in 80% of their
    characters while being different claims, so a similarity threshold scores
    an abandoned branch as surviving. Exact line identity has no such failure
    mode. It is strict -- a line that survives with a small edit counts as not
    surviving -- and `threshold` sets what fraction of a span's lines must
    match for the span to count.
    """
    lines = _norm_lines(needle)
    if not lines:
        return False
    final = set(_norm_lines(haystack))
    hits = sum(1 for ln in lines if ln in final)
    return (hits / len(lines)) >= threshold


def label_survival(steps: list[Step], final_text: str,
                   threshold: float = SURVIVAL_THRESHOLD) -> list[Step]:
    """Mark each step by whether what it wrote reached the final answer.

    This is the label the predictor is trained against, and nothing about it
    requires knowing why anything was written.
    """
    for k, s in enumerate(steps):
        if not s.added.strip():
            s.survives = None                 # pure deletion: no content to survive
        else:
            s.survives = _contained(s.added, final_text, threshold)
            s.deleted_later = not s.survives
            if not s.survives:
                # Deleted, then substantially rewritten later: a returned-to
                # branch, which is a different phenomenon from an abandoned one.
                s.returned_to = any(
                    _contained(s.added, later.added, threshold)
                    for later in steps[k + 1:] if later.added.strip()
                )
    return steps


def features(steps: list[Step]) -> list[dict]:
    """Per-step features available *before* the step's outcome is known.

    Deliberately excludes anything derived from the final text -- a predictor
    given that would be reading the answer key.
    """
    out = []
    for k, s in enumerate(steps):
        prior = steps[:k]
        out.append(dict(
            index=s.index,
            pause_before_ms=s.pause_before_ms,
            duration_ms=s.t_end - s.t_start,
            n_added=len(s.added),
            n_removed=len(s.removed),
            is_deletion_heavy=len(s.removed) > len(s.added),
            frac_through=k / max(1, len(steps) - 1),
            cum_deletions=sum(len(p.removed) for p in prior),
            prior_steps=k,
            label_survives=s.survives,
        ))
    return out


def summarise(steps: list[Step]) -> dict:
    scored = [s for s in steps if s.survives is not None]
    surv = [s for s in scored if s.survives]
    aband = [s for s in scored if not s.survives and not s.returned_to]
    ret = [s for s in scored if s.returned_to]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float('nan')

    return dict(
        n_steps=len(steps),
        n_scored=len(scored),
        n_surviving=len(surv),
        n_abandoned=len(aband),
        n_returned_to=len(ret),
        abandon_rate=len(aband) / len(scored) if scored else float('nan'),
        mean_pause_surviving_ms=mean([s.pause_before_ms for s in surv]),
        mean_pause_abandoned_ms=mean([s.pause_before_ms for s in aband]),
    )


def load(path: str) -> tuple[list[Event], str, dict]:
    with open(path) as fh:
        blob = json.load(fh)
    events = [Event(int(e['t']), e['text']) for e in blob['events']]
    final = blob.get('final_text', events[-1].text if events else '')
    meta = {k: v for k, v in blob.items() if k not in ('events', 'final_text')}
    return events, final, meta


def process(path: str, pause_ms: int = DEFAULT_PAUSE_MS) -> dict:
    events, final, meta = load(path)
    steps = label_survival(segment(events, pause_ms), final)
    return dict(meta=meta, summary=summarise(steps),
                steps=[asdict(s) for s in steps], features=features(steps))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trace', help='trace JSON from the capture page')
    ap.add_argument('--pause-ms', type=int, default=DEFAULT_PAUSE_MS)
    ap.add_argument('--sweep', action='store_true',
                    help='show how the segmentation moves with the pause threshold')
    ap.add_argument('--out', help='write the labelled trace here')
    args = ap.parse_args()

    if args.sweep:
        print(f'{"pause_ms":>9}{"steps":>7}{"scored":>8}{"abandon_rate":>14}')
        for p in (400, 800, 1200, 2000, 3000, 5000):
            r = process(args.trace, p)['summary']
            print(f'{p:>9}{r["n_steps"]:>7}{r["n_scored"]:>8}'
                  f'{r["abandon_rate"]:>14.3f}')
        return

    result = process(args.trace, args.pause_ms)
    s = result['summary']
    print(f"steps {s['n_steps']}  scored {s['n_scored']}  "
          f"surviving {s['n_surviving']}  abandoned {s['n_abandoned']}  "
          f"returned-to {s['n_returned_to']}")
    print(f"abandon rate {s['abandon_rate']:.3f}")
    print(f"mean pause before surviving  {s['mean_pause_surviving_ms']:.0f} ms")
    print(f"mean pause before abandoned  {s['mean_pause_abandoned_ms']:.0f} ms")
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(result, fh, indent=2)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
