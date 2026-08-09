# Trace capture

Collecting human problem-solving traces, and labelling them without
interpreting anyone.

## The problem this works around

You cannot read reasoning off a trace. Many internal processes emit identical
keystrokes, so the logic is underdetermined by the behaviour — the same wall
that stopped connectomics from yielding algorithms. Asking a model to
"interpret" the trace does not fix it: a fluent attribution and a correct one
are indistinguishable, and the interpreter's own priors leak into what it
claims to have found.

## What this does instead

Never ask what the person was thinking. Ask only what the document answers:

- did this span reach the final answer?
- was it written, deleted, and never returned to?
- how long was the pause before it?

Those are mechanical, and they define a supervised task with **automatic
labels**: given steps 1..k, predict whether step k+1 survives. A model that
does that has learned when a branch is going nowhere — process supervision
from human search, with the annotation bottleneck removed.

That last part is why this can scale where Newell & Simon's protocol analysis
could not. They hand-built a symbolic model per protocol. Nothing here is
hand-built.

## Use

1. Open `capture.html`, solve a problem, hit **Download trace**.
2. `python3 label.py trace.json` — summary
3. `python3 label.py trace.json --sweep` — segmentation sensitivity
4. `python3 label.py trace.json --out labelled.json` — full output + features

Capture is offline; the file never leaves the machine.

## Two design decisions worth knowing

**Full document snapshots, not keystrokes.** Diffing is the labeller's job,
and full states survive paste, undo, and multi-cursor edits that a keystroke
log misrepresents.

**Line-exact survival matching, not fuzzy.** Character similarity is actively
wrong on mathematics: `x + 6 = 10` and `2x + 6 = 20` share 80% of their
characters while being different claims, so a similarity threshold scores an
abandoned branch as surviving. This was a real bug caught on the first test —
the labeller reported an abandon rate of 0.000 on a fixture built around an
abandoned branch. Lines must match exactly after whitespace normalisation.

## What the labels do not mean

`survives` is not `correct`. A student may delete a perfectly good line for a
better one, and a wrong line can survive into a wrong answer. The label is
survival. Any paper using this has to say so, and should pair it with whether
the final answer was actually right.

## Status

Algorithm verified on a constructed fixture only: a scripted solve of
`2x + 6 = 20` that takes a wrong route, abandons it, and recovers. The
labeller isolates both wrong-branch lines, and the abandon rate is stable
across pause thresholds from 400–1200 ms. **No human trace data has been
collected.** Nothing here is a finding.
