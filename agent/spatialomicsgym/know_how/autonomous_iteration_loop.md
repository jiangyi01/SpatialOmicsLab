# Autonomous Iteration: Improving a Result Instead of Producing One

## Metadata

**Version**: 1.0
**Short Description**: Run a research loop that improves: one metric with a direction, a guard, keep-or-discard against an incumbent, an append-only ledger, and a stop reason that names what the numbers did.
**Scope**: How to run a research loop that gets BETTER each round instead of just running again -- pick one number, guard it, keep or discard each round against it, and stop for a reason you can name.
**Adapted from**: uditgoenka/autoresearch (MIT, commit 050e30dc), itself after Karpathy's autoresearch. See THIRD_PARTY_LICENSES/autoresearch-MIT.txt. Nothing is copied; the loop discipline is restated for spatial omics.
**Applies when**: a research run, an explicit "keep improving this until it stops getting better", or any task where you are about to run the same analysis a second time with different settings.
**Does not apply to**: a single requested analysis. Do not turn one question into twenty-five rounds.

---

## Overview

A loop that runs the same analysis four times with different parameters and reports the last one is
not research, it is four analyses. What makes it research is a **number that says whether round N
is better than round N-1**, and the discipline to throw away the rounds that were not.

Five things, in this order. Do not start the loop until the first three exist.

## 1. One metric, and its direction

Pick ONE scalar the analysis already produces, and say which way is better.

    Metric: median silhouette over the assigned domains   Direction: higher_is_better
    Metric: number of spots left unassigned               Direction: lower_is_better
    Metric: F1 against the held-out marker panel          Direction: higher_is_better

Rules:

- It must come out of a file the tool wrote, not out of your reading of a figure. If you cannot
  print it with five lines of Python, it is not a metric yet.
- One number. Two numbers mean you will trade them off silently and call it progress.
- Say the direction out loud in the first round and never re-derive it. A loop that flips its own
  direction mid-run reports every round as an improvement.
- A term nobody measured has no metric. If the quantity is defined by the composition of the
  output rather than measured from it, say that it is unmeasured and stop -- do not substitute a
  proxy and present it as the thing.

## 2. A guard that must always pass

The guard is not the metric. It is the thing that makes the metric mean anything, and a round that
improves the metric while failing the guard is DISCARDED.

Use the checks that already exist rather than inventing one:

    from spatialomicsgym.postanalysis import run_post_analysis
    results = run_post_analysis(OUTPUT_DIR, tool_name="the_tool_you_called")

`manifest.json` `findings` carries the guard flags. Any of these failing discards the round:

- `signal_free` -- every spot got the same value, one domain, or a constant score. A metric
  computed on a signal-free result is a number about nothing, and it is usually a very good one.
- `orientation_transposed` -- the proportion table is cell-types-by-spots. Spot IDs read as
  cell-type names.
- `celltype_names_resolved` false -- the columns are `1 2 3` or `topic_4`, not cell types.

Add one more guard of your own only if the tool has a known failure the manifest cannot see.

## 3. The predicate: what "done" means

Write down, before round 1, the condition that ends the loop, and then never re-derive it:

    Done when: median silhouette >= 0.25, or five rounds have passed with no net gain.

Pin it in the journal. Every later round compares against that exact sentence. A stop condition
recomputed each round is a stop condition that can drift into "whatever just happened".

## 4. The loop

For each round:

1. **Review.** Read the ledger rows so far. What was tried, what was kept, what was discarded and
   why. Do not re-try a setting the ledger already discarded.
2. **Change ONE thing.** One parameter, one input, one step. Two changes and the ledger cannot say
   which one moved the number.
3. **Run**, and write to a results directory of this round's own. Never into a previous round's
   directory -- the report will attribute this round's figures to that one.
4. **Measure**, in an action EARLIER than the one that writes the conclusion. An answer drafted
   before the check is treated as final even when the check then fails.
5. **Guard.** Run the post-analysis review. Quote its warnings verbatim.
6. **Decide**, and record the word:
   - `keep` -- metric moved the right way AND the guard passed. This round becomes the incumbent.
   - `discard` -- metric moved the wrong way, or the guard failed. The incumbent does not change.
   - `unknown` -- the run crashed, or the metric could not be computed. Not a zero, and not a
     discard: it is a round that says nothing.
7. **Build the next round on the INCUMBENT, not on the last round.** This is the whole point. A bad
   round otherwise poisons every round after it.

## 5. The ledger

One append-only row per round, written BEFORE the round runs and closed after it, so a run that
dies mid-round still leaves a trace of having started:

    round  started    change                    metric   delta   guard  status    results_dir
    0      ...        baseline                  0.11     -       ok     baseline  round0/...
    1      ...        n_clusters 8 -> 12        0.18     +0.07   ok     keep      round1/...
    2      ...        harmony batch correction  0.14     -0.04   ok     discard   round2/...
    3      ...        n_neighbors 15 -> 30      -        -       -      unknown   round3/...

Rewriting the whole file each round is how a crash loses everything written so far. Append.

## Stopping, and saying which

Report exactly one of these, and report the numbers with it:

- `converged` -- the predicate from step 3 is satisfied. Say which round did it.
- `plateau` -- over the last five rounds that produced a number, the metric did not net-improve.
  Oscillation that ends flat is a plateau. **Rounds recorded `unknown` are excluded from that
  window**, because a crashed round is not evidence of no progress; five `unknown` in a row is
  `blocked`, not `plateau`.
- `ceiling` -- the round budget ran out. Say what the metric was doing when it did.
- `blocked` -- the analysis cannot proceed: a missing dependency, an input that is not there, the
  same crash twice. Say what would unblock it.

Never report "the rounds were exhausted" for a loop that used one round of four. Say what stopped
it and what the number was doing.

## What to report at the end

- The metric: starting value, final value, and which round produced the final one.
- Kept versus discarded, and what the kept changes had in common.
- The stop reason from the list above, in those words.
- The guard warnings, verbatim, for every round -- including the kept ones.
- The incumbent's results directory, so the reader can open what you are describing.

## Two things this loop must never do

- **Never subsample to make a round faster.** A metric measured on a subset is a different metric,
  and comparing it to a full-dataset round makes every delta meaningless. Run the whole dataset or
  say the round could not run.
- **Never score against a denominator you invented.** If the comparison is to a published
  benchmark, the denominator is the benchmark's own. A locally recomputed one is a different
  number wearing the same name.
