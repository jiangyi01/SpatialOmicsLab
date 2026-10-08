# Third-party licences

This repository is **AGPL-3.0-only** (see `../LICENSE`). This directory holds the licence text of
third-party work that has been **vendored into our tree** — copied in and adapted as our code, rather
than installed as a dependency. Apache-2.0 §4(a) requires that recipients of a derivative work be
given a copy of the licence, and that is what these files are for.

| file | upstream | pinned commit | what we took |
|---|---|---|---|
| `ToolUniverse-Apache-2.0.txt` | [`mims-harvard/ToolUniverse`](https://github.com/mims-harvard/ToolUniverse) | `f075c2a75e8b35ae5dbb220d48d4e87e980388b1` | six `agent/spatialomicsgym/tool/` modules and their `tool_description/` companions — see `../VENDORING.md` |
| `autoresearch-MIT.txt` | [`uditgoenka/autoresearch`](https://github.com/uditgoenka/autoresearch) | `050e30dc4ba0974b03f2873111b9901ec3211390` | no code — the research loop's discipline; see below |
| `SciAgent-Skills-CC-BY-4.0.txt` | [`jaechang-hits/SciAgent-Skills`](https://github.com/jaechang-hits/SciAgent-Skills) | `fe505cae14d20b6c33be2e49666425be98f005bb` | 13 skill documents, as text, re-headed under `agent/spatialomicsgym/know_how/packs/sciagent/` — see below and `../VENDORING.md` §12 |
| `scientific-agent-skills-MIT.txt` | [`k-dense-ai/scientific-agent-skills`](https://github.com/k-dense-ai/scientific-agent-skills) | `330c8e764435a731eff571e3efdda70b363d0792` | 43 skill documents, as text, re-headed under `agent/spatialomicsgym/know_how/packs/kdense/` — see below and `../VENDORING.md` §12 |

`ToolUniverse-Apache-2.0.txt` is a byte-for-byte copy of `LICENSE` at that commit: 11,351 bytes,
sha256 `d0fd2a0c2969573ae74147b7400bbb57e6a5b334816beea15475f29fd5deeb49`, closing with
`Copyright [2025] [ToolUniverse team]`. There is no upstream `NOTICE` file (the path 404s at that
commit), so §4(d) does not attach.

Every file that carries vendored code states, in its module docstring, that we changed it and what we
changed — that is the §4(b) obligation. `../VENDORING.md` is the full record: what landed, what was
excluded and why, and how to re-sync against a newer upstream.

These files ship in the wheel as well as in a clone, via `license-files` in `../pyproject.toml`.

## `autoresearch-MIT.txt` — an attribution, not a vendoring

Nothing from that repository is in this tree. It is a Claude Code / OpenCode / Codex **skill pack**
(markdown command files and four bash scripts), so there was no package to vendor and no source to
copy; what `agent/spatialomicsgym/research/loop.py` takes is the *shape* of the loop it describes — a
mechanical metric with a direction, keep-or-discard against an incumbent, a guard that must always
pass, an append-only results ledger, plateau detection over a window that excludes uncomputable
rounds, and a stop vocabulary that names what the numbers did.

MIT does not require a notice for ideas, only for copies. This file is here anyway, because the
debt is real and a reader tracing where the design came from should not have to guess. The upstream
is itself based on [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

The copy is byte-for-byte `LICENSE` at that commit: 1,068 bytes, sha256
`36fe186f404fa529a4c290a5c671e0dd07162be0c3270c02e1cdc831e97bf127`, opening
`MIT License` / `Copyright (c) 2026 Udit Goenka`.

## Skill packs — text vendored as know-how, not code

Unlike `autoresearch`, these two **copy bytes**: 56 markdown documents were rendered from the two
pinned clones by `agent/spatialomicsgym/know_how/merge_packs.py`, driven by
`agent/spatialomicsgym/know_how/packs/MANIFEST.yaml`, and land as tier-2 know-how — prose the agent may
retrieve on a portal turn, never a routing entry in `agent/skills/` and never on a benchmark run. No script,
test, workflow or asset from either repository is in this tree. `../VENDORING.md` §12 is the full
record; `../CHINA_EXCLUSION_REPO.md` §19 is the origin sweep.

Both licences require the notice to travel with the text, and CC BY 4.0 additionally requires that
changes be indicated: every emitted document carries the upstream blob URL at the pinned commit, the
licence line, and a `Modifications` line naming what the script changed.

`SciAgent-Skills-CC-BY-4.0.txt` is a byte-for-byte copy of `LICENSE` at that commit: 1,225 bytes,
sha256 `513891915dfd73a3fd4d74b57053760e736b4f7af62f7aff813f1c7559b0f908`, opening `Creative Commons
Attribution 4.0 International (CC BY 4.0)` / `Copyright (c) 2024 jaechang-hits`. That file itself
states two inherited attributions — `K-Dense-AI/claude-scientific-skills` (MIT) and
`snap-stanford/Biomni` (Apache-2.0) — which the `**License**` line of every SciAgent document repeats.

`scientific-agent-skills-MIT.txt` is a byte-for-byte copy of `LICENSE.md` (note the extension) at
that commit: 1,068 bytes, sha256 `09b02a3c9df3053c55531d503357a9c7cde275970e6c3ceaa1ddf5f0e90b40c1`,
opening `MIT License` / `Copyright (c) 2025 K-Dense Inc.`.

`python -m spatialomicsgym.know_how.merge_packs --check` verifies both copies against the sha256 the
manifest records, and rewrites them from the clone on every merge.
