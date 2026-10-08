# SECURITY.md — what confines this platform, and what does not

**Contract document. Never edited to match what was built.** If the tree diverges from this file,
that is a failure to report, not a document to update.

This file answers three questions: **what is enforced**, **how to add a destination**, and **what
is not covered**. The third section is the one that matters most, because every control here has a
boundary and a reader who does not know where it is will assume the wrong thing.

| | |
|---|---|
| Policy file | `agent/spatialomicsgym/policy/egress.yaml` (in the package, not the repo root — it loads during `envdoctor` before any state dir exists) |
| Loader | `agent/spatialomicsgym/policy/egress.py` |
| Written by | nobody. No writer exists in `agent/spatialomicsgym/`, and a lens fails the suite if one appears |
| Measured | 2026-09-24 |

---

## 1. What the policy contains

Seeded from a scan of the shipped tree and maintained by hand since. Every entry is a destination some
shipped code names; four omics_scripts hosts were added by hand on 2026-09-30.

| | count |
|---|---|
| Hosts | 75 |
| — of them, path-granular repositories on `github.com` | 25 |
| Package indexes | 4 — `pypi.org`, `files.pythonhosted.org`, `download.pytorch.org`, `data.pyg.org` |
| Conda channels | 5 — `conda-forge`, `bioconda`, `pytorch`, `nvidia`, `defaults` |
| Git hosts | 1 — `github.com` |
| Source files the seeding scan attributed hosts to | 57 |

`scope:` on each entry says which boundary enforces it: `tool` (an in-process research module,
enforced by `http_client`'s allowlist), `worker` (reachable from a REPL or MCP tool worker),
`setup` (package installation, enforced by the broker and the recipe lens).

**Three hosts are plainly outside spatial transcriptomics** and are present because the platform
already reaches them: `clinicaltrials.gov`, `api.fda.gov`, `apiv3.iucnredlist.org`. They are
listed here for review rather than silently blessed.

---

## 2. What is enforced, and where

Module paths are relative to their package: `utils/…` and `agent/…` are inside `spatialomicsgym`
(`agent/spatialomicsgym/`).

This repository ships the agent without the web portal. The portal's privilege boundary (model-written code
running as an unprivileged `sog-agent` user, and the audited `0600`/`0700` private state) is therefore not
part of it: model-written code runs as the user who started the agent.

| Control | Where | Enforces |
|---|---|---|
| Per-module host allowlist | `utils/http_client._check_host` | An in-process research module reaches only the hosts it declares |
| Redirect re-checking | `utils/http_client.request_text` | Every hop is checked, capped at `defaults.max_redirects` (4). `requests` follows up to 30 and checks none |
| Private-destination refusal | `utils/http_client._refuse_private` | A declared name that resolves to loopback, private, link-local, reserved, multicast or unspecified space is refused |
| Response cap | `utils/http_client._read_bounded` | `defaults.max_response_bytes` (8 MB), plus a wall clock of the caller's timeout + `defaults.deadline_grace_seconds` (60 s) |
| Path-granular repositories | `agent/broker.spec_kind` | `github.com` being allowed does **not** allow every repository on it |
| Index and channel checks | `agent/broker._valid_packages`, `_valid_channels` | Installs come from the four indexes and five channels above |
| Tool-file containment | `agent/broker._tool_target` | Resolve, contain, then classify — so a tool-shaped symlink out of `agent/tools_user/` is refused |
| Transcript-tag neutralisation | `agent/execution.as_tool_data` | Tool output cannot write the agent's answer or its next cell |

---

## 3. How to add a destination

All four are operator actions on a shell. Nothing in the running platform can do any of them.

**A host a tool module needs.** Add an entry to the `hosts:` list in `egress.yaml`, with
`scope:` (`tool`, `worker` or `setup`), `owner:` naming the file that reaches it, and a `reason:` a
reviewer can check. An entry with no `owner:` is not reviewable and should not be added.

**A repository.** Add a `paths:` block to that host's entry. `github.com/owner/name` is the unit;
`github.com` is not, and an entry without `paths:` allows the whole host. Pin the commit in the
recipe as well — every `git+https://` line carries an `@<40-hex>`.

**A package index or a conda channel.** The top-level `package_indexes:` / `conda_channels:` lists.
These are the widest entries in the file: an index is every package on it.

**A limit.** The `defaults:` block — `max_redirects`, `max_response_bytes`,
`deadline_grace_seconds`, `refuse_private_ranges`. `http_client._limits()` reads them, so this is
the one place they are set. A value of zero, a negative, or a non-number falls back to the shipped
value rather than removing the bound.

**After any of them**, re-run `pytest test/test_the_egress_policy_is_central_read_only_and_path_granular.py -q`
and `bash agent/spatialomicsgym/spatialomicsgym_env/run_core_tests.sh`. `egress.fingerprint()` gives the
sha256 of the bytes actually loaded — **nothing calls it at boot today**, so a change to this file
is currently visible only by running that, or by the diff. That is a gap, not a control.

---

## 4. What this does **not** stop

Stated here rather than discovered later. Each of these is a real gap, not a hypothetical.

**A process that never goes through Python's HTTP client.** `pip`, `conda`, `git`, `curl`, and any
compiled tool. The plan's Part 2 proxy and in-worker seatbelt are what would cover them, and
**neither is built**. Until they are, the install-time checks at the broker are the only thing
between a recipe and the network.

**27 raw `requests.*` calls in `tool/` and 6 `urlopen` sites** bypass `http_client` entirely and
therefore have no allowlist, no redirect checking and no response cap. Two primitives in particular:
`extract_url_content` and `tier1_ping`, the latter reachable with a `file:` scheme.

**DNS rebinding.** `http_client` checks the address a name resolves to, then `requests` resolves
again when it connects. A record that changes between the two reaches the second answer. Closing
this means connecting to the checked IP with the hostname in SNI and the `Host` header.

**Path granularity at the network layer.** A proxy sees `CONNECT github.com:443` and nothing more,
so the 25 path-granular repository entries are enforced at the **broker and the recipe lens only**.
A process that reaches `github.com` by another route is not bound by them.

**Reads.** Model-written code can read anything the user running the agent can read. This work
restricts egress, not reads.

**Package content.** Identity, source and version are checked. There is no hash pinning, no
signature verification, and pip's build backend still executes.

**Anything that needs a separate user.** Without the portal there is no privilege boundary: the CLI,
the Python API and every benchmarking run execute model-written code as the invoking user.

**Kernel isolation is not available on this box.** No `CAP_SYS_ADMIN`, `unshare --net` is refused,
no iptables or nft. Every control above is in userspace and none of them is a sandbox.

---

## 5. Reporting

Report a finding to the maintainers with a **Where**, a **What is wrong**, an
**Attribution**, and a **Status** that says CONFIRMED or UNVERIFIED and gives the check command. A
fix appends a `FIXED`/`PARTLY FIXED` bullet with its commit; `PARTLY` must say what is still open,
because a finding marked closed while half of it stands is worse than one left open.
