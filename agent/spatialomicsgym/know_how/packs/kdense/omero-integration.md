# OMERO Integration

## Metadata

**Short Description**: Securely inspect and automate microscopy data workflows against OMERO.server with omero-py, BlitzGateway, OMERO CLI, tables, annotations, ROIs, rendering, and documented OMERO.web APIs. Use for scoped OMERO inventory, metadata export, import/export planning, or reviewed write workflows.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/omero-integration/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: GPL-2.0-or-later (omero-py, per this document's body; this skill's own text is MIT) (recorded in packs/MANIFEST.yaml)
**Commercial Use**: This text is MIT and may be used commercially. The wrapped library, omero-py, is GPL-2.0-or-later: that licence governs redistribution of the library itself, not of this description, and nothing from the library is vendored here.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 2 install fence(s) replaced by a provisioning pointer; 29 upstream script/reference path(s) marked as not vendored; 1 fence(s) left with nothing runnable replaced by a pointer; 3 manifest replacement(s) and 0 excision(s) applied; an Overview heading added above the opening prose.

---

## Overview

Use current OME documentation and the smallest explicit data scope. OMERO data
may contain unpublished images, identifiers, annotations, original files, and
derived measurements.

## Verified Baseline

This skill was refreshed on **2026-07-23**:

- **OMERO.server 5.6.18** (May 2026) is the current documented stable server.
- It was tested by OME with **OMERO.py/omero-py 5.22.1** and
  **OMERO.web 5.31.0**.
- `omero-py==5.22.1` requires Python 3.10 or newer. The OMERO support matrix
  supports 3.10 and 3.11, recommends 3.12, and still labels 3.13/3.14
  “upcoming.”
- OMERO 5.6 uses **IcePy 3.6**, with 3.6.5 prebuilt client wheels documented
  for Python versions through 3.12.

The pin above is a reproducible skill snapshot, not a promise that every
OMERO.server release accepts that client. For another server version, consult
its release entry and use the OMERO.py version tested with it. See
(upstream reference file, not included) (upstream reference file, not included).

## Operating Contract

1. Start with local validation or a dry run. Do not connect until the user has
   selected the host, group, object type, IDs, and result limit.
2. Read credentials only from `OMERO_*` environment variables the user has set for this session
   (upstream's frontmatter listed them; it is not carried into this document).
   Never search parent directories or load dotenv files.
3. Never place a password or session key in command arguments, source code,
   output JSON, logs, tracebacks, or chat. A session key is a bearer credential.
4. Default to `secure=True`. OMERO encrypts login by default, but post-login
   data and the session ID may otherwise travel unencrypted. `secure=True` does
   not by itself guarantee certificate hostname verification.
5. Bound every list, page, ROI, shape, annotation, table row, pixel plane, and
   local file scan. Do not turn an object request into a group-wide or
   cross-group export without explicit approval.
6. Treat all writes separately: annotation/link creation, rendering-default
   saves, image creation, imports, script uploads, table writes, ownership or
   group changes, and deletion require an exact reviewed target.
7. Close `BlitzGateway`, table handles, raw stores, thumbnail stores, rendering
   engines, script clients, and other stateful services in `finally` blocks or
   documented context-manager patterns.
8. Never connect to a real server merely to “test” examples.

## Choose the Interface

- **BlitzGateway (`omero-py`)**: primary Python client for object traversal,
  pixels, annotations, ROIs, rendering, and services.
- **OMERO CLI**: sessions, import scanning/import, OME-TIFF or XML export,
  scripts, and administrative plugins. Most client commands are remote; import
  also needs the matching server-side Java libraries through `OMERODIR`.
- **OMERO.web `api` and `webgateway`**: the only OMERO.web apps that official
  documentation calls stable public APIs. The documented JSON API is
  version-discovered and has limited object coverage; it is not evidence that
  every webclient URL is a supported REST endpoint.
- **OMERO.server scripts**: uploaded plugins executed by server infrastructure.
  They are different from the bundled local client helpers in (upstream helper script, not vendored).

## Install a Reproducible Client

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

OMERO.py needs the exact IcePy 3.6.5 wheel matching the interpreter, OS, architecture,
and wheel tags; neither is installed from this document.

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

Do not substitute Ice 3.7: the OMERO 5.6 support matrix marks Ice 3.6 as
recommended and 3.7 as unsupported. A plain install may attempt to compile
IcePy from source; prefer a reviewed matching wheel. The upstream package is
GPL-2.0-or-later; this skill’s own files are MIT.

For import/admin commands only, `OMERODIR` must point to a compatible extracted
OMERO.server directory. A normal remote BlitzGateway client does not require
that server tree. Read (upstream reference file, not included) (upstream reference file, not included)
before installation or authentication work.

## Credentials and Connection

Set named variables in the calling environment or secret manager. Do not put
the password on an `omero` CLI command:

```bash
export OMERO_HOST="omero.example.org"
export OMERO_PORT="4064"
export OMERO_USER="researcher"
export OMERO_SECURE="true"
# Supply OMERO_PASSWORD through the environment/secret manager, or use
# OMERO_SESSION_KEY as an alternative. Do not echo either value.
```

A password-authenticated, exception-safe read pattern is:

```python
import os
from omero.gateway import BlitzGateway

conn = None
try:
    conn = BlitzGateway(
        os.environ["OMERO_USER"],
        os.environ["OMERO_PASSWORD"],
        host=os.environ["OMERO_HOST"],
        port=int(os.environ.get("OMERO_PORT", "4064")),
        secure=True,
    )
    if not conn.connect():
        raise RuntimeError("OMERO connection failed")

    images = conn.getObjects(
        "Image",
        opts={"limit": 25, "offset": 0, "order_by": "obj.id"},
    )
    for image in images:
        print(image.getId())  # Do not print names unless requested.
finally:
    if conn is not None:
        conn.close()
```

For existing-session and CLI prompt patterns, certificate verification,
group context, and cleanup details, read
(upstream reference file, not included) (upstream reference file, not included).

## Bundled Safe Helpers

All helpers use `argparse`; `--help` works without OMERO installed. Remote
helpers are dry-run by default and require `--execute`.

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

- `validate_config.py`: validates only named endpoint/auth variables locally;
  optional DNS resolution still does not contact OMERO.
- `inventory.py`: bounded, read-only object inventory with paged JSON output.
- `export_image_metadata.py`: explicit-image annotation/ROI JSON export with
  redaction defaults and per-category limits; it never downloads file bytes or
  pixels.
- `plan_transfer.py`: local-only import scan or per-image export plan; it never
  invokes OMERO and never emits credential flags.

Read (upstream reference file, not included) (upstream reference file, not included) before using them.

## Capability Guide

- Connection, sessions, groups, TLS:
  (upstream reference file, not included) (upstream reference file, not included)
- Hierarchies, pagination, screening data, import/export:
  (upstream reference file, not included) (upstream reference file, not included)
- Tags, map/file/comment annotations, namespaces:
  (upstream reference file, not included) (upstream reference file, not included)
- Raw planes, tiles, thumbnails, rendering:
  (upstream reference file, not included) (upstream reference file, not included)
- ROI model, shape export, statistics caveat:
  (upstream reference file, not included) (upstream reference file, not included)
- Bounded table creation, paging, querying, closure:
  (upstream reference file, not included) (upstream reference file, not included)
- Local helpers and OMERO.server scripts:
  (upstream reference file, not included) (upstream reference file, not included)
- Permissions, filesets, web/public links, destructive operations:
  (upstream reference file, not included) (upstream reference file, not included)

## Final Review Before Remote Work

- Confirm server version and its tested OMERO.py pairing.
- Confirm target host, SSL router port, user/session, and one group.
- Confirm exact object IDs/types and hard limits.
- Confirm whether names, annotation values, file names, ROI labels, owner names,
  pixels, or original files may leave the server.
- Show the proposed output path and refuse overwrite unless explicitly allowed.
- For a write, show the mutation and target IDs separately from any read plan.
- Close every connection/service even after partial failure.
