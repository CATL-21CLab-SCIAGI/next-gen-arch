# Publication and local workspace policy

The publication is a curated source snapshot with a fresh root commit. Earlier
source IDs in result records identify historical experiments archived privately;
they are not commits that a public clone can resolve. Reported experiments were
not rerun by the publication cleanup.

## What belongs in Git

| Publish after review | Keep outside the tracked tree |
| --- | --- |
| Project code, tests, portable recipes | Runtime overlays, downloaded upstream checkouts |
| Licenses and attribution | Model weights, datasets, checkpoint/optimizer payloads |
| Methodology and compact result aggregates | Raw generations, job plans, node/PID receipts, live queue state |
| Sanitized examples and dependency locks | Credentials, SSH/cloud config and private service endpoints |

The root `.gitignore` admits named project directories and ignores other
workspace directories by default. Existing `results/`, `.runtime/`, and local
source snapshots may remain in the same working directory. Do not force-add
operational artifacts. Ignoring a file does not remove an already tracked copy
or clean past history.

Legacy deployment adapters retain explicit mount-layout assumptions for their
qualified environment. These are not a portable cloud service or credentials;
review their required paths before use. Private container image identity is
supplied through `NGA_CONTAINER_DIGEST`, not an embedded registry hostname.

## Review and validation

Run `python -m archlab.reporting.publication_audit --output /path/to/audit.json`
against the candidate Git tree. It records every tracked path, size and SHA256,
rejects runtime/data/secret artifacts, and checks source for high-confidence
credential markers. It is an exposure screen, not proof that all sensitive
information or third-party rights issues have been eliminated.

Validate from a fresh candidate checkout, not only the workspace containing
ignored files. Run lint, the CPU suite, package builds, and inspect wheel/sdist
members. Container-owned frameworks must not be vendored. GPU qualifications
apply only to their recorded revisions and contracts.

## History and remote surfaces

Keep a verified private Git bundle and a backup of local worktree metadata
before replacing history. Historical linked worktrees used by live jobs must
keep resolving their original revisions. A small fresh public Git history does
not itself reclaim the retained private archive's disk space.

Updating `main` alone does not remove other branches, tags, pull-request refs,
release assets, Actions artifacts/logs, or the separate Wiki history. Inventory
and review those surfaces before changing visibility. Existing GitHub caches or
third-party copies are not guaranteed to disappear after a force-push. Never
publish an exposed credential; revoke it first.

The canonical documentation lives in `docs/wiki/`. Rendered GitHub Wiki pages
are a separate publication copy. Keep the repository private until the reviewed
source tree and other exposed surfaces have been approved for publication.
