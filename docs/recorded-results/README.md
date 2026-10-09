# Machine-readable evidence

This directory contains immutable campaign contracts and result snapshots. Narrative interpretation belongs in `docs/`.

- top-level CSV/JSON files preserve published frozen campaigns;
- dated status files are historical snapshots, not live job state;
- subdirectories contain one campaign's aggregate, learning curves, and provenance;
- absolute metrics from different datasets or budget regimes must not be merged.

New live runs should remain in their configured NAS/OSS namespace. Promote only reviewed aggregates and compact provenance records into this repository; do not commit raw checkpoints, private infrastructure settings, or mutable “latest” pointers.
