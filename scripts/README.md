# Operational scripts

- `launch/`: thin DLC launch wrappers for the named Megatron contracts.
- `lib/`: shell helpers shared by those wrappers.
- `evaluation/`: prepare and verify the pinned optional evaluation environment.
- `patches/`: minimal upstream patches, pinned revisions and reproduction notes.
- `check_distribution.py`: manually check a built wheel and source archive.

Keep experiment settings in `recipes/` and reusable Python in `src/archlab`.
Scripts are operational entrypoints, so they do not belong inside recipe folders.
Retired originals are checksum-preserved in the ignored `.legacy/` archive.
