# Documentation Maintenance

## Reader structure

| Layer | Contents |
| --- | --- |
| README | Project purpose and routes into the Wiki |
| Wiki | Guides, research map, decisions and concise summaries |
| Repository docs | Versioned experiment contracts and dated evidence records |
| Run artifacts | Detailed logs, checksums, checkpoints and live state |

## Page format

For a guide: **purpose → task table or procedure → verification → related pages**.

For an experiment: **status/date → question → matched settings → results → limitations → evidence**.

Use tables for comparable facts. Keep one authoritative definition of each method or metric.

## Publishing

Page sources live in `docs/wiki/`. Publish them to the repository's separate Wiki Git repository. `Home.md`, `_Sidebar.md` and `_Footer.md` define navigation.

Review documentation with the relevant code or experiment change. Keep page text identical when publishing, but render relative Markdown links as absolute Wiki URLs for sibling pages and repository URLs for code, recipes and evidence. Plain copying leaves repository-relative links broken in the Wiki. Coordinate direct Wiki edits before the next synchronization.

Wiki availability depends on repository visibility and organization plan. An existing Wiki has its own Git history; review that history separately before making the parent repository public. For a new Wiki, create its first page on GitHub before cloning.

## Before publication

- Check Wiki page targets and repository links.
- Mark proposals and dated snapshots explicitly.
- Replace local-only artifact hyperlinks with clearly labeled artifact paths.
- Preserve source revisions and statistical limitations.
- Check for credentials, private hosts and stale launch instructions.

The Wiki explains the project; recipes and run receipts decide execution.
