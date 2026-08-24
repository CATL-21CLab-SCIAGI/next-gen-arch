# Security policy

## Supported versions

Security fixes are applied to the latest `main` branch and the latest tagged release.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature for the repository. Do not open a public issue for a vulnerability that could expose credentials, execute untrusted code, overwrite training artifacts, or compromise shared compute.

Include the affected commit, a minimal reproduction, expected impact, and any suggested mitigation. Maintainers will acknowledge a complete report as soon as practical and coordinate disclosure after a fix is available.

## Research-code boundary

The training and dataset scripts operate on user-selected local storage and can download public artifacts. Review generated training commands and set `NANOCHAT_BASE_DIR` explicitly before running them. This repository does not provide a hosted inference service or accept untrusted model files by default.
