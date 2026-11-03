# Distribution verification and publication

`publish.yml` runs verification on pull requests, selected development branches,
`v*` tags and manual dispatch. Only a `push` to a `v*` tag can publish.

The job graph is `verify -> publish`. All checks are mandatory sequential steps
inside `verify`; `publish` requires that job's successful result in the same run.
There is no `always()` or `continue-on-error` path to publication. A failed,
cancelled or skipped verification cannot publish. Manual runs and pull requests
cannot publish, even when verification passes.

## Release authorization

For a tag push, the tag must be exactly `v<project.version>`, with a canonical
PEP 440 version and no local suffix (`+...`). The tag's commit, event commit,
checkout HEAD and fetched `origin/main` tip must all agree. A missing main ref,
an older main commit or a commit from another branch is refused. The publication
job additionally requires the explicit `release_authorized` output. A branch
push, pull request or manual run produces `false` and can only verify packages.

This is deliberately stricter than checking whether a commit is an ancestor of
main. Re-running an old tag after main has advanced requires a new decision;
the workflow does not silently authorize an earlier revision. It uses main as
fetched at checkout, not a claim that main cannot subsequently move.

The present development version contains a local suffix and is intentionally
ineligible for publication. Normal release preparation must choose a publishable
version and keep `uv.lock` aligned before creating an authorized release tag.

## What is verified

- Full history and release tags v0.1.0 through v0.4.0 are present.
- Python 3.12.15 and uv 0.12.24 are fixed. `uv sync --locked` rejects a stale
  lockfile; lockfile hashes are checked before and after verification.
- Dependencies and all extras come from `uv.lock`, without installing the project
  from the checkout. Wheel and sdist are built once; uv builds the wheel from the
  sdist. Their engine contents and project metadata are compared with the checkout.
- The exact wheel is installed in the dependency environment. Its package location
  must be inside the virtual environment and its files must match the wheel.
- The complete public suite runs from a neutral directory using isolated Python.
  The JUnit gate rejects failures, collection errors, duplicate results, fewer than
  1999 passing tests and any deviation from the 59 named skips in
  `expected-onnx-skips.json`, including a different skip reason.
- The Hugging Face cache is empty and offline. Those 59 tests are **not executed**;
  this gate does not qualify real-model inference. Changes to the expected test
  inventory require review, rather than automatically accepting a new skip count.
- API/CLI, supersession, history, Attempt/Lesson/Skill, FTS and backup/restore smoke
  checks run against the installed wheel, also in a second base-only environment.

## Artifact and privilege boundaries

After all checks pass, verification uploads the two distributions, SHA-256 checksums,
source commit, lockfile hash and JUnit report. Publication downloads those immutable
artifact IDs from the same workflow run. Before uploading to PyPI it checks the
checksum manifest against the verification job's output, both file digests, and the
absence of extra distributions. It never rebuilds a package.

The verification job has repository-read permission and no publishing credential.
Checkout does not persist credentials. Only the publishing job requests
`id-token: write`; the existing `pypi` environment and Trusted Publishing action
are preserved. Repository/environment protections and PyPI publisher configuration
remain external prerequisites, not settings changed by this workflow.

Before enabling publication, configure and verify online:

- Protect main with reviewed changes and required verification checks; restrict
  bypass and force pushes. Review publishing-workflow changes explicitly.
- Restrict creation, update and deletion of `v*` tags to release maintainers.
- Require review on the `pypi` environment, restrict its deployment tags to `v*`,
  and prevent self-approval and administrative bypass where supported. The reviewer
  must confirm the exact SHA, current workflow and verification results.
- Bind the PyPI Trusted Publisher to this repository, `publish.yml` and the `pypi`
  environment; inspect other publishers and legacy API-token publishing paths.

A historical tag can run the workflow stored in that historical revision.
Therefore inspecting the current workflow alone cannot guarantee that older
workflows or a contributor's replacement workflow cannot publish. Tag rules,
environment approval and publisher restrictions are essential external controls,
not guarantees provided by this code. None of those remote settings is asserted
to be configured by the local verification.

Local syntax checks, fault injection and package tests do not replace an actual
GitHub Actions run. In particular, hosted-runner behavior, artifact transfer and
OIDC publication need separate confirmation there. The current development package
version is unchanged; this workflow does not itself authorize a release.

Reference semantics: [job dependencies](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idneeds)
and [locked environments](https://docs.astral.sh/uv/concepts/projects/sync/).
