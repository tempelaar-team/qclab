# Contributing to QC Lab

This file describes the QC Lab contribution workflow, including the CI
behavior on the `tempelaar-team/qclab` repository, so Claude can guide
contributors through the right steps and set the right expectations
about what will run when.

---

## 1. Branches

QC Lab uses two long-lived branches:

- **`main`** — release branch. Tagged releases are cut from here.
  Pushes to `main` are expected to be release-grade.
- **`dev`** — integration branch. Day-to-day development lands here.
  Periodically `dev` is merged into `main` to cut a release.

There is also a separate repo, `tempelaar-team/qclab-dev`, which hosts
the development docs site at
`https://tempelaar-team.github.io/qclab-dev/`. Contributors do not push
to `qclab-dev` directly — it is updated automatically by CI when `dev`
moves.

## 2. The contribution flow

For internal contributors with write access to `tempelaar-team/qclab`:

1. Branch off `dev`, with a descriptive branch name.
2. Make changes, commit. Push the branch to the upstream repo.
3. Open a pull request **into `dev`** (not `main`).
4. CI will run on the PR (see Section 4 below).
5. Once green and reviewed, merge to `dev`.

For external contributors without write access:

1. Fork `tempelaar-team/qclab`.
2. Branch off `dev` in the fork.
3. Open a PR from the fork's branch into `tempelaar-team/qclab`'s `dev`.
4. Note that fork PRs cannot deploy docs (the deploy step requires a
   secret token unavailable to fork PRs); the build step will still run
   and validate docs syntax.

**Never open PRs directly into `main`.** Releases are cut by merging
`dev` into `main`, which is a maintainer action.

## 3. Pre-PR local checks

Before opening a PR, contributors should run locally:

- `pytest -m "not mpi" tests` — the same command CI runs.
- For docs changes: `cd docs && make clean && make html` to confirm
  Sphinx builds without errors.
- For packaging-touching changes (`pyproject.toml`, file layout):
  `python -m build` followed by `python -m twine check dist/*`.

These are not enforced by pre-commit hooks — discipline is on the
contributor.

## 4. What CI runs, and when

The QC Lab CI is structured to give fast feedback on day-to-day work
and full coverage at integration points. The triggers below are what
contributors should expect:

| Event | Workflows that run |
|---|---|
| Push to a feature branch (no PR open) | None |
| Push to `dev` (paths: code, not docs-only) | `tests.yml` fast job |
| Push to `dev` (paths: docs-only) | `docs_dev.yml` (rebuilds and deploys to `qclab-dev`) |
| Open or update a PR into `dev` | `tests.yml` fast + full matrix, `install_source.yml` (3 OSes × 6 Python versions, with `twine check`), `pylint.yml` |
| Push to `main` | Same as PR into `dev`, plus `docs.yml` if docs-relevant paths changed |
| Open or update a PR into `main` | Same as PR into `dev` |
| Publish a GitHub Release | `publish_pypi.yml` (uploads to PyPI), then `install_pypi.yml` (cross-platform install matrix from PyPI) |

Notes on specific workflows:

- **`tests.yml` `full` job depends on `fast` job.** If `fast` fails, the
  full matrix is skipped. This saves CI minutes but means contributors
  see only the first failure.
- **Path filters.** Test workflows have `paths-ignore: ['**.md',
  '**.rst', 'docs/**']`, so pure-docs changes to `dev` skip tests
  entirely. Docs workflows have a complementary `paths:` filter that
  only triggers on docs-relevant files.
- **No TestPyPI step.** QC Lab does not publish to TestPyPI. Packaging
  is validated by running `twine check` inside `install_source.yml`.
- **MPI tests are excluded.** `pytest -m "not mpi"` skips them
  everywhere. The `mpi` marker exists for local use but is not run in
  CI.
- **Concurrency cancellation is on.** A new push to a branch cancels
  any in-flight run on the same ref.

## 5. PR description expectations

When opening a PR, the description should call out QC-Lab-specific
concerns that reviewers will look for:

- **New state-dict keys.** If the PR introduces a key the rest of the
  codebase will read or write (e.g., a new entry like `state["foo_bar"]`),
  list it. State keys are part of the public contract between tasks.
- **New ingredient slot names.** If the PR adds a new slot (a callable
  registered on `Model._ingredients`), name it and describe its
  signature. The standard slots are listed in `references/conventions.md`
  section 3.1; additions extend that contract.
- **New model `constants`.** If the PR adds attributes to a model's
  `constants`, list them and note any `_init_*` methods that consume
  them.
- **Changes to existing tasks.** If a task's `*_name` keyword arguments
  changed, mention it — these are the rebinding hooks that recipes
  depend on.
- **Performance flags.** If `update_dh_qc_dzc` or `update_h_q` semantics
  changed, flag it explicitly.

A short "How tested" section is also expected: which tests were added
or modified, which worked example was used to manually verify, what
edge cases were checked.

## 6. Things Claude should not do without explicit confirmation

When acting as a contributor on behalf of a user, Claude must not:

- Push directly to `dev` or `main`. All changes go through PR.
- Force-push to a shared branch (`dev`, `main`, or any branch with an
  open PR from another contributor).
- Open PRs into `main` from feature branches. PRs land in `dev`;
  release PRs (`dev` → `main`) are a maintainer action.
- Modify `.github/workflows/` without flagging the change in the PR
  description, since CI changes affect everyone.
- Bump the version in `pyproject.toml` outside of an explicit release PR.
- Delete branches, even merged ones, without confirmation.
- Modify `pytest.ini` markers or exclusion patterns without flagging.

## 7. Things Claude should ask about before doing

- Whether a change to `dev`'s public CI behavior (paths, triggers,
  matrix) is intended.
- Whether to open a draft PR vs. a ready-for-review PR.
- The branch name to use, if not obvious from the request.
- Whether to add a corresponding test, when modifying tasks/ingredients
  that have existing test coverage.
