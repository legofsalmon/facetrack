# Release notes

One file per release, named after its tag: `v1.0.0.md`. The **release**
workflow (`.github/workflows/release.yml`) publishes the file as that
release's notes and refuses to run without it. Buyers read these notes on
the GitHub release page, and the letissier.ie product page links to them.

To cut a release:

1. Set `__version__` in `yewee/__init__.py` to the new version.
2. Add `docs/releases/v<version>.md` in the same PR.
3. Once that PR is merged, run Actions → release → "Run workflow" on
   `main` with the version (`v1.0.1`). The workflow builds both platforms,
   signs and notarises the Mac dmg, creates the tag at that commit, and
   publishes the release with both files attached.

Leave the Downloads section out of the notes: the workflow always attaches
the same two files, `Yewee-<version>.dmg` and `yewee-setup-<version>.exe`.
