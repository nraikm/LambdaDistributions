# Proved matrix groups

This is the canonical home for the completed matrix-group proof and
verification work in this workspace.

- [`all_new_matrix_groups_proofs.pdf`](all_new_matrix_groups_proofs.pdf) is the
  single consolidated proof anthology. It contains each canonical proof once.
- [`packages/`](packages/) contains executable sources: verification code,
  marimo notebooks, LaTeX, and tests.
- [`documents/`](documents/) retains the unique editable proof sources that do
  not already live in a source package; standalone proof PDFs have been
  removed in favor of the anthology.
- [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) summarizes the mathematical and
  computational outcomes and records the latest regression run.
- [`all_new_matrix_groups_formulas.tex`](all_new_matrix_groups_formulas.tex)
  and its compiled PDF form the cross-family formula compendium.

`all_new_matrix_groups_proofs.tex` is a standalone monolithic source: all proof
bodies are physically inlined, and it contains no `input`, `include`, or
embedded-PDF dependency. `all_new_matrix_groups_proofs.pdf` is the verified
canonical build of that source, including the power-character and
bounded-support stability theorem. Its table of contents lists proof-level
sections only; numbered internal headings are omitted from the table of
contents.

The legacy paths `for_this_guy/` and `output/pdf/for_this_guy/` are retained as
compatibility links. Existing imports, notebooks, and older task references
therefore continue to work, while the real files now live here.
