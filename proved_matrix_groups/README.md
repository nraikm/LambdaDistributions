# Proved matrix groups

This is the canonical home for the consolidated matrix-group proof documents
in this workspace.

- [`all_new_matrix_groups_proofs_integrated.pdf`](all_new_matrix_groups_proofs_integrated.pdf)
  is the single consolidated proof anthology. It contains each canonical proof
  once, including the power-character and bounded-support stability theorem.
  [`all_new_matrix_groups_proofs_integrated.tex`](all_new_matrix_groups_proofs_integrated.tex)
  is its standalone monolithic source: all proof bodies are physically inlined,
  and it contains no `input`, `include`, or embedded-PDF dependency.
- [`all_new_matrix_groups_formulas.tex`](all_new_matrix_groups_formulas.tex)
  and its compiled PDF form the cross-family formula compendium.
- [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) summarizes the mathematical and
  computational outcomes and records the latest regression run.

Executable sources live in the library and the notebook collection:

- `lambda_distributions/proofs/` holds the per-family verification backends,
  regression suites, shared modules, and (for now) the standalone
  proof-and-verification tex/pdf sources for each proved group family.
- `lambda_distributions/dists/` holds the same for the new-distribution
  families, including the compact Spin/Pin checks.
- `marimo_notebooks/` holds every interactive verification and laboratory
  notebook, named after its group or family.
