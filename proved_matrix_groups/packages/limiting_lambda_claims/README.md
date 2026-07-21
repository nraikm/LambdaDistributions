# Three limiting Lambda-distribution claims

This directory turns the three claims in the supplied note into separate,
reproducible group-family checks.

- `symmetric_k_subsets/` proves and exhaustively checks the complete claim for
  the action of `S_n` on `k`-subsets.
- `rooted_tree_automorphisms/` proves and checks the exact wreath/plethystic
  recursion.  It explicitly does **not** claim the still-open general random
  tree Lambda-limit.
- The related Barnes--Wall files remain grouped in
  `../barnes_wall_lattice_automorphism_groups/`, separated into extraspecial
  Pauli, real Clifford, and Barnes--Wall lattice layers.

Each group folder contains a plain Python verifier, a marimo notebook, and a
LaTeX proof-and-verification note.  Compiled PDFs are copied beneath
`output/pdf/for_this_guy/` with the same family structure.
