# Barnes--Wall lattice automorphism groups

This folder separates the three group layers in Problem 11 while sharing the
same transparent matrix routines in `common.py`.

- `extraspecial_real_pauli/`: formula (11.5), tested for `m=1,2,3`.
- `real_clifford/`: formulas (11.7)--(11.13), tested for `m=1,2`.
- `barnes_wall/`: the index-two/twisted formula (11.19), tested on
  `BW_2 = Z^2` and `BW_4 = D4`. The exceptional `m=3` case is not inferred.

Run a verifier directly with `.venv/bin/python <group>/verification.py`. Open
the corresponding canonical app listed in the top-level
`marimo_notebooks/README.md` with `.venv/bin/marimo edit` from the repository
root.
Each group also has a compiled proof-and-verification PDF under
`output/pdf/for_this_guy/barnes_wall_lattice_automorphism_groups/`.
