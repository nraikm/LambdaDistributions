# Association-scheme permutation sigma-MGFs

This package accompanies the standalone proof note
`association_scheme_sigma_mgfs_proof_and_verification.pdf`.

It groups the formulas by the mechanism used to compute power fixed points:

- homogeneous and parabolic actions (Johnson, Grassmann, polar, building);
- product and wreath actions (Hamming and Doob factors);
- translation actions (form schemes, affine polar, folded and halved cubes);
- graph automorphism actions (Shrikhande, rook, and Doob examples).

Run the exact suite with:

```bash
python3 -m lambda_distributions.dists.association_scheme_sigma_mgfs.verification
```

Run the marimo laboratory with:

```bash
marimo run marimo_notebooks/association_schemes.py
```

The verification constructs the finite permutation matrices and compares
matrix traces, cycle-factor coefficients, direct configuration orbits, and
the family-specific power-fixed-point formulas.  The default suite performs
99 exact coefficient comparisons.
