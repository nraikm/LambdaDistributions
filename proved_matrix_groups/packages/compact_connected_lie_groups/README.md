# Compact connected Lie-group verification

This folder is the proof-and-computation companion for the compact-group
sigma-MGF statement supplied on 17 July 2026.

Each group has its own marimo notebook and LaTeX/PDF note.  The shared
`verification_core.py` constructs root and weight systems, constructs explicit
diagonal maximal-torus matrices, evaluates low-degree Schur characters, and
applies the Weyl integral exactly via the Weyl-denominator identity.

Run any notebook with, for example:

```bash
.venv/bin/marimo edit marimo_notebooks/lie_group_g2.py
```

The directly executed representative cases are:

- `G2`: minimal 7, through total degree 3;
- `Spin(7)`: spin 8, through total degree 4 (including the Cayley form);
- `F4`: minimal 26, through total degree 3;
- `E6`: minimal 27, through total degree 3;
- `E7`: minimal 56, through total degree 2, with the quartic strategy documented;
- `E8`: adjoint 248, through total degree 2, with the alternating cubic strategy documented;
- `SU(n)`: adjoint modules for `n=2,3,4`, through total degree 3.

Every notebook also checks an explicit unitary torus matrix and verifies the
determinant integrand against the product over its eigenvalues.

The tables distinguish two bases.  The Schur coefficient indexed by
`lambda` is `dim(S_lambda V)^G`.  The coefficient requested in the supplied
formula is the monomial coefficient `[m_tau]`, computed independently as
`dim(tensor_i Sym^{tau_i}(V))^G`.  Both are obtained from exact Weyl constant
terms, not from the decomposition tables quoted in the PDFs.
