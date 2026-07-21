# Association-scheme permutation representations

This folder tests the four families in the supplied note as genuinely finite
matrix groups.  Each family has its own marimo notebook, calculation module,
LaTeX proof, and compiled PDF.

The coefficient

\[
[m_\tau]\,\mathcal S_{G,X}
=\dim\left(\bigotimes_j\operatorname{Sym}^{\tau_j}\mathbb C[X]\right)^G
\]

is computed in three independent ways:

1. Newton's recurrence applied to traces of powers of the explicitly built
   permutation matrices;
2. coefficient extraction from the proposed cycle product;
3. direct orbit enumeration on tuples of multisets of vertices.

In addition, every cycle count is reconstructed from the family-specific
fixed-point formula by Möbius inversion.  The sweep contains 66 exact
coefficient comparisons, all through total degree three.

Run everything from the repository root:

```sh
pytest -q for_this_guy/association_scheme_permutation_representations/test_verification.py
```

Run an individual notebook:

```sh
marimo run marimo_notebooks/johnson_scheme.py
marimo run marimo_notebooks/hamming_scheme.py
marimo run marimo_notebooks/grassmann_scheme.py
marimo run marimo_notebooks/polar_scheme.py
```

The stable copies of the PDFs are under
`output/pdf/for_this_guy/association_scheme_permutation_representations/`.

