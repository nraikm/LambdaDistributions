# Exact finite-n Molien verification

The material is separated by matrix-group family.

## Monomial reflection groups `G(r,p,n)`

- Marimo notebook: `marimo_notebooks/g_r_p_n.py`
- Standalone verification engine: `g_r_p_n/verification.py`
- Proof PDF: `../../output/pdf/for_this_guy/exact_finite_n_molien/g_r_p_n/g_r_p_n_proof_and_verification.pdf`
- LaTeX source: `../../output/pdf/for_this_guy/exact_finite_n_molien/g_r_p_n/g_r_p_n_proof_and_verification.tex`

This includes the signed permutation groups `W(B_n/C_n)=G(2,1,n)` and
`W(D_n)=G(2,2,n)`.

## General wreath products `H wr S_n`

- Marimo notebook: `marimo_notebooks/wreath_product.py`
- Standalone verification engine: `wreath_products/verification.py`
- Proof PDF: `../../output/pdf/for_this_guy/exact_finite_n_molien/wreath_products/wreath_product_proof_and_verification.pdf`
- LaTeX source: `../../output/pdf/for_this_guy/exact_finite_n_molien/wreath_products/wreath_product_proof_and_verification.tex`

## Run

From the repository root:

```sh
marimo edit marimo_notebooks/g_r_p_n.py
marimo edit marimo_notebooks/wreath_product.py
```

The adjacent `verification.py` files can also be run directly for the fixed
representative suites.
