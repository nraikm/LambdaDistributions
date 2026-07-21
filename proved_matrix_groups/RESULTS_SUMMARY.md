# Results summary

## Overall result

All proved matrix-group work is now consolidated across two homes: the
executable verification backends, regression suites, and standalone proof
sources live in the library under `lambda_distributions/proofs/` (new-dist
families under `lambda_distributions/dists/`), the interactive notebooks live
in `marimo_notebooks/`, and the consolidated documents live in
`proved_matrix_groups/`. The complete proof anthology is
`all_new_matrix_groups_proofs_integrated.pdf`. The cross-family formula index
is `all_new_matrix_groups_formulas.tex` (with a compiled PDF beside it).

The regression suites were rerun independently on 18 July 2026 to avoid
pytest module-name collisions between packages. **All 78 tests passed:** 64
tests in 14 matrix-group package suites and 14 core workspace tests. No active
task paths were excluded. The merged machine-readable record is
`verification_results.xml`; environment, lockfile, tolerance, and external
data checksums are in `verification_manifest.json`.

Across the collection, the recurring proof pattern is to construct the
stated matrices, compute the generalized Molien/sigma-MGF coefficient by a
Reynolds projector, invariant-space kernel, constant term, or orbit count,
and compare it with an independent character, cycle-index, fixed-point,
weight-lattice, Weyl-integral, or plethystic formula.

## Rerun suites

| Family | Tests | Result | Main verified content |
|---|---:|---|---|
| Association-scheme permutation representations | 4 | Pass | Johnson, Hamming, Grassmann, and polar actions; 66 exact coefficient comparisons plus fixed-point/Mobius cycle reconstruction. |
| Compact connected Lie groups | 4 | Pass | Weyl constant terms for `G2`, `Spin(7)`, `F4`, `E6`, `E7`, `E8`, and adjoint `SU(n)`, with torus-determinant checks. |
| Finite braid images | 3 | Pass | Cyclic scalar, `S3` permutation, and Ising/Clifford images; Reynolds ranks agree with elementwise and class Molien routes. |
| Finite-group exact formulas | 17 | Pass | Dicyclic/generalized quaternion, symmetric, and alternating groups; projectors agree with character, cycle/weight, and orbit formulas. |
| Hyperoctahedral action on the binary cube | 1 | Pass | Explicit `2^n`-dimensional permutation matrices agree with the signed-cycle law for every tested `n <= 5`. |
| Finite-group sigma-MGF | 6 | Pass | Regular representations, selected `S_n` representations, and `S_n/A_n` subset actions. |
| Finite Lie-type permutation actions | 3 | Pass | Linear, projective, flag, symplectic, orthogonal, and unitary actions over finite fields, including degree-three polar formed-space comparisons in consecutive Witt ranks. |
| Generalized Molien extensions | 3 | Pass | Code-monomial groups, `Delta(3n^2)/Delta(6n^2)`, and `PSL(2,q)` projective-line modules. |
| Local-unitary tensor products | 5 | Pass | One-sided and balanced two-sided sigma-MGFs for tensor-product representations of products of unitary groups. |
| Matrix-group formula verification | 3 | Pass | 928 comparisons across compact classical groups, tori, `H3`/polyhedral groups, restricted monomial groups, representation variants, `A5`, and binary polyhedral groups. |
| Nonuniform Lambda distributions | 3 | Pass | Cyclic walks, nonuniform measures on `S_n`, and `U(1)` heat-kernel interpolation. |
| Schur functors for `U/O/USp` | 4 | Pass | 120 Young-symmetrizer/plethystic comparisons and the stable classical-group formulas, with finite-rank qualifications. |
| Unitary conjugation on `End(C^n)` | 4 | Pass | Direct kernels agree with Littlewood--Richardson, stable orbit-count, and necklace/Euler-product formulas. |
| Sporadic matrix groups | 2 | Pass | `M24` generator-orbit checks agree with the checksummed ATLAS class table; the signed-coordinate `Co0` stress test verifies central parity and even coefficients. |
| Core workspace regression | 14 | Pass | Baseline Lambda-distribution functionality remains intact after the moves. |

## Consolidated formula and group families

- **Finite abelian and monomial:** the diagonal zero-weight/congruence theorem;
  `G(r,p,n)`; types `B/C` and `D`; and general wreath products `H wr S_n`.
- **Dihedral and permutation:** generalized dihedral and dicyclic groups;
  exact `S_n` and `A_n` formulas; sign, standard, subset, and general
  permutation actions; and the hyperoctahedral group on the binary cube.
- **Affine, Lie-type, and exceptional finite actions:** `AGL_n(q)`, finite
  classical point/line/flag actions, `PSL(2,q)`, `Delta(3n^2)`,
  `Delta(6n^2)`, and code-monomial groups.
- **Quantum and Clifford:** finite braid images, qudit Pauli--Heisenberg and
  scalar-extended Pauli groups, extraspecial groups, Clifford groups, real
  Clifford groups, and Barnes--Wall layers.
- **Compact and classical:** `U(n)`, `SU(n)`, `SO(n)`, `O(n)`, tori,
  `G2`, `Spin(7)`, `F4`, `E6`, `E7`, `E8`, adjoint `SU(n)`, and Schur
  functors for `U/O/USp`.
- **Tensor and conjugation actions:** local-unitary tensor products, balanced
  two-sided series, and unitary conjugation on endomorphism spaces.
- **Qualified noncompact and sporadic cases:** Proctor's odd symplectic group
  via maximal-compact/finite-field substitutes; polyhedral groups, `M24`, and
  `Co0`/Leech frame-shape analysis.
- **Probability and limits:** nonuniform cyclic and symmetric-group measures,
  heat kernels, fixed-size subset limits, and exact rooted-tree recursions.

## Principal formulas

The common finite-group object is

```text
M_G(u,t) = |G|^{-1} sum_g [u^0] exp(sum_{r>=1} u^r tr(g^r)/r)
                              / det(I-tg),
```

with the compact analogue obtained by Haar integration. The main reductions
proved across the packages include the diagonal zero-weight condition, the
`G(r,p,n)` constrained cycle/partition sum, the wreath-product cycle-index
composition, the generalized-dihedral spectral decomposition, the universal
permutation-action fixed-point formula, Weyl constant terms for compact
groups, and plethystic Schur-functor substitutions. The complete statement of
every new formula, including stable-range and probabilistic qualifications, is
in `all_new_matrix_groups_formulas.pdf`.

## Corrections and boundaries retained

- Orthogonal Schur shapes require `length(lambda) <= n`.
- A three-dimensional rotation contributes
  `(1-t)^(-1)(1-2 cos(theta)t+t^2)^(-1)`.
- The scalar-extended qubit Pauli noncentral factor has a plus sign.
- One-sided holomorphic series collapse when a continuous scalar center acts
  nontrivially; balanced two-sided series retain the invariant content.
- Proctor's complex odd symplectic group is noncompact, so no normalized Haar
  probability is asserted.
- Full `Co0` claims remain distinct from signed-coordinate subgroup checks.
- Exact finite-group, finite-rank, stable-range, numerical-Haar, and limiting
  claims are labeled separately throughout the compendium.
- For `S_n` on `k`-subsets, both quadratic coefficients equal `k+1` once
  `n >= 2k`. For `A_n`, the safe range is `n >= 2k+2`; at `(n,k)=(4,2)` the
  unordered coefficient is `3` and the ordered coefficient is `4`.
- Code-monomial formulas are stated for coordinate-permutation automorphisms;
  non-permutation monomial inputs are rejected by the verifier.
- E7 quartic and optional E8 sparse-cubic computations are not reported as
  executed; the E8 cubic row is instead proved algebraically.
- Numerical common-kernel and floating group-reconstruction diagnostics now
  report thresholds, spectral gaps, tolerances, and residuals.

## Organization

- `lambda_distributions/proofs/`: executable verification backends, regression
  suites, shared modules, and standalone proof sources for the proved
  matrix-group families.
- `lambda_distributions/dists/`: the same for the new-distribution families,
  including the compact Spin/Pin checks.
- `marimo_notebooks/`: every interactive verification and laboratory notebook,
  named after its group or family.
- `all_new_matrix_groups_proofs_integrated.pdf`: the single consolidated proof
  anthology; `all_new_matrix_groups_proofs_integrated.tex` is the standalone
  monolithic source.
- `all_new_matrix_groups_formulas.tex` and `.pdf`: cross-family theorem and
  formula compendium.
