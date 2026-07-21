"""Tools for experimenting with Lambda-distributions of finite matrix groups.

The public API is intentionally small: construct a group, evaluate a
distribution, and compare the result with a proposed formula.
"""

from .core import Partition, integer_partitions, normalize_partition, partitions_up_to, z_partition
from .distribution import (
    average_homogeneous,
    homogeneous_value,
    lambda_distribution,
    power_sum_moment,
    power_sum_value,
    scalar_sigma_monomial_coefficient,
    scalar_sigma_monomial_coefficients,
    sigma_mgf_coefficients,
)
from .groups import (
    FiniteMatrixGroup,
    alternating_group,
    cyclic_character,
    cyclic_permutation_group,
    cyclic_real_2d,
    dicyclic_group,
    dihedral_group,
    generalized_symmetric_group,
    pauli_group,
)
from .verification import Check, VerificationReport, check_formula
from .sampling import is_symplectic, is_unitary, log_sigma_mgf, random_symplectic, random_unitary, sigma_mgf

__all__ = [
    "Check",
    "FiniteMatrixGroup",
    "Partition",
    "VerificationReport",
    "alternating_group",
    "average_homogeneous",
    "check_formula",
    "cyclic_character",
    "cyclic_permutation_group",
    "cyclic_real_2d",
    "dicyclic_group",
    "dihedral_group",
    "generalized_symmetric_group",
    "homogeneous_value",
    "integer_partitions",
    "is_symplectic",
    "is_unitary",
    "lambda_distribution",
    "normalize_partition",
    "partitions_up_to",
    "pauli_group",
    "power_sum_moment",
    "power_sum_value",
    "scalar_sigma_monomial_coefficient",
    "scalar_sigma_monomial_coefficients",
    "random_symplectic",
    "random_unitary",
    "log_sigma_mgf",
    "sigma_mgf",
    "sigma_mgf_coefficients",
    "z_partition",
]
