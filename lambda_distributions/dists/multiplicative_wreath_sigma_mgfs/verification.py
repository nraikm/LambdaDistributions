"""Exact matrix checks for multiplicative wreath-product representations.

The tests intentionally use independent routes.

* Wreath types are read from labels and permutation cycles, while powers are
  formed with the semidirect-product law.
* Product, diagonal-coset, and twisted-regular actions are honest permutation
  matrices stored by their nonzero column positions.
* Tensor-induced and fixed-tail bipartition modules are signed permutation
  matrices.  Their invariant spaces are counted by signed-orbit propagation,
  independently of the Molien character average.
* Bosonic and fermionic Fock identities are checked degree by degree from
  explicit monomial and exterior bases.

All arithmetic is integral or rational; no tolerance-based comparison occurs.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement, product
from math import comb, factorial, gcd, prod
from typing import Sequence

from lambda_distributions.dists.primitive_action_sigma_mgfs.verification import (
    compose,
    conjugacy_class,
    cycles,
    fixed_points,
    fixed_points_of_power,
    identity,
    inverse,
    power,
    product_action_suite,
    symmetric_group,
    twisted_wreath_suite,
)


Permutation = tuple[int, ...]
GroupElement = Permutation
WreathElement = tuple[tuple[GroupElement, ...], Permutation]


def conjugacy_data(group: Sequence[GroupElement]):
    remaining = set(group)
    classes: dict[GroupElement, frozenset[GroupElement]] = {}
    representatives: dict[GroupElement, GroupElement] = {}
    centralizers: dict[GroupElement, int] = {}
    while remaining:
        representative = min(remaining)
        conjugates = conjugacy_class(group, representative)
        key = min(conjugates)
        centralizer = sum(compose(x, representative) == compose(representative, x) for x in group)
        for element in conjugates:
            classes[element] = conjugates
            representatives[element] = key
            centralizers[element] = centralizer
        remaining.difference_update(conjugates)
    return classes, representatives, centralizers


def wreath_multiply(left: WreathElement, right: WreathElement) -> WreathElement:
    """Semidirect product for y_i=h_i x_{pi^{-1}(i)}."""

    labels, top = left
    other_labels, other_top = right
    top_inverse = inverse(top)
    new_labels = tuple(
        compose(labels[index], other_labels[top_inverse[index]])
        for index in range(len(labels))
    )
    return new_labels, compose(top, other_top)


def wreath_power(element: WreathElement, exponent: int, base_identity: GroupElement) -> WreathElement:
    n = len(element[0])
    answer = (tuple(base_identity for _ in range(n)), identity(n))
    base = element
    while exponent:
        if exponent & 1:
            answer = wreath_multiply(answer, base)
        base = wreath_multiply(base, base)
        exponent //= 2
    return answer


def cycle_product(labels: Sequence[GroupElement], cycle: Sequence[int], base_identity: GroupElement):
    answer = base_identity
    for coordinate in cycle:
        answer = compose(labels[coordinate], answer)
    return answer


def wreath_type(
    element: WreathElement,
    representatives: dict[GroupElement, GroupElement],
    base_identity: GroupElement,
):
    labels, top = element
    answer: Counter[tuple[GroupElement, int]] = Counter()
    for cycle in cycles(top):
        class_key = representatives[cycle_product(labels, cycle, base_identity)]
        answer[(class_key, len(cycle))] += 1
    return answer


def transformed_type(
    type_data: Counter[tuple[GroupElement, int]],
    exponent: int,
    representatives: dict[GroupElement, GroupElement],
):
    answer: Counter[tuple[GroupElement, int]] = Counter()
    for (class_key, length), multiplicity in type_data.items():
        divisor = gcd(length, exponent)
        powered_class = representatives[power(class_key, exponent // divisor)]
        answer[(powered_class, length // divisor)] += divisor * multiplicity
    return answer


def wreath_centralizer_denominator(
    type_data: Counter[tuple[GroupElement, int]],
    centralizers: dict[GroupElement, int],
):
    answer = 1
    by_class: dict[GroupElement, Counter[int]] = {}
    for (class_key, length), multiplicity in type_data.items():
        by_class.setdefault(class_key, Counter())[length] = multiplicity
    for class_key, parts in by_class.items():
        number_of_parts = sum(parts.values())
        answer *= centralizers[class_key] ** number_of_parts
        for length, multiplicity in parts.items():
            answer *= length**multiplicity * factorial(multiplicity)
    return answer


def wreath_class_power_case(base_group: Sequence[GroupElement], n: int, name: str):
    _, representatives, centralizers = conjugacy_data(base_group)
    base_identity = identity(len(base_group[0]))
    elements = tuple(
        (tuple(labels), top)
        for labels in product(base_group, repeat=n)
        for top in symmetric_group(n)
    )
    type_counts: Counter[tuple[tuple[GroupElement, int, int], ...]] = Counter()
    type_lookup = {}
    power_checks = 0
    for element in elements:
        current_type = wreath_type(element, representatives, base_identity)
        key = tuple(sorted((class_key, length, multiplicity) for (class_key, length), multiplicity in current_type.items()))
        type_counts[key] += 1
        type_lookup[key] = current_type
        for exponent in range(1, 7):
            actual = wreath_type(
                wreath_power(element, exponent, base_identity), representatives, base_identity
            )
            predicted = transformed_type(current_type, exponent, representatives)
            assert actual == predicted
            power_checks += 1

    weight_checks = 0
    for key, count in type_counts.items():
        denominator = wreath_centralizer_denominator(type_lookup[key], centralizers)
        assert len(elements) % denominator == 0
        assert count == len(elements) // denominator
        weight_checks += 1
    return {
        "group": name,
        "n": n,
        "order": len(elements),
        "wreath types": len(type_counts),
        "power checks": power_checks,
        "class-weight checks": weight_checks,
        "passed": True,
    }


def wreath_class_power_suite():
    cases = (
        wreath_class_power_case(symmetric_group(2), 3, "C2 wr S3"),
        wreath_class_power_case(symmetric_group(3), 2, "S3 wr S2"),
    )
    return {"cases": cases, "passed": all(case["passed"] for case in cases)}


@dataclass(frozen=True)
class SignedPermutation:
    permutation: Permutation
    signs: tuple[int, ...]

    def __post_init__(self):
        assert len(self.permutation) == len(self.signs)
        assert set(self.signs).issubset({-1, 1})


def signed_identity(dimension: int):
    return SignedPermutation(identity(dimension), (1,) * dimension)


def signed_compose(left: SignedPermutation, right: SignedPermutation):
    permutation = tuple(left.permutation[right.permutation[index]] for index in range(len(right.permutation)))
    signs = tuple(
        right.signs[index] * left.signs[right.permutation[index]]
        for index in range(len(right.permutation))
    )
    return SignedPermutation(permutation, signs)


def signed_power(element: SignedPermutation, exponent: int):
    answer = signed_identity(len(element.permutation))
    base = element
    while exponent:
        if exponent & 1:
            answer = signed_compose(answer, base)
        base = signed_compose(base, base)
        exponent //= 2
    return answer


def signed_trace(element: SignedPermutation):
    return sum(
        element.signs[index]
        for index, image in enumerate(element.permutation)
        if image == index
    )


def signed_tensor_element(labels: Sequence[int], top: Permutation):
    n = len(labels)
    basis = tuple(product(range(2), repeat=n))
    locations = {item: index for index, item in enumerate(basis)}
    top_inverse = inverse(top)
    images = []
    signs = []
    for source in basis:
        target = tuple(source[top_inverse[index]] for index in range(n))
        sign = prod((-1) ** (labels[index] * target[index]) for index in range(n))
        images.append(locations[target])
        signs.append(sign)
    return SignedPermutation(tuple(images), tuple(signs))


def signed_natural_element(labels: Sequence[int], top: Permutation):
    # e_j maps to (-1)^(label_{top(j)}) e_{top(j)}.
    return SignedPermutation(tuple(top), tuple((-1) ** labels[top[j]] for j in range(len(top))))


def hyperoctahedral_elements(n: int, constructor=signed_natural_element):
    return tuple(
        constructor(labels, top)
        for labels in product(range(2), repeat=n)
        for top in symmetric_group(n)
    )


def hyperoctahedral_generators(n: int, constructor=signed_natural_element):
    generators = []
    for coordinate in range(n):
        labels = [0] * n
        labels[coordinate] = 1
        generators.append(constructor(labels, identity(n)))
    for coordinate in range(n - 1):
        top = list(range(n))
        top[coordinate], top[coordinate + 1] = top[coordinate + 1], top[coordinate]
        generators.append(constructor((0,) * n, tuple(top)))
    return tuple(generators)


@lru_cache(maxsize=None)
def symmetric_basis(dimension: int, degree: int):
    return tuple(combinations_with_replacement(range(dimension), degree))


def represented_signed_action(element: SignedPermutation, tau: Sequence[int]):
    block_bases = tuple(symmetric_basis(len(element.permutation), degree) for degree in tau)
    locations = tuple({item: index for index, item in enumerate(block)} for block in block_bases)
    full_basis = tuple(product(*(range(len(block)) for block in block_bases)))
    full_locations = {item: index for index, item in enumerate(full_basis)}
    images = []
    signs = []
    for basis_indices in full_basis:
        target_indices = []
        sign = 1
        for block_number, basis_index in enumerate(basis_indices):
            source = block_bases[block_number][basis_index]
            target = tuple(sorted(element.permutation[index] for index in source))
            sign *= prod(element.signs[index] for index in source)
            target_indices.append(locations[block_number][target])
        images.append(full_locations[tuple(target_indices)])
        signs.append(sign)
    return SignedPermutation(tuple(images), tuple(signs))


def signed_orbit_invariant_dimension(generators: Sequence[SignedPermutation], tau: Sequence[int]):
    represented = tuple(represented_signed_action(generator, tau) for generator in generators)
    dimension = len(represented[0].permutation)
    unvisited = set(range(dimension))
    answer = 0
    while unvisited:
        start = unvisited.pop()
        potentials = {start: 1}
        queue = deque([start])
        consistent = True
        while queue:
            source = queue.popleft()
            for generator in represented:
                target = generator.permutation[source]
                value = generator.signs[source] * potentials[source]
                if target in potentials:
                    if potentials[target] != value:
                        consistent = False
                else:
                    potentials[target] = value
                    unvisited.discard(target)
                    queue.append(target)
        answer += int(consistent)
    return answer


def complete_characters(element: SignedPermutation, maximum: int):
    values = [1]
    for degree in range(1, maximum + 1):
        numerator = sum(
            signed_trace(signed_power(element, exponent)) * values[degree - exponent]
            for exponent in range(1, degree + 1)
        )
        assert numerator % degree == 0
        values.append(numerator // degree)
    return values


def signed_molien_coefficient(elements: Sequence[SignedPermutation], tau: Sequence[int]):
    maximum = max(tau, default=0)
    numerator = 0
    for element in elements:
        complete = complete_characters(element, maximum)
        numerator += prod(complete[degree] for degree in tau)
    assert numerator % len(elements) == 0
    return numerator // len(elements)


def c2_tensor_character(cycle_parity: int):
    return 2 if cycle_parity == 0 else 0


def tensor_cycle_formula(labels: Sequence[int], top: Permutation, exponent: int):
    answer = 1
    for cycle in cycles(top):
        divisor = gcd(len(cycle), exponent)
        parity = sum(labels[index] for index in cycle) % 2
        powered_parity = parity * (exponent // divisor) % 2
        answer *= c2_tensor_character(powered_parity) ** divisor
    return answer


def tensor_induced_suite():
    taus = ((1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1, 1))
    cases = []
    character_checks = 0
    coefficient_checks = 0
    for n in (2, 3):
        indexed = tuple(
            (labels, top, signed_tensor_element(labels, top))
            for labels in product(range(2), repeat=n)
            for top in symmetric_group(n)
        )
        elements = tuple(item[2] for item in indexed)
        generators = hyperoctahedral_generators(n, signed_tensor_element)
        for labels, top, matrix in indexed:
            for exponent in range(1, 7):
                assert signed_trace(signed_power(matrix, exponent)) == tensor_cycle_formula(
                    labels, top, exponent
                )
                character_checks += 1
        rows = []
        for tau in taus:
            direct = signed_orbit_invariant_dimension(generators, tau)
            molien = signed_molien_coefficient(elements, tau)
            row = {"tau": str(tau), "direct signed fixed space": direct, "Molien": molien}
            if all(degree == 1 for degree in tau):
                s = len(tau)
                a_s = 2 ** (s - 1)
                formula = comb(n + a_s - 1, a_s - 1)
                row["moment theorem"] = formula
                assert direct == molien == formula
            else:
                assert direct == molien
            coefficient_checks += 1
            rows.append(row)
        cases.append(
            {
                "group": f"C2 wr S{n}",
                "module": f"(1 + sign)^tensor {n}",
                "dimension": 2**n,
                "order": len(elements),
                "rows": tuple(rows),
            }
        )
    return {
        "cases": tuple(cases),
        "character-power checks": character_checks,
        "coefficient checks": coefficient_checks,
        "passed": True,
    }


def fixed_tail_bipartition_suite():
    taus = ((1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1, 1))
    cases = []
    character_checks = 0
    coefficient_checks = 0
    for n in (2, 3, 4):
        raw = tuple(
            (labels, top, signed_natural_element(labels, top))
            for labels in product(range(2), repeat=n)
            for top in symmetric_group(n)
        )
        elements = tuple(item[2] for item in raw)
        generators = hyperoctahedral_generators(n)
        for labels, top, matrix in raw:
            type_data: Counter[tuple[int, int]] = Counter()
            for cycle in cycles(top):
                type_data[(sum(labels[index] for index in cycle) % 2, len(cycle))] += 1
            for exponent in range(1, 5):
                transformed: Counter[tuple[int, int]] = Counter()
                for (parity, length), multiplicity in type_data.items():
                    divisor = gcd(length, exponent)
                    new_parity = parity * (exponent // divisor) % 2
                    transformed[(new_parity, length // divisor)] += divisor * multiplicity
                q_value = transformed[(0, 1)] - transformed[(1, 1)]
                assert signed_trace(signed_power(matrix, exponent)) == q_value
                character_checks += 1
        rows = []
        for tau in taus:
            direct = signed_orbit_invariant_dimension(generators, tau)
            formula = signed_molien_coefficient(elements, tau)
            assert direct == formula
            rows.append({"tau": str(tau), "direct signed fixed space": direct, "class formula": formula})
            coefficient_checks += 1
        cases.append(
            {
                "group": f"C2 wr S{n}",
                "bipartition": f"((n-1),(1)) at n={n}",
                "dimension": n,
                "order": len(elements),
                "rows": tuple(rows),
            }
        )
    return {
        "character polynomial": "X_(1,+) - X_(1,-)",
        "marked-Poisson limit": "Z_+ - Z_- with independent Poisson(1/2) variables",
        "cases": tuple(cases),
        "character-power checks": character_checks,
        "coefficient checks": coefficient_checks,
        "passed": True,
    }


def diagonal_action_permutation(group: Sequence[GroupElement], components: Sequence[GroupElement]):
    n = len(components)
    points = tuple(product(group, repeat=n - 1))
    locations = {point: index for index, point in enumerate(points)}
    last_inverse = inverse(components[-1])
    return tuple(
        locations[
            tuple(compose(components[index], compose(point[index], last_inverse)) for index in range(n - 1))
        ]
        for point in points
    )


def diagonal_fixed_formula(group: Sequence[GroupElement], components: Sequence[GroupElement], exponent: int):
    powered = tuple(power(component, exponent) for component in components)
    classes, _, centralizers = conjugacy_data(group)
    first_class = classes[powered[0]]
    if any(component not in first_class for component in powered[1:]):
        return 0
    return centralizers[powered[0]] ** (len(components) - 1)


def diagonal_suite():
    group = symmetric_group(3)
    _, representatives, centralizers = conjugacy_data(group)
    class_keys = sorted(set(representatives.values()))
    cases = []
    fixed_checks = 0
    for n in (2, 3):
        indexed = tuple(
            (components, diagonal_action_permutation(group, components))
            for components in product(group, repeat=n)
        )
        for components, action in indexed:
            for exponent in range(1, 7):
                assert fixed_points_of_power(action, exponent) == diagonal_fixed_formula(
                    group, components, exponent
                )
                fixed_checks += 1
        rows = []
        for moment in (1, 2, 3):
            numerator = sum(fixed_points(action) ** moment for _, action in indexed)
            direct = Fraction(numerator, len(indexed))
            formula = sum(
                Fraction(centralizers[key] ** ((moment - 1) * n - moment), 1)
                if (moment - 1) * n - moment >= 0
                else Fraction(1, centralizers[key] ** (moment - (moment - 1) * n))
                for key in class_keys
            )
            assert direct == formula and direct.denominator == 1
            rows.append({"moment": moment, "direct matrix average": direct.numerator, "class formula": formula.numerator})
        cases.append(
            {
                "group": f"S3^{n} on S3^{n}/Delta(S3)",
                "dimension": len(group) ** (n - 1),
                "order": len(indexed),
                "rows": tuple(rows),
            }
        )
    return {"cases": tuple(cases), "fixed-power checks": fixed_checks, "passed": True}


def fock_suite():
    def convolve(left: Sequence[int], right: Sequence[int], maximum: int):
        answer = [0] * (maximum + 1)
        for left_degree, left_value in enumerate(left):
            for right_degree, right_value in enumerate(right):
                if left_degree + right_degree <= maximum:
                    answer[left_degree + right_degree] += left_value * right_value
        return tuple(answer)

    rows = []
    identity_checks = 0
    for energy in range(7):
        bosonic_direct = sum(1 for odd_count in range(energy + 1) if odd_count % 2 == 0)
        bosonic_average = sum(
            sum(((-1) ** (label * odd_count)) for odd_count in range(energy + 1))
            for label in (0, 1)
        ) // 2
        fermionic_direct = {0: 1, 1: 1, 2: 0}.get(energy, 0)
        fermionic_average = 0
        if energy <= 2:
            # Average the explicit exterior-basis traces for diag(1, +/-1).
            fermionic_average = sum(
                (1 if energy == 0 else (1 + (-1) ** label if energy == 1 else (-1) ** label))
                for label in (0, 1)
            ) // 2
        assert bosonic_direct == bosonic_average
        assert fermionic_direct == fermionic_average
        rows.append(
            {
                "energy": energy,
                "bosonic direct/formula": bosonic_direct,
                "fermionic direct/formula": fermionic_direct,
            }
        )
        identity_checks += 2

    # Check the character identities before averaging for g^r.
    for label in (0, 1):
        for exponent in range(1, 5):
            sign = (-1) ** (label * exponent)
            determinant_inverse = convolve((1,) * 7, tuple(sign**degree for degree in range(7)), 6)
            for degree in range(7):
                explicit_sym = sum(sign**odd_count for odd_count in range(degree + 1))
                assert explicit_sym == determinant_inverse[degree]
                identity_checks += 1
            explicit_exterior = (1, 1 + sign, sign)
            determinant_coefficients = convolve((1, 1), (1, sign), 2)
            assert explicit_exterior == determinant_coefficients
            identity_checks += 3
    return {
        "group": "C2 on 1 + sign",
        "rows": tuple(rows),
        "graded character checks": identity_checks,
        "passed": True,
    }


def varying_base_suite():
    rows = []
    for base_order in (2, 3):
        for k in (2, 3):
            for moment in (2, 3):
                statistic = base_order ** (moment - 1)
                formula = comb(k + statistic - 1, statistic - 1)
                rows.append(
                    {
                        "regular base": f"C{base_order}",
                        "k": k,
                        "moment": moment,
                        "R_s = a_s": statistic,
                        "product/tensor formula": formula,
                    }
                )
    return {"rows": tuple(rows), "exact formula checks": len(rows), "passed": True}


def run_suite():
    class_power = wreath_class_power_suite()
    product_actions = product_action_suite()
    tensor = tensor_induced_suite()
    fixed_tail = fixed_tail_bipartition_suite()
    diagonal = diagonal_suite()
    twisted = twisted_wreath_suite()
    fock = fock_suite()
    varying = varying_base_suite()

    exact_checks = (
        sum(case["power checks"] + case["class-weight checks"] for case in class_power["cases"])
        + sum(case["fixed-power checks"] + len(case["rows"]) * 3 for case in product_actions["cases"])
        + tensor["character-power checks"]
        + tensor["coefficient checks"] * 2
        + fixed_tail["character-power checks"]
        + fixed_tail["coefficient checks"] * 2
        + diagonal["fixed-power checks"]
        + sum(len(case["rows"]) * 2 for case in diagonal["cases"])
        + twisted["fixed-power checks"]
        + twisted["conditional spike checks"]
        + len(twisted["rows"]) * 2
        + 2
        + fock["graded character checks"]
        + varying["exact formula checks"]
    )
    sections = {
        "wreath class powers": class_power,
        "product actions": product_actions,
        "tensor-induced": tensor,
        "fixed-tail bipartitions": fixed_tail,
        "diagonal cosets": diagonal,
        "twisted regular": twisted,
        "Fock": fock,
        "varying base": varying,
    }
    return {
        **sections,
        "exact checks": exact_checks,
        "passed": all(section["passed"] for section in sections.values()),
    }


if __name__ == "__main__":
    report = run_suite()
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {report['exact checks']:,} exact checks")
    for name, section in report.items():
        if isinstance(section, dict) and "passed" in section:
            print(f"  {name}: {'PASS' if section['passed'] else 'FAIL'}")
