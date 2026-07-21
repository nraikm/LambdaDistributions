"""Exact finite verification of primitive-action sigma-MGF formulas.

Every complex matrix representation in this module is a permutation
representation on ``C[X]``.  A permutation is stored by its row-sparse data
``p[i]`` rather than as a dense zero-one matrix.  This is exactly the matrix
representation: the corresponding matrix sends the basis vector ``e_i`` to
``e_{p[i]}``.

The suite deliberately uses two scales.

* ``A_5`` supplies the smallest nonabelian-simple examples for almost-simple,
  diagonal, and holomorph-simple actions.
* ``S_3`` supplies small exhaustive wreath-product benchmarks.  The fixed
  point identities used there are group-theoretic and do not require
  simplicity; using ``S_3`` lets the program enumerate every group element,
  every relevant power, and every target orbit.
"""

from __future__ import annotations

from collections import Counter, deque
from functools import lru_cache
from itertools import combinations_with_replacement, permutations, product
from math import comb, factorial, gcd, prod
from typing import Callable, Iterable, Sequence


Permutation = tuple[int, ...]
GroupElement = Permutation


def identity(size: int) -> Permutation:
    return tuple(range(size))


def compose(left: Permutation, right: Permutation) -> Permutation:
    """Return ``left o right``."""

    return tuple(left[right[index]] for index in range(len(left)))


def inverse(permutation: Permutation) -> Permutation:
    answer = [0] * len(permutation)
    for source, target in enumerate(permutation):
        answer[target] = source
    return tuple(answer)


def power(permutation: Permutation, exponent: int) -> Permutation:
    answer = identity(len(permutation))
    base = permutation
    while exponent:
        if exponent & 1:
            answer = compose(answer, base)
        base = compose(base, base)
        exponent //= 2
    return answer


def parity(permutation: Permutation) -> int:
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


@lru_cache(maxsize=None)
def symmetric_group(degree: int) -> tuple[Permutation, ...]:
    return tuple(permutations(range(degree)))


@lru_cache(maxsize=None)
def alternating_group(degree: int) -> tuple[Permutation, ...]:
    return tuple(g for g in symmetric_group(degree) if parity(g) == 1)


def cycle_counts(permutation: Permutation) -> Counter[int]:
    seen: set[int] = set()
    answer: Counter[int] = Counter()
    for start in range(len(permutation)):
        if start in seen:
            continue
        current = start
        length = 0
        while current not in seen:
            seen.add(current)
            current = permutation[current]
            length += 1
        answer[length] += 1
    return answer


def cycles(permutation: Permutation) -> tuple[tuple[int, ...], ...]:
    seen: set[int] = set()
    answer = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        cycle = []
        current = start
        while current not in seen:
            seen.add(current)
            cycle.append(current)
            current = permutation[current]
        answer.append(tuple(cycle))
    return tuple(answer)


def divisors(value: int) -> tuple[int, ...]:
    return tuple(d for d in range(1, value + 1) if value % d == 0)


def mobius(value: int) -> int:
    remaining = value
    prime_count = 0
    candidate = 2
    while candidate * candidate <= remaining:
        if remaining % candidate == 0:
            remaining //= candidate
            prime_count += 1
            if remaining % candidate == 0:
                return 0
            while remaining % candidate == 0:
                remaining //= candidate
        candidate += 1
    if remaining > 1:
        prime_count += 1
    return -1 if prime_count % 2 else 1


def fixed_points(permutation: Permutation) -> int:
    return sum(image == index for index, image in enumerate(permutation))


def fixed_points_of_power(permutation: Permutation, exponent: int) -> int:
    return sum(length * count for length, count in cycle_counts(permutation).items() if exponent % length == 0)


def mobius_cycles(permutation: Permutation) -> Counter[int]:
    maximum = max(cycle_counts(permutation), default=0)
    answer: Counter[int] = Counter()
    for length in range(1, maximum + 1):
        numerator = sum(
            mobius(length // divisor) * fixed_points_of_power(permutation, divisor)
            for divisor in divisors(length)
        )
        if numerator:
            assert numerator % length == 0
            answer[length] = numerator // length
    return answer


def complete_from_cycles(counts: Counter[int], degree: int) -> int:
    """Coefficient of t^degree in prod_r (1-t^r)^(-c_r)."""

    coefficients = [0] * (degree + 1)
    coefficients[0] = 1
    for length, count in counts.items():
        updated = [0] * (degree + 1)
        for old_degree, old_value in enumerate(coefficients):
            for copies in range((degree - old_degree) // length + 1):
                updated[old_degree + copies * length] += old_value * comb(
                    count + copies - 1, copies
                )
        coefficients = updated
    return coefficients[degree]


def molien_coefficient(action_group: Sequence[Permutation], tau: Sequence[int]) -> int:
    numerator = sum(
        prod(complete_from_cycles(cycle_counts(g), degree) for degree in tau)
        for g in action_group
    )
    assert numerator % len(action_group) == 0
    return numerator // len(action_group)


@lru_cache(maxsize=None)
def multisets(size: int, degree: int) -> tuple[tuple[int, ...], ...]:
    return tuple(combinations_with_replacement(range(size), degree))


def direct_configuration_orbits(
    action_group: Sequence[Permutation], tau: Sequence[int]
) -> int:
    blocks = tuple(multisets(len(action_group[0]), degree) for degree in tau)
    remaining = set(product(*blocks))
    answer = 0
    while remaining:
        representative = next(iter(remaining))
        orbit = {
            tuple(tuple(sorted(g[index] for index in block)) for block in representative)
            for g in action_group
        }
        remaining.difference_update(orbit)
        answer += 1
    return answer


def orbit_count_from_generators(
    size: int, generators: Sequence[Permutation], tuple_power: int = 1
) -> int:
    """Count orbits on ordered ``tuple_power``-tuples by graph traversal."""

    remaining = set(product(range(size), repeat=tuple_power))
    number = 0
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        while queue:
            point = queue.popleft()
            for generator in generators:
                image = tuple(generator[index] for index in point)
                if image in remaining:
                    remaining.remove(image)
                    queue.append(image)
        number += 1
    return number


def action_permutation(
    group_element, points: Sequence, action: Callable[[object, object], object]
) -> Permutation:
    locations = {point: index for index, point in enumerate(points)}
    return tuple(locations[action(group_element, point)] for point in points)


def conjugate(group_element: GroupElement, by: GroupElement) -> GroupElement:
    return compose(inverse(by), compose(group_element, by))


def conjugacy_class(group: Sequence[GroupElement], element: GroupElement):
    return frozenset(conjugate(element, by) for by in group)


def conjugacy_classes(group: Sequence[GroupElement]):
    remaining = set(group)
    answer = []
    while remaining:
        representative = next(iter(remaining))
        cls = conjugacy_class(group, representative)
        remaining.difference_update(cls)
        answer.append(cls)
    return tuple(sorted(answer, key=lambda cls: (len(cls), min(cls))))


def centralizer_size(group: Sequence[GroupElement], element: GroupElement) -> int:
    return sum(compose(x, element) == compose(element, x) for x in group)


def group_generators(name: str) -> tuple[Permutation, ...]:
    if name == "A5":
        return ((1, 2, 0, 3, 4), (1, 2, 3, 4, 0))
    if name == "S3":
        return ((1, 0, 2), (1, 2, 0))
    raise ValueError(name)


def representation_law(group: Sequence[Permutation]) -> bool:
    actions = set(group)
    return all(compose(left, right) in actions for left in group for right in group)


def almost_simple_suite() -> dict:
    group = alternating_group(5)
    subgroup = tuple(g for g in group if g[0] == 0)
    class_formula_checks = 0
    for g in group:
        for exponent in range(1, 7):
            powered = power(g, exponent)
            cls = conjugacy_class(group, powered)
            predicted_numerator = centralizer_size(group, powered) * len(cls.intersection(subgroup))
            assert predicted_numerator % len(subgroup) == 0
            predicted = predicted_numerator // len(subgroup)
            observed = fixed_points_of_power(g, exponent)
            assert observed == predicted
            assert mobius_cycles(g) == cycle_counts(g)
            class_formula_checks += 1

    coefficient_rows = []
    for tau in ((1,), (2,), (1, 1), (3,), (2, 1)):
        formula = molien_coefficient(group, tau)
        direct = direct_configuration_orbits(group, tau)
        coefficient_rows.append(
            {"tau": str(tau), "Molien/cycle": formula, "direct orbits": direct, "passed": formula == direct}
        )

    return {
        "group": "A5 on A5/A4 (degree 5)",
        "order": len(group),
        "matrix dimension": 5,
        "class-power checks": class_formula_checks,
        "representation law": representation_law(group),
        "rows": tuple(coefficient_rows),
        "passed": representation_law(group) and all(row["passed"] for row in coefficient_rows),
    }


def left_right_action(group: Sequence[GroupElement], left: GroupElement, right: GroupElement) -> Permutation:
    locations = {element: index for index, element in enumerate(group)}
    left_inverse = inverse(left)
    return tuple(
        locations[compose(left_inverse, compose(x, right))]
        for x in group
    )


def diagonal_action(
    group: Sequence[GroupElement], components: Sequence[GroupElement]
) -> Permutation:
    """T^k on T^(k-1), a concrete model of T^k/diag(T)."""

    locations = {element: index for index, element in enumerate(group)}
    rank = len(components) - 1
    points = tuple(product(group, repeat=rank))
    point_locations = {point: index for index, point in enumerate(points)}
    last_inverse = inverse(components[-1])
    return tuple(
        point_locations[
            tuple(compose(components[i], compose(point[i], last_inverse)) for i in range(rank))
        ]
        for point in points
    )


def diagonal_fixed_formula(group: Sequence[GroupElement], components: Sequence[GroupElement], exponent: int) -> int:
    powered = tuple(power(component, exponent) for component in components)
    first_class = conjugacy_class(group, powered[0])
    if not all(component in first_class for component in powered[1:]):
        return 0
    return centralizer_size(group, powered[0]) ** (len(components) - 1)


def conjugation_action(group: Sequence[GroupElement], by: GroupElement, tuple_power: int = 1) -> Permutation:
    points = tuple(product(group, repeat=tuple_power))
    locations = {point: index for index, point in enumerate(points)}
    return tuple(
        locations[tuple(conjugate(component, by) for component in point)]
        for point in points
    )


def coordinate_swap(group: Sequence[GroupElement], tuple_power: int) -> Permutation:
    points = tuple(product(group, repeat=tuple_power))
    locations = {point: index for index, point in enumerate(points)}
    if tuple_power != 2:
        raise ValueError("only the two-block check needs a coordinate swap")
    return tuple(locations[(point[1], point[0])] for point in points)


def diagonal_and_holomorph_suite() -> dict:
    group = alternating_group(5)
    classes = conjugacy_classes(group)
    centralizers = tuple(sorted(centralizer_size(group, next(iter(cls))) for cls in classes))

    # Every element of A5 x A5, and powers 1 through 6, on the 60-point set T.
    action_group = []
    fixed_checks = 0
    for left in group:
        for right in group:
            action = left_right_action(group, left, right)
            action_group.append(action)
            for exponent in range(1, 7):
                direct = fixed_points_of_power(action, exponent)
                predicted = diagonal_fixed_formula(group, (inverse(left), right), exponent)
                assert direct == predicted
                fixed_checks += 1

    # Rank of A5 x A5 on T is checked by direct orbits on T x T.
    lr_generators = []
    identity_element = identity(5)
    for generator in group_generators("A5"):
        lr_generators.append(left_right_action(group, generator, identity_element))
        lr_generators.append(left_right_action(group, identity_element, generator))
    direct_rank = orbit_count_from_generators(len(group), lr_generators, tuple_power=2)
    holomorph_formula = len(classes)

    # k=3 diagonal moment: rank equals simultaneous-conjugacy orbits on T^2.
    conjugation_generators = tuple(
        conjugation_action(group, generator, tuple_power=2)
        for generator in group_generators("A5")
    )
    k3_direct = orbit_count_from_generators(len(group) ** 2, conjugation_generators)
    k3_formula = sum(centralizers)

    # Representative direct fixed-point checks on the degree-3600 k=3 action.
    sample = tuple(group[index] for index in (0, 1, 7, 13, 29, 41, 59))
    k3_fixed_checks = 0
    for components in (
        (sample[0], sample[1], sample[2]),
        (sample[3], sample[3], sample[3]),
        (sample[4], sample[5], sample[6]),
        (sample[1], sample[4], sample[1]),
    ):
        action = diagonal_action(group, components)
        for exponent in (1, 2, 3, 5):
            assert fixed_points_of_power(action, exponent) == diagonal_fixed_formula(group, components, exponent)
            k3_fixed_checks += 1

    rows = (
        {"case": "simple diagonal k=2", "direct [m_(1,1)]": direct_rank, "formula": len(classes)},
        {"case": "simple diagonal k=3", "direct [m_(1,1)]": k3_direct, "formula": k3_formula},
        {"case": "holomorph simple", "direct [m_(1,1)]": direct_rank, "formula": holomorph_formula},
    )
    return {
        "group": "A5 diagonal and left-right actions",
        "conjugacy classes": len(classes),
        "centralizer sizes": centralizers,
        "degree-60 fixed checks": fixed_checks,
        "degree-3600 sample checks": k3_fixed_checks,
        "rows": rows,
        "passed": all(row["direct [m_(1,1)]"] == row["formula"] for row in rows),
    }


def compound_diagonal_suite() -> dict:
    group = alternating_group(5)
    base_generators = group_generators("A5")
    points = tuple(product(group, repeat=2))
    locations = {point: index for index, point in enumerate(points)}
    separate_generators = []
    for coordinate in range(2):
        for generator in base_generators:
            separate_generators.append(
                tuple(
                    locations[
                        tuple(
                            conjugate(value, generator) if index == coordinate else value
                            for index, value in enumerate(point)
                        )
                    ]
                    for point in points
                )
            )
    without_swap = orbit_count_from_generators(len(points), separate_generators)
    with_swap = orbit_count_from_generators(
        len(points), tuple(separate_generators) + (coordinate_swap(group, 2),)
    )
    rank = len(conjugacy_classes(group))
    rows = (
        {"case": "two diagonal blocks, no block permutations", "direct": without_swap, "formula": rank**2},
        {"case": "two diagonal blocks with S2", "direct": with_swap, "formula": comb(2 + rank - 1, rank - 1)},
    )
    return {
        "group": "A5, ell=2, b=2 (degree 3600)",
        "rows": rows,
        "passed": all(row["direct"] == row["formula"] for row in rows),
    }


def act_on_word(base_action: Sequence[Permutation], labels: Sequence[int], top: Permutation, word: tuple[int, ...]):
    return tuple(base_action[labels[index]][word[top[index]]] for index in range(len(word)))


def product_action_permutation(
    base_action: Sequence[Permutation], labels: Sequence[int], top: Permutation
) -> Permutation:
    alphabet_size = len(base_action[0])
    words = tuple(product(range(alphabet_size), repeat=len(labels)))
    locations = {word: index for index, word in enumerate(words)}
    return tuple(locations[act_on_word(base_action, labels, top, word)] for word in words)


def cycle_product_element(
    base_group: Sequence[GroupElement], labels: Sequence[int], cycle: Sequence[int]
) -> GroupElement:
    answer = identity(len(base_group[0]))
    for coordinate in cycle:
        answer = compose(answer, base_group[labels[coordinate]])
    return answer


def product_fixed_formula(
    base_group: Sequence[GroupElement], labels: Sequence[int], top: Permutation, exponent: int
) -> int:
    answer = 1
    for cycle in cycles(top):
        divisor = gcd(len(cycle), exponent)
        cycle_product = cycle_product_element(base_group, labels, cycle)
        fixed = fixed_points_of_power(cycle_product, exponent // divisor)
        answer *= fixed**divisor
    return answer


def product_action_generators(base_action: Sequence[Permutation], base_generator_indices: Sequence[int], k: int):
    generators = []
    identity_top = identity(k)
    identity_label = base_action.index(identity(len(base_action[0])))
    for coordinate in range(k):
        for generator_index in base_generator_indices:
            labels = [identity_label] * k
            labels[coordinate] = generator_index
            generators.append(product_action_permutation(base_action, labels, identity_top))
    for coordinate in range(k - 1):
        top = list(range(k))
        top[coordinate], top[coordinate + 1] = top[coordinate + 1], top[coordinate]
        generators.append(
            product_action_permutation(base_action, [identity_label] * k, tuple(top))
        )
    return tuple(generators)


def base_orbit_count(base_generators: Sequence[Permutation], alphabet_size: int, tuple_power: int) -> int:
    return orbit_count_from_generators(alphabet_size, base_generators, tuple_power)


def exhaustive_product_case(base_group: Sequence[Permutation], generator_names: Sequence[Permutation], k: int) -> dict:
    top_group = symmetric_group(k)
    base_locations = {element: index for index, element in enumerate(base_group)}
    generator_indices = tuple(base_locations[g] for g in generator_names)
    actions = []
    fixed_checks = 0
    for labels in product(range(len(base_group)), repeat=k):
        for top in top_group:
            action = product_action_permutation(base_group, labels, top)
            actions.append(action)
            for exponent in range(1, 7):
                assert fixed_points_of_power(action, exponent) == product_fixed_formula(
                    base_group, labels, top, exponent
                )
                fixed_checks += 1
    generators = product_action_generators(base_group, generator_indices, k)
    moment_rows = []
    for moment in (2, 3):
        direct = orbit_count_from_generators(len(base_group[0]) ** k, generators, moment)
        base_rank = base_orbit_count(generator_names, len(base_group[0]), moment)
        formula = comb(k + base_rank - 1, base_rank - 1)
        burnside_numerator = sum(fixed_points(action) ** moment for action in actions)
        assert burnside_numerator % len(actions) == 0
        burnside = burnside_numerator // len(actions)
        moment_rows.append(
            {"moment": moment, "base orbit types": base_rank, "direct orbits": direct, "Burnside": burnside, "formula": formula}
        )
    return {
        "k": k,
        "degree": len(base_group[0]) ** k,
        "group order": len(actions),
        "fixed-power checks": fixed_checks,
        "rows": tuple(moment_rows),
        "passed": all(row["direct orbits"] == row["Burnside"] == row["formula"] for row in moment_rows),
    }


def product_action_suite() -> dict:
    base = symmetric_group(3)
    cases = tuple(exhaustive_product_case(base, group_generators("S3"), k) for k in (2, 3))
    return {
        "group": "S3 wr S_k on {0,1,2}^k",
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }


def coordinate_automorphism(word: tuple[GroupElement, ...], top: Permutation):
    return tuple(word[top[index]] for index in range(len(word)))


def affine_action_permutation(
    group: Sequence[GroupElement], translation: tuple[GroupElement, ...], top: Permutation
) -> Permutation:
    points = tuple(product(group, repeat=len(translation)))
    locations = {point: index for index, point in enumerate(points)}
    return tuple(
        locations[
            tuple(
                compose(translation[index], point[top[index]])
                for index in range(len(translation))
            )
        ]
        for point in points
    )


def affine_power_translation(
    group: Sequence[GroupElement], translation: tuple[GroupElement, ...], top: Permutation, exponent: int
):
    result = tuple(identity(len(group[0])) for _ in translation)
    top_power = identity(len(top))
    for _ in range(exponent):
        shifted = coordinate_automorphism(translation, top_power)
        result = tuple(compose(result[index], shifted[index]) for index in range(len(result)))
        top_power = compose(top_power, top)
    return result, top_power


def lang_value(word: tuple[GroupElement, ...], top: Permutation):
    transformed = coordinate_automorphism(word, top)
    return tuple(compose(word[index], inverse(transformed[index])) for index in range(len(word)))


def twisted_fixed_formula(
    group: Sequence[GroupElement], translation: tuple[GroupElement, ...], top: Permutation, exponent: int
) -> int:
    translation_power, top_power = affine_power_translation(group, translation, top, exponent)
    points = tuple(product(group, repeat=len(translation)))
    image = {lang_value(point, top_power) for point in points}
    if translation_power not in image:
        return 0
    return len(group) ** len(cycles(top_power))


def twisted_wreath_suite() -> dict:
    group = symmetric_group(3)
    k = 2
    top_group = symmetric_group(k)
    translations = tuple(product(group, repeat=k))
    actions = []
    fixed_checks = 0
    conditional_checks = 0
    for top in top_group:
        conditional = []
        expected_spike = len(group) ** len(cycles(top))
        for translation in translations:
            action = affine_action_permutation(group, translation, top)
            actions.append(action)
            value = fixed_points(action)
            conditional.append(value)
            for exponent in range(1, 7):
                assert fixed_points_of_power(action, exponent) == twisted_fixed_formula(
                    group, translation, top, exponent
                )
                fixed_checks += 1
        assert set(conditional).issubset({0, expected_spike})
        assert conditional.count(expected_spike) * expected_spike == len(conditional)
        conditional_checks += 1

    rows = []
    for moment in (1, 2, 3):
        numerator = sum(fixed_points(action) ** moment for action in actions)
        assert numerator % len(actions) == 0
        direct = numerator // len(actions)
        if moment == 1:
            formula = 1
        else:
            types = len(group) ** (moment - 1)
            formula = comb(k + types - 1, types - 1)
        rows.append({"moment": moment, "direct average": direct, "formula": formula})

    # [m_(1,1)] is independently P\N, i.e. multisets of k elements of T.
    top_swap = tuple(reversed(range(k)))
    points = tuple(product(group, repeat=k))
    locations = {point: index for index, point in enumerate(points)}
    swap_action = tuple(locations[coordinate_automorphism(point, top_swap)] for point in points)
    direct_pair = orbit_count_from_generators(len(points), (swap_action,))
    formula_pair = comb(k + len(group) - 1, len(group) - 1)
    return {
        "group": "S3^2 semidirect S2 on its regular socle (degree 36)",
        "group order": len(actions),
        "fixed-power checks": fixed_checks,
        "conditional spike checks": conditional_checks,
        "rows": tuple(rows),
        "pair direct": direct_pair,
        "pair formula": formula_pair,
        "passed": direct_pair == formula_pair and all(row["direct average"] == row["formula"] for row in rows),
    }


def holomorph_compound_suite() -> dict:
    group = symmetric_group(3)
    base_actions = tuple(
        left_right_action(group, left, right) for left in group for right in group
    )
    base_action_locations = {action: index for index, action in enumerate(base_actions)}
    identity_element = identity(3)
    base_generators = tuple(
        left_right_action(group, generator, identity_element)
        for generator in group_generators("S3")
    ) + tuple(
        left_right_action(group, identity_element, generator)
        for generator in group_generators("S3")
    )
    generator_indices = tuple(base_action_locations[action] for action in base_generators)
    k = 2
    top_group = symmetric_group(k)
    actions = []
    fixed_checks = 0
    # The generic product formula only needs the base action permutations as
    # an abstract group.  Their composition table is represented by the same
    # permutation tuples.
    for labels in product(range(len(base_actions)), repeat=k):
        for top in top_group:
            action = product_action_permutation(base_actions, labels, top)
            actions.append(action)
            for exponent in (1, 2, 3, 4, 6):
                assert fixed_points_of_power(action, exponent) == product_fixed_formula(
                    base_actions, labels, top, exponent
                )
                fixed_checks += 1

    generators = product_action_generators(base_actions, generator_indices, k)
    direct = orbit_count_from_generators(len(group) ** k, generators, 2)
    class_count = len(conjugacy_classes(group))
    formula = comb(k + class_count - 1, class_count - 1)
    burnside_numerator = sum(fixed_points(action) ** 2 for action in actions)
    assert burnside_numerator % len(actions) == 0
    burnside = burnside_numerator // len(actions)
    return {
        "group": "((S3 x S3)^2 semidirect S2) on S3^2",
        "degree": len(group) ** k,
        "group order": len(actions),
        "fixed-power checks": fixed_checks,
        "direct pair orbits": direct,
        "Burnside pair coefficient": burnside,
        "formula": formula,
        "passed": direct == burnside == formula,
    }


def universal_audit(action_groups: Iterable[Sequence[Permutation]]) -> dict:
    elements = 0
    for action_group in action_groups:
        for action in action_group:
            assert mobius_cycles(action) == cycle_counts(action)
            elements += 1
    return {"permutation matrices audited": elements, "passed": True}


@lru_cache(maxsize=1)
def run_suite() -> dict:
    almost = almost_simple_suite()
    diagonal = diagonal_and_holomorph_suite()
    compound = compound_diagonal_suite()
    product_result = product_action_suite()
    twisted = twisted_wreath_suite()
    holomorph_compound = holomorph_compound_suite()

    # A separate universal Möbius audit on complete small matrix groups.
    universal = universal_audit((alternating_group(5), symmetric_group(3)))
    sections = (almost, diagonal, compound, product_result, twisted, holomorph_compound, universal)
    total_checks = (
        almost["class-power checks"]
        + diagonal["degree-60 fixed checks"]
        + diagonal["degree-3600 sample checks"]
        + sum(case["fixed-power checks"] for case in product_result["cases"])
        + twisted["fixed-power checks"]
        + holomorph_compound["fixed-power checks"]
        + universal["permutation matrices audited"]
    )
    return {
        "passed": all(section["passed"] for section in sections),
        "exact checks": total_checks,
        "almost simple": almost,
        "diagonal and holomorph simple": diagonal,
        "compound diagonal": compound,
        "product action": product_result,
        "twisted wreath": twisted,
        "holomorph compound": holomorph_compound,
        "universal": universal,
    }


if __name__ == "__main__":
    report = run_suite()
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {report['exact checks']} exact checks")
    for name in (
        "almost simple",
        "diagonal and holomorph simple",
        "compound diagonal",
        "product action",
        "twisted wreath",
        "holomorph compound",
    ):
        print(name, report[name])
