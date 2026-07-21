"""Exact tests of the full Hamming wreath-product formulas (16)--(20)."""

from __future__ import annotations

from itertools import permutations, product

from ..shared import fixed_formula_reconstructs, numerical_determinant_check, verify_action


def _perm_compose(left, right):
    return tuple(left[right[x]] for x in range(len(right)))


def _coordinate_map_power(sigmas, pi, exponent: int):
    """Power a map source i -> pi(i), with alphabet label sigma_i."""

    d = len(pi)
    q = len(sigmas[0])
    answer_dest = tuple(range(d))
    answer_labels = tuple(tuple(range(q)) for _ in range(d))
    base_dest = tuple(pi)
    base_labels = tuple(sigmas)

    def compose_maps(after_dest, after_labels, before_dest, before_labels):
        dest = tuple(after_dest[before_dest[i]] for i in range(d))
        labels = tuple(
            _perm_compose(after_labels[before_dest[i]], before_labels[i])
            for i in range(d)
        )
        return dest, labels

    while exponent:
        if exponent & 1:
            answer_dest, answer_labels = compose_maps(
                base_dest, base_labels, answer_dest, answer_labels
            )
        base_dest, base_labels = compose_maps(base_dest, base_labels, base_dest, base_labels)
        exponent //= 2
    return answer_dest, answer_labels


def claimed_fixed_words(sigmas, pi, exponent: int) -> int:
    """Equation (16), using the coordinate cycles and their label products."""

    dest, labels = _coordinate_map_power(sigmas, pi, exponent)
    seen = set()
    answer = 1
    for start in range(len(dest)):
        if start in seen:
            continue
        current = start
        cycle_product = tuple(range(len(labels[0])))
        while current not in seen:
            seen.add(current)
            cycle_product = _perm_compose(labels[current], cycle_product)
            current = dest[current]
        answer *= sum(cycle_product[x] == x for x in range(len(cycle_product)))
    return answer


def hamming_action(q: int, d: int):
    alphabet_permutations = tuple(permutations(range(q)))
    coordinate_permutations = tuple(permutations(range(d)))
    words = tuple(product(range(q), repeat=d))
    position = {word: index for index, word in enumerate(words)}
    elements = []
    for sigmas in product(alphabet_permutations, repeat=d):
        sigmas = tuple(tuple(sigma) for sigma in sigmas)
        for pi in coordinate_permutations:
            pi = tuple(pi)
            action = []
            for word in words:
                moved = [0] * d
                for source in range(d):
                    moved[pi[source]] = sigmas[source][word[source]]
                action.append(position[tuple(moved)])
            elements.append((sigmas, pi, tuple(action)))
    return tuple(elements)


def _case(q: int, d: int):
    elements = hamming_action(q, d)
    actions = tuple(element[2] for element in elements)
    reconstruction = all(
        fixed_formula_reconstructs(
            action,
            lambda exponent, sigmas=sigmas, pi=pi: claimed_fixed_words(sigmas, pi, exponent),
        )
        for sigmas, pi, action in elements
    )
    rows = verify_action(f"S_{q} wr S_{d} on Q^{d}", actions, pair_orbits=d + 1)
    return rows, reconstruction, actions


def run_suite():
    rows = []
    reconstructions = []
    actions = []
    for q, d in ((2, 2), (2, 3), (3, 2)):
        case_rows, reconstruction, action = _case(q, d)
        rows.extend(case_rows)
        reconstructions.append(reconstruction)
        actions.append(action)
    numerical = numerical_determinant_check(actions[-1])
    return {
        "family": "Hamming",
        "cases": ((2, 2), (2, 3), (3, 2)),
        "rows": tuple(rows),
        "fixed-point reconstruction": all(reconstructions),
        "numerical determinant": numerical,
        "passed": all(row["pass"] for row in rows) and all(reconstructions) and numerical[2] < 1e-10,
    }


if __name__ == "__main__":
    result = run_suite()
    print(f"Hamming: {'PASS' if result['passed'] else 'FAIL'} ({len(result['rows'])} comparisons)")

