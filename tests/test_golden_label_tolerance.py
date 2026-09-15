"""Boundary checks for the golden label acceptance rule."""

import pandas as pd
import pytest


@pytest.mark.parametrize("mismatches", [14, 15])
def test_golden_label_tolerance_is_strictly_below_one_percent(
    assert_golden_labels, mismatches
):
    expected = pd.Series("a", index=pd.Index([f"cell_{i}" for i in range(1500)]))
    observed = expected.copy()
    observed.iloc[:mismatches] = "b"
    if mismatches < 15:
        assert_golden_labels(observed, expected)
    else:
        with pytest.raises(AssertionError, match="required < 1%"):
            assert_golden_labels(observed, expected)


def test_golden_label_tolerance_keeps_cell_alignment_strict(assert_golden_labels):
    expected = pd.Series("a", index=["cell_1", "cell_2"])
    with pytest.raises(AssertionError):
        assert_golden_labels(expected.iloc[::-1], expected)
