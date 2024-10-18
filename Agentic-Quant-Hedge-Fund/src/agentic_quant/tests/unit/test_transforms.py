"""
Basic unit tests for mathematical transforms.
"""
import numpy as np
import pytest

from ...core.registry import get


def test_add_transform():
    """Test the add transform function."""
    add_fn = get("add")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    result = add_fn(a, b)
    expected = np.array([5.0, 7.0, 9.0])
    np.testing.assert_array_equal(result, expected)


def test_sub_transform():
    """Test the sub transform function."""
    sub_fn = get("sub")
    a = np.array([5.0, 7.0, 9.0])
    b = np.array([1.0, 2.0, 3.0])
    result = sub_fn(a, b)
    expected = np.array([4.0, 5.0, 6.0])
    np.testing.assert_array_equal(result, expected) 