import pytest

from biomol.two_numbers import TwoNumbers


@pytest.fixture(scope="session")
def zero_zero():
    return TwoNumbers(0, 0)
