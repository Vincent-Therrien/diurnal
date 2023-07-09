"""
    Test te diurnal.families module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: July 2023
    - License: MIT
"""

from diurnal import family


def test_exclude_one_family():
    """Ensure that the function `all_but` works as expected."""
    test_family = "SRP"
    train_families = [
        "5s",
        "16s",
        "23s",
        "grp1",
        "grp2",
        "RNaseP",
        "telomerase",
        "tmRNA",
        "tRNA"
    ]
    excluded_families = family.all_but(test_family)
    assert excluded_families == train_families, \
        f"Selected: {excluded_families}. Expected: {train_families}"


def test_exclude_multiple_families():
    """Ensure that the function `all_but` works as expected."""
    test_families = ["grp1", "tmRNA"]
    train_families = [
        "5s",
        "16s",
        "23s",
        "grp2",
        "RNaseP",
        "SRP",
        "telomerase",
        "tRNA"
    ]
    excluded_families = family.all_but(test_families)
    assert excluded_families == train_families, \
        f"Selected: {excluded_families}. Expected: {train_families}"
