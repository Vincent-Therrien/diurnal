"""
    Test the diurnal.database module.
"""

import requests

import diurnal.database as database


def test_repository_availability():
    """Test the availability of the database repository."""
    response = requests.get(database.URL_PREFIX)
    assert response.status_code == 200, \
        f"The address {database.URL_PREFIX} is unaccessible."
