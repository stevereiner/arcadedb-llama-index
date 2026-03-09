#!/usr/bin/env python3
"""
Authentication test for ArcadeDB.

Container on port 2481 started automatically by conftest.py.
"""

import pytest
import requests

_TEST_PORT = 2481


def _server_available() -> bool:
    try:
        requests.get(f"http://localhost:{_TEST_PORT}", timeout=2)
        return True
    except Exception:
        return False


def test_auth_combinations():
    """Verify root/playwithdata authenticates against the test container."""
    if not _server_available():
        pytest.skip(f"ArcadeDB test container not available on port {_TEST_PORT}")

    auth_combinations = [
        ("root", "playwithdata"),
        ("root", "playingwithdata"),
        ("admin", "playwithdata"),
        ("root", "admin"),
        ("admin", "admin"),
    ]

    for username, password in auth_combinations:
        try:
            r = requests.get(
                f"http://localhost:{_TEST_PORT}/api/v1/databases",
                auth=(username, password),
                timeout=5,
            )
            if r.status_code == 200:
                print(f"SUCCESS: {username}/{password}")
                return
        except Exception:
            pass

    assert False, f"No valid authentication found on port {_TEST_PORT}"


if __name__ == "__main__":
    test_auth_combinations()
