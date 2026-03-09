#!/usr/bin/env python3
"""
Diagnostic script for verifying ArcadeDB credentials and basic SQL operations.

Not collected by pytest (function is named check_*, not test_*).
Run directly:
    python tests/test_correct_auth.py
    python tests/test_correct_auth.py 2481
"""

import requests


def check_correct_auth(host: str = "localhost", port: int = 2480, database: str = "flexible_graphrag") -> None:
    """Verify credentials and run a quick SELECT/INSERT/SELECT cycle."""

    base_url = f"http://{host}:{port}"
    auth = ("root", "playwithdata")

    print(f"Testing ArcadeDB at {base_url}, database={database} ...")

    def sql(endpoint, command):
        return requests.post(
            f"{base_url}/api/v1/{endpoint}/{database}",
            json={"command": command, "language": "sql"},
            auth=auth,
            timeout=10,
        )

    r = sql("query", "SELECT count(*) as total FROM schema:types")
    print(f"Schema query: {r.status_code} — {r.text[:120]}")

    r = sql("command", "INSERT INTO PERSON SET name = 'TestUser', role = 'Tester'")
    print(f"INSERT:       {r.status_code} — {r.text[:120]}")

    r = sql("query", "SELECT name, role FROM PERSON WHERE name = 'TestUser'")
    print(f"SELECT:       {r.status_code} — {r.text[:120]}")


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 2480
    db = sys.argv[2] if len(sys.argv) > 2 else "flexible_graphrag"
    check_correct_auth(port=port, database=db)
