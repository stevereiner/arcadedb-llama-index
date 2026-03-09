"""
Root pytest configuration for the arcadedb-llama-index project.

Docker container
----------------
A single ArcadeDB container is started once for the whole pytest session
(port 2481) and torn down after all tests finish.  The three test files
that need it (test_arcadedb_integration.py, test_final_integration.py,
test_auth.py) all share it — no per-test container start/stop cycles.

JVM lifecycle
-------------
Some tests use ``arcadedb_embedded`` which starts a JVM via JPype.  JPype
cannot restart the JVM once it has been shut down.  ``pytest_unconfigure``
calls ``os._exit(0)`` to bypass Python cleanup (including JPype's atexit
shutdown hook) after all tests finish — taken from arcadedb-embedded-python's
own test suite.
"""

import os
import sys
import time

_container = None
_TEST_PORT = 2481
_CONTAINER_NAME = "arcadedb_test_session"


def pytest_sessionstart(session) -> None:  # noqa: ANN001
    """Start a shared ArcadeDB container once for the entire test session.

    The container is ephemeral (no volume) so every run starts with a clean
    database and the root password set via JAVA_OPTS is always honoured.
    Skipped during --collect-only since no tests actually run.
    """
    global _container

    # Don't start a container just for collection
    if session.config.option.collectonly:
        return

    try:
        import docker
    except ImportError:
        return  # docker package not installed — tests that need it will skip

    try:
        import requests
        requests.get(f"http://localhost:{_TEST_PORT}", timeout=2)
        return  # already running (e.g. manually started) — don't manage it
    except Exception:
        pass

    try:
        client = docker.from_env()
        # Remove any leftover container from a previous crashed run
        try:
            client.containers.get(_CONTAINER_NAME).remove(force=True)
        except Exception:
            pass

        _container = client.containers.run(
            "arcadedata/arcadedb:latest",
            detach=True,
            name=_CONTAINER_NAME,
            ports={"2480/tcp": _TEST_PORT, "2424/tcp": 2425},
            environment={"JAVA_OPTS": "-Darcadedb.server.rootPassword=playwithdata"},
        )
        print(f"\n[conftest] ArcadeDB container started on port {_TEST_PORT}, waiting for ready...")
        # Poll /api/v1/ready (no auth needed, returns 204 when up) instead of blind sleep
        import requests as _requests
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                if _requests.get(f"http://localhost:{_TEST_PORT}/api/v1/ready", timeout=2).status_code == 204:
                    break
            except Exception:
                pass
            time.sleep(1)
        time.sleep(2)  # brief extra settle time after ready signal
        print(f"[conftest] ArcadeDB container ready.")
    except Exception as e:
        print(f"\n[conftest] Could not start ArcadeDB container: {e}")


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001
    """Stop and remove the shared ArcadeDB container after all tests."""
    global _container
    if _container is not None:
        try:
            _container.stop()
            _container.remove()
            print(f"\n[conftest] ArcadeDB container stopped.")
        except Exception as e:
            print(f"\n[conftest] Error stopping container: {e}")
        _container = None


def pytest_unconfigure(config) -> None:  # noqa: ANN001
    """Force-exit after tests complete to prevent JVM thread hangs/restart crashes."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
