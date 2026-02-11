#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Portable PostgreSQL launcher for the Option C Q&A training DB.

Uses:
    POSTGRES_BIN_DIR  -> folder containing pg_ctl.exe
    POSTGRES_DATA_DIR -> data directory for the cluster
    POSTGRES_TRAIN_HOST
    POSTGRES_TRAIN_PORT

If PostgreSQL is already running, it does nothing.
If not, it calls pg_ctl start -w and waits until the port is accepting connections.
"""

import os
import socket
import subprocess
import time
from pathlib import Path

from option_c_qna.configuration.logging_config import get_qna_logger

log = get_qna_logger("start_postgres")


def _env_or_raise(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set")
    return value


def is_postgres_running(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return True if something is listening on (host, port)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        sock.close()
        return True
    except OSError:
        return False


def start_postgres() -> None:
    """
    Ensure PostgreSQL is running for the training DB.
    If it's already up, we just log and return.
    Otherwise we call pg_ctl start and wait until it's reachable.
    """
    host = os.getenv("POSTGRES_TRAIN_HOST", "127.0.0.1")
    port_str = os.getenv("POSTGRES_TRAIN_PORT", "5432")
    try:
        port = int(port_str)
    except ValueError:
        raise RuntimeError(f"Invalid POSTGRES_TRAIN_PORT value: {port_str!r}")

    bin_dir = Path(_env_or_raise("POSTGRES_BIN_DIR"))
    data_dir = Path(_env_or_raise("POSTGRES_DATA_DIR"))
    pg_ctl = bin_dir / "pg_ctl.exe"

    if not pg_ctl.exists():
        raise FileNotFoundError(f"pg_ctl not found at: {pg_ctl}")

    if not data_dir.exists():
        raise FileNotFoundError(f"PostgreSQL data directory does not exist: {data_dir}")

    log.info("Checking PostgreSQL status...")

    if is_postgres_running(host, port):
        log.info(f"PostgreSQL already running on {host}:{port}")
        return

    log.info("PostgreSQL is NOT running — starting now...")

    env = os.environ.copy()
    env["PGPORT"] = str(port)

    log_file = data_dir / "server.log"

    cmd = [
        str(pg_ctl),
        "start",
        "-D",
        str(data_dir),
        "-w",              # wait until startup completes
        "-l",
        str(log_file),
    ]

    log.info("Running pg_ctl command:")
    log.info(" ".join(cmd))

    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as e:
        log.error(f"pg_ctl start failed with exit code {e.returncode}")
        log.error(f"Check PostgreSQL log file: {log_file}")
        raise

    # Extra safety loop – verify that we can actually connect to the port
    log.info("Waiting for PostgreSQL socket to become available...")
    for i in range(10):
        if is_postgres_running(host, port):
            log.info(f"PostgreSQL is now accepting connections on {host}:{port}")
            return
        time.sleep(1)

    raise RuntimeError(
        f"PostgreSQL did not start listening on {host}:{port} within the timeout; "
        f"check log file: {log_file}"
    )


if __name__ == "__main__":
    # Manual test helper
    start_postgres()
