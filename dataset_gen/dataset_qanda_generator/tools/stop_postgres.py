import subprocess
from pathlib import Path

PG_BIN = Path(r"E:\emtac\databases\postgresql\pgsql\bin")
PG_DATA = Path(r"E:\emtac\databases\postgresql\data")
PG_CTL = PG_BIN / "pg_ctl.exe"

def stop_postgres():
    print("[PG_CTL] Stopping PostgreSQL...")
    subprocess.call([
        str(PG_CTL),
        "-D", str(PG_DATA),
        "stop",
        "-m", "fast"
    ])
    print("[OK] PostgreSQL stopped.")

if __name__ == "__main__":
    stop_postgres()
