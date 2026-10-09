from pathlib import Path
import json
import sqlite3

db_path = Path(__file__).resolve().parent / "livestock_app.db"

if not db_path.exists():
    raise FileNotFoundError(
        f"Database sapadla nahi: {db_path}\n"
        "Adhi app start kar ani kamit kami ek image analyze kar."
    )

connection = sqlite3.connect(db_path)
connection.row_factory = sqlite3.Row

for table in ("care_guidance", "predictions"):
    print(f"\n===== {table} =====")
    rows = connection.execute(f"SELECT * FROM {table}").fetchall()

    if not rows:
        print("Ya table madhye ajun data nahi.")
    else:
        for row in rows:
            print(json.dumps(dict(row), indent=2, ensure_ascii=False))

connection.close()