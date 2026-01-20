from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime, timezone
import sqlite3, os

DB = os.getenv("LICENSE_DB", "/tmp/licenses.db")

def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = db()
    con.executescript("""
    CREATE TABLE IF NOT EXISTS licenses(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      tenant TEXT NOT NULL,
      subject TEXT,            -- e.g. "user:alice" or "client:orchestrator"
      status TEXT NOT NULL DEFAULT 'active',
      expires_at TEXT          -- ISO8601
    );
    CREATE TABLE IF NOT EXISTS license_features(
      license_id INTEGER NOT NULL,
      feature_key TEXT NOT NULL,
      enabled INTEGER NOT NULL,
      FOREIGN KEY(license_id) REFERENCES licenses(id)
    );
    """)
    con.commit()
    con.close()

class LicenseIn(BaseModel):
    tenant: str
    subject: Optional[str] = None
    features: Dict[str, bool]
    expires_at: Optional[str] = None  # ISO8601

class VerifyIn(BaseModel):
    tenant: str
    subject: Optional[str] = None
    feature: str

app = FastAPI(on_startup=[init_db])

# --- Admin endpoints (protect with Keycloak later) ---
@app.post("/admin/licenses")
def create_license(payload: LicenseIn):
    con = db(); cur = con.cursor()
    cur.execute("INSERT INTO licenses(tenant, subject, expires_at) VALUES(?,?,?)",
                (payload.tenant, payload.subject, payload.expires_at))
    lid = cur.lastrowid
    for k, v in payload.features.items():
        cur.execute("INSERT INTO license_features(license_id, feature_key, enabled) VALUES(?,?,?)",
                    (lid, k, 1 if v else 0))
    con.commit(); con.close()
    return {"id": lid}

@app.patch("/admin/licenses/{lid}")
def update_license(lid: int, payload: LicenseIn):
    con = db(); cur = con.cursor()
    cur.execute("UPDATE licenses SET tenant=?, subject=?, expires_at=? WHERE id=?",
                (payload.tenant, payload.subject, payload.expires_at, lid))
    cur.execute("DELETE FROM license_features WHERE license_id=?", (lid,))
    for k, v in payload.features.items():
        cur.execute("INSERT INTO license_features(license_id, feature_key, enabled) VALUES(?,?,?)",
                    (lid, k, 1 if v else 0))
    con.commit(); con.close()
    return {"ok": True}

@app.post("/admin/licenses/{lid}/revoke")
def revoke_license(lid: int):
    con = db(); cur = con.cursor()
    cur.execute("UPDATE licenses SET status='revoked' WHERE id=?", (lid,))
    con.commit(); con.close()
    return {"ok": True}

# --- Runtime verification ---
@app.post("/license/verify")
def verify_license(payload: VerifyIn):
    con = db(); cur = con.cursor()
    cur.execute("""
      SELECT l.id, l.status, l.expires_at, lf.enabled
      FROM licenses l
      LEFT JOIN license_features lf ON lf.license_id = l.id AND lf.feature_key = ?
      WHERE l.tenant=? AND (l.subject=? OR l.subject IS NULL)
      ORDER BY CASE WHEN l.subject IS NULL THEN 1 ELSE 0 END, l.id DESC
      LIMIT 1;
    """, (payload.feature, payload.tenant, payload.subject))
    row = cur.fetchone(); con.close()

    if not row: return {"valid": False}
    if row["status"] != "active": return {"valid": False}
    if row["enabled"] != 1: return {"valid": False}

    if row["expires_at"]:
        try:
            exp = datetime.fromisoformat(row["expires_at"].replace("Z","+00:00"))
            if exp < datetime.now(timezone.utc):
                return {"valid": False, "expires_at": row["expires_at"]}
        except Exception:
            pass

    return {"valid": True, "features": {payload.feature: True}, "expires_at": row["expires_at"]}
