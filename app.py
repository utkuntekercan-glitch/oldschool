import streamlit as st
# ----- FORCE LIGHT THEME -----
st.set_page_config(
    page_title="Oldschool Finans",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #ffffff;
}
[data-testid="stSidebar"] * {
    color: #1f2937 !important;
}
</style>
""", unsafe_allow_html=True)
# ----- END THEME -----

import sqlite3
from pathlib import Path
from datetime import date, datetime
import re
import unicodedata
from html import escape
import hashlib
import pandas as pd

import io
import os
import shutil
import importlib.util
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

try:
    import psycopg2
except Exception:
    psycopg2 = None
try:
    import psycopg
except Exception:
    psycopg = None
try:
    import pg8000.dbapi as pg8000
except Exception:
    pg8000 = None

APP_TITLE = "Oldschool Espor Center"
DB_PATH = Path("oldschool_finance.db")
DATABASE_URL = str(st.secrets.get("DATABASE_URL", os.getenv("DATABASE_URL", ""))).strip()
USE_POSTGRES = bool(DATABASE_URL)

# Keep chart labels readable for Turkish text in generated PDF images.
plt.rcParams["font.family"] = "DejaVu Sans"

def ensure_backup():
    """Her acilista veritabanini backups/ klasorune kopyalar."""
    try:
        if USE_POSTGRES:
            return
        if not DB_PATH.exists():
            return
        backups = Path("backups")
        backups.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dst = backups / f"oldschool_finance_{ts}.db"
        shutil.copy2(DB_PATH, dst)
    except Exception:
        pass


# ---------- DB ----------
class DBConn:
    def __init__(self, driver: str, raw_conn):
        self.driver = driver
        self._conn = raw_conn

    def _sql(self, q: str) -> str:
        if self.driver == "postgres":
            return q.replace("?", "%s")
        return q

    def execute(self, q: str, params=()):
        cur = self._conn.cursor()
        cur.execute(self._sql(q), tuple(params))
        return cur

    def executemany(self, q: str, seq):
        cur = self._conn.cursor()
        cur.executemany(self._sql(q), [tuple(x) for x in seq])
        return cur

    def commit(self):
        self._conn.commit()
        try:
            st.cache_data.clear()
        except Exception:
            pass

    def close(self):
        self._conn.close()


def get_conn():
    if USE_POSTGRES:
        if psycopg2 is not None:
            if "sslmode=" in DATABASE_URL:
                raw = psycopg2.connect(DATABASE_URL)
            else:
                raw = psycopg2.connect(DATABASE_URL, sslmode="require")
        elif psycopg is not None:
            if "sslmode=" in DATABASE_URL:
                raw = psycopg.connect(DATABASE_URL)
            else:
                raw = psycopg.connect(DATABASE_URL, sslmode="require")
        elif pg8000 is not None:
            url = DATABASE_URL
            if "sslmode=" not in url:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}sslmode=require"
            raw = pg8000.connect(url)
        else:
            raise RuntimeError("Postgres surucusu bulunamadi. requirements.txt icine psycopg2-binary veya psycopg[binary] ekleyin.")
        raw.autocommit = False
        return DBConn("postgres", raw)

    raw = sqlite3.connect(DB_PATH, check_same_thread=False)
    raw.execute("PRAGMA foreign_keys = ON;")
    return DBConn("sqlite", raw)


def init_db(conn: DBConn):
    id_col = "BIGSERIAL PRIMARY KEY" if conn.driver == "postgres" else "INTEGER PRIMARY KEY AUTOINCREMENT"

    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS daily_cash (
        id {id_col},
        d TEXT NOT NULL UNIQUE,
        cash REAL NOT NULL DEFAULT 0,
        card REAL NOT NULL DEFAULT 0,
        note TEXT
    );
    """)
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS expense (
        id {id_col},
        d TEXT NOT NULL,
        category TEXT NOT NULL,
        amount REAL NOT NULL,
        pay_method TEXT NOT NULL, -- Nakit/Kart/Havale
        note TEXT,
        source TEXT NOT NULL DEFAULT 'manual' -- manual/auto
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS categories (
        name TEXT PRIMARY KEY,
        active INTEGER NOT NULL DEFAULT 1
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS month_lock (
        month TEXT PRIMARY KEY, -- YYYY-MM
        locked INTEGER NOT NULL DEFAULT 0,
        locked_at TEXT
    );
    """)
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS recurring_rule (
        id {id_col},
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        amount REAL NOT NULL,
        day_of_month INTEGER NOT NULL,
        pay_method TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1
    );
    """)
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS loan (
        id {id_col},
        name TEXT NOT NULL,
        monthly_amount REAL NOT NULL,
        start_month TEXT NOT NULL, -- YYYY-MM
        months_total INTEGER NOT NULL,
        pay_method TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1
    );
    """)
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS installment_plan (
        id {id_col},
        name TEXT NOT NULL,
        total_amount REAL NOT NULL,
        months_total INTEGER NOT NULL,
        start_month TEXT NOT NULL, -- YYYY-MM
        pay_method TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS month_close (
        month TEXT PRIMARY KEY, -- YYYY-MM
        opened_at TEXT NOT NULL
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS app_user_auth (
        username TEXT PRIMARY KEY,
        role TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)
    # Query performance indexes (especially important on Postgres as data grows).
    conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_cash_d ON daily_cash(d);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_expense_d ON expense(d);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_expense_source_note_d ON expense(source, note, d);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_recurring_rule_active ON recurring_rule(active);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_loan_active ON loan(active);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_installment_plan_active ON installment_plan(active);")
    conn.commit()

def sync_postgres_sequences(conn: DBConn):
    """Ensure BIGSERIAL sequences are aligned with current max(id) after data imports."""
    if conn.driver != "postgres":
        return

    id_tables = ["daily_cash", "expense", "recurring_rule", "loan", "installment_plan"]
    for table in id_tables:
        conn.execute(f"""
            SELECT setval(
                pg_get_serial_sequence('{table}', 'id'),
                COALESCE((SELECT MAX(id) FROM {table}), 0) + 1,
                false
            )
        """)
    conn.commit()


def ym(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"

def parse_ym(s: str):
    y, m = s.split("-")
    return int(y), int(m)

def is_valid_ym(s: str) -> bool:
    if not re.fullmatch(r"\d{4}-\d{2}", str(s).strip()):
        return False
    try:
        _, m = parse_ym(str(s).strip())
        return 1 <= m <= 12
    except Exception:
        return False

def shift_ym(ym_str: str, delta_months: int) -> str:
    y, m = parse_ym(ym_str)
    idx = (y * 12 + (m - 1)) + int(delta_months)
    ny = idx // 12
    nm = (idx % 12) + 1
    return f"{ny:04d}-{nm:02d}"

def month_range(ym_str: str):
    y, m = parse_ym(ym_str)
    start = date(y, m, 1)
    if m == 12:
        end = date(y+1, 1, 1)
    else:
        end = date(y, m+1, 1)
    return start, end

def day_in_month(ym_str: str, day: int) -> date:
    y, m = parse_ym(ym_str)
    safe_day = min(max(day, 1), 28)
    return date(y, m, safe_day)

def ensure_month_open(conn, ym_str: str):
    row = conn.execute("SELECT month FROM month_close WHERE month = ?", (ym_str,)).fetchone()
    if row:
        return False
    conn.execute("INSERT INTO month_close(month, opened_at) VALUES(?,?)",
                 (ym_str, datetime.now().isoformat(timespec="seconds")))
    conn.commit()
    return True

def auto_generate_for_month(conn, ym_str: str):
    """Adds or updates auto expense rows for rules, loans, and installments in the selected month."""
    def upsert_auto_expense(d: str, category: str, amount: float, pay_method: str, note_prefix: str, entity_id: int, name: str):
        note = f"{note_prefix}:{entity_id}:{name}"
        legacy_pattern = f"{note_prefix}:{entity_id}:%"
        existing_rows = conn.execute("""
            SELECT id FROM expense
            WHERE d=? AND source='auto' AND (note=? OR note LIKE ?)
            ORDER BY id
        """, (d, note, legacy_pattern)).fetchall()

        if existing_rows:
            keep_id = int(existing_rows[0][0])
            conn.execute("""
                UPDATE expense
                SET category=?, amount=?, pay_method=?, note=?
                WHERE id=?
            """, (category, float(amount), pay_method, note, keep_id))

            if len(existing_rows) > 1:
                dup_ids = [int(r[0]) for r in existing_rows[1:]]
                placeholders = ",".join("?" for _ in dup_ids)
                conn.execute(f"DELETE FROM expense WHERE id IN ({placeholders})", dup_ids)
        else:
            conn.execute(
                "INSERT INTO expense(d, category, amount, pay_method, note, source) VALUES(?,?,?,?,?, 'auto')",
                (d, category, float(amount), pay_method, note),
            )

    # Recurring rules
    rules = conn.execute("SELECT id, name, category, amount, day_of_month, pay_method FROM recurring_rule WHERE active=1").fetchall()
    for rid, name, category, amount, dom, pm in rules:
        d = day_in_month(ym_str, dom).isoformat()
        upsert_auto_expense(d, category, amount, pm, "RULE", int(rid), str(name))

    # Loans
    loans = conn.execute("SELECT id, name, monthly_amount, start_month, months_total, pay_method FROM loan WHERE active=1").fetchall()
    for lid, name, monthly, start_month, months_total, pm in loans:
        sy, sm = parse_ym(start_month)
        cy, cm = parse_ym(ym_str)
        idx = (cy - sy) * 12 + (cm - sm)  # 0-based
        if 0 <= idx < months_total:
            d = day_in_month(ym_str, 1).isoformat()
            upsert_auto_expense(d, "Kredi", monthly, pm, "LOAN", int(lid), str(name))

    # Installments
    plans = conn.execute("SELECT id, name, total_amount, months_total, start_month, pay_method FROM installment_plan WHERE active=1").fetchall()
    for pid, name, total, months_total, start_month, pm in plans:
        sy, sm = parse_ym(start_month)
        cy, cm = parse_ym(ym_str)
        idx = (cy - sy) * 12 + (cm - sm)
        if 0 <= idx < months_total:
            monthly = float(total) / float(months_total)
            d = day_in_month(ym_str, 1).isoformat()
            upsert_auto_expense(d, "Kart Taksit", monthly, pm, "INST", int(pid), str(name))
    conn.commit()

def rebuild_auto_for_month(conn, ym_str: str):
    """Rebuild auto-generated rows for one month to clean stale/duplicate auto expenses."""
    start, end = month_range(ym_str)
    conn.execute("""
        DELETE FROM expense
        WHERE d >= ? AND d < ?
          AND source='auto'
          AND (
              note LIKE 'RULE:%'
              OR note LIKE 'LOAN:%'
              OR note LIKE 'INST:%'
          )
    """, (start.isoformat(), end.isoformat()))
    conn.commit()
    auto_generate_for_month(conn, ym_str)

def df_query(conn, q, params=()):
    cur = conn.execute(q, params)
    rows = cur.fetchall()
    cols = [c[0] for c in cur.description] if cur.description else []
    return pd.DataFrame(rows, columns=cols)

@st.cache_data(show_spinner=False, ttl=20)
def load_dashboard_month_data(ym_str: str, db_mode: str):
    start, end = month_range(ym_str)
    start_s, end_s = start.isoformat(), end.isoformat()
    c = get_conn()
    try:
        rev = df_query(c, """
            SELECT d, cash, card, (cash+card) AS total
            FROM daily_cash
            WHERE d >= ? AND d < ?
            ORDER BY d
        """, (start_s, end_s))
        exp = df_query(c, """
            SELECT d, category, amount, pay_method, source, note
            FROM expense
            WHERE d >= ? AND d < ?
            ORDER BY d
        """, (start_s, end_s))
        return rev, exp
    finally:
        try:
            c.close()
        except Exception:
            pass

def is_month_locked(conn, ym_str: str) -> bool:
    row = conn.execute("SELECT locked FROM month_lock WHERE month=?", (ym_str,)).fetchone()
    return bool(row and int(row[0]) == 1)

def set_month_lock(conn, ym_str: str, locked: bool):
    conn.execute(
        "INSERT INTO month_lock(month, locked, locked_at) VALUES(?,?,?) "
        "ON CONFLICT(month) DO UPDATE SET locked=excluded.locked, locked_at=excluded.locked_at",
        (ym_str, 1 if locked else 0, datetime.now().isoformat(timespec="seconds"))
    )
    conn.commit()

def get_active_categories(conn) -> list[str]:
    rows = conn.execute("SELECT name FROM categories WHERE active=1 ORDER BY name").fetchall()
    return [r[0] for r in rows]

def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        str(password).encode("utf-8"),
        str(salt).encode("utf-8"),
        200_000,
    ).hex()

def _upsert_user_auth(conn, username: str, role: str, password: str):
    username = str(username).strip()
    role = str(role).strip()
    if not username or not password:
        return
    salt = os.urandom(16).hex()
    ph = _hash_password(password, salt)
    existing = conn.execute("SELECT username FROM app_user_auth WHERE username=?", (username,)).fetchone()
    if existing:
        conn.execute(
            "UPDATE app_user_auth SET role=?, password_hash=?, salt=?, updated_at=? WHERE username=?",
            (role, ph, salt, datetime.now().isoformat(timespec="seconds"), username),
        )
    else:
        conn.execute(
            "INSERT INTO app_user_auth(username, role, password_hash, salt, updated_at) VALUES(?,?,?,?,?)",
            (username, role, ph, salt, datetime.now().isoformat(timespec="seconds")),
        )
    conn.commit()

def _seed_auth_users_if_missing(conn):
    users = [
        (str(st.secrets.get("APP_USER", "")).strip(), "admin", str(st.secrets.get("APP_PASSWORD", ""))),
        (str(st.secrets.get("CASH_USER", "")).strip(), "cash_only", str(st.secrets.get("CASH_PASSWORD", ""))),
    ]
    for username, role, password in users:
        if not username or not password:
            continue
        exists = conn.execute("SELECT username FROM app_user_auth WHERE username=?", (username,)).fetchone()
        if not exists:
            _upsert_user_auth(conn, username, role, password)

def authenticate_user(conn, username: str, password: str):
    _seed_auth_users_if_missing(conn)
    row = conn.execute(
        "SELECT role, password_hash, salt FROM app_user_auth WHERE username=?",
        (str(username).strip(),),
    ).fetchone()
    if not row:
        return None
    role, password_hash, salt = str(row[0]), str(row[1]), str(row[2])
    if _hash_password(password, salt) == password_hash:
        return role
    return None

def change_user_password(conn, username: str, role: str, new_password: str):
    _upsert_user_auth(conn, username, role, new_password)

def set_undo_action(action: dict):
    st.session_state["undo_action"] = action

def apply_undo_action(conn) -> tuple[bool, str]:
    action = st.session_state.get("undo_action")
    if not action:
        return False, "Geri alınacak işlem yok."

    t = action.get("type")
    try:
        if t == "expense_add":
            conn.execute("DELETE FROM expense WHERE id=?", (int(action["id"]),))
            msg = "Eklenen gider geri alındı."
        elif t == "expense_update":
            old = action["old"]
            conn.execute(
                "UPDATE expense SET d=?, category=?, amount=?, pay_method=?, note=? WHERE id=? AND source='manual'",
                (old["d"], old["category"], float(old["amount"]), old["pay_method"], old["note"], int(old["id"])),
            )
            msg = "Gider güncellemesi geri alındı."
        elif t == "expense_delete":
            row = action["row"]
            conn.execute(
                "INSERT INTO expense(id, d, category, amount, pay_method, note, source) VALUES(?,?,?,?,?,?,?)",
                (int(row["id"]), row["d"], row["category"], float(row["amount"]), row["pay_method"], row["note"], row.get("source", "manual")),
            )
            msg = "Silinen gider geri yüklendi."
        elif t == "daily_cash_upsert":
            d = action["d"]
            prev = action.get("previous")
            if prev is None:
                conn.execute("DELETE FROM daily_cash WHERE d=?", (d,))
            else:
                conn.execute("""
                    INSERT INTO daily_cash(d, cash, card, note)
                    VALUES(?,?,?,?)
                    ON CONFLICT(d) DO UPDATE SET cash=excluded.cash, card=excluded.card, note=excluded.note
                """, (prev["d"], float(prev["cash"]), float(prev["card"]), prev["note"]))
            msg = "Günlük kasa işlemi geri alındı."
        else:
            return False, "Bilinmeyen işlem tipi."
        conn.commit()
        st.session_state.pop("undo_action", None)
        return True, msg
    except Exception as e:
        return False, f"Geri alma başarısız: {e}"


def seed_defaults_if_empty(conn, start_month: str):
    """Program ilk açıldığında senin verdiğin sabitleri/kredileri/taksitleri otomatik ekler (sadece DB boşsa)."""
    rules_count = conn.execute("SELECT COUNT(*) FROM recurring_rule").fetchone()[0]
    loans_count = conn.execute("SELECT COUNT(*) FROM loan").fetchone()[0]
    inst_count  = conn.execute("SELECT COUNT(*) FROM installment_plan").fetchone()[0]

    # Kategoriler
    cat_count = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    if cat_count == 0:
        default_cats = [
            ("Elektrik",1),("Gaz",1),("Su",1),("Alışveriş",1),("SSK+Vergi",1),
            ("Gündüz Maaş",1),("Maaş",1),("Kira",1),("İnternet",1),("Muhasebe",1),
            ("Sigorta",1),("Alarm",1),("Kredi",1),("Kart Taksit",1),("Diğer",1)
        ]
        conn.executemany("INSERT INTO categories(name, active) VALUES(?,?) ON CONFLICT(name) DO NOTHING", default_cats)
        conn.commit()


    # Sadece tamamen boş başlangıçta seed edelim:
    if rules_count == 0:
        defaults = [
            ("Kira", "Kira", 110000, 1, "Havale"),
            ("İnternet", "İnternet", 12000, 1, "Havale"),
            ("Muhasebe", "Muhasebe", 12000, 15, "Havale"),
            ("Sigorta", "Sigorta", 400, 15, "Havale"),
            ("Alarm", "Alarm", 500, 15, "Havale"),
            ("Maaş Toplam (Sabit)", "Maaş", 315000, 1, "Havale"),
        ]
        conn.executemany("""
            INSERT INTO recurring_rule(name, category, amount, day_of_month, pay_method, active)
            VALUES(?,?,?,?,?,1)
        """, defaults)
        conn.commit()

    if loans_count == 0:
        defaults_loans = [
            ("Araba Kredisi", 70000, start_month, 14, "Havale"),
            ("Okul Kredisi", 70000, start_month, 4, "Havale"),
        ]
        conn.executemany("""
            INSERT INTO loan(name, monthly_amount, start_month, months_total, pay_method, active)
            VALUES(?,?,?,?,?,1)
        """, defaults_loans)
        conn.commit()

    if inst_count == 0:
        defaults_inst = [
            ("Kredi Kartı Taksitleri (Toplam)", 36000, 6, start_month, "Kart"),
        ]
        conn.executemany("""
            INSERT INTO installment_plan(name, total_amount, months_total, start_month, pay_method, active)
            VALUES(?,?,?,?,?,1)
        """, defaults_inst)
        conn.commit()




# ---------- MONEY FORMAT ----------
def tr_money(x: float) -> str:
    """₺150.000 format (TR)"""
    try:
        s = format(float(x), ",.0f").replace(",", ".")
    except Exception:
        s = "0"
    return f"₺{s}"

def parse_amount_token(token: str) -> float | None:
    try:
        s = str(token).strip()
        s = s.replace("₺", "").replace("TL", "").replace("tl", "")
        s = re.sub(r"[^0-9,.\-]", "", s)
        if not s:
            return None

        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            parts = s.split(",")
            if len(parts) == 2 and len(parts[1]) <= 2:
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        else:
            parts = s.split(".")
            if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
                s = "".join(parts)

        return float(s)
    except Exception:
        return None

def extract_amount_from_line(line: str) -> float | None:
    matches = re.findall(r"[-+]?\d[\d.,]*", str(line))
    for m in reversed(matches):
        val = parse_amount_token(m)
        if val is not None:
            return val
    return None

def extract_bank_visa_amounts(ocr_text: str) -> tuple[float | None, float | None]:
    lines = [ln.strip() for ln in str(ocr_text).splitlines() if ln.strip()]
    bank_primary_keywords = ("banka", "bank")
    visa_primary_keywords = ("visa",)
    bank_fallback_keywords = ("nakit", "cash")
    visa_fallback_keywords = ("kredi kart", "kart", "card")

    bank_val = None
    visa_val = None

    # Primary mapping requested:
    # - "Banka" line -> Nakit
    # - "Visa" line  -> Kredi Kartı
    for ln in lines:
        low = ln.lower()
        if bank_val is None and any(k in low for k in bank_primary_keywords):
            bank_val = extract_amount_from_line(ln)
        if visa_val is None and any(k in low for k in visa_primary_keywords):
            visa_val = extract_amount_from_line(ln)

    # Fallbacks only if primary labels are not found.
    if bank_val is None:
        for ln in lines:
            low = ln.lower()
            if any(k in low for k in bank_fallback_keywords):
                bank_val = extract_amount_from_line(ln)
                if bank_val is not None:
                    break
    if visa_val is None:
        for ln in lines:
            low = ln.lower()
            if any(k in low for k in visa_fallback_keywords):
                visa_val = extract_amount_from_line(ln)
                if visa_val is not None:
                    break

    if bank_val is None:
        m = re.search(r"(?:banka|bank)[^0-9]{0,20}([-+]?\d[\d.,]*)", "\n".join(lines), flags=re.I)
        if m:
            bank_val = parse_amount_token(m.group(1))
    if visa_val is None:
        m = re.search(r"(?:visa)[^0-9]{0,20}([-+]?\d[\d.,]*)", "\n".join(lines), flags=re.I)
        if m:
            visa_val = parse_amount_token(m.group(1))

    # Last fallback patterns if OCR heavily corrupts labels.
    if bank_val is None:
        m = re.search(r"(?:nakit|cash)[^0-9]{0,20}([-+]?\d[\d.,]*)", "\n".join(lines), flags=re.I)
        if m:
            bank_val = parse_amount_token(m.group(1))
    if visa_val is None:
        m = re.search(r"(?:kredi kart|kart|card)[^0-9]{0,20}([-+]?\d[\d.,]*)", "\n".join(lines), flags=re.I)
        if m:
            visa_val = parse_amount_token(m.group(1))

    return bank_val, visa_val

def _google_vision_config() -> tuple[str, str]:
    api_key = str(st.secrets.get("GOOGLE_VISION_API_KEY", "")).strip()
    svc_json = str(st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")).strip()
    if (not svc_json) and ("gcp_service_account" in st.secrets):
        try:
            svc_json = json.dumps(dict(st.secrets["gcp_service_account"]))
        except Exception:
            svc_json = ""
    return api_key, svc_json

@st.cache_resource(show_spinner=False)
def get_google_vision_client(api_key: str, svc_json: str):
    from google.cloud import vision
    if svc_json:
        from google.oauth2 import service_account
        info = json.loads(svc_json)
        creds = service_account.Credentials.from_service_account_info(info)
        return vision.ImageAnnotatorClient(credentials=creds)
    if api_key:
        return vision.ImageAnnotatorClient(client_options={"api_key": api_key})
    raise RuntimeError("Google Vision secrets tanımlı değil.")

def is_ocr_available() -> tuple[bool, str]:
    has_google_lib = importlib.util.find_spec("google.cloud.vision") is not None
    gv_api_key, gv_svc_json = _google_vision_config()
    has_google_cfg = bool(gv_api_key or gv_svc_json)
    if has_google_lib and has_google_cfg:
        return True, "Kullanılabilir OCR motoru: google-vision"
    if has_google_cfg and not has_google_lib:
        return False, "OCR devre dışı: Google Vision secrets var ama 'google-cloud-vision' paketi kurulu değil."
    return False, "OCR devre dışı: Google Vision secrets eksik."

def read_ocr_text(uploaded_file) -> str:
    gv_api_key, gv_svc_json = _google_vision_config()
    if importlib.util.find_spec("google.cloud.vision") is not None and (gv_api_key or gv_svc_json):
        from google.cloud import vision
        client = get_google_vision_client(gv_api_key, gv_svc_json)
        image_bytes = uploaded_file.getvalue()
        response = client.document_text_detection(image=vision.Image(content=image_bytes))
        if response.error.message:
            raise RuntimeError(f"Google Vision hatası: {response.error.message}")
        full_text = ""
        if response.full_text_annotation and response.full_text_annotation.text:
            full_text = response.full_text_annotation.text
        elif response.text_annotations:
            full_text = response.text_annotations[0].description
        if full_text.strip():
            return full_text
    raise RuntimeError("Google Vision kullanılamıyor. Secrets ayarını kontrol et.")

# ---------- DISPLAY HELPERS ----------
def clean_category_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.split(":").str[-1]

def clean_auto_notes(df: pd.DataFrame, note_col: str, source_col: str = "source") -> pd.Series:
    s = df[note_col].fillna("").astype(str)
    if source_col in df.columns:
        mask = (df[source_col].astype(str) == "auto") & s.str.startswith(("RULE:", "LOAN:", "INST:"))
        s = s.mask(mask, "")
    s = s.replace("", pd.NA)
    return s

def format_expense_for_display(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Tarih
    if "d" in df.columns and "Tarih" not in df.columns:
        df["Tarih"] = df["d"]
    # Kategori
    if "category" in df.columns:
        df["Kategori"] = clean_category_series(df["category"])
    elif "Kategori" in df.columns:
        df["Kategori"] = clean_category_series(df["Kategori"])

    # Tutar
    if "amount" in df.columns and "Tutar" not in df.columns:
        df["Tutar"] = df["amount"]

    # Ödeme
    if "pay_method" in df.columns:
        df["Ödeme"] = df["pay_method"]
    elif "Odeme" in df.columns:
        df["Ödeme"] = df["Odeme"]

    # Kaynak
    if "source" in df.columns and "Kaynak" not in df.columns:
        df["Kaynak"] = df["source"]

    # Notlar (otomatik satırlarda RULE/LOAN/INST gizle)
    if "note" in df.columns:
        df["Notlar"] = clean_auto_notes(df, "note", "source" if "source" in df.columns else "")
    elif "Notlar" in df.columns:
        # bazen SQL zaten Notlar olarak getirebilir
        col = "Notlar"
        df["Notlar"] = df[col].fillna("").astype(str)
        if "Kaynak" in df.columns:
            mask = (df["Kaynak"].astype(str) == "auto") & df["Notlar"].str.startswith(("RULE:", "LOAN:", "INST:"))
            df.loc[mask, "Notlar"] = ""
        df["Notlar"] = df["Notlar"].replace("", pd.NA)

    cols = [c for c in ["Tarih", "Kategori", "Tutar", "Ödeme", "Kaynak", "Notlar"] if c in df.columns]
    return df[cols]

def render_mobile_cards(
    df: pd.DataFrame,
    fields: list[str],
    empty_text: str,
    amount_fields: set[str] | None = None,
    title_fields: list[str] | None = None,
    badge_fields: set[str] | None = None,
):
    if df is None or len(df) == 0:
        st.info(empty_text)
        return

    amount_fields = amount_fields or set()
    title_fields = title_fields or []
    badge_fields = badge_fields or set()

    def category_badge_class(category_text: str) -> str:
        c = category_text.strip().lower()
        if "kira" in c:
            return "mobile-badge-rent"
        if "maaş" in c or "maas" in c:
            return "mobile-badge-salary"
        if "kredi" in c:
            return "mobile-badge-loan"
        if "taksit" in c:
            return "mobile-badge-installment"
        if "elektrik" in c or "su" in c or "gaz" in c or "internet" in c:
            return "mobile-badge-bills"
        if "alışveriş" in c or "alisveris" in c:
            return "mobile-badge-shopping"
        return "mobile-badge-category"

    for _, row in df.iterrows():
        card_lines = []
        title_parts = []
        for t in title_fields:
            if t in df.columns and pd.notna(row[t]) and str(row[t]).strip() != "":
                title_parts.append(escape(str(row[t])))

        for field in fields:
            if field not in df.columns or field in title_fields:
                continue
            val = row[field]
            if pd.isna(val) or str(val).strip() == "":
                continue

            if field in amount_fields:
                try:
                    val_str = tr_money(float(val))
                except Exception:
                    val_str = str(val)
            else:
                val_str = str(val)

            value_html = escape(val_str)
            if field in badge_fields:
                badge_cls = "mobile-badge-default"
                normalized = val_str.strip().lower()
                if field == "Ödeme":
                    if normalized == "nakit":
                        badge_cls = "mobile-badge-cash"
                    elif normalized == "kart":
                        badge_cls = "mobile-badge-card"
                    elif normalized == "havale":
                        badge_cls = "mobile-badge-transfer"
                elif field == "Kaynak":
                    if normalized == "manual":
                        badge_cls = "mobile-badge-manual"
                    elif normalized == "auto":
                        badge_cls = "mobile-badge-auto"
                elif field == "Kategori":
                    badge_cls = category_badge_class(val_str)
                value_html = f"<span class='mobile-badge {badge_cls}'>{escape(val_str)}</span>"

            card_lines.append(
                f"<div class='mobile-card-row'><span class='mobile-card-label'>{escape(field)}</span><span class='mobile-card-value'>{value_html}</span></div>"
            )

        title_html = ""
        if title_parts:
            title_html = f"<div class='mobile-card-title'>{' | '.join(title_parts)}</div>"

        body_html = "".join(card_lines)
        st.markdown(
            f"<div class='mobile-card'>{title_html}<div class='mobile-card-body'>{body_html}</div></div>",
            unsafe_allow_html=True,
        )

def render_mobile_category_progress(by_cat: pd.DataFrame, top_n: int = 5):
    if by_cat is None or len(by_cat) == 0 or "Kategori" not in by_cat.columns or "Tutar" not in by_cat.columns:
        return
    top = by_cat.head(top_n).copy()
    max_val = float(top["Tutar"].max()) if len(top) else 0.0
    if max_val <= 0:
        return

    rows = []
    for _, r in top.iterrows():
        cat = escape(str(r["Kategori"]))
        val = float(r["Tutar"])
        pct = max(4.0, min(100.0, (val / max_val) * 100.0))
        rows.append(
            f"<div class='mobile-progress-item'>"
            f"<div class='mobile-progress-head'><span>{cat}</span><span>{escape(tr_money(val))}</span></div>"
            f"<div class='mobile-progress-track'><div class='mobile-progress-fill' style='width:{pct:.1f}%'></div></div>"
            f"</div>"
        )
    st.markdown(
        "<div class='mobile-progress-wrap'><div class='mobile-progress-title'>Top 5 Kategori</div>"
        + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


# ---------- PDF REPORTS ----------
def repair_text(s) -> str:
    if s is None:
        return ""
    text = str(s)
    if any(token in text for token in ("Ã", "Ä", "Å", "â", "ğŸ", "ï")):
        try:
            text = text.encode("latin1").decode("utf-8")
        except Exception:
            pass
    return unicodedata.normalize("NFC", text)

def pdf_draw(c, x, y, text, right: bool = False):
    clean = repair_text(text)
    if right:
        c.drawRightString(x, y, clean)
    else:
        c.drawString(x, y, clean)

def get_pdf_fonts() -> tuple[str, str]:
    mpl_font_dir = Path(mpl.get_data_path()) / "fonts" / "ttf"
    candidates = [
        # Matplotlib bundled fonts (cross-platform, Unicode)
        (mpl_font_dir / "DejaVuSans.ttf", mpl_font_dir / "DejaVuSans-Bold.ttf", "PDFDejaVuSans", "PDFDejaVuSansBold"),
        # Windows
        (Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\arialbd.ttf"), "PDFArial", "PDFArialBold"),
        (Path(r"C:\Windows\Fonts\DejaVuSans.ttf"), Path(r"C:\Windows\Fonts\DejaVuSans-Bold.ttf"), "PDFDejaVuSans", "PDFDejaVuSansBold"),
        # Linux
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"), "PDFDejaVuSans", "PDFDejaVuSansBold"),
    ]
    for regular_path, bold_path, regular_name, bold_name in candidates:
        if regular_path.exists():
            if not bold_path.exists():
                bold_path = regular_path
            if regular_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(regular_name, str(regular_path)))
            if bold_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))
            return regular_name, bold_name
    raise RuntimeError("Unicode PDF font bulunamadı. DejaVu Sans veya Arial TTF erişilebilir olmalı.")

def _fig_to_imagereader(fig) -> ImageReader:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return ImageReader(buf)



def _pdf_money_plain(x: float) -> str:
    return re.sub(r"^[^0-9\-]+", "", tr_money(float(x))).strip()

def _pdf_draw_title_band(c, w, h, pdf_regular: str, pdf_bold: str, title: str, subtitle: str):
    band_x = 1.6 * cm
    band_y = h - 4.0 * cm
    band_w = w - 3.2 * cm
    band_h = 2.3 * cm
    c.setFillColorRGB(0.94, 0.96, 0.99)
    c.setStrokeColorRGB(0.84, 0.89, 0.96)
    c.roundRect(band_x, band_y, band_w, band_h, 10, stroke=1, fill=1)
    c.setFillColorRGB(0.09, 0.15, 0.28)
    c.setFont(pdf_bold, 16)
    pdf_draw(c, band_x + 0.45 * cm, band_y + 1.45 * cm, title)
    c.setFont(pdf_regular, 10.5)
    c.setFillColorRGB(0.24, 0.31, 0.45)
    pdf_draw(c, band_x + 0.45 * cm, band_y + 0.65 * cm, subtitle)
    c.setFillColorRGB(0, 0, 0)

def _pdf_draw_metric_cards(c, w, y_top, pdf_regular: str, pdf_bold: str, metrics: list[tuple[str, float]]):
    left = 2.0 * cm
    gap = 0.7 * cm
    card_w = (w - 2 * left - gap) / 2
    card_h = 1.55 * cm
    for idx, (label, value) in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        x = left + col * (card_w + gap)
        y = y_top - row * (card_h + 0.35 * cm)
        c.setFillColorRGB(1, 1, 1)
        c.setStrokeColorRGB(0.88, 0.9, 0.93)
        c.roundRect(x, y - card_h, card_w, card_h, 8, stroke=1, fill=1)

        c.setFillColorRGB(0.42, 0.46, 0.52)
        c.setFont(pdf_regular, 9)
        pdf_draw(c, x + 0.35 * cm, y - 0.48 * cm, label)

        value_color = (0.05, 0.39, 0.23)
        if "Net" in label and float(value) < 0:
            value_color = (0.67, 0.12, 0.09)
        c.setFillColorRGB(*value_color)
        c.setFont(pdf_bold, 12)
        pdf_draw(c, x + 0.35 * cm, y - 1.10 * cm, tr_money(float(value)))
        c.setFillColorRGB(0, 0, 0)

    rows = (len(metrics) + 1) // 2
    return y_top - rows * (card_h + 0.35 * cm) - 0.25 * cm


def build_monthly_pdf(conn, ym_str: str) -> bytes:
    pdf_regular, pdf_bold = get_pdf_fonts()
    start, end = month_range(ym_str)
    start_s, end_s = start.isoformat(), end.isoformat()

    rev = df_query(conn, """
        SELECT d, cash, card, (cash+card) AS total
        FROM daily_cash
        WHERE d >= ? AND d < ?
        ORDER BY d
    """, (start_s, end_s))

    exp = df_query(conn, """
        SELECT d, category, amount, pay_method, source, note
        FROM expense
        WHERE d >= ? AND d < ?
        ORDER BY d
    """, (start_s, end_s))

    cash_sum = float(rev["cash"].sum()) if len(rev) else 0.0
    card_sum = float(rev["card"].sum()) if len(rev) else 0.0
    total_rev = float(rev["total"].sum()) if len(rev) else 0.0
    total_exp = float(exp["amount"].sum()) if len(exp) else 0.0
    net = total_rev - total_exp

    exp_disp = format_expense_for_display(exp) if len(exp) else pd.DataFrame(columns=["Tarih", "Kategori", "Tutar", "Odeme", "Kaynak", "Notlar"])
    by_cat = (exp_disp.groupby("Kategori", as_index=False)["Tutar"].sum().sort_values("Tutar", ascending=False)) if len(exp_disp) else pd.DataFrame(columns=["Kategori", "Tutar"])
    if len(by_cat):
        by_cat["Kategori"] = by_cat["Kategori"].apply(repair_text)

    charts = []
    if len(rev):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(pd.to_datetime(rev["d"]), rev["total"], color="#1f4e79", linewidth=2.2)
        ax.set_title("Gunluk Toplam Gelir", fontsize=11, fontweight="bold")
        ax.set_xlabel("Tarih")
        ax.set_ylabel("TL")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        fig.tight_layout()
        charts.append(_fig_to_imagereader(fig))

    if len(by_cat):
        fig, ax = plt.subplots(figsize=(8, 3))
        top10 = by_cat.head(10)
        ax.bar(top10["Kategori"], top10["Tutar"], color="#2a7f62")
        ax.set_title("Kategori Bazli Gider (Top 10)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Kategori")
        ax.set_ylabel("TL")
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()
        charts.append(_fig_to_imagereader(fig))

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    _pdf_draw_title_band(
        c, w, h, pdf_regular, pdf_bold,
        "Oldschool Esports Center Finans Raporu",
        f"Ay: {ym_str}   |   Olusturma: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    )
    y = h - 4.6 * cm

    y = _pdf_draw_metric_cards(
        c, w, y, pdf_regular, pdf_bold,
        [
            ("Nakit Gelir", cash_sum),
            ("Kart Gelir", card_sum),
            ("Toplam Gelir", total_rev),
            ("Toplam Gider", total_exp),
            ("Net", net),
        ],
    )

    c.setFillColorRGB(0.08, 0.13, 0.24)
    c.setFont(pdf_bold, 12)
    pdf_draw(c, 2 * cm, y, "Gider Dagilimi (Kategori)")
    c.setFillColorRGB(0, 0, 0)
    y -= 0.45 * cm

    top = by_cat.head(12) if len(by_cat) else by_cat
    if len(top):
        table_x = 2 * cm
        table_w = w - 4 * cm
        row_h = 0.58 * cm

        c.setFillColorRGB(0.92, 0.95, 0.99)
        c.setStrokeColorRGB(0.82, 0.87, 0.94)
        c.roundRect(table_x, y - row_h, table_w, row_h, 4, stroke=1, fill=1)
        c.setFillColorRGB(0.1, 0.19, 0.35)
        c.setFont(pdf_bold, 10)
        pdf_draw(c, table_x + 0.25 * cm, y - 0.38 * cm, "Kategori")
        pdf_draw(c, table_x + table_w - 0.25 * cm, y - 0.38 * cm, "Tutar", right=True)
        y -= row_h

        for i, (_, r) in enumerate(top.iterrows()):
            if y < 4.8 * cm:
                c.showPage()
                _pdf_draw_title_band(c, w, h, pdf_regular, pdf_bold, "Oldschool Esports Center Finans Raporu", f"Ay: {ym_str}   |   Kategori Tablosu (devam)")
                y = h - 5.0 * cm

            shade = 0.985 if i % 2 == 0 else 0.965
            c.setFillColorRGB(shade, shade, shade)
            c.setStrokeColorRGB(0.9, 0.9, 0.9)
            c.rect(table_x, y - row_h, table_w, row_h, stroke=1, fill=1)
            c.setFillColorRGB(0.1, 0.1, 0.1)
            c.setFont(pdf_regular, 10)
            pdf_draw(c, table_x + 0.25 * cm, y - 0.38 * cm, str(r["Kategori"])[:40])
            pdf_draw(c, table_x + table_w - 0.25 * cm, y - 0.38 * cm, _pdf_money_plain(float(r["Tutar"])), right=True)
            y -= row_h
    else:
        c.setFont(pdf_regular, 10)
        c.setFillColorRGB(0.45, 0.45, 0.45)
        pdf_draw(c, 2 * cm, y, "Bu ay icin gider kaydi bulunamadi.")
        c.setFillColorRGB(0, 0, 0)

    if charts:
        c.showPage()
        y = h - 2 * cm
        c.setFillColorRGB(0.08, 0.13, 0.24)
        c.setFont(pdf_bold, 13)
        pdf_draw(c, 2 * cm, y, "Grafikler")
        c.setFillColorRGB(0, 0, 0)
        y -= 0.9 * cm
        for img in charts:
            c.setFillColorRGB(0.97, 0.98, 1.0)
            c.setStrokeColorRGB(0.85, 0.88, 0.93)
            c.roundRect(1.7 * cm, y - 9.9 * cm, w - 3.4 * cm, 9.7 * cm, 8, stroke=1, fill=1)
            c.drawImage(img, 2 * cm, y - 9.6 * cm, width=w - 4 * cm, height=9.1 * cm, preserveAspectRatio=True, anchor='n')
            y -= 10.8 * cm
            if y < 4 * cm:
                c.showPage()
                y = h - 2 * cm

    c.save()
    buf.seek(0)
    return buf.read()


def build_yearly_pdf(conn, year: int) -> bytes:
    pdf_regular, pdf_bold = get_pdf_fonts()
    start = date(year, 1, 1)
    end = date(year + 1, 1, 1)
    start_s, end_s = start.isoformat(), end.isoformat()

    rev = df_query(conn, """
        SELECT substr(d,1,7) AS ym, SUM(cash) AS cash, SUM(card) AS card, SUM(cash+card) AS total
        FROM daily_cash
        WHERE d >= ? AND d < ?
        GROUP BY substr(d,1,7)
        ORDER BY ym
    """, (start_s, end_s))

    exp = df_query(conn, """
        SELECT substr(d,1,7) AS ym, SUM(amount) AS expense
        FROM expense
        WHERE d >= ? AND d < ?
        GROUP BY substr(d,1,7)
        ORDER BY ym
    """, (start_s, end_s))

    df = pd.merge(rev, exp, on="ym", how="outer").fillna(0)
    df["net"] = df["total"] - df["expense"]

    cash_sum = float(df["cash"].sum()) if len(df) else 0.0
    card_sum = float(df["card"].sum()) if len(df) else 0.0
    total_rev = float(df["total"].sum()) if len(df) else 0.0
    total_exp = float(df["expense"].sum()) if len(df) else 0.0
    net = total_rev - total_exp

    charts = []
    if len(df):
        fig = plt.figure()
        plt.plot(df["ym"], df["total"], label="Gelir")
        plt.plot(df["ym"], df["expense"], label="Gider")
        plt.title("Aylık Gelir & Gider")
        plt.xlabel("Ay")
        plt.ylabel("₺")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        charts.append(_fig_to_imagereader(fig))

        fig = plt.figure()
        plt.bar(df["ym"], df["net"])
        plt.title("Aylık Net")
        plt.xlabel("Ay")
        plt.ylabel("₺")
        plt.xticks(rotation=45, ha="right")
        charts.append(_fig_to_imagereader(fig))

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    y = h - 2*cm
    c.setFont(pdf_bold, 16)
    pdf_draw(c, 2*cm, y, "Oldschool Esports Center Finans Raporu")
    y -= 0.8*cm
    c.setFont(pdf_regular, 11)
    pdf_draw(c, 2*cm, y, f"Yıl: {year}    Oluşturma: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 1.2*cm

    c.setFont(pdf_bold, 12)
    pdf_draw(c, 2*cm, y, "Yıllık Özet")
    y -= 0.6*cm
    c.setFont(pdf_regular, 11)
    for ln in [
        f"Nakit Gelir:  {tr_money(cash_sum)}",
        f"Kart Gelir:   {tr_money(card_sum)}",
        f"Toplam Gelir: {tr_money(total_rev)}",
        f"Toplam Gider: {tr_money(total_exp)}",
        f"Net:          {tr_money(net)}",
    ]:
        pdf_draw(c, 2*cm, y, ln)
        y -= 0.5*cm

    y -= 0.3*cm
    c.setFont(pdf_bold, 12)
    pdf_draw(c, 2*cm, y, "Aylık Tablo")
    y -= 0.6*cm
    c.setFont(pdf_regular, 10)
    for _, r in df.iterrows():
        pdf_draw(c, 2*cm, y, str(r["ym"]))
        pdf_draw(c, w-2*cm, y,
                 f"Gelir {tr_money(float(r['total']))[1:]}  |  Gider {tr_money(float(r['expense']))[1:]}  |  Net {tr_money(float(r['net']))[1:]}",
                 right=True)
        y -= 0.45*cm
        if y < 5*cm:
            c.showPage()
            y = h - 2*cm

    if charts:
        c.showPage()
        y = h - 2*cm
        c.setFont(pdf_bold, 12)
        pdf_draw(c, 2*cm, y, "Grafikler")
        y -= 0.8*cm
        for img in charts:
            c.drawImage(img, 2*cm, y-10*cm, width=w-4*cm, height=9.5*cm, preserveAspectRatio=True, anchor='n')
            y -= 11*cm
            if y < 4*cm:
                c.showPage()
                y = h - 2*cm

    c.save()
    buf.seek(0)
    return buf.read()

@st.cache_data(show_spinner=False)
def build_monthly_pdf_cached(ym_str: str, db_mode: str) -> bytes:
    c = get_conn()
    try:
        return build_monthly_pdf(c, ym_str)
    finally:
        try:
            c.close()
        except Exception:
            pass

@st.cache_data(show_spinner=False)
def build_yearly_pdf_cached(year: int, db_mode: str) -> bytes:
    c = get_conn()
    try:
        return build_yearly_pdf(c, year)
    finally:
        try:
            c.close()
        except Exception:
            pass


# ---------- UI ----------

# ---------- THEME / STYLE ----------
SOFT_CSS = """<style>
/* Soft Professional theme */
.stApp { background: #f4f6f8; color: #1f2937; }
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid rgba(31,41,55,0.08); }
h1, h2, h3, h4 { color: #111827; }
div[data-testid="stMetric"] {
  background: #ffffff;
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid rgba(31,41,55,0.10);
  box-shadow: 0 2px 8px rgba(17,24,39,0.05);
}
div[data-testid="stMetric"] label { color: rgba(17,24,39,0.70) !important; }
.stButton>button, .stFormSubmitButton>button, .stDownloadButton>button {
  border-radius: 12px;
  padding: 0.62rem 0.95rem;
  border: 1px solid rgba(31,41,55,0.12);
  background: #ffffff;
  color: #1f2937;
}
.stButton>button:hover, .stFormSubmitButton>button:hover, .stDownloadButton>button:hover {
  background: rgba(17,24,39,0.03);
  border-color: rgba(31,41,55,0.18);
}
.stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background: #c0392b;
  border-color: #a93226;
  color: #ffffff !important;
}
.stButton>button[kind="primary"]:hover, .stFormSubmitButton>button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
  background: #a93226;
  border-color: #922b21;
}
.stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background: #c0392b !important;
  border-color: #a93226 !important;
}
.stButton>button[kind="primary"]:disabled, .stFormSubmitButton>button[kind="primary"]:disabled, button[data-testid="stBaseButton-primary"]:disabled {
  background: #d98880 !important;
  border-color: #d98880 !important;
  color: #ffffff !important;
  opacity: 1 !important;
}
div[data-testid="stDataFrame"] {
  background: #ffffff;
  border: 1px solid rgba(31,41,55,0.10);
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 2px 10px rgba(17,24,39,0.04);
}
hr { border-color: rgba(31,41,55,0.08); }
.mobile-card {
  background: linear-gradient(180deg, #ffffff 0%, #fbfcfd 100%);
  border: 1px solid rgba(31,41,55,0.12);
  border-radius: 14px;
  padding: 0.75rem 0.85rem;
  box-shadow: 0 4px 14px rgba(17,24,39,0.06);
  margin-bottom: 0.65rem;
}
.mobile-card-title {
  font-weight: 700;
  font-size: 0.95rem;
  color: #111827;
  border-left: 3px solid #c0392b;
  padding-left: 0.5rem;
  margin-bottom: 0.45rem;
}
.mobile-card-body {
  display: grid;
  row-gap: 0.25rem;
}
.mobile-card-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.8rem;
}
.mobile-card-label {
  font-size: 0.82rem;
  color: #6b7280;
}
.mobile-card-value {
  font-size: 0.88rem;
  font-weight: 600;
  color: #1f2937;
  text-align: right;
}
.mobile-badge {
  display: inline-block;
  padding: 0.16rem 0.5rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 700;
  line-height: 1.25;
  border: 1px solid transparent;
}
.mobile-badge-default { background: #f3f4f6; color: #374151; border-color: #d1d5db; }
.mobile-badge-category { background: #fff7ed; color: #9a3412; border-color: #fed7aa; }
.mobile-badge-cash { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
.mobile-badge-card { background: #eff6ff; color: #1e3a8a; border-color: #bfdbfe; }
.mobile-badge-transfer { background: #fff7ed; color: #9a3412; border-color: #fed7aa; }
.mobile-badge-manual { background: #f5f3ff; color: #5b21b6; border-color: #ddd6fe; }
.mobile-badge-auto { background: #f3f4f6; color: #374151; border-color: #d1d5db; }
.mobile-badge-rent { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
.mobile-badge-salary { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
.mobile-badge-loan { background: #eff6ff; color: #1e3a8a; border-color: #bfdbfe; }
.mobile-badge-installment { background: #fff7ed; color: #9a3412; border-color: #fed7aa; }
.mobile-badge-bills { background: #eef2ff; color: #3730a3; border-color: #c7d2fe; }
.mobile-badge-shopping { background: #fdf2f8; color: #9d174d; border-color: #fbcfe8; }
.mobile-progress-wrap {
  margin-top: 0.6rem;
  margin-bottom: 0.5rem;
  background: #ffffff;
  border: 1px solid rgba(31,41,55,0.10);
  border-radius: 12px;
  padding: 0.65rem 0.7rem;
}
.mobile-progress-title {
  font-size: 0.84rem;
  font-weight: 700;
  color: #111827;
  margin-bottom: 0.45rem;
}
.mobile-progress-item { margin-bottom: 0.45rem; }
.mobile-progress-item:last-child { margin-bottom: 0; }
.mobile-progress-head {
  display: flex;
  justify-content: space-between;
  gap: 0.6rem;
  font-size: 0.78rem;
  color: #374151;
  margin-bottom: 0.16rem;
}
.mobile-progress-track {
  width: 100%;
  height: 7px;
  background: #eef2f7;
  border-radius: 999px;
  overflow: hidden;
}
.mobile-progress-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #c0392b 0%, #e67e22 100%);
}

/* Mobile optimizations */
@media (max-width: 768px) {
  .block-container {
    padding-top: 0.8rem;
    padding-left: 0.7rem;
    padding-right: 0.7rem;
    padding-bottom: 1rem;
  }
  h1 { font-size: 1.35rem !important; }
  h2, h3 { font-size: 1.1rem !important; }
  div[data-testid="stMetric"] {
    padding: 10px 12px;
    border-radius: 12px;
  }
  .stButton>button, .stFormSubmitButton>button, .stDownloadButton>button {
    width: 100%;
    min-height: 2.4rem;
  }
  div[data-testid="stDataFrame"] {
    border-radius: 10px;
  }
  .mobile-card {
    border-radius: 12px;
    padding: 0.68rem 0.72rem;
  }
  .mobile-card-title {
    font-size: 0.9rem;
  }
  .mobile-card-label {
    font-size: 0.8rem;
  }
  .mobile-card-value {
    font-size: 0.86rem;
  }
  .mobile-badge {
    font-size: 0.72rem;
  }
  .mobile-progress-head {
    font-size: 0.75rem;
  }
}
</style>"""

# ---------- LOGIN SYSTEM ----------
def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_role" not in st.session_state:
        st.session_state.user_role = None
    if "auth_username" not in st.session_state:
        st.session_state.auth_username = None
    if "_login_submit_by_enter" not in st.session_state:
        st.session_state._login_submit_by_enter = False

    if st.session_state.authenticated:
        return True

    st.title("🔐 Giriş Yap")

    def _mark_login_submit():
        st.session_state._login_submit_by_enter = True

    username = st.text_input("Kullanıcı Adı", key="login_username")
    password = st.text_input("Şifre", type="password", key="login_password", on_change=_mark_login_submit)
    clicked = st.button("Giriş", type="primary")
    submitted = clicked or bool(st.session_state._login_submit_by_enter)

    if submitted:
        st.session_state._login_submit_by_enter = False
        auth_conn = get_conn()
        try:
            init_db(auth_conn)
            role = authenticate_user(auth_conn, username, password)
            if role in ("admin", "cash_only"):
                st.session_state.authenticated = True
                st.session_state.user_role = role
                st.session_state.auth_username = str(username).strip()
                st.rerun()
            else:
                st.error("Hatalı kullanıcı adı veya şifre")
        except Exception as e:
            st.error(f"Giriş doğrulaması başarısız: {e}")
        finally:
            try:
                auth_conn.close()
            except Exception:
                pass

    return False


if not check_login():
    st.stop()
if st.session_state.get("user_role") not in ("admin", "cash_only"):
    st.session_state.user_role = "admin"
user_role = st.session_state.get("user_role", "admin")

# ---------- END LOGIN ----------
st.markdown(SOFT_CSS, unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align:center;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
if "undo_flash" in st.session_state:
    ok, msg = st.session_state.pop("undo_flash")
    if ok:
        st.success(msg)
    else:
        st.error(msg)

def get_session_conn():
    conn = st.session_state.get("_db_conn")
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
    conn = get_conn()
    st.session_state["_db_conn"] = conn
    return conn

conn = get_session_conn()
if "_db_bootstrapped" not in st.session_state:
    init_db(conn)
    sync_postgres_sequences(conn)
    ensure_backup()
    # Seed defaults only once per session; rerun cost is high on remote Postgres.
    today = date.today()
    default_month = ym(today)
    seed_defaults_if_empty(conn, default_month)
    st.session_state["_db_bootstrapped"] = True

today = date.today()
default_month = ym(today)

with st.sidebar:
    st.header("Ay Seçimi")
    if "selected_month_ui" not in st.session_state or not is_valid_ym(st.session_state.selected_month_ui):
        st.session_state.selected_month_ui = default_month

    nav_l, nav_r = st.columns([1, 1])
    with nav_l:
        if st.button("◀ Önceki", use_container_width=True):
            st.session_state.selected_month_ui = shift_ym(st.session_state.selected_month_ui, -1)
    with nav_r:
        if st.button("Sonraki ▶", use_container_width=True):
            st.session_state.selected_month_ui = shift_ym(st.session_state.selected_month_ui, 1)

    month_options = [shift_ym(default_month, -i) for i in range(0, 18)]
    current_sel = st.session_state.selected_month_ui if st.session_state.selected_month_ui in month_options else default_month
    quick_sel = st.selectbox("Hızlı Seçim", month_options, index=month_options.index(current_sel))
    st.session_state.selected_month_ui = quick_sel

    with st.expander("Manuel giriş (YYYY-AA)"):
        manual_month = st.text_input("Ay", value=st.session_state.selected_month_ui, key="manual_month_input")
        if st.button("Uygula", use_container_width=True):
            if is_valid_ym(manual_month.strip()):
                st.session_state.selected_month_ui = manual_month.strip()
                st.rerun()
            else:
                st.warning(f"Geçersiz ay formatı: '{manual_month}'.")

    selected_month = st.session_state.selected_month_ui.strip()
    st.caption("Bu ay üzerinden raporlar gösterilir.")
    st.divider()
    if user_role == "admin":
        locked_now = is_month_locked(conn, selected_month)
        lock_label = "🔒 Bu ay kilitli" if locked_now else "🔓 Bu ay açık"
        new_locked = st.toggle(lock_label, value=locked_now, help="Kilitliyken bu ay için yeni kayıt ekleme/düzenleme kapatılır.")
        if new_locked != locked_now:
            set_month_lock(conn, selected_month, new_locked)
            st.rerun()
        if st.button("Bu ay oto giderleri yenile", use_container_width=True, help="Otomatik satirlari silip mevcut kurallardan yeniden olusturur."):
            rebuild_auto_for_month(conn, selected_month)
            st.success("Bu ayin otomatik giderleri yenilendi.")
            st.rerun()
    else:
        st.caption("Yetki: Sadece Günlük Kasa")
    st.divider()
    st.header("Menü")
    if user_role == "admin":
        page_options = ["🏠 Dashboard", "💰 Günlük Kasa", "🧾 Gider Yönetimi", "🔁 Sabitler (Kurallar)", "🏦 Krediler", "💳 Kart Taksitleri", "📤 Veri Dökümü", "⚙️ Ayarlar"]
    else:
        page_options = ["💰 Günlük Kasa"]
    if st.session_state.get("page_ui") not in page_options:
        st.session_state["page_ui"] = page_options[0]
    page = st.radio("Sayfa", page_options, key="page_ui")
    st.divider()
    mobile_mode = st.toggle("Mobil görünüm", value=False, help="Dar ekranlarda daha rahat kullanım için düzeni sadeleştirir.")
    undo_action = st.session_state.get("undo_action")
    if undo_action:
        st.caption(f"Son işlem: {undo_action.get('label', 'İşlem')}")
        if st.button("↩️ Son işlemi geri al", use_container_width=True):
            ok, msg = apply_undo_action(conn)
            st.session_state["undo_flash"] = (ok, msg)
            st.rerun()

if st.session_state.get("_auto_generated_month") != selected_month:
    ensure_month_open(conn, selected_month)
    auto_generate_for_month(conn, selected_month)
    st.session_state["_auto_generated_month"] = selected_month

# --------- DASHBOARD ----------
if page == "🏠 Dashboard":
    st.subheader(f"📌 {selected_month} Özeti")

    db_mode = "postgres" if USE_POSTGRES else "sqlite"
    rev, exp = load_dashboard_month_data(selected_month, db_mode)
    cash_sum = float(rev["cash"].sum()) if len(rev) else 0.0
    card_sum = float(rev["card"].sum()) if len(rev) else 0.0
    total_rev = float(rev["total"].sum()) if len(rev) else 0.0
    total_exp = float(exp["amount"].sum()) if len(exp) else 0.0
    net = total_rev - total_exp

    if mobile_mode:
        m1, m2 = st.columns(2)
        m1.metric("Toplam Gelir", tr_money(total_rev))
        m2.metric("Toplam Gider", tr_money(total_exp))
        m3, m4 = st.columns(2)
        m3.metric("Net", tr_money(net))
        m4.metric("Bakiye (Ay)", tr_money(net))
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam Gelir", tr_money(total_rev))
        c2.metric("Toplam Gider", tr_money(total_exp))
        c3.metric("Net", tr_money(net))
        c4.metric("Bakiye (Ay)", tr_money(net))

    year_sel = int(selected_month.split("-")[0])
    if mobile_mode:
        if st.button("🔄 Yenile"):
            st.rerun()
        st.download_button("Aylık Rapor", data=build_monthly_pdf_cached(selected_month, db_mode),
                           file_name=f"finans_raporu_{selected_month}.pdf", use_container_width=True)
        st.download_button("Yıllık Rapor", data=build_yearly_pdf_cached(year_sel, db_mode),
                           file_name=f"finans_raporu_{year_sel}.pdf", use_container_width=True)
    else:
        a1, a3, a4 = st.columns([1, 1.2, 1.2])
        with a1:
            if st.button("🔄 Yenile"):
                st.rerun()
        with a3:
            st.download_button("Aylık Rapor", data=build_monthly_pdf_cached(selected_month, db_mode),
                               file_name=f"finans_raporu_{selected_month}.pdf")
        with a4:
            st.download_button("Yıllık Rapor", data=build_yearly_pdf_cached(year_sel, db_mode),
                               file_name=f"finans_raporu_{year_sel}.pdf")

    st.divider()

    def render_income_table():
        st.markdown("<h3 style='text-align:center;'>Gelir</h3>", unsafe_allow_html=True)
        if len(rev):
            rev_display = rev.copy()
            rev_display.rename(columns={
                "d": "Tarih",
                "cash": "Nakit",
                "card": "Kart",
                "total": "Toplam"
            }, inplace=True)
            if mobile_mode:
                render_mobile_cards(
                    rev_display,
                    fields=["Tarih", "Toplam", "Nakit", "Kart"],
                    empty_text="Bu ay için günlük kasa girişi yok.",
                    amount_fields={"Toplam", "Nakit", "Kart"},
                    title_fields=["Tarih"],
                )
            else:
                st.dataframe(rev_display, use_container_width=True, hide_index=True)
        else:
            st.info("Bu ay için günlük kasa girişi yok.")

    def render_expense_breakdown():
        st.markdown("<h3 style='text-align:center;'>Gider</h3>", unsafe_allow_html=True)
        if len(exp):
            exp_disp = format_expense_for_display(exp)
            by_cat = exp_disp.groupby("Kategori", as_index=False)["Tutar"].sum().sort_values("Tutar", ascending=False)
            if mobile_mode:
                render_mobile_cards(
                    by_cat,
                    fields=["Kategori", "Tutar"],
                    empty_text="Bu ay için gider kaydı yok.",
                    amount_fields={"Tutar"},
                    title_fields=["Kategori"],
                )
                render_mobile_category_progress(by_cat, top_n=5)
            else:
                st.dataframe(by_cat, use_container_width=True, hide_index=True)
            chart_df = by_cat.set_index("Kategori")
            st.bar_chart(chart_df)
        else:
            st.info("Bu ay için gider kaydı yok.")

    if mobile_mode:
        render_income_table()
        render_expense_breakdown()
    else:
        left, right = st.columns([1.2, 1])
        with left:
            render_income_table()
        with right:
            render_expense_breakdown()

    st.divider()
    st.markdown("### Gider Listesi")
    if len(exp):
        show = format_expense_for_display(exp)
        if mobile_mode:
            render_mobile_cards(
                show,
                fields=["Tarih", "Kategori", "Tutar", "Ödeme", "Kaynak", "Notlar"],
                empty_text="Gider yok.",
                amount_fields={"Tutar"},
                title_fields=["Tarih"],
                badge_fields={"Kategori", "Ödeme", "Kaynak"},
            )
        else:
            st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("Gider yok.")

# --------- DAILY CASH ----------
elif page == "💰 Günlük Kasa":
    st.subheader("💰 Günlük Kasa")
    if is_month_locked(conn, selected_month):
        st.warning("Bu ay kilitli. Kayıt ekleme/düzenleme kapalı.")
    if "cash_form_date" not in st.session_state:
        st.session_state.cash_form_date = today
    if "cash_form_cash" not in st.session_state:
        st.session_state.cash_form_cash = 0.0
    if "cash_form_card" not in st.session_state:
        st.session_state.cash_form_card = 0.0
    if "cash_form_note" not in st.session_state:
        st.session_state.cash_form_note = ""
    if "cash_form_last_loaded_date" not in st.session_state:
        st.session_state.cash_form_last_loaded_date = None
    d = st.date_input("Tarih", key="cash_form_date")

    selected_cash_date = d.isoformat()
    if st.session_state.cash_form_last_loaded_date != selected_cash_date:
        existing = conn.execute(
            "SELECT cash, card, note FROM daily_cash WHERE d=?",
            (selected_cash_date,),
        ).fetchone()
        if existing:
            st.session_state.cash_form_cash = float(existing[0] or 0.0)
            st.session_state.cash_form_card = float(existing[1] or 0.0)
            st.session_state.cash_form_note = "" if existing[2] is None else str(existing[2])
        else:
            st.session_state.cash_form_cash = 0.0
            st.session_state.cash_form_card = 0.0
            st.session_state.cash_form_note = ""
        st.session_state.cash_form_last_loaded_date = selected_cash_date
        st.rerun()

    cash = st.number_input("Nakit (₺)", min_value=0.0, step=100.0, format="%.2f", key="cash_form_cash")
    card = st.number_input("Kart (₺)", min_value=0.0, step=100.0, format="%.2f", key="cash_form_card")
    note = st.text_input("Not (opsiyonel)", key="cash_form_note")
    st.caption(f"Toplam: {tr_money(float(cash) + float(card))}")

    locked = is_month_locked(conn, selected_month)

    if st.button("Kaydet / Güncelle", type="primary", disabled=locked):
        prev = conn.execute("SELECT d, cash, card, note FROM daily_cash WHERE d=?", (d.isoformat(),)).fetchone()
        conn.execute("""
            INSERT INTO daily_cash(d, cash, card, note)
            VALUES(?,?,?,?)
            ON CONFLICT(d) DO UPDATE SET cash=excluded.cash, card=excluded.card, note=excluded.note
        """, (d.isoformat(), float(cash), float(card), note.strip() or None))
        conn.commit()
        prev_payload = None
        if prev:
            prev_payload = {"d": prev[0], "cash": float(prev[1]), "card": float(prev[2]), "note": prev[3]}
        set_undo_action({
            "type": "daily_cash_upsert",
            "d": d.isoformat(),
            "previous": prev_payload,
            "label": f"Günlük Kasa ({d.isoformat()})",
        })
        st.success("Kaydedildi ✅")

    st.divider()
    st.markdown("### Seçili Ay Kayıtları")
    start, end = month_range(selected_month)
    rev = df_query(conn, """
        SELECT d AS Tarih, cash AS Nakit, card AS Kart, (cash+card) AS Toplam, note AS Notlar
        FROM daily_cash
        WHERE d >= ? AND d < ?
        ORDER BY d DESC
    """, (start.isoformat(), end.isoformat()))
    if mobile_mode:
        render_mobile_cards(
            rev,
            fields=["Tarih", "Toplam", "Nakit", "Kart", "Notlar"],
            empty_text="Bu ay için günlük kasa kaydı yok.",
            amount_fields={"Toplam", "Nakit", "Kart"},
            title_fields=["Tarih"],
        )
    else:
        st.dataframe(rev, use_container_width=True, hide_index=True)

# --------- MANUAL EXPENSE ----------
elif page == "🧾 Gider Yönetimi":
    st.subheader("🧾 Gider Yönetimi")
    if is_month_locked(conn, selected_month):
        st.warning("Bu ay kilitli. Manuel gider ekleme/düzenleme kapalı.")
    active_categories = get_active_categories(conn)
    if len(active_categories) == 0:
        st.error("Aktif kategori yok. Ayarlar bölümünden en az bir kategori aktif et.")
        st.stop()

    if "exp_form_date" not in st.session_state:
        st.session_state.exp_form_date = today
    if "exp_form_category" not in st.session_state or st.session_state.exp_form_category not in active_categories:
        st.session_state.exp_form_category = active_categories[0]
    if "exp_form_amount" not in st.session_state:
        st.session_state.exp_form_amount = 0.0
    if "exp_form_pay_method" not in st.session_state:
        st.session_state.exp_form_pay_method = "Nakit"
    if "exp_form_note" not in st.session_state:
        st.session_state.exp_form_note = ""

    templates = {
        "Kira (Havale)": {"category": "Kira", "pay_method": "Havale", "note": "Aylık kira"},
        "İnternet (Havale)": {"category": "İnternet", "pay_method": "Havale", "note": "İnternet faturası"},
        "Elektrik (Havale)": {"category": "Elektrik", "pay_method": "Havale", "note": "Elektrik faturası"},
        "Alışveriş (Kart)": {"category": "Alışveriş", "pay_method": "Kart", "note": "Market/temel ihtiyaç"},
    }

    t1, t2 = st.columns([2, 1])
    with t1:
        chosen_template = st.selectbox("Hızlı Şablon", ["Seçiniz"] + list(templates.keys()), key="exp_template_select")
    with t2:
        st.markdown("&nbsp;")
        if st.button("Şablonu Uygula", use_container_width=True):
            if chosen_template in templates:
                tpl = templates[chosen_template]
                tpl_cat = tpl["category"] if tpl["category"] in active_categories else active_categories[0]
                st.session_state.exp_form_category = tpl_cat
                st.session_state.exp_form_pay_method = tpl["pay_method"]
                st.session_state.exp_form_note = tpl["note"]
                st.rerun()

    d = st.date_input("Tarih", key="exp_form_date")
    category = st.selectbox("Kategori", active_categories, key="exp_form_category")
    amount = st.number_input("Tutar (₺)", min_value=0.0, step=100.0, format="%.2f", key="exp_form_amount")
    pay_method = st.selectbox("Ödeme Tipi", ["Nakit", "Kart", "Havale"], key="exp_form_pay_method")
    note = st.text_input("Not (opsiyonel)", key="exp_form_note")

    locked = is_month_locked(conn, selected_month)

    if st.button("Gider Kaydet", type="primary", disabled=locked):
        cur = conn.execute(
            "INSERT INTO expense(d, category, amount, pay_method, note, source) VALUES(?,?,?,?,?, 'manual')",
            (d.isoformat(), category, float(amount), pay_method, note.strip() or None),
        )
        conn.commit()
        set_undo_action({
            "type": "expense_add",
            "id": int(cur.lastrowid),
            "label": f"Gider Ekle ({d.isoformat()} | {category})",
        })
        st.session_state.exp_form_amount = 0.0
        st.session_state.exp_form_note = ""
        st.success("Gider eklendi ✅")
        st.rerun()

    st.divider()
    st.markdown("### Seçili Ay Giderleri")
    start, end = month_range(selected_month)

    exp_raw = df_query(conn, """
        SELECT id, d, category, amount, pay_method, source, note
        FROM expense
        WHERE d >= ? AND d < ?
        ORDER BY d DESC
    """, (start.isoformat(), end.isoformat()))

    # Ekranda gösterilecek tablo (otomatik notları gizler, kategoriyi temizler, Türkçe kolonlar)
    exp_disp = format_expense_for_display(exp_raw)

    if mobile_mode:
        render_mobile_cards(
            exp_disp,
            fields=["Tarih", "Kategori", "Tutar", "Ödeme", "Kaynak", "Notlar"],
            empty_text="Bu ay için gider kaydı yok.",
            amount_fields={"Tutar"},
            title_fields=["Tarih"],
            badge_fields={"Kategori", "Ödeme", "Kaynak"},
        )
    else:
        st.dataframe(exp_disp, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### ✏️ Manuel Gider Düzenle / Sil")

    manual = exp_raw[exp_raw["source"].astype(str) == "manual"].copy() if len(exp_raw) else exp_raw

    if manual is None or len(manual) == 0:
        st.info("Bu ay için düzenlenebilir (manuel) gider yok.")
    else:
        # Seçim listesi için okunabilir etiket üretelim
        manual["note_disp"] = manual["note"].fillna("").astype(str)
        manual["label"] = (
            manual["d"].astype(str)
            + " | "
            + clean_category_series(manual["category"]).fillna("").astype(str)
            + " | ₺"
            + manual["amount"].astype(float).round(2).astype(str)
            + " | "
            + manual["pay_method"].astype(str)
        )
        # Not varsa etikete ekle
        has_note = manual["note_disp"].str.len() > 0
        manual.loc[has_note, "label"] = manual.loc[has_note, "label"] + " | " + manual.loc[has_note, "note_disp"]

        # En üstte en yeni gözüksün
        manual = manual.reset_index(drop=True)

        sel_id = st.selectbox(
            "Düzenlenecek gideri seç",
            options=manual["id"].tolist(),
            format_func=lambda _id: manual.loc[manual["id"] == _id, "label"].iloc[0],
        )

        row = manual.loc[manual["id"] == sel_id].iloc[0]

        # Form
        with st.form("edit_manual_expense"):
            ed = st.date_input("Tarih (düzenle)", value=date.fromisoformat(str(row["d"])))
            ecat = st.text_input("Kategori (düzenle)", value=str(row["category"]))
            eamount = st.number_input("Tutar (₺) (düzenle)", min_value=0.0, step=100.0, format="%.2f", value=float(row["amount"]))
            epm = st.selectbox("Ödeme Tipi (düzenle)", ["Nakit", "Kart", "Havale"], index=["Nakit","Kart","Havale"].index(str(row["pay_method"])) if str(row["pay_method"]) in ["Nakit","Kart","Havale"] else 0)
            enote = st.text_input("Not (düzenle)", value=str(row["note"]) if pd.notna(row["note"]) else "")

            if mobile_mode:
                save = st.form_submit_button("Kaydet (Güncelle)", type="primary")
                delete = st.form_submit_button("Sil", type="secondary")
            else:
                c1, c2 = st.columns(2)
                save = c1.form_submit_button("Kaydet (Güncelle)", type="primary")
                delete = c2.form_submit_button("Sil", type="secondary")

        if save:
            old_payload = {
                "id": int(row["id"]),
                "d": str(row["d"]),
                "category": str(row["category"]),
                "amount": float(row["amount"]),
                "pay_method": str(row["pay_method"]),
                "note": None if pd.isna(row["note"]) else str(row["note"]),
            }
            conn.execute(
                "UPDATE expense SET d=?, category=?, amount=?, pay_method=?, note=? WHERE id=? AND source='manual'",
                (ed.isoformat(), ecat.strip(), float(eamount), epm, enote.strip() or None, int(sel_id)),
            )
            conn.commit()
            set_undo_action({
                "type": "expense_update",
                "old": old_payload,
                "label": f"Gider Güncelle (#{int(sel_id)})",
            })
            st.success("Gider güncellendi ✅")
            st.rerun()

        if delete:
            row_payload = {
                "id": int(row["id"]),
                "d": str(row["d"]),
                "category": str(row["category"]),
                "amount": float(row["amount"]),
                "pay_method": str(row["pay_method"]),
                "note": None if pd.isna(row["note"]) else str(row["note"]),
                "source": str(row["source"]),
            }
            conn.execute("DELETE FROM expense WHERE id=? AND source='manual'", (int(sel_id),))
            conn.commit()
            set_undo_action({
                "type": "expense_delete",
                "row": row_payload,
                "label": f"Gider Sil (#{int(sel_id)})",
            })
            st.success("Gider silindi ✅")
            st.rerun()


# --------- RECURRING RULES ----------
elif page == "🔁 Sabitler (Kurallar)":
    st.subheader("🔁 Sabit Gider Kuralları")
    st.caption("Her ay otomatik gider kaydı oluşturur. Bu sürüm ilk açılışta senin sabitlerini otomatik ekler.")
    with st.form("rule_form"):
        name = st.text_input("Kural Adı (örn: Kira)")
        category = st.text_input("Kategori (örn: Kira)")
        amount = st.number_input("Aylık Tutar (₺)", min_value=0.0, step=100.0, format="%.2f")
        dom = st.number_input("Ayın Kaçı", min_value=1, max_value=28, step=1)
        pay_method = st.selectbox("Ödeme Tipi", ["Nakit", "Kart", "Havale"])
        submitted = st.form_submit_button("Kural Ekle", type="primary")
    if submitted:
        conn.execute("INSERT INTO recurring_rule(name, category, amount, day_of_month, pay_method, active) VALUES(?,?,?,?,?,1)",
                     (name.strip(), category.strip(), float(amount), int(dom), pay_method))
        conn.commit()
        st.success("Kural eklendi ✅")
        auto_generate_for_month(conn, selected_month)

    st.divider()
    rules = df_query(conn, "SELECT id, name AS Ad, category AS Kategori, amount AS Tutar, day_of_month AS Gun, pay_method AS Odeme, active AS Aktif FROM recurring_rule ORDER BY id DESC")
    st.dataframe(rules, use_container_width=True, hide_index=True)


    st.divider()
    st.markdown("### ✏️ Sabit Kural Düzenle / Sil")
    raw_rules = df_query(conn, "SELECT id, name, category, amount, day_of_month, pay_method, active FROM recurring_rule ORDER BY id DESC")
    if len(raw_rules) == 0:
        st.info("Düzenlenecek kural yok.")
    else:
        sel = st.selectbox(
            "Düzenlenecek kuralı seç",
            raw_rules["id"].tolist(),
            format_func=lambda rid: f"#{rid} • {raw_rules.loc[raw_rules['id']==rid, 'name'].iloc[0]}",
            key="rule_edit_select",
        )
        r = raw_rules.loc[raw_rules["id"] == sel].iloc[0]

        with st.form("rule_edit_form"):
            e_name = st.text_input("Kural Adı", value=str(r["name"]), key="rule_edit_name")
            e_cat  = st.text_input("Kategori", value=str(r["category"]), key="rule_edit_cat")
            e_amt  = st.number_input("Aylık Tutar (₺)", min_value=0.0, step=100.0, format="%.2f", value=float(r["amount"]), key="rule_edit_amt")
            e_dom  = st.number_input("Ayın Kaçı", min_value=1, max_value=28, step=1, value=int(r["day_of_month"]), key="rule_edit_dom")
            e_pm   = st.selectbox("Ödeme Tipi", ["Nakit", "Kart", "Havale"], index=["Nakit","Kart","Havale"].index(str(r["pay_method"])), key="rule_edit_pm")
            e_active = st.checkbox("Aktif", value=bool(int(r["active"])), key="rule_edit_active")

            if mobile_mode:
                save = st.form_submit_button("Kaydet (Güncelle)", type="primary")
                delete = st.form_submit_button("Sil (Kalıcı)")
                toggle = st.form_submit_button("Sadece Aktif/Pasif Değiştir")
            else:
                c1, c2, c3 = st.columns([1,1,1])
                save = c1.form_submit_button("Kaydet (Güncelle)", type="primary")
                delete = c2.form_submit_button("Sil (Kalıcı)")
                toggle = c3.form_submit_button("Sadece Aktif/Pasif Değiştir")

        if save:
            conn.execute(
                "UPDATE recurring_rule SET name=?, category=?, amount=?, day_of_month=?, pay_method=?, active=? WHERE id=?",
                (e_name.strip(), e_cat.strip(), float(e_amt), int(e_dom), e_pm, 1 if e_active else 0, int(sel))
            )
            conn.commit()
            auto_generate_for_month(conn, selected_month)
            st.success("Kural güncellendi ✅")
            st.rerun()

        if toggle:
            conn.execute("UPDATE recurring_rule SET active=? WHERE id=?", (0 if bool(int(r['active'])) else 1, int(sel)))
            conn.commit()
            auto_generate_for_month(conn, selected_month)
            st.success("Kural durumu güncellendi ✅")
            st.rerun()

        if delete:
            conn.execute("DELETE FROM recurring_rule WHERE id=?", (int(sel),))
            conn.commit()
            st.success("Kural silindi ✅")
            st.rerun()

# --------- LOANS ----------
elif page == "🏦 Krediler":
    st.subheader("🏦 Krediler (Belirli süreli)")
    st.caption("Her ayın 1'inde otomatik gider kaydı düşer. İlk açılışta Araba/Okul kredisi otomatik eklenir.")
    with st.form("loan_form"):
        name = st.text_input("Kredi Adı")
        monthly = st.number_input("Aylık Taksit (₺)", min_value=0.0, step=100.0, format="%.2f")
        start_month = st.text_input("Başlangıç Ayı (YYYY-AA)", value=selected_month)
        months_total = st.number_input("Toplam Ay", min_value=1, max_value=240, step=1, value=12)
        pay_method = st.selectbox("Ödeme Tipi", ["Nakit", "Kart", "Havale"])
        submitted = st.form_submit_button("Kredi Ekle", type="primary")
    if submitted:
        conn.execute("INSERT INTO loan(name, monthly_amount, start_month, months_total, pay_method, active) VALUES(?,?,?,?,?,1)",
                     (name.strip(), float(monthly), start_month.strip(), int(months_total), pay_method))
        conn.commit()
        st.success("Kredi eklendi ✅")
        auto_generate_for_month(conn, selected_month)

    st.divider()
    loans = df_query(conn, "SELECT id, name AS Ad, monthly_amount AS Aylik, start_month AS Baslangic, months_total AS Ay, pay_method AS Odeme, active AS Aktif FROM loan ORDER BY id DESC")
    st.dataframe(loans, use_container_width=True, hide_index=True)


    st.divider()
    st.markdown("### ✏️ Kredi Düzenle / Sil")
    raw_loans = df_query(conn, "SELECT id, name, monthly_amount, start_month, months_total, pay_method, active FROM loan ORDER BY id DESC")
    if len(raw_loans) == 0:
        st.info("Düzenlenecek kredi yok.")
    else:
        sel = st.selectbox(
            "Düzenlenecek krediyi seç",
            raw_loans["id"].tolist(),
            format_func=lambda lid: f"#{lid} • {raw_loans.loc[raw_loans['id']==lid, 'name'].iloc[0]}",
            key="loan_edit_select",
        )
        r = raw_loans.loc[raw_loans["id"] == sel].iloc[0]

        with st.form("loan_edit_form"):
            e_name = st.text_input("Kredi Adı", value=str(r["name"]), key="loan_edit_name")
            e_monthly = st.number_input("Aylık Taksit (₺)", min_value=0.0, step=100.0, format="%.2f", value=float(r["monthly_amount"]), key="loan_edit_monthly")
            e_start = st.text_input("Başlangıç Ayı (YYYY-AA)", value=str(r["start_month"]), key="loan_edit_start")
            e_total = st.number_input("Toplam Ay", min_value=1, max_value=240, step=1, value=int(r["months_total"]), key="loan_edit_total")
            pm_opts = ["Nakit", "Kart", "Havale"]
            e_pm = st.selectbox("Ödeme Tipi", pm_opts, index=pm_opts.index(str(r["pay_method"])), key="loan_edit_pm")
            e_active = st.checkbox("Aktif", value=bool(int(r["active"])), key="loan_edit_active")

            if mobile_mode:
                save = st.form_submit_button("Kaydet (Güncelle)", type="primary")
                delete = st.form_submit_button("Sil (Kalıcı)")
                toggle = st.form_submit_button("Sadece Aktif/Pasif Değiştir")
            else:
                c1, c2, c3 = st.columns([1,1,1])
                save = c1.form_submit_button("Kaydet (Güncelle)", type="primary")
                delete = c2.form_submit_button("Sil (Kalıcı)")
                toggle = c3.form_submit_button("Sadece Aktif/Pasif Değiştir")

        if save:
            conn.execute(
                "UPDATE loan SET name=?, monthly_amount=?, start_month=?, months_total=?, pay_method=?, active=? WHERE id=?",
                (e_name.strip(), float(e_monthly), e_start.strip(), int(e_total), e_pm, 1 if e_active else 0, int(sel))
            )
            conn.commit()
            auto_generate_for_month(conn, selected_month)
            st.success("Kredi güncellendi ✅")
            st.rerun()

        if toggle:
            conn.execute("UPDATE loan SET active=? WHERE id=?", (0 if bool(int(r['active'])) else 1, int(sel)))
            conn.commit()
            auto_generate_for_month(conn, selected_month)
            st.success("Kredi durumu güncellendi ✅")
            st.rerun()

        if delete:
            conn.execute("DELETE FROM loan WHERE id=?", (int(sel),))
            conn.commit()
            st.success("Kredi silindi ✅")
            st.rerun()

# --------- INSTALLMENTS ----------
elif page == "💳 Kart Taksitleri":
    st.subheader("💳 Kart Taksitli Alımlar")
    st.caption("Toplam tutar / taksit sayısı girilir. Sistem her ay eşit payı otomatik gider yazar. İlk açılışta 36.000/6 planı otomatik eklenir.")
    with st.form("inst_form"):
        name = st.text_input("Plan Adı (örn: Malzeme alımı)")
        total = st.number_input("Toplam Tutar (₺)", min_value=0.0, step=100.0, format="%.2f")
        months_total = st.number_input("Taksit Sayısı (Ay)", min_value=1, max_value=60, step=1, value=6)
        start_month = st.text_input("Başlangıç Ayı (YYYY-AA)", value=selected_month)
        pay_method = st.selectbox("Ödeme Tipi", ["Kart", "Nakit", "Havale"], index=0)
        submitted = st.form_submit_button("Plan Ekle", type="primary")
    if submitted:
        conn.execute("INSERT INTO installment_plan(name, total_amount, months_total, start_month, pay_method, active) VALUES(?,?,?,?,?,1)",
                     (name.strip(), float(total), int(months_total), start_month.strip(), pay_method))
        conn.commit()
        st.success("Taksit planı eklendi ✅")
        auto_generate_for_month(conn, selected_month)

    st.divider()
    plans = df_query(conn, "SELECT id, name AS Ad, total_amount AS Toplam, months_total AS Ay, start_month AS Baslangic, pay_method AS Odeme, active AS Aktif FROM installment_plan ORDER BY id DESC")
    st.dataframe(plans, use_container_width=True, hide_index=True)


    st.divider()
    st.markdown("### ✏️ Taksit Planı Düzenle / Sil")
    raw_plans = df_query(conn, "SELECT id, name, total_amount, months_total, start_month, pay_method, active FROM installment_plan ORDER BY id DESC")
    if len(raw_plans) == 0:
        st.info("Düzenlenecek taksit planı yok.")
    else:
        sel = st.selectbox(
            "Düzenlenecek taksit planını seç",
            raw_plans["id"].tolist(),
            format_func=lambda pid: f"#{pid} • {raw_plans.loc[raw_plans['id']==pid, 'name'].iloc[0]}",
            key="plan_edit_select",
        )
        r = raw_plans.loc[raw_plans["id"] == sel].iloc[0]

        with st.form("plan_edit_form"):
            e_name = st.text_input("Plan Adı", value=str(r["name"]), key="plan_edit_name")
            e_total = st.number_input("Toplam Tutar (₺)", min_value=0.0, step=100.0, format="%.2f", value=float(r["total_amount"]), key="plan_edit_total")
            e_months = st.number_input("Taksit Sayısı (Ay)", min_value=1, max_value=60, step=1, value=int(r["months_total"]), key="plan_edit_months")
            e_start = st.text_input("Başlangıç Ayı (YYYY-AA)", value=str(r["start_month"]), key="plan_edit_start")
            pm_opts = ["Kart", "Nakit", "Havale"]
            e_pm = st.selectbox("Ödeme Tipi", pm_opts, index=pm_opts.index(str(r["pay_method"])), key="plan_edit_pm")
            e_active = st.checkbox("Aktif", value=bool(int(r["active"])), key="plan_edit_active")

            if mobile_mode:
                save = st.form_submit_button("Kaydet (Güncelle)", type="primary")
                delete = st.form_submit_button("Sil (Kalıcı)")
                toggle = st.form_submit_button("Sadece Aktif/Pasif Değiştir")
            else:
                c1, c2, c3 = st.columns([1,1,1])
                save = c1.form_submit_button("Kaydet (Güncelle)", type="primary")
                delete = c2.form_submit_button("Sil (Kalıcı)")
                toggle = c3.form_submit_button("Sadece Aktif/Pasif Değiştir")

        if save:
            conn.execute(
                "UPDATE installment_plan SET name=?, total_amount=?, months_total=?, start_month=?, pay_method=?, active=? WHERE id=?",
                (e_name.strip(), float(e_total), int(e_months), e_start.strip(), e_pm, 1 if e_active else 0, int(sel))
            )
            conn.commit()
            auto_generate_for_month(conn, selected_month)
            st.success("Taksit planı güncellendi ✅")
            st.rerun()

        if toggle:
            conn.execute("UPDATE installment_plan SET active=? WHERE id=?", (0 if bool(int(r['active'])) else 1, int(sel)))
            conn.commit()
            auto_generate_for_month(conn, selected_month)
            st.success("Taksit planı durumu güncellendi ✅")
            st.rerun()

        if delete:
            conn.execute("DELETE FROM installment_plan WHERE id=?", (int(sel),))
            conn.commit()
            st.success("Taksit planı silindi ✅")
            st.rerun()

# --------- EXPORT ----------
elif page == "📤 Veri Dökümü":
    st.subheader("📤 Veri Dökümü (CSV)")
    start, end = month_range(selected_month)

    rev = df_query(conn, """
        SELECT d AS Tarih, cash AS NakitGelir, card AS KartGelir, (cash+card) AS ToplamGelir, note AS Notlar
        FROM daily_cash
        WHERE d >= ? AND d < ?
        ORDER BY d
    """, (start.isoformat(), end.isoformat()))

    exp = df_query(conn, """
        SELECT d AS Tarih, category AS Kategori, amount AS Tutar, pay_method AS Odeme, source AS Kaynak, note AS Notlar
        FROM expense
        WHERE d >= ? AND d < ?
        ORDER BY d
    """, (start.isoformat(), end.isoformat()))

    st.markdown("### Gelir (Günlük Kasa)")
    st.dataframe(rev, use_container_width=True, hide_index=True)
    st.markdown("### Gider")
    exp_disp = format_expense_for_display(exp)
    st.dataframe(exp_disp, use_container_width=True, hide_index=True)

    if mobile_mode:
        st.download_button("📥 Gelir CSV indir", data=rev.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"gelir_{selected_month}.csv", use_container_width=True)
        st.download_button("📥 Gider CSV indir", data=exp_disp.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"gider_{selected_month}.csv", use_container_width=True)
    else:
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("📥 Gelir CSV indir", data=rev.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"gelir_{selected_month}.csv")
        with d2:
            st.download_button("📥 Gider CSV indir", data=exp_disp.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"gider_{selected_month}.csv")


# --------- SETTINGS ----------
elif page == "⚙️ Ayarlar":
    st.subheader("⚙️ Ayarlar")
    st.markdown("### Şifre Değiştir")
    current_user = str(st.session_state.get("auth_username", "")).strip()
    current_role = str(st.session_state.get("user_role", "admin")).strip() or "admin"
    if current_user:
        with st.form("change_password_form"):
            old_pw = st.text_input("Mevcut Şifre", type="password")
            new_pw = st.text_input("Yeni Şifre", type="password")
            new_pw2 = st.text_input("Yeni Şifre (Tekrar)", type="password")
            pw_submit = st.form_submit_button("Şifreyi Güncelle", type="primary")

        if pw_submit:
            if not old_pw or not new_pw or not new_pw2:
                st.warning("Tüm alanları doldur.")
            elif new_pw != new_pw2:
                st.warning("Yeni şifreler eşleşmiyor.")
            elif len(new_pw) < 4:
                st.warning("Yeni şifre en az 4 karakter olmalı.")
            else:
                role_ok = authenticate_user(conn, current_user, old_pw)
                if role_ok is None:
                    st.error("Mevcut şifre yanlış.")
                else:
                    change_user_password(conn, current_user, current_role, new_pw)
                    st.success("Şifre güncellendi ✅")
    else:
        st.info("Aktif kullanıcı bulunamadı.")

    st.divider()
    st.markdown("### Kategoriler")
    st.caption("Kategorileri buradan ekleyip/pasif yapabilirsin. Pasif olanlar gider ekleme ekranında görünmez.")

    cats = df_query(conn, "SELECT name AS Kategori, active AS Aktif FROM categories ORDER BY name")
    edited = st.data_editor(
        cats,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "Kategori": st.column_config.TextColumn(required=True),
            "Aktif": st.column_config.CheckboxColumn()
        }
    )

    save_container = st.container() if mobile_mode else st.columns([1, 3])[0]
    with save_container:
        if st.button("💾 Kaydet", type="primary"):
            df = edited.copy()
            df["Kategori"] = df["Kategori"].astype(str).str.strip()
            df = df[df["Kategori"] != ""].drop_duplicates(subset=["Kategori"])
            conn.execute("DELETE FROM categories")
            conn.executemany(
                "INSERT INTO categories(name, active) VALUES(?,?)",
                [(r["Kategori"], 1 if bool(r["Aktif"]) else 0) for _, r in df.iterrows()]
            )
            conn.commit()
            st.success("Kaydedildi ✅")
            st.rerun()

    st.divider()
    st.markdown("### Yedekleme")
    if st.button("🧩 Şimdi yedek al"):
        ensure_backup()
        st.success("Yedek alındı ✅ (backups/ klasörüne)")


st.caption("FINAL v4 • Soft Professional tema • PDF rapor (ay/yıl) • kategori yönetimi • ay kilitleme • otomatik yedekleme.")
