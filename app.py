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
import pandas as pd

import io
import os
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

APP_TITLE = "Oldschool Esports Center • Finans Paneli (FINAL v4)"
DB_PATH = Path("oldschool_finance.db")

# Keep chart labels readable for Turkish text in generated PDF images.
plt.rcParams["font.family"] = "DejaVu Sans"

def ensure_backup():
    """Her açılışta veritabanını backups/ klasörüne kopyalar."""
    try:
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
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS daily_cash (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        d TEXT NOT NULL UNIQUE,
        cash REAL NOT NULL DEFAULT 0,
        card REAL NOT NULL DEFAULT 0,
        note TEXT
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS expense (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    conn.execute("""
    CREATE TABLE IF NOT EXISTS recurring_rule (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        amount REAL NOT NULL,
        day_of_month INTEGER NOT NULL,
        pay_method TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS loan (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        monthly_amount REAL NOT NULL,
        start_month TEXT NOT NULL, -- YYYY-MM
        months_total INTEGER NOT NULL,
        pay_method TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS installment_plan (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    """Sabitler + kredi + kart taksitlerini o ay için otomatik gider kaydı olarak ekler (zaten varsa tekrar eklemez)."""
    # Recurring rules
    rules = conn.execute("SELECT id, name, category, amount, day_of_month, pay_method FROM recurring_rule WHERE active=1").fetchall()
    for rid, name, category, amount, dom, pm in rules:
        d = day_in_month(ym_str, dom).isoformat()
        note = f"RULE:{rid}:{name}"
        exists = conn.execute("""
            SELECT 1 FROM expense
            WHERE d=? AND category=? AND amount=? AND pay_method=? AND source='auto' AND note=?
            LIMIT 1
        """, (d, category, amount, pm, note)).fetchone()
        if not exists:
            conn.execute("INSERT INTO expense(d, category, amount, pay_method, note, source) VALUES(?,?,?,?,?, 'auto')",
                         (d, category, amount, pm, note))

    # Loans
    loans = conn.execute("SELECT id, name, monthly_amount, start_month, months_total, pay_method FROM loan WHERE active=1").fetchall()
    for lid, name, monthly, start_month, months_total, pm in loans:
        sy, sm = parse_ym(start_month)
        cy, cm = parse_ym(ym_str)
        idx = (cy - sy) * 12 + (cm - sm)  # 0-based
        if 0 <= idx < months_total:
            d = day_in_month(ym_str, 1).isoformat()
            note = f"LOAN:{lid}:{name}"
            exists = conn.execute("""
                SELECT 1 FROM expense
                WHERE d=? AND category=? AND amount=? AND pay_method=? AND source='auto' AND note=?
                LIMIT 1
            """, (d, "Kredi", monthly, pm, note)).fetchone()
            if not exists:
                conn.execute("INSERT INTO expense(d, category, amount, pay_method, note, source) VALUES(?,?,?,?,?, 'auto')",
                             (d, "Kredi", monthly, pm, note))

    # Installments
    plans = conn.execute("SELECT id, name, total_amount, months_total, start_month, pay_method FROM installment_plan WHERE active=1").fetchall()
    for pid, name, total, months_total, start_month, pm in plans:
        sy, sm = parse_ym(start_month)
        cy, cm = parse_ym(ym_str)
        idx = (cy - sy) * 12 + (cm - sm)
        if 0 <= idx < months_total:
            monthly = float(total) / float(months_total)
            d = day_in_month(ym_str, 1).isoformat()
            note = f"INST:{pid}:{name}"
            exists = conn.execute("""
                SELECT 1 FROM expense
                WHERE d=? AND category=? AND amount=? AND pay_method=? AND source='auto' AND note=?
                LIMIT 1
            """, (d, "Kart Taksit", monthly, pm, note)).fetchone()
            if not exists:
                conn.execute("INSERT INTO expense(d, category, amount, pay_method, note, source) VALUES(?,?,?,?,?, 'auto')",
                             (d, "Kart Taksit", monthly, pm, note))
    conn.commit()

def df_query(conn, q, params=()):
    return pd.read_sql_query(q, conn, params=params)

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
        conn.executemany("INSERT OR IGNORE INTO categories(name, active) VALUES(?,?)", default_cats)
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


# ---------- PDF REPORTS ----------
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

    exp_disp = format_expense_for_display(exp) if len(exp) else pd.DataFrame(columns=["Tarih","Kategori","Tutar","Ödeme","Kaynak","Notlar"])
    by_cat = (exp_disp.groupby("Kategori", as_index=False)["Tutar"].sum()
              .sort_values("Tutar", ascending=False)) if len(exp_disp) else pd.DataFrame(columns=["Kategori","Tutar"])

    charts = []
    if len(rev):
        fig = plt.figure()
        plt.plot(pd.to_datetime(rev["d"]), rev["total"])
        plt.title("Günlük Toplam Gelir")
        plt.xlabel("Tarih")
        plt.ylabel("₺")
        charts.append(_fig_to_imagereader(fig))

    if len(by_cat):
        fig = plt.figure()
        plt.bar(by_cat["Kategori"], by_cat["Tutar"])
        plt.title("Kategori Bazlı Gider")
        plt.xlabel("Kategori")
        plt.ylabel("₺")
        plt.xticks(rotation=45, ha="right")
        charts.append(_fig_to_imagereader(fig))

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    y = h - 2*cm
    c.setFont(pdf_bold, 16)
    c.drawString(2*cm, y, "Oldschool Esports Center Finans Raporu")
    y -= 0.8*cm
    c.setFont(pdf_regular, 11)
    c.drawString(2*cm, y, f"Ay: {ym_str}    Oluşturma: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 1.2*cm

    c.setFont(pdf_bold, 12)
    c.drawString(2*cm, y, "Özet")
    y -= 0.6*cm
    c.setFont(pdf_regular, 11)
    for ln in [
        f"Nakit Gelir:  {tr_money(cash_sum)}",
        f"Kart Gelir:   {tr_money(card_sum)}",
        f"Toplam Gelir: {tr_money(total_rev)}",
        f"Toplam Gider: {tr_money(total_exp)}",
        f"Net:          {tr_money(net)}",
    ]:
        c.drawString(2*cm, y, ln)
        y -= 0.5*cm

    y -= 0.3*cm
    c.setFont(pdf_bold, 12)
    c.drawString(2*cm, y, "Gider Kırılımı (Kategori)")
    y -= 0.6*cm
    c.setFont(pdf_regular, 10)
    top = by_cat.head(12) if len(by_cat) else by_cat
    for _, r in top.iterrows():
        c.drawString(2*cm, y, str(r["Kategori"])[:38])
        c.drawRightString(w-2*cm, y, f"{tr_money(float(r['Tutar']))[1:]}")
        y -= 0.45*cm
        if y < 5*cm:
            c.showPage()
            y = h - 2*cm

    if charts:
        c.showPage()
        y = h - 2*cm
        c.setFont(pdf_bold, 12)
        c.drawString(2*cm, y, "Grafikler")
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
    c.drawString(2*cm, y, "Oldschool Esports Center Finans Raporu")
    y -= 0.8*cm
    c.setFont(pdf_regular, 11)
    c.drawString(2*cm, y, f"Yıl: {year}    Oluşturma: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 1.2*cm

    c.setFont(pdf_bold, 12)
    c.drawString(2*cm, y, "Yıllık Özet")
    y -= 0.6*cm
    c.setFont(pdf_regular, 11)
    for ln in [
        f"Nakit Gelir:  {tr_money(cash_sum)}",
        f"Kart Gelir:   {tr_money(card_sum)}",
        f"Toplam Gelir: {tr_money(total_rev)}",
        f"Toplam Gider: {tr_money(total_exp)}",
        f"Net:          {tr_money(net)}",
    ]:
        c.drawString(2*cm, y, ln)
        y -= 0.5*cm

    y -= 0.3*cm
    c.setFont(pdf_bold, 12)
    c.drawString(2*cm, y, "Aylık Tablo")
    y -= 0.6*cm
    c.setFont(pdf_regular, 10)
    for _, r in df.iterrows():
        c.drawString(2*cm, y, str(r["ym"]))
        c.drawRightString(w-2*cm, y,
                          f"Gelir {tr_money(float(r['total']))[1:]}  |  Gider {tr_money(float(r['expense']))[1:]}  |  Net {tr_money(float(r['net']))[1:]}")
        y -= 0.45*cm
        if y < 5*cm:
            c.showPage()
            y = h - 2*cm

    if charts:
        c.showPage()
        y = h - 2*cm
        c.setFont(pdf_bold, 12)
        c.drawString(2*cm, y, "Grafikler")
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
.stButton>button {
  border-radius: 12px;
  padding: 0.62rem 0.95rem;
  border: 1px solid rgba(31,41,55,0.12);
  background: #ffffff;
}
.stButton>button:hover {
  background: rgba(17,24,39,0.03);
  border-color: rgba(31,41,55,0.18);
}
div[data-testid="stDataFrame"] {
  background: #ffffff;
  border: 1px solid rgba(31,41,55,0.10);
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 2px 10px rgba(17,24,39,0.04);
}
hr { border-color: rgba(31,41,55,0.08); }

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
  .stButton>button, .stDownloadButton>button {
    width: 100%;
    min-height: 2.4rem;
  }
  div[data-testid="stDataFrame"] {
    border-radius: 10px;
  }
}
</style>"""

# ---------- LOGIN SYSTEM ----------
def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔐 Giriş Yap")

    username = st.text_input("Kullanıcı Adı")
    password = st.text_input("Şifre", type="password")

    if st.button("Giriş"):
        if (
            username == st.secrets.get("APP_USER")
            and password == st.secrets.get("APP_PASSWORD")
        ):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Hatalı kullanıcı adı veya şifre")

    return False


if not check_login():
    st.stop()

# ---------- END LOGIN ----------
st.markdown(SOFT_CSS, unsafe_allow_html=True)
st.title(APP_TITLE)

conn = get_conn()
init_db(conn)
ensure_backup()

today = date.today()
default_month = ym(today)

# Seed defaults on first run
seed_defaults_if_empty(conn, default_month)

with st.sidebar:
    st.header("Ay Seçimi")
    selected_month_input = st.text_input("Ay (YYYY-AA)", value=default_month, help="Örn: 2026-02")
    selected_month = selected_month_input.strip()
    if not is_valid_ym(selected_month):
        st.warning(f"Geçersiz ay formatı: '{selected_month_input}'. Varsayılan ay kullanılıyor: {default_month}")
        selected_month = default_month
    st.caption("Bu ay üzerinden raporlar gösterilir.")
    st.divider()
    locked_now = is_month_locked(conn, selected_month)
    lock_label = "🔒 Bu ay kilitli" if locked_now else "🔓 Bu ay açık"
    new_locked = st.toggle(lock_label, value=locked_now, help="Kilitliyken bu ay için yeni kayıt ekleme/düzenleme kapatılır.")
    if new_locked != locked_now:
        set_month_lock(conn, selected_month, new_locked)
        st.rerun()
    st.divider()
    st.header("Menü")
    page = st.radio("Sayfa", ["🏠 Dashboard", "💰 Günlük Kasa", "🧾 Gider Yönetimi", "🔁 Sabitler (Kurallar)", "🏦 Krediler", "💳 Kart Taksitleri", "📤 Veri Dökümü", "⚙️ Ayarlar"])
    st.divider()
    mobile_mode = st.toggle("Mobil görünüm", value=False, help="Dar ekranlarda daha rahat kullanım için düzeni sadeleştirir.")

ensure_month_open(conn, selected_month)
auto_generate_for_month(conn, selected_month)

# --------- DASHBOARD ----------
if page == "🏠 Dashboard":
    st.subheader(f"📌 {selected_month} Özeti")
    start, end = month_range(selected_month)
    start_s, end_s = start.isoformat(), end.isoformat()

    rev = df_query(conn, """
        SELECT d, cash, card, (cash+card) AS total
        FROM daily_cash
        WHERE d >= ? AND d < ?
        ORDER BY d
    """, (start_s, end_s))
    cash_sum = float(rev["cash"].sum()) if len(rev) else 0.0
    card_sum = float(rev["card"].sum()) if len(rev) else 0.0
    total_rev = float(rev["total"].sum()) if len(rev) else 0.0

    exp = df_query(conn, """
        SELECT d, category, amount, pay_method, source, note
        FROM expense
        WHERE d >= ? AND d < ?
        ORDER BY d
    """, (start_s, end_s))
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
        st.caption("PDF Rapor")
        st.download_button("📄 Ay Raporu (PDF)", data=build_monthly_pdf(conn, selected_month),
                           file_name=f"finans_raporu_{selected_month}.pdf", use_container_width=True)
        st.download_button("📄 Yıl Raporu (PDF)", data=build_yearly_pdf(conn, year_sel),
                           file_name=f"finans_raporu_{year_sel}.pdf", use_container_width=True)
    else:
        a1, a2, a3, a4 = st.columns([1, 1, 1.2, 1.2])
        with a1:
            if st.button("🔄 Yenile"):
                st.rerun()
        with a2:
            st.caption("PDF Rapor")
        with a3:
            st.download_button("📄 Ay Raporu (PDF)", data=build_monthly_pdf(conn, selected_month),
                               file_name=f"finans_raporu_{selected_month}.pdf")
        with a4:
            st.download_button("📄 Yıl Raporu (PDF)", data=build_yearly_pdf(conn, year_sel),
                               file_name=f"finans_raporu_{year_sel}.pdf")

    st.divider()

    def render_income_table():
        st.markdown("### Günlük Gelir (Nakit/Kart)")
        if len(rev):
            rev_display = rev.copy()
            rev_display.rename(columns={
                "d": "Tarih",
                "cash": "Nakit",
                "card": "Kart",
                "total": "Toplam"
            }, inplace=True)
            st.dataframe(rev_display, use_container_width=True, hide_index=True)
        else:
            st.info("Bu ay için günlük kasa girişi yok.")

    def render_expense_breakdown():
        st.markdown("### Gider Kırılımı (Kategori)")
        if len(exp):
            exp_disp = format_expense_for_display(exp)
            by_cat = exp_disp.groupby("Kategori", as_index=False)["Tutar"].sum().sort_values("Tutar", ascending=False)
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
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("Gider yok.")

# --------- DAILY CASH ----------
elif page == "💰 Günlük Kasa":
    st.subheader("💰 Günlük Kasa Girişi (Toplam Nakit / Toplam Kart)")
    if is_month_locked(conn, selected_month):
        st.warning("Bu ay kilitli. Kayıt ekleme/düzenleme kapalı.")
    d = st.date_input("Tarih", value=today)
    cash = st.number_input("Nakit (₺)", min_value=0.0, step=100.0, format="%.2f")
    card = st.number_input("Kart (₺)", min_value=0.0, step=100.0, format="%.2f")
    note = st.text_input("Not (opsiyonel)")

    locked = is_month_locked(conn, selected_month)

    if st.button("Kaydet / Güncelle", type="primary", disabled=locked):
        conn.execute("""
            INSERT INTO daily_cash(d, cash, card, note)
            VALUES(?,?,?,?)
            ON CONFLICT(d) DO UPDATE SET cash=excluded.cash, card=excluded.card, note=excluded.note
        """, (d.isoformat(), float(cash), float(card), note.strip() or None))
        conn.commit()
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
    st.dataframe(rev, use_container_width=True, hide_index=True)

# --------- MANUAL EXPENSE ----------
elif page == "🧾 Gider Yönetimi":
    st.subheader("🧾 Gider Yönetimi")
    if is_month_locked(conn, selected_month):
        st.warning("Bu ay kilitli. Manuel gider ekleme/düzenleme kapalı.")
    d = st.date_input("Tarih", value=today)
    category = st.selectbox("Kategori", get_active_categories(conn))
    amount = st.number_input("Tutar (₺)", min_value=0.0, step=100.0, format="%.2f")
    pay_method = st.selectbox("Ödeme Tipi", ["Nakit", "Kart", "Havale"])
    note = st.text_input("Not (opsiyonel)")

    locked = is_month_locked(conn, selected_month)

    if st.button("Gider Kaydet", type="primary", disabled=locked):
        conn.execute(
            "INSERT INTO expense(d, category, amount, pay_method, note, source) VALUES(?,?,?,?,?, 'manual')",
            (d.isoformat(), category, float(amount), pay_method, note.strip() or None),
        )
        conn.commit()
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
            conn.execute(
                "UPDATE expense SET d=?, category=?, amount=?, pay_method=?, note=? WHERE id=? AND source='manual'",
                (ed.isoformat(), ecat.strip(), float(eamount), epm, enote.strip() or None, int(sel_id)),
            )
            conn.commit()
            st.success("Gider güncellendi ✅")
            st.rerun()

        if delete:
            conn.execute("DELETE FROM expense WHERE id=? AND source='manual'", (int(sel_id),))
            conn.commit()
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
