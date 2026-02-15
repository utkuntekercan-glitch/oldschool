# Oldschool Finans Paneli

Streamlit tabanli finans paneli.

## Kurulum

```bash
pip install -r requirements.txt
```

## Calistirma

```bash
streamlit run app.py
```

## Giris Bilgileri

Uygulama `st.secrets` icinden asagidaki alanlari bekler:

- `APP_USER`
- `APP_PASSWORD`

## Notlar

- Uygulama SQLite veritabani kullanir: `oldschool_finance.db`
- Acilista otomatik yedek alir: `backups/`
- Aylik ve yillik PDF raporlar Unicode font ile uretildigi icin Turkce karakterler desteklenir.

