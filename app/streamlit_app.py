import urllib.parse
from pathlib import Path

import streamlit as st
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import load_all_models, run_pipeline_on_image
from monitoring import compute_stats, get_recent_fallback_trend, load_logs

# Kleine, fest ins Repo committete Auswahl an Beispielbelegen - der komplette
# SROIE-Datensatz ist zu gross fuer GitHub und liegt daher nicht auf dem
# Streamlit-Server. Diese 6 Bilder reichen fuer eine Demo voellig aus.
SAMPLE_RECEIPTS_DIR = Path(__file__).parent.parent / "data" / "sample_receipts"

DEFAULT_CURRENCY = "RM"

st.set_page_config(page_title="InsightScan", page_icon="🧾", layout="centered")


def build_watermark_data_uri() -> str:
    svg = """
    <svg xmlns='http://www.w3.org/2000/svg' width='240' height='240'>
        <text x='15' y='45' font-size='36' fill='#B8860B' opacity='0.10' font-family='Arial'>€</text>
        <text x='140' y='90' font-size='32' fill='#B8860B' opacity='0.10' font-family='Arial'>$</text>
        <text x='40' y='150' font-size='34' fill='#B8860B' opacity='0.10' font-family='Arial'>£</text>
        <text x='160' y='200' font-size='30' fill='#B8860B' opacity='0.10' font-family='Arial'>¥</text>
        <text x='190' y='40' font-size='24' fill='#B8860B' opacity='0.10' font-family='Arial'>RM</text>
    </svg>
    """
    encoded = urllib.parse.quote(svg.strip())
    return f"data:image/svg+xml,{encoded}"


def inject_custom_style():
    watermark_uri = build_watermark_data_uri()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #FFFCF2;
            background-image: url("{watermark_uri}");
            background-repeat: repeat;
            background-size: 240px 240px;
        }}

        h1, h2, h3 {{
            color: #92660A;
        }}

        p, .stCaption, [data-testid="stCaptionContainer"] {{
            color: #6B5424;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: #FDE68A !important;
            border-radius: 6px 6px 0 0;
            color: #78350F !important;
            font-weight: 600;
        }}

        div.stButton > button {{
            background-color: #FBBF24;
            color: #78350F;
            border: 1px solid #D9A422;
            font-weight: 600;
        }}
        div.stButton > button:hover {{
            background-color: #F59E0B;
            border-color: #B8860B;
            color: white;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            background-color: #FFF9E6;
            border: 2px dashed #F4D778;
        }}

        [data-testid="stExpander"] {{
            background-color: #FFFCF2;
            border: 1px solid #F4D778;
            border-radius: 8px;
        }}

        .field-box {{
            background-color: #FFF7DA;
            border: 1px solid #F4D778;
            border-radius: 10px;
            padding: 12px 14px;
            height: 100%;
        }}
        .field-label {{
            font-size: 0.8rem;
            color: #92660A;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 4px;
        }}
        .field-value {{
            font-size: 1.3rem;
            color: #4A3A12;
            font-weight: 700;
            word-wrap: break-word;
            overflow-wrap: break-word;
            line-height: 1.3;
        }}

        [data-testid="stMetric"] {{
            background-color: #FFF7DA;
            border: 1px solid #F4D778;
            border-radius: 10px;
            padding: 12px;
        }}
        [data-testid="stMetricValue"] {{
            color: #92660A;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_field_box(label: str, value: str):
    display_value = value if value else "nicht erkannt"
    st.markdown(
        f"""
        <div class="field-box">
            <div class="field-label">{label}</div>
            <div class="field-value">{display_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_models():
    return load_all_models()


def get_sample_receipts():
    """Liest die kleine, fest im Repo enthaltene Beispielbeleg-Auswahl ein."""
    if not SAMPLE_RECEIPTS_DIR.exists():
        return []
    return sorted(SAMPLE_RECEIPTS_DIR.glob("*.jpg"))


def display_result(result: dict):
    fields = result["fields"]
    currency = result["currency"]
    rule_results = result["rule_results"]
    amount_check = rule_results["amount_check"]
    date_check = rule_results["date_check"]

    st.subheader("Erkannte Belegdaten")
    col1, col2, col3 = st.columns(3)
    with col1:
        render_field_box("Firma", fields["company"])
    with col2:
        render_field_box("Datum", fields["date"])
    with col3:
        total_display = f"{fields['total']} {currency}" if fields["total"] else ""
        render_field_box("Betrag", total_display)

    if fields["used_fallback"]["date"] or fields["used_fallback"]["total"]:
        fallback_fields = [
            name for name, used in [("Datum", fields["used_fallback"]["date"]),
                                      ("Betrag", fields["used_fallback"]["total"])]
            if used
        ]
        st.caption(f"ℹ️ Hinweis: {', '.join(fallback_fields)} wurde über die Regex-Ersatzerkennung "
                   f"ermittelt, da LayoutLM kein eindeutiges Ergebnis lieferte.")

    st.subheader("Prüfergebnis")

    if amount_check["requires_approval"]:
        st.warning(f"⚠️ {amount_check['explanation']}")
    elif amount_check["amount_known"]:
        st.success(f"✅ {amount_check['explanation']}")
    else:
        st.info(f"ℹ️ {amount_check['explanation']}")

    if date_check["within_deadline"] is False:
        st.error(f"⏰ {date_check['explanation']}")
    elif date_check["within_deadline"] is True:
        st.success(f"✅ {date_check['explanation']}")
    else:
        st.info(f"ℹ️ {date_check['explanation']}")

    st.subheader("Zusammenfassung")
    st.write(result["summary"])

    with st.expander("🔍 Details zur Richtlinien-Prüfung anzeigen"):
        st.caption("Diese Abschnitte der Spesenrichtlinie wurden per semantischer Suche "
                   "als am relevantesten für diesen Beleg identifiziert:")
        for similarity, text in result["relevant_chunks"]:
            st.markdown(f"**Ähnlichkeit: {similarity:.2f}**")
            st.text(text)
            st.divider()


def display_monitoring_dashboard():
    logs = load_logs()
    stats = compute_stats(logs)

    if stats["total"] == 0:
        st.info("Noch keine Belege verarbeitet. Sobald du über die Tabs oben Belege "
                 "prüfst, erscheinen hier Kennzahlen zur Erkennungsqualität.")
        return

    st.subheader(f"Bisher verarbeitete Belege: {stats['total']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Firma erkannt", f"{stats['company_recognition_rate']:.0%}")
    col2.metric("Datum: Fallback-Quote", f"{stats['date_fallback_rate']:.0%}")
    col3.metric("Betrag: Fallback-Quote", f"{stats['total_fallback_rate']:.0%}")

    st.caption(
        "Die Fallback-Quote zeigt, wie oft die einfache Regex-Ersatzerkennung statt "
        "des trainierten LayoutLM-Modells einspringen musste. Eine dauerhaft steigende "
        "Quote deutet auf 'Data Drift' hin - die eingehenden Belege weichen zunehmend "
        "von den Trainingsdaten ab, was ein Signal für ein Nachtraining mit neueren, "
        "vielfältigeren Beispielen wäre."
    )

    trend = get_recent_fallback_trend(logs, window_size=5)
    if trend:
        st.subheader("Trend: Betrag-Fallback-Quote über Zeit")
        st.line_chart(trend)
        st.caption("Jeder Punkt zeigt die Fallback-Quote der letzten 5 verarbeiteten Belege "
                   "(gleitendes Fenster) - so werden Trends sichtbar, auch wenn die "
                   "Gesamtquote oben noch stabil wirkt.")
    else:
        st.caption("Trend-Diagramm erscheint, sobald mindestens 5 Belege verarbeitet wurden.")


def main():
    inject_custom_style()

    st.title("🧾 InsightScan")
    st.caption("Automatisierte Belegprüfung: Computer Vision + RAG + Regelbasierte Compliance-Prüfung")

    models = get_models()

    st.divider()

    currency_input = st.text_input(
        "Währung der Belege",
        value=DEFAULT_CURRENCY,
        max_chars=6,
        help="Beliebiges Währungssymbol oder -kürzel, z.B. RM, EUR, USD, CAD, ¥, Fr. "
             "Die Betragsgrenzen der Richtlinie (100/500/2.000) werden als reine Zahl "
             "geprüft, unabhängig von der eingegebenen Währung.",
    )
    currency = currency_input.strip() or DEFAULT_CURRENCY

    tab_upload, tab_sample, tab_monitoring = st.tabs(
        ["📤 Eigenen Beleg hochladen", "📋 Beispielbeleg auswählen", "📊 Monitoring"]
    )

    image_to_process = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Belegfoto hochladen (JPG oder PNG)", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)

    with tab_sample:
        sample_files = get_sample_receipts()

        if not sample_files:
            st.warning("Keine Beispielbelege gefunden.")
        else:
            sample_names = [f.stem for f in sample_files]
            selected_name = st.selectbox("Beispielbeleg auswählen", sample_names)

            if selected_name:
                selected_path = next(f for f in sample_files if f.stem == selected_name)
                if st.button("Diesen Beispielbeleg verwenden"):
                    image_to_process = Image.open(selected_path)

    with tab_monitoring:
        display_monitoring_dashboard()

    if image_to_process is not None:
        st.divider()
        col_img, col_result = st.columns([1, 2])

        with col_img:
            st.image(image_to_process, caption="Zu prüfender Beleg", use_container_width=True)

        with col_result:
            with st.spinner("Analysiere Beleg (OCR, Felderkennung, Richtlinien-Prüfung) ..."):
                result = run_pipeline_on_image(image_to_process, models, currency=currency)
            display_result(result)


if __name__ == "__main__":
    main()