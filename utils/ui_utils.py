import streamlit as st

COLOR_PRIMARY = "#D0001B"


def inject_custom_css():
    st.markdown(
        """
        <style>
            .stApp { max-width: 1200px; margin: 0 auto; }
            .stButton>button { border-radius: 8px; font-weight: bold; }
            .css-1r6slb0, .e134j13f0 { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_header():
    st.markdown(
        f"""
        <div style='text-align: center; padding: 1rem 0; border-bottom: 2px solid {COLOR_PRIMARY}; margin-bottom: 1.5rem;'>
            <h1 style='color: {COLOR_PRIMARY}; margin: 0;'>🧥 KI-Fundbüro</h1>
            <p style='color: #666; margin: 0;'>Katharineum zu Lübeck</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_footer():
    st.markdown(
        """
        <br><hr>
        <div style='text-align: center; color: #888; font-size: 0.85rem; padding: 1rem 0;'>
            KI-Fundbüro • Katharineum zu Lübeck
        </div>
        """,
        unsafe_allow_html=True
    )


def render_home_card(title: str, description: str, icon_name: str, button_key: str):
    st.subheader(title)
    st.write(description)
    st.button(f"Öffnen: {title}", key=button_key, use_container_width=True)


def render_item_card(item: dict):
    st.markdown(f"### {item.get('category', 'Fundstück')}")
    st.caption(f"ID: {item.get('id')} | Status: {item.get('status', 'Verfügbar')}")
    st.write(f"**Farbe:** {item.get('color', '-')}")
    st.write(f"**Fundort:** {item.get('location_found', '-')}")


def render_prediction_result(pred: dict):
    conf = float(pred.get("confidence", 0)) * 100
    st.info(f"🤖 **KI-Erkennung:** {pred.get('label')} (Sicherheit: {conf:.1f}%)")


def render_status_badge(status: str):
    st.markdown(f"**Status:** `{status}`")
