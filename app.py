import streamlit as st
from pathlib import Path
from datetime import datetime

from utils.ui_utils import (
    inject_custom_css,
    render_header,
    render_footer,
    render_home_card,
    render_item_card,
    render_prediction_result,
    render_status_badge,
)
from utils.data_utils import (
    load_items,
    save_item,
    update_item,
    delete_item,
    get_item_by_id,
    filter_items,
    get_unique_categories,
    get_unique_colors,
    get_unique_locations,
    export_data_as_json
)
from utils.model_utils import load_labels, predict_clothing, get_model_info
from utils.image_utils import save_uploaded_image

# 1. Konfiguration
st.set_page_config(
    page_title="Fundbüro - Katharineum zu Lübeck",
    page_icon="🧥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

inject_custom_css()

# Session State
if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_item_id" not in st.session_state:
    st.session_state.selected_item_id = None
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False


def navigate_to(page_name: str, item_id: str = None):
    st.session_state.page = page_name
    st.session_state.selected_item_id = item_id
    st.rerun()


render_header()

# Quick Navigation Bar
cols_nav = st.columns([1, 1, 1, 1])
with cols_nav[0]:
    if st.button("🏠 STARTSEITE", use_container_width=True):
        navigate_to("home")
with cols_nav[1]:
    if st.button("🔍 SUCHEN", use_container_width=True):
        navigate_to("search")
with cols_nav[2]:
    if st.button("➕ HOCHLADEN", use_container_width=True):
        navigate_to("upload")
with cols_nav[3]:
    if st.button("🔒 ADMIN", use_container_width=True):
        navigate_to("admin")

st.markdown("<hr style='border: 1px solid #FFE6E6; margin-bottom: 2rem;'>", unsafe_allow_html=True)


# ==========================================
# SEITE: STARTSEITE
# ==========================================
if st.session_state.page == "home":
    st.markdown("<h2 style='text-align: center; color: #D0001B;'>Wir helfen dir, Verlorenes wiederzufinden.</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        render_home_card(
            title="KLAMOTTEN SUCHEN",
            description="Durchsuche die aktuell im Fundbüro registrierten Kleidungsstücke schnell und einfach.",
            icon_name="search",
            button_key="home_btn_search"
        )
        if st.session_state.get("home_btn_search"):
            navigate_to("search")

    with col2:
        render_home_card(
            title="KLAMOTTEN HOCHLADEN",
            description="Lade ein gefundenes Kleidungsstück hoch. Die KI unterstützt dich automatisch bei der Kategorisierung.",
            icon_name="hanger",
            button_key="home_btn_upload"
        )
        if st.session_state.get("home_btn_upload"):
            navigate_to("upload")

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    st.markdown("### ℹ️ Wie funktioniert das KI-Fundbüro?")
    st.write("Unsere App nutzt moderne Machine-Learning-Algorithmen über Hugging Face. Sobald du ein Bild hochlädst, analysiert die KI das Foto und ordnet es automatisch zu.")

    items = load_items()
    total_items = len(items)
    returned_items = len([i for i in items if i.get("status") == "Abgeholt"])
    available_items = len([i for i in items if i.get("status") == "Verfügbar"])

    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.metric("📦 Gefundene Kleidungsstücke", total_items)
    s_col2.metric("✅ Verfügbar im Fundbüro", available_items)
    s_col3.metric("🎉 Erfolgreich zurückgegeben", returned_items)


# ==========================================
# SEITE: HOCHLADEN
# ==========================================
elif st.session_state.page == "upload":
    st.markdown("<h2 style='color: #D0001B;'>➕ Kleidungsstück hochladen</h2>", unsafe_allow_html=True)
    st.write("Lade ein Foto des gefundenen Gegenstands hoch. Die KI hilft bei der Erkennung.")

    uploaded_file = st.file_uploader("Bild auswählen (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Hochgeladenes Foto", width=300)

        if st.button("🔍 KLEIDUNGSSTÜCK ANALYSIEREN", type="primary"):
            with st.spinner("KI analysiert das Kleidungsstück..."):
                prediction = predict_clothing(uploaded_file)
                st.session_state.current_prediction = prediction

        if "current_prediction" in st.session_state:
            pred = st.session_state.current_prediction
            render_prediction_result(pred)

            st.markdown("### 📝 Details zum Fundstück eingeben")

            labels = load_labels()
            default_cat = pred["label"] if pred["label"] in labels else (labels[0] if labels else "Sonstiges")
            cat_index = labels.index(default_cat) if default_cat in labels else 0

            with st.form("upload_form"):
                category = st.selectbox("Kategorie", options=labels if labels else ["Sonstiges"], index=cat_index)
                color = st.selectbox("Farbe", ["Schwarz", "Weiß", "Grau", "Rot", "Blau", "Grün", "Gelb", "Orange", "Braun", "Rosa", "Lila", "Mehrfarbig", "Unbekannt"])
                brand = st.text_input("Marke / Label", value="Unbekannt")
                size = st.text_input("Größe (z.B. M, L, 140)", value="Unbekannt")
                location_found = st.text_input("Fundort (z.B. Sporthalle, Pausenhof)", value="Schulgelände")
                date_found = st.date_input("Funddatum", value=datetime.now())
                description = st.text_area("Beschreibung", placeholder="z.B. Reißverschluss leicht beschädigt...")
                special_features = st.text_input("Besonderheiten", placeholder="z.B. Schlüssel in der Tasche")
                contact = st.text_input("Kontakt für Rückfragen (optional)", placeholder="z.B. Hausmeister")

                submit = st.form_submit_button("💾 FUNDSTÜCK SPEICHERN")

                if submit:
                    saved_img_path = save_uploaded_image(uploaded_file)
                    new_item = {
                        "category": category,
                        "ai_prediction": pred["label"],
                        "ai_confidence": pred["confidence"],
                        "color": color,
                        "brand": brand,
                        "size": size,
                        "location_found": location_found,
                        "date_found": str(date_found),
                        "description": description,
                        "special_features": special_features,
                        "contact": contact,
                        "status": "Verfügbar",
                        "image_path": saved_img_path,
                        "created_at": datetime.now().isoformat()
                    }
                    item_id = save_item(new_item)
                    st.success(f"Erfolgreich gespeichert! Fundstück-ID: **{item_id}**")
                    if "current_prediction" in st.session_state:
                        del st.session_state.current_prediction


# ==========================================
# SEITE: SUCHEN
# ==========================================
elif st.session_state.page == "search":
    st.markdown("<h2 style='color: #D0001B;'>🔍 Klamotten suchen</h2>", unsafe_allow_html=True)

    items = load_items()

    st.markdown("### 📸 KI-Bildsuche (Verlorenes Kleidungsstück fotografieren)")
    search_image = st.file_uploader(
        "Lade ein Foto hoch – die KI filtert automatisch nach passenden Fundstücken",
        type=["jpg", "jpeg", "png", "webp"],
        key="search_img_upload"
    )

    ai_matched_category = None

    if search_image is not None:
        with st.spinner("KI vergleicht das Foto mit dem Fundbüro-Bestand..."):
            prediction = predict_clothing(search_image)
            ai_matched_category = prediction["label"]
            conf = prediction["confidence"] * 100

            if conf >= 30:
                st.success(f"Erkannte Kategorie für Suche: **{ai_matched_category}** (Sicherheit: {conf:.1f}%)")
            else:
                st.warning(f"Kategorie unsicher ({conf:.1f}%). Gefiltert nach bester Vermutung: **{ai_matched_category}**")

    with st.expander("🔎 Manuelle Filter & Suche anpassen", expanded=(search_image is None)):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            search_query = st.text_input("Freitextsuche", placeholder="Suchbegriff...")
            cats = get_unique_categories(items)
            default_cat_idx = 0
            if ai_matched_category in cats:
                default_cat_idx = cats.index(ai_matched_category) + 1

            selected_cat = st.selectbox("Kategorie", options=["Alle"] + cats, index=default_cat_idx)
        with f_col2:
            selected_color = st.selectbox("Farbe", options=["Alle"] + get_unique_colors(items))
            selected_location = st.selectbox("Fundort", options=["Alle"] + get_unique_locations(items))
        with f_col3:
            selected_status = st.selectbox("Status", options=["Alle", "Verfügbar", "Reserviert", "Abgeholt", "Archiviert"])

    filtered = filter_items(
        items,
        query=search_query,
        category=selected_cat if selected_cat != "Alle" else None,
        color=selected_color if selected_color != "Alle" else None,
        location=selected_location if selected_location != "Alle" else None,
        status=selected_status if selected_status != "Alle" else None
    )

    st.markdown(f"**{len(filtered)}** Fundstücke gefunden.")

    if not filtered:
        st.info("Keine passenden Kleidungsstücke gefunden.")
    else:
        cols = st.columns(3)
        for idx, item in enumerate(filtered):
            col = cols[idx % 3]
            with col:
                render_item_card(item)
                if st.button(f"Details ansehen ## {item['id']}", key=f"btn_det_{item['id']}", use_container_width=True):
                    navigate_to("detail", item["id"])


# ==========================================
# SEITE: DETAILANSICHT
# ==========================================
elif st.session_state.page == "detail":
    item = get_item_by_id(st.session_state.selected_item_id)
    if not item:
        st.error("Fundstück nicht gefunden.")
        if st.button("Zurück zur Suche"):
            navigate_to("search")
    else:
        st.markdown(f"<h2 style='color: #D0001B;'>Fundstück Details: {item['id']}</h2>", unsafe_allow_html=True)

        col_img, col_info = st.columns([1, 1])
        with col_img:
            img_p = Path(item.get("image_path", ""))
            if img_p.exists():
                st.image(str(img_p), use_container_width=True)
            else:
                st.warning("Kein Bild verfügbar.")

        with col_info:
            render_status_badge(item.get("status", "Verfügbar"))
            st.markdown(f"**Kategorie:** {item.get('category')}")
            st.markdown(f"**Farbe:** {item.get('color')}")
            st.markdown(f"**Größe:** {item.get('size')}")
            st.markdown(f"**Marke:** {item.get('brand')}")
            st.markdown(f"**Fundort:** {item.get('location_found')}")
            st.markdown(f"**Funddatum:** {item.get('date_found')}")
            st.markdown(f"**Beschreibung:** {item.get('description', '-')}")
            st.markdown(f"**Besonderheiten:** {item.get('special_features', '-')}")

        st.markdown("<br><hr><br>", unsafe_allow_html=True)

        st.markdown("### 🙋‍♂️ Gehört dieses Kleidungsstück dir?")
        with st.form("claim_form"):
            claim_name = st.text_input("Dein Name")
            claim_email = st.text_input("E-Mail oder Telefonnummer")
            claim_reason = st.text_area("Warum gehört das Kleidungsstück dir?")
            submit_claim = st.form_submit_button("✉️ ANFRAGE ABSENDEN")

            if submit_claim:
                if claim_name and claim_email and claim_reason:
                    st.success("Vielen Dank! Deine Anfrage wurde aufgenommen.")
                else:
                    st.error("Bitte fülle alle Pflichtfelder aus.")


# ==========================================
# SEITE: ADMIN
# ==========================================
elif st.session_state.page == "admin":
    st.markdown("<h2 style='color: #D0001B;'>🔒 Administration Fundbüro</h2>", unsafe_allow_html=True)

    secret_pass = st.secrets.get("admin", {}).get("password", "admin")

    if not st.session_state.admin_authenticated:
        input_pass = st.text_input("Admin-Passwort eingeben", type="password")
        if st.button("Anmelden"):
            if input_pass == secret_pass:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Falsches Passwort.")
    else:
        st.success("Erfolgreich als Administrator angemeldet.")
        if st.button("Abmelden"):
            st.session_state.admin_authenticated = False
            st.rerun()

        tabs = st.tabs(["📋 Fundstücke verwalten", "🤖 KI-Modell Status", "📥 Daten-Export"])

        with tabs[0]:
            items = load_items()
            st.markdown(f"Gesamt: **{len(items)}** Einträge")
            for item in items:
                with st.expander(f"[{item['status']}] {item['id']} - {item['category']} ({item['color']})"):
                    c1, c2 = st.columns(2)
                    with c1:
                        new_status = st.selectbox(
                            "Status ändern",
                            ["Verfügbar", "Reserviert", "Abgeholt", "Nicht zugeordnet", "Archiviert"],
                            index=["Verfügbar", "Reserviert", "Abgeholt", "Nicht zugeordnet", "Archiviert"].index(item.get("status", "Verfügbar")),
                            key=f"stat_{item['id']}"
                        )
                        if st.button("Status Speichern", key=f"sav_{item['id']}"):
                            update_item(item["id"], {"status": new_status})
                            st.success("Status aktualisiert!")
                            st.rerun()
                    with c2:
                        if st.button("🗑️ Fundstück Löschen", key=f"del_{item['id']}", type="primary"):
                            delete_item(item["id"])
                            st.warning("Fundstück gelöscht!")
                            st.rerun()

        with tabs[1]:
            info = get_model_info()
            st.json(info)

        with tabs[2]:
            st.download_button(
                label="📥 JSON-Daten herunterladen",
                data=export_data_as_json(),
                file_name="lost_items_export.json",
                mime="application/json"
            )

render_footer()
