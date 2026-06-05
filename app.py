"""
Interface Streamlit - Hackathon INTELO2026
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

from fraud_detection import (
    ML_AVAILABLE,
    _apply_level1,
    _apply_level2,
    _build_user_history,
    detect_fraud,
    load_transactions,
)

try:
    from fraud_ml import get_ml_scores
except ImportError:
    def get_ml_scores(transactions):
        return {}

SAMPLE_CSV = Path(__file__).parent / "data" / "sample_transactions.csv"
TABLE_INITIAL_ROWS = 10
TABLE_ROW_STEP = 10
TABLE_SCROLL_HEIGHT = 420

# Premium Minimalist Theme Colors (Linear/Vercel inspired)
COLOR_ALERT = "#ef4444"
COLOR_TEXT = "#ededed"
COLOR_MUTED = "#a1a1aa"
COLOR_HEADING = "#ffffff"
COLOR_BORDER = "#27272a"
COLOR_CARD_BG = "#09090b"


def _heading(text: str, icon: str, level: int = 2) -> None:
    st.markdown(f"{'#' * level} :material/{icon}: {text}")


def _inject_styles() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400&display=swap');

            html, body, [class*="css"] {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }}

            h1, h2, h3 {{
                font-family: 'Inter', sans-serif;
                font-weight: 600;
                color: {COLOR_HEADING};
                letter-spacing: -0.03em;
            }}

            .section-caption {{
                color: {COLOR_MUTED};
                font-size: 0.9rem;
                margin-bottom: 2rem;
                font-weight: 400;
                letter-spacing: -0.01em;
            }}

            .kpi-card {{
                background: {COLOR_CARD_BG};
                border: 1px solid {COLOR_BORDER};
                border-radius: 6px;
                padding: 1.5rem;
                transition: border-color 0.15s ease;
            }}
            .kpi-card:hover {{
                border-color: #3f3f46;
            }}

            .kpi-label {{
                color: {COLOR_MUTED};
                font-size: 0.75rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }}

            .kpi-value {{
                color: {COLOR_HEADING};
                font-size: 2rem;
                font-weight: 600;
                line-height: 1.2;
                margin-top: 0.5rem;
                letter-spacing: -0.02em;
            }}

            .kpi-value-alert {{
                color: {COLOR_ALERT};
            }}

            .kpi-sub {{
                color: {COLOR_MUTED};
                font-size: 0.8rem;
                margin-top: 0.5rem;
                font-weight: 400;
            }}

            .kpi-sub-alert {{
                color: {COLOR_ALERT};
            }}

            .ml-formula {{
                color: {COLOR_MUTED};
                font-family: 'JetBrains Mono', 'Menlo', monospace;
                font-size: 0.85rem;
                padding: 1.2rem;
                border: 1px solid {COLOR_BORDER};
                border-radius: 6px;
                background: {COLOR_CARD_BG};
                margin-bottom: 1.5rem;
                text-align: left;
            }}
            
            hr {{
                border-color: {COLOR_BORDER};
                margin: 3rem 0;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _kpi_card(label: str, value: str, sub: str = "", alert: bool = False) -> None:
    value_class = "kpi-value kpi-value-alert" if alert else "kpi-value"
    sub_class = "kpi-sub kpi-sub-alert" if alert else "kpi-sub"
    sub_html = f'<div class="{sub_class}">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="{value_class}">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_results_df(transactions: list[dict], results: list[dict]) -> pd.DataFrame:
    results_by_id = {r.get("transaction_id"): r for r in results}
    rows = []
    for tx in transactions:
        tid = tx.get("transaction_id")
        res = results_by_id.get(tid, {})
        suspicious = res.get("is_suspicious", False)
        rows.append({
            "transaction_id": tid,
            "timestamp": tx.get("timestamp"),
            "user_id": tx.get("user_id"),
            "amount": tx.get("amount"),
            "country": tx.get("country"),
            "fraud_score": float(res.get("fraud_score", 0.0)),
            "statut": "Suspect" if suspicious else "Conforme",
            "motif": res.get("reason", ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("fraud_score", ascending=False)
    return df


def _prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    if "timestamp" in display.columns:
        ts = pd.to_datetime(display["timestamp"], errors="coerce", utc=True)
        display["timestamp"] = ts.dt.strftime("%Y-%m-%d %H:%M").fillna("")
    return display


def _style_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def _row_style(row: pd.Series) -> list[str]:
        is_suspect = row.get("statut") == "Suspect"
        styles = []
        for col in df.columns:
            if is_suspect:
                styles.append(f"color: {COLOR_ALERT}; background-color: transparent;")
            elif col == "fraud_score":
                styles.append(f"color: {COLOR_HEADING}; background-color: transparent;")
            else:
                styles.append(f"color: {COLOR_TEXT}; background-color: transparent;")
        return styles

    fmt = {"fraud_score": "{:.2f}"}
    if "amount" in df.columns:
        fmt["amount"] = "{:.2f}"

    return df.style.apply(_row_style, axis=1).format(fmt)


def _render_paginated_table(df: pd.DataFrame, table_key: str) -> None:
    if df.empty:
        st.info("Aucune transaction à afficher.")
        return

    limit_key = f"{table_key}_row_limit"
    if limit_key not in st.session_state:
        st.session_state[limit_key] = TABLE_INITIAL_ROWS

    total = len(df)
    visible_count = min(st.session_state[limit_key], total)
    visible_df = _prepare_display_df(df.head(visible_count))

    st.caption(f"{visible_count} / {total} lignes affichées")
    st.dataframe(
        _style_table(visible_df),
        use_container_width=True,
        hide_index=True,
        height=TABLE_SCROLL_HEIGHT,
    )

    if visible_count < total:
        remaining = total - visible_count
        step = min(TABLE_ROW_STEP, remaining)
        if st.button(
            f"Charger {step} ligne(s) de plus",
            key=f"{table_key}_load_more",
        ):
            st.session_state[limit_key] = visible_count + step
            st.rerun()


def _build_ml_breakdown_df(transactions: list[dict]) -> pd.DataFrame:
    ml_scores = get_ml_scores(transactions)
    histories = _build_user_history(transactions)
    rows = []
    for tx, history in zip(transactions, histories):
        tid = tx.get("transaction_id")
        rules_score, _ = _apply_level1(tx)
        if rules_score == 0.0:
            rules_score, _ = _apply_level2(tx, history)
        ml_score = ml_scores.get(tid, 0.0) if tid else 0.0
        if ML_AVAILABLE:
            final = 0.6 * rules_score + 0.4 * ml_score
        else:
            final = rules_score
        final = max(0.0, min(1.0, final))
        rows.append({
            "transaction_id": tid,
            "score_regles": round(rules_score, 2),
            "score_ml": round(ml_score, 2),
            "score_final": round(final, 2),
            "statut": "Suspect" if final >= 0.5 else "Conforme",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score_final", ascending=False)
    return df


def _style_ml_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def _row_style(row: pd.Series) -> list[str]:
        is_suspect = row.get("statut") == "Suspect"
        styles = []
        for col in df.columns:
            if is_suspect:
                styles.append(f"color: {COLOR_ALERT}; background-color: transparent;")
            else:
                styles.append(f"color: {COLOR_TEXT}; background-color: transparent;")
        return styles

    return df.style.apply(_row_style, axis=1)


def _render_ml_panel(transactions: list[dict], df_results: pd.DataFrame) -> None:
    st.divider()
    _heading("Intelligence Artificielle", "psychology")
    
    ml_status = "Actif (Isolation Forest)" if ML_AVAILABLE else "Désactivé (Règles seules)"
    st.markdown(
        f'<div class="ml-formula">Modèle Hybride: Score_Final = σ (0.6 × Z-Score_Règles + 0.4 × ML_Anomaly_Score)<br>Statut: {ml_status}</div>',
        unsafe_allow_html=True,
    )
    
    ml_df = _build_ml_breakdown_df(transactions)
    if ml_df.empty:
        st.info("Aucune donnée disponible.")
        return

    tab1, tab2 = st.tabs(["Données", "Déviation"])
    
    with tab1:
        st.dataframe(_style_ml_table(ml_df), use_container_width=True, hide_index=True)
        
    with tab2:
        if not ml_df.empty and ML_AVAILABLE:
            chart_data = ml_df.melt(id_vars=["transaction_id", "statut"], value_vars=["score_regles", "score_ml"], var_name="Type de Score", value_name="Score")
            scatter_chart = alt.Chart(chart_data).mark_circle(size=60, opacity=0.8).encode(
                x=alt.X("transaction_id:N", axis=alt.Axis(domain=False, ticks=False, grid=False, labels=False, title=None)),
                y=alt.Y("Score:Q", axis=alt.Axis(domain=False, ticks=False, grid=True, gridColor="#27272a", labelColor=COLOR_MUTED, title=None)),
                color=alt.Color("Type de Score:N", scale=alt.Scale(domain=["score_regles", "score_ml"], range=["#52525b", "#ffffff"]), legend=None),
                tooltip=["transaction_id", "Type de Score", "Score", "statut"]
            ).properties(height=300).configure_view(strokeOpacity=0).interactive()
            st.altair_chart(scatter_chart, use_container_width=True)
        elif not ML_AVAILABLE:
            st.info("Le modèle ML est inactif.")

def _render_charts(df: pd.DataFrame) -> None:
    st.divider()
    _heading("Analytique", "analytics")
    
    if df.empty:
        return
        
    col1, col2 = st.columns(2)
    
    axis_config = alt.Axis(domain=False, ticks=False, grid=False, labelColor=COLOR_MUTED, title=None)
    
    with col1:
        st.markdown('<p style="color: #a1a1aa; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1rem;">Tendance du Risque (Temps Réel)</p>', unsafe_allow_html=True)
        if "timestamp" in df.columns and not df["timestamp"].isnull().all():
            trend_chart = alt.Chart(df).mark_area(
                line={'color': COLOR_ALERT},
                color=alt.Gradient(
                    gradient='linear',
                    stops=[alt.GradientStop(color=COLOR_ALERT, offset=0),
                           alt.GradientStop(color='rgba(239, 68, 68, 0)', offset=1)],
                    x1=1, x2=1, y1=1, y2=0
                ),
                interpolate='monotone',
                opacity=0.8
            ).encode(
                x=alt.X("timestamp:T", axis=alt.Axis(domain=False, ticks=False, grid=True, gridColor="#27272a", labelColor=COLOR_MUTED, title=None)),
                y=alt.Y("sum(fraud_score):Q", axis=axis_config),
                tooltip=[alt.Tooltip("timestamp:T", format="%Y-%m-%d %H:%M"), alt.Tooltip("sum(fraud_score):Q", format=".2f", title="Score Cumulé")]
            ).properties(height=250).configure_view(strokeOpacity=0).interactive()
            st.altair_chart(trend_chart, use_container_width=True)
        else:
            st.write("Données temporelles non disponibles pour tracer la tendance.")
        
    with col2:
        st.markdown('<p style="color: #a1a1aa; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1rem;">Exposition Géographique</p>', unsafe_allow_html=True)
        suspects_df = df[df["statut"] == "Suspect"]
        if not suspects_df.empty:
            country_chart = alt.Chart(suspects_df).mark_bar(color=COLOR_ALERT, opacity=0.9, cornerRadiusTopRight=2, cornerRadiusBottomRight=2).encode(
                x=alt.X("count()", axis=axis_config),
                y=alt.Y("country:N", sort="-x", axis=axis_config),
                tooltip=["country", "count()"]
            ).properties(height=250).configure_view(strokeOpacity=0).interactive()
            st.altair_chart(country_chart, use_container_width=True)
        else:
            st.write("Aucune anomalie géographique.")


def render_interface(transactions: list[dict], results: list[dict]) -> None:
    total = len(transactions)
    alerts = sum(1 for r in results if r.get("is_suspicious"))
    risk_rate = (alerts / total * 100) if total > 0 else 0.0
    avg_score = (
        sum(r.get("fraud_score", 0.0) for r in results) / len(results)
        if results
        else 0.0
    )
    threshold = 10.0
    risk_high = risk_rate > threshold

    _heading("Synthèse", "dashboard")
    st.markdown(
        '<p class="section-caption">Monitoring des flux et détection d\'anomalies.</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _kpi_card("Transactions", str(total))
    with col2:
        _kpi_card(
            "Anomalies",
            str(alerts),
            sub=f"{alerts} transaction(s) suspecte(s)" if alerts else "Aucune",
            alert=alerts > 0,
        )
    with col3:
        _kpi_card(
            "Exposition",
            f"{risk_rate:.1f} %",
            sub=f"{risk_rate - threshold:+.1f} pts (seuil {threshold:.0f} %)",
            alert=risk_high,
        )
    with col4:
        _kpi_card("Score Moyen", f"{avg_score:.2f}")

    df = _build_results_df(transactions, results)
    
    _render_charts(df)

    st.divider()
    _heading("Journal", "receipt_long")
    alerts_only = st.checkbox(
        "N'afficher que les alertes",
        value=False,
        key="journal_alerts_only",
    )
    journal_df = df[df["statut"] == "Suspect"].copy() if alerts_only else df
    if alerts_only and journal_df.empty:
        st.info("Aucune alerte à afficher.")
    else:
        _render_paginated_table(journal_df, table_key="journal")

    st.divider()
    _heading("Profils", "group")

    if not results:
        st.info("Aucun résultat d'analyse.")
    else:
        tx_by_id = {tx.get("transaction_id"): tx for tx in transactions}
        users_with_alerts: dict[str, int] = {}
        for r in results:
            if r.get("is_suspicious"):
                tx = tx_by_id.get(r.get("transaction_id"), {})
                uid = tx.get("user_id")
                if uid:
                    users_with_alerts[uid] = users_with_alerts.get(uid, 0) + 1

        if not users_with_alerts:
            st.write("Aucun profil utilisateur suspect détecté.")
        else:
            results_by_id = {r.get("transaction_id"): r for r in results}

            for user_id in sorted(users_with_alerts):
                n_alerts = users_with_alerts[user_id]
                label = f"{user_id} - {n_alerts} anomalie(s)"
                with st.expander(label):
                    user_rows = []
                    for tx in transactions:
                        if tx.get("user_id") != user_id:
                            continue
                        tid = tx.get("transaction_id")
                        res = results_by_id.get(tid, {})
                        user_rows.append({
                            "transaction_id": tid,
                            "timestamp": tx.get("timestamp"),
                            "amount": tx.get("amount"),
                            "country": tx.get("country"),
                            "fraud_score": float(res.get("fraud_score", 0.0)),
                            "statut": "Suspect" if res.get("is_suspicious") else "Conforme",
                            "motif": res.get("reason", ""),
                        })
                    user_df = pd.DataFrame(user_rows)
                    if not user_df.empty:
                        user_df["_ts_sort"] = pd.to_datetime(
                            user_df["timestamp"], errors="coerce", utc=True
                        )
                        user_df = user_df.sort_values("_ts_sort").drop(columns=["_ts_sort"])
                    _render_paginated_table(user_df, table_key=f"user_{user_id}")

    _render_ml_panel(transactions, df)


def main() -> None:
    st.set_page_config(
        page_title="Sentinel | INTELO2026",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    _inject_styles()

    _heading("Sentinel", "shield", level=1)
    st.caption("Moteur d'inférence et détection comportementale")

    with st.sidebar:
        st.markdown("### :material/cloud_upload: Ingestion")
        st.markdown('<p style="color: #a1a1aa; font-size: 0.8rem; margin-bottom: 2rem;">Endpoints supportés: REST, GraphQL, Kafka, S3 (CSV, Parquet, JSON)</p>', unsafe_allow_html=True)
        use_sample = st.toggle("Mock Stream (Dev)", value=True)
        transactions: list[dict] = []

        if use_sample:
            transactions = load_transactions(str(SAMPLE_CSV))
            st.write(f"{len(transactions)} événements ingérés.")
        else:
            uploaded = st.file_uploader("Upload Batch", type=["csv", "json", "parquet"])
            if uploaded:
                try:
                    tmp = Path(".streamlit_upload.csv")
                    tmp.write_bytes(uploaded.getvalue())
                    transactions = load_transactions(str(tmp))
                    tmp.unlink(missing_ok=True)
                    st.write(f"{len(transactions)} événements ingérés.")
                except Exception:
                    st.error("Erreur de parsing (schema non conforme).")

    if not transactions:
        st.info("En attente de flux d'événements.")
        return

    if st.button("Initialiser inférence", type="primary", icon=":material/play_arrow:"):
        with st.spinner("Exécution du pipeline analytique..."):
            try:
                results = detect_fraud(transactions)
            except NotImplementedError:
                st.error("Règles d'inférence non implémentées (`fraud_detection.py`).")
                return
            except Exception as exc:
                st.error(f"Échec de l'inférence : {exc}")
                return

        st.session_state["analysis_transactions"] = transactions
        st.session_state["analysis_results"] = results
        st.session_state["journal_row_limit"] = TABLE_INITIAL_ROWS
        for key in list(st.session_state.keys()):
            if key.startswith("user_") and key.endswith("_row_limit"):
                del st.session_state[key]

    if "analysis_results" in st.session_state and "analysis_transactions" in st.session_state:
        render_interface(
            st.session_state["analysis_transactions"],
            st.session_state["analysis_results"],
        )


if __name__ == "__main__":
    main()
