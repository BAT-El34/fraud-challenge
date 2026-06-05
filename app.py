"""
Interface Streamlit - Hackathon INTELO2026
"""

from pathlib import Path

import pandas as pd
import streamlit as st

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

COLOR_ALERT = "#B91C1C"
COLOR_ALERT_BG = "#FEF2F2"
COLOR_TEXT = "#374151"
COLOR_MUTED = "#6B7280"
COLOR_HEADING = "#111827"


def _inject_styles() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

            html, body, [class*="css"] {{
                font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
            }}

            h1, h2, h3 {{
                font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
                font-weight: 600;
                color: {COLOR_HEADING};
                letter-spacing: -0.02em;
            }}

            .section-caption {{
                color: {COLOR_MUTED};
                font-size: 0.9rem;
                margin-bottom: 1rem;
            }}

            .kpi-card {{
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                padding: 0.85rem 1rem;
            }}

            .kpi-label {{
                color: {COLOR_MUTED};
                font-size: 0.8rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }}

            .kpi-value {{
                color: {COLOR_HEADING};
                font-size: 1.75rem;
                font-weight: 600;
                line-height: 1.2;
                margin-top: 0.25rem;
            }}

            .kpi-value-alert {{
                color: {COLOR_ALERT};
            }}

            .kpi-sub {{
                color: {COLOR_MUTED};
                font-size: 0.78rem;
                margin-top: 0.35rem;
            }}

            .kpi-sub-alert {{
                color: {COLOR_ALERT};
            }}

            div[data-testid="stSidebar"] {{
                background-color: #F9FAFB;
            }}

            div[data-testid="stSidebar"] .stMarkdown {{
                color: {COLOR_TEXT};
            }}

            .ml-formula {{
                color: {COLOR_MUTED};
                font-size: 0.85rem;
                margin-bottom: 0.75rem;
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
            "user_id": tx.get("user_id"),
            "amount": tx.get("amount"),
            "country": tx.get("country"),
            "fraud_score": float(res.get("fraud_score", 0.0)),
            "statut": "Suspect" if suspicious else "Conforme",
            "motif": res.get("reason", ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("fraud_score", ascending=False)
    return df


def _style_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def _row_style(row: pd.Series) -> list[str]:
        is_suspect = row.get("statut") == "Suspect"
        styles = []
        for col in df.columns:
            if is_suspect and col in ("statut", "fraud_score"):
                styles.append(
                    f"color: {COLOR_ALERT}; font-weight: 600; background-color: {COLOR_ALERT_BG}"
                )
            elif is_suspect:
                styles.append(f"color: {COLOR_TEXT}; background-color: {COLOR_ALERT_BG}")
            elif col == "fraud_score":
                styles.append(f"color: {COLOR_MUTED}; background-color: #FFFFFF")
            else:
                styles.append(f"color: {COLOR_TEXT}; background-color: #FFFFFF")
        return styles

    fmt = {"fraud_score": "{:.2f}"}
    if "amount" in df.columns:
        fmt["amount"] = "{:.2f}"

    return df.style.apply(_row_style, axis=1).format(fmt)


def _render_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Aucune transaction à afficher.")
        return
    st.dataframe(_style_table(df), use_container_width=True, hide_index=True)


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
            if is_suspect and col in ("statut", "score_final", "score_ml"):
                styles.append(
                    f"color: {COLOR_ALERT}; font-weight: 600; background-color: {COLOR_ALERT_BG}"
                )
            elif is_suspect:
                styles.append(f"color: {COLOR_TEXT}; background-color: {COLOR_ALERT_BG}")
            else:
                styles.append(f"color: {COLOR_TEXT}; background-color: #FFFFFF")
        return styles

    return df.style.apply(_row_style, axis=1)


def _render_ml_panel(transactions: list[dict]) -> None:
    st.divider()
    st.subheader("Apport machine learning")
    ml_status = "actif" if ML_AVAILABLE else "indisponible (règles seules)"
    st.markdown(
        f'<p class="ml-formula">Score final = 0.6 × règles + 0.4 × ML — module {ml_status}</p>',
        unsafe_allow_html=True,
    )
    ml_df = _build_ml_breakdown_df(transactions)
    if ml_df.empty:
        st.info("Aucune donnée à analyser.")
        return
    st.dataframe(_style_ml_table(ml_df), use_container_width=True, hide_index=True)


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

    st.subheader("Vue d'ensemble")
    st.markdown(
        '<p class="section-caption">Indicateurs clés après analyse du lot de transactions.</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _kpi_card("Transactions", str(total))
    with col2:
        _kpi_card(
            "Alertes",
            str(alerts),
            sub=f"{alerts} transaction(s) signalée(s)" if alerts else "Aucune alerte",
            alert=alerts > 0,
        )
    with col3:
        _kpi_card(
            "Taux de risque",
            f"{risk_rate:.1f} %",
            sub=f"{risk_rate - threshold:+.1f} pts vs seuil {threshold:.0f} %",
            alert=risk_high,
        )
    with col4:
        _kpi_card("Score moyen", f"{avg_score:.2f}")

    st.divider()

    df = _build_results_df(transactions, results)
    st.subheader("Transactions")
    alerts_only = st.checkbox("Afficher uniquement les alertes", value=False)
    display_df = df[df["statut"] == "Suspect"] if alerts_only and not df.empty else df
    _render_table(display_df)

    st.divider()
    st.subheader("Clients signalés")

    if not results:
        st.info("Aucun résultat d'analyse disponible.")
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
            st.info("Aucun client avec alerte.")
        else:
            results_by_id = {r.get("transaction_id"): r for r in results}

            for user_id in sorted(users_with_alerts):
                n_alerts = users_with_alerts[user_id]
                label = f"{user_id} — {n_alerts} alerte(s)"
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
                    _render_table(user_df)

    _render_ml_panel(transactions)


def main() -> None:
    st.set_page_config(
        page_title="Détection de fraude — INTELO2026",
        layout="wide",
    )
    _inject_styles()

    st.title("Détection de fraude financière")
    st.caption("Hackathon INTELO2026 — analyse des transactions suspectes")

    with st.sidebar:
        st.header("Données")
        use_sample = st.toggle("Utiliser le fichier d'exemple", value=True)
        transactions: list[dict] = []

        if use_sample:
            transactions = load_transactions(str(SAMPLE_CSV))
            st.caption(f"{len(transactions)} transactions chargées")
        else:
            uploaded = st.file_uploader("Importer un CSV", type=["csv"])
            if uploaded:
                tmp = Path(".streamlit_upload.csv")
                tmp.write_bytes(uploaded.getvalue())
                transactions = load_transactions(str(tmp))
                tmp.unlink(missing_ok=True)
                st.caption(f"{len(transactions)} transactions importées")

    if not transactions:
        st.info("Chargez des transactions puis lancez l'analyse.")
        return

    if st.button("Analyser", type="primary"):
        try:
            results = detect_fraud(transactions)
        except NotImplementedError:
            st.error("Implémentez d'abord `detect_fraud` dans `fraud_detection.py`.")
            return
        except Exception as exc:
            st.error(f"Erreur : {exc}")
            return

        render_interface(transactions, results)


if __name__ == "__main__":
    main()
