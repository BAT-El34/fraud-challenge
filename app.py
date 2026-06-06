"""
Interface Streamlit - Hackathon INTELO2026
"""

from pathlib import Path

import altair as alt
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
TABLE_INITIAL_ROWS = 10
TABLE_ROW_STEP = 10
TABLE_SCROLL_HEIGHT = 380
RISK_THRESHOLD = 10.0

COLOR_ALERT = "#f87171"
COLOR_MUTED = "#71717a"
COLOR_ACCENT = "#fafafa"
COLOR_GRID = "#27272a"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
            html, body, [class*="css"] {
                font-family: 'Inter', system-ui, sans-serif;
            }
            h1, h2, h3, h4 {
                letter-spacing: -0.03em;
                font-weight: 600;
            }
            .app-badge {
                display: inline-block;
                font-size: 0.7rem;
                font-weight: 500;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #a1a1aa;
                border: 1px solid #3f3f46;
                border-radius: 999px;
                padding: 0.2rem 0.65rem;
                margin-bottom: 0.75rem;
            }
            div[data-testid="stMetric"] {
                background: #18181b;
                border: 1px solid #27272a;
                border-radius: 10px;
                padding: 0.85rem 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _section(title: str, icon: str, caption: str = "") -> None:
    st.markdown(f"### :material/{icon}: {title}")
    if caption:
        st.caption(caption)


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
        df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("fraud_score", ascending=False)
    return df


def _format_display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "_ts" in out.columns:
        out["timestamp"] = out["_ts"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
        out = out.drop(columns=["_ts"], errors="ignore")
    elif "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True).dt.strftime(
            "%Y-%m-%d %H:%M"
        ).fillna("")
    return out


def _table_columns(include_timestamp: bool = False) -> dict:
    cols = {
        "transaction_id": st.column_config.TextColumn("ID", width="small"),
        "user_id": st.column_config.TextColumn("Client", width="small"),
        "amount": st.column_config.NumberColumn("Montant", format="%.2f"),
        "country": st.column_config.TextColumn("Pays", width="small"),
        "fraud_score": st.column_config.ProgressColumn(
            "Score",
            format="%.2f",
            min_value=0.0,
            max_value=1.0,
        ),
        "statut": st.column_config.TextColumn("Statut", width="small"),
        "motif": st.column_config.TextColumn("Motif", width="large"),
    }
    if include_timestamp:
        cols = {
            "transaction_id": cols["transaction_id"],
            "timestamp": st.column_config.TextColumn("Date", width="medium"),
            **{k: v for k, v in cols.items() if k != "transaction_id"},
        }
    return cols


def _render_paginated_table(
    df: pd.DataFrame,
    table_key: str,
    include_timestamp: bool = False,
) -> None:
    if df.empty:
        st.info("Aucune transaction a afficher.")
        return

    limit_key = f"{table_key}_row_limit"
    if limit_key not in st.session_state:
        st.session_state[limit_key] = TABLE_INITIAL_ROWS

    total = len(df)
    visible_count = min(st.session_state[limit_key], total)
    visible_df = _format_display_df(df.head(visible_count))

    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.caption(f"{visible_count} sur {total} lignes")
    with col_b:
        st.progress(visible_count / total if total else 0.0)

    st.dataframe(
        visible_df,
        column_config=_table_columns(include_timestamp),
        use_container_width=True,
        hide_index=True,
        height=TABLE_SCROLL_HEIGHT,
    )

    if visible_count < total:
        remaining = total - visible_count
        step = min(TABLE_ROW_STEP, remaining)
        if st.button(
            f"Afficher {step} lignes supplementaires",
            key=f"{table_key}_load_more",
            use_container_width=True,
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


def _render_charts(df: pd.DataFrame) -> None:
    if df.empty:
        return

    axis = alt.Axis(
        domain=False,
        ticks=False,
        gridColor=COLOR_GRID,
        labelColor=COLOR_MUTED,
        title=None,
    )
    col1, col2 = st.columns(2)

    with col1:
        if "_ts" in df.columns and df["_ts"].notna().any():
            chart_df = df.dropna(subset=["_ts"])
            chart = (
                alt.Chart(chart_df)
                .mark_line(color=COLOR_ACCENT, strokeWidth=2)
                .encode(
                    x=alt.X("_ts:T", axis=axis),
                    y=alt.Y("fraud_score:Q", axis=axis),
                    tooltip=[
                        alt.Tooltip("_ts:T", title="Date", format="%Y-%m-%d %H:%M"),
                        alt.Tooltip("fraud_score:Q", format=".2f", title="Score"),
                    ],
                )
                .properties(height=220)
                .configure_view(strokeOpacity=0)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("Serie temporelle indisponible.")

    with col2:
        suspects = df[df["statut"] == "Suspect"]
        if not suspects.empty:
            chart = (
                alt.Chart(suspects)
                .mark_bar(color=COLOR_ALERT, cornerRadiusEnd=4)
                .encode(
                    x=alt.X("count()", axis=axis),
                    y=alt.Y("country:N", sort="-x", axis=axis),
                    tooltip=["country", "count()"],
                )
                .properties(height=220)
                .configure_view(strokeOpacity=0)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("Aucune alerte geographique.")


def _render_ml_panel(transactions: list[dict]) -> None:
    ml_df = _build_ml_breakdown_df(transactions)
    if ml_df.empty:
        st.info("Donnees ML indisponibles.")
        return

    status = "Isolation Forest actif" if ML_AVAILABLE else "Regles seules"
    st.caption(f"Formule hybride : 0.6 x regles + 0.4 x ML | {status}")

    tab_data, tab_chart = st.tabs(["Scores", "Comparaison"])

    with tab_data:
        st.dataframe(
            ml_df,
            column_config={
                "transaction_id": st.column_config.TextColumn("ID"),
                "score_regles": st.column_config.NumberColumn("Regles", format="%.2f"),
                "score_ml": st.column_config.NumberColumn("ML", format="%.2f"),
                "score_final": st.column_config.ProgressColumn(
                    "Final", format="%.2f", min_value=0.0, max_value=1.0
                ),
                "statut": st.column_config.TextColumn("Statut"),
            },
            use_container_width=True,
            hide_index=True,
        )

    with tab_chart:
        if ML_AVAILABLE:
            melted = ml_df.melt(
                id_vars=["transaction_id"],
                value_vars=["score_regles", "score_ml"],
                var_name="source",
                value_name="score",
            )
            chart = (
                alt.Chart(melted)
                .mark_circle(size=70, opacity=0.85)
                .encode(
                    x=alt.X("transaction_id:N", title=None),
                    y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                    color=alt.Color(
                        "source:N",
                        scale=alt.Scale(
                            domain=["score_regles", "score_ml"],
                            range=[COLOR_MUTED, COLOR_ACCENT],
                        ),
                        legend=alt.Legend(title="Source"),
                    ),
                    tooltip=["transaction_id", "source", "score"],
                )
                .properties(height=280)
                .configure_view(strokeOpacity=0)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Module ML non charge.")


def render_interface(transactions: list[dict], results: list[dict]) -> None:
    total = len(transactions)
    alerts = sum(1 for r in results if r.get("is_suspicious"))
    risk_rate = (alerts / total * 100) if total > 0 else 0.0
    avg_score = (
        sum(r.get("fraud_score", 0.0) for r in results) / len(results) if results else 0.0
    )

    with st.container(border=True):
        _section("Vue d'ensemble", "dashboard", "Indicateurs apres analyse du lot.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Transactions", total)
        with c2:
            st.metric(
                "Alertes",
                alerts,
                delta=f"{alerts} detectee(s)" if alerts else "Aucune",
                delta_color="inverse" if alerts else "off",
            )
        with c3:
            st.metric(
                "Taux de risque",
                f"{risk_rate:.1f} %",
                delta=f"{risk_rate - RISK_THRESHOLD:+.1f} pts vs {RISK_THRESHOLD:.0f} %",
                delta_color="inverse" if risk_rate > RISK_THRESHOLD else "off",
            )
        with c4:
            st.metric("Score moyen", f"{avg_score:.2f}")

    df = _build_results_df(transactions, results)

    with st.container(border=True):
        _section("Analytique", "analytics")
        _render_charts(df)

    with st.container(border=True):
        _section("Journal des transactions", "receipt_long")
        alerts_only = st.toggle(
            "Afficher uniquement les alertes",
            value=False,
            key="journal_alerts_only",
        )
        journal_df = df[df["statut"] == "Suspect"].copy() if alerts_only else df
        if alerts_only and journal_df.empty:
            st.success("Aucune alerte sur ce lot.")
        else:
            _render_paginated_table(journal_df, table_key="journal")

    with st.container(border=True):
        _section("Clients signales", "group")
        if not results:
            st.info("Aucun resultat.")
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
                st.success("Aucun client avec alerte.")
            else:
                results_by_id = {r.get("transaction_id"): r for r in results}
                for user_id in sorted(users_with_alerts):
                    n = users_with_alerts[user_id]
                    with st.expander(f"{user_id} | {n} alerte(s)"):
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
                            user_df["_ts"] = pd.to_datetime(
                                user_df["timestamp"], errors="coerce", utc=True
                            )
                            user_df = user_df.sort_values("_ts")
                        _render_paginated_table(
                            user_df,
                            table_key=f"user_{user_id}",
                            include_timestamp=True,
                        )

    with st.container(border=True):
        _section("Machine learning", "psychology")
        _render_ml_panel(transactions)


def main() -> None:
    st.set_page_config(
        page_title="Detection de fraude | INTELO2026",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    st.markdown('<span class="app-badge">INTELO2026</span>', unsafe_allow_html=True)
    st.markdown("# :material/shield: Detection de fraude")
    st.caption("Analyse des transactions financieres | regles metier + Isolation Forest")

    with st.sidebar:
        st.markdown("### :material/cloud_upload: Donnees")
        use_sample = st.toggle("Fichier d'exemple", value=True)
        transactions: list[dict] = []

        if use_sample:
            transactions = load_transactions(str(SAMPLE_CSV))
            st.caption(f"{len(transactions)} transactions chargees")
        else:
            uploaded = st.file_uploader("Importer CSV", type=["csv"])
            if uploaded:
                tmp = Path(".streamlit_upload.csv")
                tmp.write_bytes(uploaded.getvalue())
                transactions = load_transactions(str(tmp))
                tmp.unlink(missing_ok=True)
                st.caption(f"{len(transactions)} transactions importees")

        st.divider()
        st.caption("Module ML : actif" if ML_AVAILABLE else "Module ML : absent")

    if not transactions:
        st.info("Chargez des transactions dans la barre laterale.")
        return

    if st.button(
        "Lancer l'analyse",
        type="primary",
        icon=":material/play_arrow:",
        use_container_width=True,
    ):
        with st.spinner("Analyse en cours..."):
            try:
                results = detect_fraud(transactions)
            except NotImplementedError:
                st.error("Implementez detect_fraud dans fraud_detection.py.")
                return
            except Exception as exc:
                st.error(f"Erreur : {exc}")
                return

        st.session_state["analysis_transactions"] = transactions
        st.session_state["analysis_results"] = results
        st.session_state["journal_row_limit"] = TABLE_INITIAL_ROWS
        for key in list(st.session_state.keys()):
            if key.endswith("_row_limit") and key != "journal_row_limit":
                del st.session_state[key]

    if "analysis_results" in st.session_state:
        render_interface(
            st.session_state["analysis_transactions"],
            st.session_state["analysis_results"],
        )


if __name__ == "__main__":
    main()
