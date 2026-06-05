"""
Interface Streamlit — À CRÉER PAR VOUS pour le jury.

Le jury lancera :  streamlit run app.py

Règles :
  - Ne modifiez pas l'appel à detect_fraud / load_transactions (contrat technique).
  - Personnalisez render_interface() : clarté, intuitivité, compréhension pour un public non technique.
  - L'interface n'est PAS notée par la CI ; elle sert au jury pour repêcher et comparer les candidats.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from fraud_detection import detect_fraud, load_transactions

SAMPLE_CSV = Path(__file__).parent / "data" / "sample_transactions.csv"


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
    delta_risk = risk_rate - threshold

    st.subheader("Résumé")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total transactions", total)
    with col2:
        st.metric("Alertes détectées", alerts, delta=f"{alerts} alerte(s)" if alerts > 0 else None)
    with col3:
        st.metric(
            "Taux de risque",
            f"{risk_rate:.1f} %",
            delta=f"{delta_risk:+.1f} pts vs seuil {threshold:.0f} %",
            delta_color="inverse" if risk_rate > threshold else "normal",
        )
    with col4:
        st.metric("Score moyen", f"{avg_score:.2f}")

    st.divider()
    st.subheader("Transactions")

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
            "fraud_score": res.get("fraud_score", 0.0),
            "verdict": "🔴 Suspect" if suspicious else "🟢 Normal",
            "reason": res.get("reason", ""),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("fraud_score", ascending=False)

    alerts_only = st.checkbox("Afficher uniquement les alertes", value=False)
    display_df = df[df["verdict"] == "🔴 Suspect"] if alerts_only and not df.empty else df

    if display_df.empty:
        st.info("Aucune transaction à afficher.")
    else:
        styled = display_df.style.background_gradient(
            cmap="Reds", subset=["fraud_score"]
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Détail par client")

    if not results:
        st.info("Aucun résultat d'analyse disponible.")
        return

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
        return

    for user_id in sorted(users_with_alerts):
        n_alerts = users_with_alerts[user_id]
        with st.expander(f"👤 {user_id} — {n_alerts} alerte(s)"):
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
                    "fraud_score": res.get("fraud_score", 0.0),
                    "verdict": "🔴 Suspect" if res.get("is_suspicious") else "🟢 Normal",
                    "reason": res.get("reason", ""),
                })
            user_df = pd.DataFrame(user_rows)
            if not user_df.empty:
                user_df["_ts_sort"] = pd.to_datetime(
                    user_df["timestamp"], errors="coerce", utc=True
                )
                user_df = user_df.sort_values("_ts_sort").drop(columns=["_ts_sort"])
            st.dataframe(user_df, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Détection de fraude — Hackathon INTELO2026",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("Détection de fraude financière")
    st.caption("Hackathon INTELO2026 — interface participant · évaluée par le jury")

    with st.sidebar:
        st.header("Charger des données")
        use_sample = st.toggle("Utiliser le fichier d'exemple", value=True)
        transactions: list[dict] = []

        if use_sample:
            transactions = load_transactions(str(SAMPLE_CSV))
            st.success(f"{len(transactions)} transactions (exemple)")
        else:
            uploaded = st.file_uploader("Importer un CSV", type=["csv"])
            if uploaded:
                tmp = Path(".streamlit_upload.csv")
                tmp.write_bytes(uploaded.getvalue())
                transactions = load_transactions(str(tmp))
                tmp.unlink(missing_ok=True)
                st.success(f"{len(transactions)} transactions importées")

        st.divider()
        st.markdown(
            "**Jury :** évaluez l'ergonomie et la clarté de l'écran principal, "
            "pas seulement le score des tests."
        )

    if not transactions:
        st.info("Chargez des transactions (barre latérale) puis lancez l'analyse.")
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
