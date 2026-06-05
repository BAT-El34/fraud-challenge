import datetime
import math
import numpy as np
from sklearn.ensemble import IsolationForest

MIN_SAMPLES_FOR_MODEL = 5

def _parse_timestamp(ts_str: str) -> datetime.datetime | None:
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        # Replaces 'Z' with '+00:00' to cleanly parse ISO 8601 UTC string across Python versions
        ts_clean = ts_str.replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(ts_clean)
    except Exception:
        return None

def _compute_user_stats(transactions: list[dict]) -> dict:
    user_stats = {}
    for tx in transactions:
        if not tx or not isinstance(tx, dict):
            continue
        user_id = tx.get("user_id")
        if not user_id:
            continue
        amount = tx.get("amount")
        if amount is not None and isinstance(amount, (int, float)) and amount > 0:
            if user_id not in user_stats:
                user_stats[user_id] = {"sum": 0.0, "count": 0}
            user_stats[user_id]["sum"] += float(amount)
            user_stats[user_id]["count"] += 1
    return user_stats

def _extract_features(tx: dict, user_stats: dict) -> list[float] | None:
    try:
        dt = _parse_timestamp(tx.get("timestamp"))
        amount = tx.get("amount")
        user_id = tx.get("user_id")

        # f1_amount_log
        try:
            amt_val = float(amount) if amount is not None else 0.0
        except (ValueError, TypeError):
            amt_val = 0.0

        if amt_val <= 0.0:
            f1_amount_log = 0.0
        else:
            f1_amount_log = math.log1p(amt_val)

        # f2_hour
        if dt is not None:
            f2_hour = float(dt.hour)
        else:
            f2_hour = 12.0

        # f3_amount_ratio & user_mean_amount
        user_mean_amount = 50.0
        has_history = False
        if user_id and user_id in user_stats:
            stats = user_stats[user_id]
            total_sum = stats["sum"]
            total_count = stats["count"]

            is_valid = amount is not None and isinstance(amount, (int, float)) and amount > 0
            if is_valid:
                sum_other = total_sum - float(amount)
                count_other = total_count - 1
            else:
                sum_other = total_sum
                count_other = total_count

            if count_other > 0:
                user_mean_amount = sum_other / count_other
                has_history = True

        if not has_history:
            f3_amount_ratio = 1.0
        else:
            if user_mean_amount <= 0.0:
                f3_amount_ratio = 1.0
            else:
                f3_amount_ratio = amt_val / user_mean_amount

        # f4_card_absent
        card_present = tx.get("card_present")
        if card_present is True:
            f4_card_absent = 0.0
        elif card_present is False:
            f4_card_absent = 1.0
        else:
            f4_card_absent = 0.5

        # f5_is_night
        if dt is not None:
            h = dt.hour
            if h in [0, 1, 2, 3, 4, 5, 22, 23]:
                f5_is_night = 1.0
            else:
                f5_is_night = 0.0
        else:
            f5_is_night = 0.0

        return [f1_amount_log, f2_hour, f3_amount_ratio, f4_card_absent, f5_is_night]
    except Exception:
        return None

def get_ml_scores(transactions: list[dict]) -> dict[str, float]:
    """
    Analyse une liste complète de transactions avec Isolation Forest.
    """
    try:
        if not transactions or not isinstance(transactions, list):
            return {}

        user_stats = _compute_user_stats(transactions)

        transactions_valides = []
        X = []

        for tx in transactions:
            if tx is None or not isinstance(tx, dict):
                continue
            tid = tx.get("transaction_id")
            if tid is None:
                continue
            
            feats = _extract_features(tx, user_stats)
            if feats is not None:
                X.append(feats)
                transactions_valides.append(tx)

        # Si le nombre de transactions featurisables est insuffisant, retourne 0.0 pour toutes.
        if len(X) < MIN_SAMPLES_FOR_MODEL:
            result = {}
            for tx in transactions:
                if tx and isinstance(tx, dict):
                    tid = tx.get("transaction_id")
                    if tid is not None:
                        result[str(tid)] = 0.0
            return result

        # Conversion en matrice numpy
        X_arr = np.array(X, dtype=float)

        # Entraînement de l'Isolation Forest global
        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_arr)
        raw_scores = model.decision_function(X_arr)

        # Normalisation des scores bruts vers [0.0, 1.0]
        # decision_function retourne des valeurs négatives pour les anomalies et positives pour le normal.
        inverted = -raw_scores
        min_val = inverted.min()
        max_val = inverted.max()

        if max_val - min_val > 1e-9:
            normalized = (inverted - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(inverted)

        normalized = np.clip(normalized, 0.0, 1.0)

        # Construction du dictionnaire des résultats
        result = {}
        for i, tx in enumerate(transactions_valides):
            tid = tx.get("transaction_id")
            if tid is not None:
                result[str(tid)] = float(normalized[i])

        return result
    except Exception:
        # Jamais d'exception levée
        try:
            return {str(tx.get("transaction_id")): 0.0 for tx in transactions if tx and isinstance(tx, dict) and tx.get("transaction_id") is not None}
        except Exception:
            return {}
