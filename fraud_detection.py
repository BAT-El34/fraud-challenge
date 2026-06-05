"""
Défi — Détection de fraude financière.

Vous devez implémenter la fonction `detect_fraud`.
La fonction `load_transactions` vous est FOURNIE (ne la modifiez pas).
"""

import csv
import statistics
from datetime import datetime, timezone

try:
    from fraud_ml import get_ml_scores
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

    def get_ml_scores(transactions):
        return {}


_CONTINENT_EU = {
    "FR", "DE", "ES", "IT", "BE", "NL", "PT", "CH", "AT", "PL",
    "SE", "NO", "DK", "FI", "IE", "GR", "CZ", "RO", "HU", "SK",
    "BG", "HR", "SI", "LT", "LV", "EE", "LU", "MT", "CY", "GB",
}
_CONTINENT_AS = {
    "JP", "CN", "KR", "IN", "SG", "TH", "MY", "ID", "PH", "VN",
    "TW", "HK", "PK", "BD", "LK", "NP", "MM", "KH", "LA", "MN",
}
_CONTINENT_AM = {
    "US", "CA", "MX", "BR", "AR", "CL", "CO", "PE", "VE", "EC",
    "UY", "PY", "BO", "CR", "PA", "GT", "CU", "DO", "HN", "NI",
}
_CONTINENT_AF = {
    "TG", "SN", "CI", "MA", "NG", "ZA", "GH", "CM", "KE", "ET",
    "TZ", "UG", "DZ", "TN", "EG", "AO", "MZ", "ZW", "RW", "BF",
}
_CONTINENT_GROUPS = [_CONTINENT_EU, _CONTINENT_AS, _CONTINENT_AM, _CONTINENT_AF]

_GEO_TRAVELER_MINUTES = 4320  # 3 days


def load_transactions(path):
    """Lit un fichier CSV de transactions et renvoie une liste de dicts."""
    transactions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transactions.append(_clean_row(row))
    return transactions


def _clean_row(row):
    def get(key):
        v = row.get(key)
        return v.strip() if isinstance(v, str) and v.strip() != "" else None

    amount_raw = get("amount")
    try:
        amount = float(amount_raw) if amount_raw is not None else None
    except ValueError:
        amount = None

    card_raw = get("card_present")
    if card_raw is None:
        card_present = None
    else:
        card_present = card_raw.lower() in ("true", "1", "yes", "oui")

    return {
        "transaction_id": get("transaction_id"),
        "timestamp": get("timestamp"),
        "user_id": get("user_id"),
        "amount": amount,
        "currency": get("currency"),
        "merchant": get("merchant"),
        "country": get("country"),
        "card_present": card_present,
    }


def _parse_ts(ts_str):
    """Parse un timestamp ISO 8601, retourne datetime ou None sans jamais planter."""
    if ts_str is None or not isinstance(ts_str, str):
        return None
    try:
        normalized = ts_str.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _same_continent(country_a, country_b):
    """Retourne True si les deux pays sont sur le même continent (heuristique)."""
    if not country_a or not country_b:
        return False
    a, b = country_a.upper(), country_b.upper()
    if a == b:
        return True
    for group in _CONTINENT_GROUPS:
        if a in group and b in group:
            return True
    return False


def _build_user_history(transactions):
    """Construit un dict {user_id: [transactions précédentes]} dans l'ordre de la liste."""
    history = {}
    result = []
    for tx in transactions:
        uid = tx.get("user_id")
        prev = list(history.get(uid, [])) if uid else []
        result.append(prev)
        if uid:
            history.setdefault(uid, []).append(tx)
    return result


def _apply_level1(tx):
    """Règles fondamentales — retourne (score, reason)."""
    signals = []
    amount = tx.get("amount")
    country = tx.get("country")
    user_id = tx.get("user_id")

    if amount is None:
        signals.append((0.85, "Montant manquant"))
    elif amount <= 0:
        signals.append((0.90, "Montant nul ou négatif"))

    if country is None and user_id is None:
        signals.append((0.85, "Champs obligatoires manquants: country, user_id"))
    elif country is None:
        signals.append((0.85, "Champs obligatoires manquants: country"))
    elif user_id is None:
        signals.append((0.80, "Champs obligatoires manquants: user_id"))

    if not signals:
        return 0.0, ""

    score = max(s for s, _ in signals)
    reason = " | ".join(r for _, r in signals)
    return score, reason


def _signal_amount_anomaly(tx, history):
    """Signal A — montant anormal vs historique."""
    amount = tx.get("amount")
    if amount is None or amount <= 0:
        return None

    amounts = [
        h["amount"] for h in history
        if h.get("amount") is not None and h["amount"] > 0
    ]
    if len(amounts) < 3:
        return None

    mean = statistics.mean(amounts)
    std = statistics.stdev(amounts) if len(amounts) >= 2 else 0.0

    if std == 0:
        if mean <= 0:
            return None
        ratio = amount / mean
        if ratio > 10:
            return 0.90, "Montant très supérieur à l'habitude du client"
        if ratio > 3:
            return 0.75, "Montant anormalement élevé pour ce client"
        return None

    z_score = (amount - mean) / max(std, 1.0)
    if z_score > 5:
        return 0.90, "Montant très supérieur à l'habitude du client"
    if z_score > 3:
        return 0.75, "Montant anormalement élevé pour ce client"
    return None


def _signal_velocity(tx, history):
    """Signal B — fréquence suspecte dans les 60 dernières minutes."""
    ts_current = _parse_ts(tx.get("timestamp"))
    if ts_current is None:
        return None

    count = 1
    window_start = ts_current.timestamp() - 3600
    for h in history:
        ts_h = _parse_ts(h.get("timestamp"))
        if ts_h is None:
            continue
        if window_start <= ts_h.timestamp() <= ts_current.timestamp():
            count += 1

    if count >= 5:
        return 0.80, "Fréquence de transactions anormalement élevée"
    return None


def _signal_geography(tx, history):
    """Signal C — incohérence géographique."""
    country = tx.get("country")
    ts_current = _parse_ts(tx.get("timestamp"))
    if not country or ts_current is None:
        return None

    prev_different = None
    for h in reversed(history):
        h_country = h.get("country")
        if h_country and h_country != country:
            prev_different = h
            break

    if prev_different is None:
        return None

    ts_prev = _parse_ts(prev_different.get("timestamp"))
    if ts_prev is None:
        return None

    delay_minutes = abs((ts_current - ts_prev).total_seconds()) / 60.0
    if delay_minutes >= _GEO_TRAVELER_MINUTES:
        return None

    prev_country = prev_different.get("country")
    threshold = 120 if _same_continent(country, prev_country) else 600
    if delay_minutes < threshold:
        return 0.88, "Deux pays différents en trop peu de temps"
    return None


def _apply_level2(tx, history):
    """Logique métier — retourne (score, reason)."""
    signals = []

    for signal_fn in (_signal_amount_anomaly, _signal_velocity, _signal_geography):
        result = signal_fn(tx, history)
        if result is not None:
            signals.append(result)

    if not signals:
        return 0.0, ""

    score = max(s for s, _ in signals)
    reason = " | ".join(r for _, r in signals)
    return score, reason


def detect_fraud(transactions):
    """Analyse une liste de transactions et renvoie un verdict pour chacune.

    Retour : list[dict] avec transaction_id, fraud_score (0-1),
    is_suspicious (bool), reason (str) — un résultat par transaction, même ordre.
    """
    if not transactions:
        return []

    ml_scores = get_ml_scores(transactions)
    histories = _build_user_history(transactions)
    results = []

    for tx, history in zip(transactions, histories):
        try:
            tid = tx.get("transaction_id")
            rules_score, reason = _apply_level1(tx)

            if rules_score == 0.0:
                rules_score, reason = _apply_level2(tx, history)

            if rules_score == 0.0:
                reason = "Transaction conforme au profil du client"

            ml_score = ml_scores.get(tid, 0.0) if tid else 0.0
            if ML_AVAILABLE:
                final = 0.6 * rules_score + 0.4 * ml_score
            else:
                final = rules_score
            final = max(0.0, min(1.0, final))

            results.append({
                "transaction_id": tid,
                "fraud_score": final,
                "is_suspicious": final >= 0.5,
                "reason": reason,
            })
        except Exception:
            results.append({
                "transaction_id": tx.get("transaction_id"),
                "fraud_score": 0.0,
                "is_suspicious": False,
                "reason": "Erreur d'analyse",
            })

    return results
