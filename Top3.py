# app.py
"""
AI Horse Racing Predictor - Single-file Streamlit app
- Optuna intégré (10 essais par défaut)
- joblib.Memory caching pour fetch_url + parsing
- st.cache_data pour UI-level caching
- Mode demo / upload CSV / fetch URL (best-effort)
- Local & Streamlit Cloud ready
"""

import re
import json
import time
from dataclasses import dataclass
from typing import Any
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import streamlit as st
from joblib import Memory
import optuna
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, precision_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import logging
import os

# -----------------------------
# Configuration
# -----------------------------
@dataclass
class Config:
    RANDOM_SEED: int = 42
    CACHE_DIR: str = ".cache_joblib"
    OPTUNA_TRIALS: int = 10
    ST_CACHE_TTL: int = 3600
    DEFAULT_MODEL: str = "lightgbm"
    MAX_OPTUNA_TRIALS_UI: int = 40

cfg = Config()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deep_race_app")

# Ensure cache dir exists
os.makedirs(cfg.CACHE_DIR, exist_ok=True)
memory = Memory(cfg.CACHE_DIR, verbose=0)

# -----------------------------
# Utilities
# -----------------------------
def clean_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    return re.sub(r"[^\w\s\-,.;:']", "", s).strip()

def safe_float(x, default=10.0):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default

def safe_int(x, default=1):
    try:
        m = re.search(r"\d+", str(x))
        return int(m.group()) if m else default
    except Exception:
        return default

# -----------------------------
# Data fetching/parsing (cached)
# -----------------------------
@memory.cache
def fetch_url_raw(url: str, timeout: int = 15) -> str:
    logger.info(f"Fetching {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; DeepRace/1.0; +https://example.local)"
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

def extract_from_table(soup: BeautifulSoup):
    horses = []
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all(["td", "th"])
            if len(cols) < 3:
                continue
            txts = [clean_text(c.get_text()) for c in cols]
            horse = {}
            for t in txts:
                if t.isdigit() and "Numéro de corde" not in horse:
                    horse["Numéro de corde"] = t
                elif re.match(r"^\d+[.,]?\d*\s*(kg|KG)?$", t) and "Poids" not in horse:
                    horse["Poids"] = t.split()[0]
                elif re.match(r"^\d+[.,]?\d*$", t) and "Cote" not in horse:
                    horse["Cote"] = t
                elif len(t) > 2 and "Nom" not in horse:
                    horse["Nom"] = t
                elif re.match(r"^[0-9a-zA-Z]{2,8}$", t) and "Musique" not in horse:
                    horse["Musique"] = t
            if "Nom" in horse and "Cote" in horse:
                horses.append(horse)
    return horses

@memory.cache
def parse_html_to_df(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    # try JSON-LD
    scripts = soup.find_all("script", type="application/ld+json")
    for s in scripts:
        try:
            data = json.loads(s.string)
            # heuristic scanning
            if isinstance(data, dict):
                for key in ["runners", "horse_list", "itemListElement", "horseList"]:
                    arr = data.get(key)
                    if isinstance(arr, list) and arr:
                        horses = []
                        for item in arr:
                            h = {}
                            h["Nom"] = item.get("name") or item.get("horseName")
                            h["Cote"] = item.get("odds") or item.get("price")
                            h["Numéro de corde"] = item.get("number") or item.get("post")
                            horses.append(h)
                        if horses:
                            return pd.DataFrame(horses)
        except Exception:
            continue
    # fallback table parsing
    horses = extract_from_table(soup)
    if not horses:
        return pd.DataFrame()
    df = pd.DataFrame(horses)
    if "Cote" in df.columns:
        df["Cote"] = df["Cote"].astype(str).str.replace(",", ".")
    if "Poids" in df.columns:
        df["Poids"] = df["Poids"].astype(str).str.replace(",", ".")
    return df

@st.cache_data(ttl=cfg.ST_CACHE_TTL)
def load_from_url(url: str) -> pd.DataFrame:
    try:
        html = fetch_url_raw(url)
        return parse_html_to_df(html)
    except Exception as e:
        logger.warning(f"load_from_url failed: {e}")
        return pd.DataFrame()

# -----------------------------
# Demo data generator
# -----------------------------
def demo_df(n_runners=12):
    base = [
        "Galopin des Champs","Hippomene","Quick Thunder","Flash du Gite","Roi du Vent",
        "Saphir Etoile","Tonnerre Royal","Jupiter Force","Ouragan Bleu","Sprint Final",
        "Eclair Volant","Meteorite","Pegase Rapide","Foudre Noire","Vent du Nord"
    ]
    n = max(8, min(n_runners, len(base)))
    odds = np.clip(np.random.pareto(1.5, n) * 5 + 2, 2, 30)
    data = {
        "Nom": base[:n],
        "Numéro de corde": [str(i+1) for i in range(n)],
        "Cote": [f"{o:.1f}" for o in odds],
        "Poids": [f"{p:.1f}" for p in np.random.normal(58, 3, n)],
        "Musique": np.random.choice(["1a2a","2a1a","3a2a","1a3a","4a2a"], n)
    }
    return pd.DataFrame(data)

# -----------------------------
# Features
# -----------------------------
def prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Cote" in df.columns:
        df["odds_numeric"] = df["Cote"].apply(lambda x: safe_float(x, default=10.0))
    else:
        df["odds_numeric"] = 10.0
    if "Numéro de corde" in df.columns:
        df["draw_numeric"] = df["Numéro de corde"].apply(lambda x: safe_int(x, default=1))
    else:
        df["draw_numeric"] = range(1, len(df)+1)
    if "Poids" in df.columns:
        df["weight_kg"] = df["Poids"].apply(lambda x: safe_float(x, default=60.0))
    else:
        df["weight_kg"] = 60.0
    if "Musique" not in df.columns:
        df["Musique"] = "5a5a"
    return df

def weight_score_series(weights):
    wm = np.mean(weights)
    ws = np.std(weights) if np.std(weights) > 0 else 1.0
    z = (wm - np.array(weights)) / ws
    return 1.0 / (1.0 + np.exp(-z))

def parse_musique_score(musique):
    try:
        if pd.isna(musique) or not musique:
            return 0.3
        positions = [int(ch) for ch in str(musique) if ch.isdigit()]
        if not positions:
            return 0.3
        weights = np.linspace(1.0, 0.5, len(positions))
        weighted_positions = sum(p*w for p,w in zip(positions, weights))
        total_weight = sum(weights)
        score = 1.0 / (weighted_positions / total_weight)
        return min(score / 5.0, 1.0)
    except Exception:
        return 0.3

def consistency_score(musique):
    try:
        positions = [int(ch) for ch in str(musique) if ch.isdigit()]
        if len(positions) < 2:
            return 0.3
        var = np.var(positions)
        return 1.0 / (1.0 + var)
    except Exception:
        return 0.3

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_base(df)
    n = len(df)
    feats = pd.DataFrame(index=df.index)
    feats["odds_reciprocal"] = 1.0 / df["odds_numeric"].replace(0, 1)
    feats["odds_log"] = np.log1p(df["odds_numeric"])
    feats["odds_rank_pct"] = df["odds_numeric"].rank(pct=True)
    feats["draw_position_pct"] = df["draw_numeric"] / max(1, n)
    feats["draw_sector"] = pd.cut(df["draw_numeric"], bins=3, labels=[0,1,2]).astype(int)
    feats["weight_score"] = weight_score_series(df["weight_kg"])
    feats["weight_advantage"] = (df["weight_kg"].max() - df["weight_kg"]).fillna(0)
    feats["recent_perf_score"] = df["Musique"].apply(parse_musique_score)
    feats["consistency_score"] = df["Musique"].apply(consistency_score)
    feats["odds_draw_synergy"] = feats["odds_reciprocal"] * feats["draw_position_pct"]
    feats["odds_weight_synergy"] = feats["odds_reciprocal"] * feats["weight_score"]
    feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
    return feats

# -----------------------------
# Modeling with Optuna
# -----------------------------
MODEL_REG = {
    "lightgbm": lgb.LGBMClassifier,
    "xgboost": xgb.XGBClassifier,
    "random_forest": RandomForestClassifier
}

class ModelManager:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or cfg.DEFAULT_MODEL
        self.model = None
        self.scaler = StandardScaler()

    def _default_inst(self):
        cls = MODEL_REG.get(self.model_name, lgb.LGBMClassifier)
        if self.model_name == "xgboost":
            return cls(use_label_encoder=False, eval_metric="logloss", random_state=cfg.RANDOM_SEED)
        return cls(random_state=cfg.RANDOM_SEED)

    def train(self, X: pd.DataFrame, y: pd.Series, use_optuna: bool = True, optuna_trials: int = None):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=cfg.RANDOM_SEED)
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        if use_optuna:
            trials = optuna_trials or cfg.OPTUNA_TRIALS
            best_params = self._optuna_tune(X_train_s, y_train, n_trials=trials)
            model = self._instantiate_with_params(best_params)
        else:
            model = self._default_inst()

        calibrated = CalibratedClassifierCV(model, method="isotonic", cv=3)
        calibrated.fit(X_train_s, y_train)
        self.model = calibrated

        preds = calibrated.predict_proba(X_test_s)[:, 1]
        metrics = {
            "test_auc": roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5,
            "test_log_loss": log_loss(y_test, preds) if len(np.unique(y_test)) > 1 else float("inf"),
            "precision_top3": self._precision_at_k(y_test, preds, k=3)
        }
        return metrics

    def predict_proba(self, X: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained")
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)[:, 1]

    def persist(self, path: str):
        joblib.dump({"model": self.model, "scaler": self.scaler, "model_name": self.model_name}, path)

    def load(self, path: str):
        d = joblib.load(path)
        self.model = d["model"]
        self.scaler = d["scaler"]
        self.model_name = d.get("model_name", self.model_name)

    def _precision_at_k(self, y_true, y_prob, k=3):
        if len(y_true) < k:
            return 0.0
        idx = np.argsort(y_prob)[-k:]
        return precision_score(y_true.iloc[idx], [1] * len(idx), zero_division=0)

    def _instantiate_with_params(self, params: dict):
        cls = MODEL_REG.get(self.model_name, lgb.LGBMClassifier)
        if self.model_name == "xgboost":
            return cls(**params, use_label_encoder=False, eval_metric="logloss", random_state=cfg.RANDOM_SEED)
        return cls(**params, random_state=cfg.RANDOM_SEED)

    def _optuna_tune(self, X, y, n_trials: int = 10):
        def objective(trial: optuna.Trial):
            if self.model_name == "lightgbm":
                params = {
                    "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0)
                }
                model = lgb.LGBMClassifier(**params, random_state=cfg.RANDOM_SEED)
            elif self.model_name == "xgboost":
                params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0)
                }
                model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss", random_state=cfg.RANDOM_SEED)
            else:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 20)
                }
                model = RandomForestClassifier(**params, random_state=cfg.RANDOM_SEED)

            scores = cross_val_score(model, X, y, cv=3, scoring="roc_auc")
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        logger.info(f"Optuna best value: {study.best_value} / params: {study.best_trial.params}")
        return study.best_trial.params

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🤖 Pronostics Hippiques IA", layout="wide")
st.title("🏇 Pronostics Hippiques IA — App unique (Optuna intégré)")

# Sidebar
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Modèle", ["lightgbm", "xgboost", "random_forest"], index=0)
use_optuna = st.sidebar.checkbox("Activer AutoML (Optuna)", value=True)
optuna_trials = st.sidebar.number_input("Nombre d'essais Optuna", min_value=1, max_value=cfg.MAX_OPTUNA_TRIALS_UI, value=cfg.OPTUNA_TRIALS)
demo_runners = st.sidebar.slider("Partants (démo)", 8, 16, 12)
save_dir = st.sidebar.text_input("Répertoire pour sauvegarder modèle", value="models")
os.makedirs(save_dir, exist_ok=True)

# Data input
st.header("Chargement des données")
col1, col2 = st.columns(2)
with col1:
    url = st.text_input("URL de la course (optionnel, Geny.com ou similaire)")
    uploaded_file = st.file_uploader("Ou téléversez un CSV", type=["csv"])
with col2:
    st.info("Si aucune source fournie, la démo sera utilisée.")
    st.write("📝 Format CSV recommandé: colonnes 'Nom','Cote','Numéro de corde','Poids','Musique'")

# Run controls
run_button = st.button("🚀 Lancer l'analyse")

if run_button:
    # Load data
    with st.spinner("Chargement des données..."):
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                source = "Fichier CSV"
            except Exception as e:
                st.error(f"Impossible de lire le CSV: {e}")
                df = pd.DataFrame()
                source = "erreur"
        elif url:
            df = load_from_url(url)
            source = url
        else:
            df = demo_df(demo_runners)
            source = "Données de démo"

    if df is None or df.empty:
        st.error("Aucune donnée valide chargée.")
    else:
        st.success(f"Données chargées depuis: {source} — {len(df)} lignes")
        st.dataframe(df.head(10))

        # Feature engineering
        with st.spinner("Génération des features..."):
            X = build_features(df)

        # Weak labels (heuristic) : below median odds => label 1 (potentiel)
        try:
            median_odds = df["Cote"].apply(lambda x: safe_float(x, 10.0)).median()
            y = df["Cote"].apply(lambda x: 1 if safe_float(x, 10.0) < median_odds else 0)
        except Exception:
            y = pd.Series([0]*len(df))

        st.write("Extrait des features:")
        st.dataframe(X.head(10))

        # Train model
        mm = ModelManager(model_choice)
        with st.spinner("Entraînement du modèle (Optuna si activé)..."):
            metrics = mm.train(X, y, use_optuna=use_optuna, optuna_trials=int(optuna_trials))
        st.success("Entraînement terminé")
        st.write("Métriques:", metrics)

        # Predict
        probs = mm.predict_proba(X)
        # Adjust probabilities for race context
        probs = np.clip(probs, 0.02, 0.98)
        if probs.sum() > 0:
            probs = probs / probs.sum() * min(0.95, len(probs) * 0.12)

        df_out = df.copy()
        df_out["probability"] = probs
        df_out["score_final"] = probs
        df_out = df_out.sort_values("score_final", ascending=False).reset_index(drop=True)
        df_out["rank"] = df_out.index + 1

        st.subheader("Résultats — Top 20")
        display_cols = [c for c in ["rank", "Nom", "Cote", "Numéro de corde", "Poids", "probability"] if c in df_out.columns]
        st.dataframe(df_out[display_cols].head(20))

        st.subheader("Recommandations (Top 3)")
        for i, row in df_out.head(3).iterrows():
            st.markdown(f"**{i+1}. {row.get('Nom','N/A')}** — Prob: {row['probability']*100:.1f}% | Cote: {row.get('Cote','N/A')}")

        # Save model
        model_path = os.path.join(save_dir, f"model_{model_choice}_{int(time.time())}.joblib")
        if st.button("💾 Sauvegarder le modèle sur le disque"):
            mm.persist(model_path)
            st.success(f"Modèle sauvegardé: {model_path}")

        # Option to export results CSV
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger les résultats (CSV)", data=csv, file_name="results.csv", mime="text/csv")

# Footer notes
st.markdown("---")
st.markdown("**Notes:** App démo. Pour production, remplace les labels heuristiques par des labels réels, "
            "affine la validation et surveille les essais Optuna (coûteux en CPU).")
