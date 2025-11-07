# Project: AI Horse Racing Predictor - Modular Refactor (Local)
# Layout: src/ (modules) + run.py + requirements.txt

"""
This single-file package textdoc contains multiple Python modules separated by
"### FILE: <path>" markers. Save each section into the indicated path to build
the project locally.

Selected options:
1) Split modules
2) Advanced caching (joblib.Memory + st.cache_resource)
3) AutoML (Optuna hyperparameter tuning)
4) Skip SHAP
5) Local use

How to use:
- Create a folder (e.g. `deep_race_app`) and save each "### FILE" block to that path.
- Create a virtualenv and `pip install -r requirements.txt`.
- Run `python run.py` to start the Streamlit app locally: `streamlit run run.py`.

"""

###############################################################################
### FILE: requirements.txt
###############################################################################
# core


###############################################################################
### FILE: run.py
###############################################################################
from src.ui import main

if __name__ == "__main__":
    main()

###############################################################################
### FILE: src/__init__.py
###############################################################################
# Package init

###############################################################################
### FILE: src/config.py
###############################################################################
from dataclasses import dataclass

@dataclass
class Config:
    RANDOM_SEED: int = 42
    CACHE_DIR: str = ".cache_joblib"
    OPTUNA_TRIALS: int = 40
    CACHE_TTL: int = 3600  # seconds for st.cache_data if used
    DEFAULT_MODEL: str = "lightgbm"

cfg = Config()

###############################################################################
### FILE: src/utils.py
###############################################################################
import re
import json
import logging
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deep_race")


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    return re.sub(r"[^\w\s.,-]", "", s).strip()


def safe_float(x, default=10.0):
    try:
        return float(str(x).replace(',', '.'))
    except Exception:
        return default


def safe_int(x, default=1):
    try:
        import re
        m = re.search(r"\d+", str(x))
        return int(m.group()) if m else default
    except Exception:
        return default


def to_json(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)

###############################################################################
### FILE: src/data.py
###############################################################################
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from .utils import clean_text, safe_float, safe_int, logger
from joblib import Memory
from .config import cfg

# persistent disk cache for expensive fetches / parsing
memory = Memory(cfg.CACHE_DIR, verbose=0)


@memory.cache
def fetch_url(url: str, timeout: int = 15) -> str:
    logger.info(f"Fetching URL: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; DeepRace/1.0; +https://example.local)'
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def extract_from_table(soup: BeautifulSoup):
    horses = []
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all(['td','th'])
            if len(cols) < 3:
                continue
            # naive mapping
            txts = [clean_text(c.get_text()) for c in cols]
            # best-effort column mapping
            horse = {}
            for t in txts:
                if t.isdigit() and 'Numéro de corde' not in horse:
                    horse['Numéro de corde'] = t
                elif re_weight := re.search(r"^(\d+[.,]?\d*)\s*(kg|KG)?$", t):
                    horse['Poids'] = re_weight.group(1)
                elif re_odds := re.match(r"^\d+[.,]?\d*$", t) and 'Cote' not in horse:
                    horse['Cote'] = t
                elif len(t) > 1 and 'Nom' not in horse:
                    horse['Nom'] = t
            if 'Nom' in horse and 'Cote' in horse:
                horses.append(horse)
    return horses


@memory.cache
def parse_geny_like_html(html: str):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    # try JSON-LD first
    scripts = soup.find_all('script', type='application/ld+json')
    for s in scripts:
        try:
            data = json.loads(s.string)
            # heuristic: find array of runners
            if isinstance(data, dict):
                # flatten common structures
                for k in ['race', 'runners', 'horse_list', 'itemListElement']:
                    if k in data:
                        arr = data[k]
                        if isinstance(arr, list) and len(arr) > 0:
                            # basic mapping
                            horses = []
                            for item in arr:
                                h = {}
                                h['Nom'] = item.get('name') or item.get('horseName')
                                h['Cote'] = item.get('odds') or item.get('price')
                                h['Numéro de corde'] = item.get('number') or item.get('post')
                                horses.append(h)
                            if horses:
                                return horses
        except Exception:
            continue

    # fallback to table extraction
    horses = extract_from_table(soup)
    return horses


def html_to_df(html: str):
    horses = parse_geny_like_html(html)
    if not horses:
        return pd.DataFrame()

    df = pd.DataFrame(horses)
    # basic normalization
    if 'Cote' in df.columns:
        df['Cote'] = df['Cote'].apply(lambda x: str(x).replace(',', '.'))
    if 'Poids' in df.columns:
        df['Poids'] = df['Poids'].apply(lambda x: str(x).replace(',', '.'))

    return df


def load_from_url(url: str):
    try:
        html = fetch_url(url)
        return html_to_df(html)
    except Exception as e:
        logger.warning(f"fetch/load failed: {e}")
        return pd.DataFrame()


def demo_df(n_runners=12):
    import numpy as np
    base = [
        'Galopin des Champs','Hippomene','Quick Thunder','Flash du Gite','Roi du Vent',
        'Saphir Etoile','Tonnerre Royal','Jupiter Force','Ouragan Bleu','Sprint Final',
        'Eclair Volant','Meteorite','Pegase Rapide','Foudre Noire','Vent du Nord'
    ]
    n = max(8, min(n_runners, len(base)))
    odds = np.clip(np.random.pareto(1.5, n) * 5 + 2, 2, 30)
    data = {
        'Nom': base[:n],
        'Numéro de corde': [str(i+1) for i in range(n)],
        'Cote': [f"{o:.1f}" for o in odds],
        'Poids': [f"{p:.1f}" for p in np.random.normal(58, 3, n)],
        'Musique': ["1a2a","2a1a","3a2a","1a3a","4a2a"] * (n//5 + 1)
    }
    import pandas as pd
    return pd.DataFrame(data)

###############################################################################
### FILE: src/features.py
###############################################################################
import numpy as np
import pandas as pd
from .utils import safe_float, safe_int


def prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    # basic normalization and safe columns
    df = df.copy()
    if 'Cote' in df.columns:
        df['odds_numeric'] = df['Cote'].apply(lambda x: safe_float(x, default=10.0))
    else:
        df['odds_numeric'] = 10.0

    if 'Numéro de corde' in df.columns:
        df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: safe_int(x, default=1))
    else:
        df['draw_numeric'] = range(1, len(df)+1)

    if 'Poids' in df.columns:
        df['weight_kg'] = df['Poids'].apply(lambda x: safe_float(x, default=60.0))
    else:
        df['weight_kg'] = 60.0

    # music default
    if 'Musique' not in df.columns:
        df['Musique'] = '5a5a'

    return df


def advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_base(df)
    n = len(df)

    features = pd.DataFrame(index=df.index)
    # odds
    features['odds_reciprocal'] = 1.0 / df['odds_numeric'].replace(0, 1)
    features['odds_log'] = np.log1p(df['odds_numeric'])
    features['odds_rank_pct'] = df['odds_numeric'].rank(pct=True)

    # draw
    features['draw_position_pct'] = df['draw_numeric'] / max(1, n)
    # draw sector as categorical numeric
    features['draw_sector'] = pd.cut(df['draw_numeric'], bins=3, labels=[0,1,2]).astype(int)

    # weight
    features['weight_score'] = _weight_score(df['weight_kg'])
    features['weight_advantage'] = (df['weight_kg'].max() - df['weight_kg']).fillna(0)

    # musique parsing
    features['recent_perf_score'] = df['Musique'].apply(_parse_musique)
    features['consistency_score'] = df['Musique'].apply(_consistency)

    # interactions
    features['odds_draw_synergy'] = features['odds_reciprocal'] * features['draw_position_pct']
    features['odds_weight_synergy'] = features['odds_reciprocal'] * features['weight_score']

    # normalize
    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    return features


def _weight_score(w):
    import numpy as np
    w_mean = np.mean(w)
    w_std = np.std(w) if np.std(w) > 0 else 1.0
    z = (w_mean - w) / w_std
    return 1.0 / (1.0 + np.exp(-z))


def _parse_musique(musique):
    try:
        if pd.isna(musique) or not musique:
            return 0.3
        positions = [int(ch) for ch in str(musique) if ch.isdigit()]
        if not positions:
            return 0.3
        weights = np.linspace(1.0, 0.5, len(positions))
        weighted = sum(p*w for p,w in zip(positions, weights))
        total = sum(weights)
        score = 1.0 / (weighted / total)
        return min(score/5.0, 1.0)
    except Exception:
        return 0.3


def _consistency(musique):
    try:
        positions = [int(ch) for ch in str(musique) if ch.isdigit()]
        if len(positions) < 2:
            return 0.3
        var = np.var(positions)
        return 1.0 / (1.0 + var)
    except Exception:
        return 0.3

###############################################################################
### FILE: src/models.py
###############################################################################
import numpy as np
import optuna
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, precision_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from .config import cfg
from .utils import logger

MODEL_REGISTRY = {
    'lightgbm': lgb.LGBMClassifier,
    'xgboost': xgb.XGBClassifier,
    'random_forest': RandomForestClassifier
}


class ModelWrapper:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or cfg.DEFAULT_MODEL
        self.model = None
        self.scaler = StandardScaler()

    def _get_default(self):
        cls = MODEL_REGISTRY.get(self.model_name, lgb.LGBMClassifier)
        return cls(random_state=cfg.RANDOM_SEED)

    def train(self, X, y, use_optuna=True):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=cfg.RANDOM_SEED)
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        if use_optuna:
            logger.info("Starting Optuna tuning")
            best_params = self._optuna_tune(X_train_s, y_train)
            logger.info(f"Best params: {best_params}")
            model = self._instantiate_with_params(best_params)
        else:
            model = self._get_default()

        # Calibrated model for reliable probabilities
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrated.fit(X_train_s, y_train)
        self.model = calibrated

        preds = calibrated.predict_proba(X_test_s)[:, 1]
        metrics = {
            'test_auc': roc_auc_score(y_test, preds) if len(np.unique(y_test))>1 else 0.5,
            'test_log_loss': log_loss(y_test, preds) if len(np.unique(y_test))>1 else float('inf'),
            'precision_top3': self._precision_at_k(y_test, preds, k=3)
        }
        return metrics

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)[:, 1]

    def persist(self, path: str):
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'model_name': self.model_name}, path)

    def load(self, path: str):
        d = joblib.load(path)
        self.model = d['model']
        self.scaler = d['scaler']
        self.model_name = d.get('model_name', self.model_name)

    def _precision_at_k(self, y_true, y_prob, k=3):
        if len(y_true) < k:
            return 0.0
        idx = np.argsort(y_prob)[-k:]
        return precision_score(y_true.iloc[idx], [1]*len(idx), zero_division=0)

    def _instantiate_with_params(self, params: dict):
        cls = MODEL_REGISTRY.get(self.model_name, lgb.LGBMClassifier)
        return cls(**params, random_state=cfg.RANDOM_SEED)

    def _optuna_tune(self, X, y):
        def objective(trial: optuna.Trial):
            if self.model_name == 'lightgbm':
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                }
                model = lgb.LGBMClassifier(**params, random_state=cfg.RANDOM_SEED)
            elif self.model_name == 'xgboost':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                }
                model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', random_state=cfg.RANDOM_SEED)
            else:
                # random_forest
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20)
                }
                model = RandomForestClassifier(**params, random_state=cfg.RANDOM_SEED)

            # cross-val proxy
            from sklearn.model_selection import cross_val_score
            import numpy as np
            scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc')
            return float(np.mean(scores))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=cfg.OPTUNA_TRIALS)
        best_trial = study.best_trial
        best_params = best_trial.params
        # adjust types for LightGBM/XGBoost if needed
        return best_params

###############################################################################
### FILE: src/ui.py
###############################################################################
import streamlit as st
from .data import load_from_url, demo_df
from .features import advanced_feature_engineering
from .models import ModelWrapper
from .config import cfg
from .utils import logger
from joblib import Memory

# Streamlit-level caching for expensive ops (app-level)
st.set_page_config(page_title="AI Horse Racing Predictor", layout='wide')

memory = Memory(cfg.CACHE_DIR, verbose=0)

@st.cache_data(ttl=cfg.CACHE_TTL)
def fetch_table_from_url(url: str):
    return load_from_url(url)


def sidebar_controls():
    st.sidebar.header("Configuration")
    model_choice = st.sidebar.selectbox("Model", ['lightgbm', 'xgboost', 'random_forest'], index=0)
    use_optuna = st.sidebar.checkbox("Use AutoML (Optuna)", value=True)
    n_demo = st.sidebar.slider("Demo runners", 8, 16, 12)
    return model_choice, use_optuna, n_demo


def main():
    st.title("🏇 AI Horse Racing Predictor — Modular Refactor")
    st.write("Local Mode — Advanced caching + Optuna AutoML")

    url = st.text_input("Race URL (Geny or similar)")
    uploaded = st.file_uploader("Or upload CSV", type=['csv'])

    model_choice, use_optuna, n_demo = sidebar_controls()

    if st.button("Run Analysis"):
        with st.spinner("Loading data..."):
            if uploaded is not None:
                import pandas as pd
                df = pd.read_csv(uploaded)
                source = 'uploaded CSV'
            elif url:
                df = fetch_table_from_url(url)
                source = url
            else:
                df = demo_df(n_demo)
                source = 'demo'

        if df is None or df.empty:
            st.error("No data loaded — please check URL or upload a CSV")
            return

        st.success(f"Loaded {len(df)} rows from {source}")
        st.dataframe(df.head(10))

        with st.spinner("Feature engineering..."):
            X = advanced_feature_engineering(df)

        # intelligent label generation (weak supervision)
        st.info("Generating weak labels from odds median + heuristics")
        y = (df['Cote'].apply(lambda x: float(str(x).replace(',', '.'))) < float(df['Cote'].apply(lambda x: float(str(x).replace(',', '.'))).median())).astype(int)

        st.write("Features:")
        st.dataframe(X.head(10))

        # train
        model = ModelWrapper(model_choice)
        with st.spinner("Training model (this may take a while with Optuna)..."):
            metrics = model.train(X, y, use_optuna=use_optuna)

        st.success("Training finished")
        st.write(metrics)

        # predict
        probs = model.predict_proba(X)
        # probability adjustment & normalization per race
        import numpy as np
        probs = np.clip(probs, 0.02, 0.98)
        if probs.sum() > 0:
            probs = probs / probs.sum() * min(0.95, len(probs) * 0.12)

        df_out = df.copy()
        df_out['probability'] = probs
        df_out['score_final'] = probs
        df_out = df_out.sort_values('score_final', ascending=False).reset_index(drop=True)
        df_out['rank'] = df_out.index + 1

        st.subheader("Results")
        st.dataframe(df_out[['rank','Nom','Cote','Numéro de corde','probability']].head(20))

        st.subheader("Top 3 Recommendations")
        for i, row in df_out.head(3).iterrows():
            st.markdown(f"**{i+1}. {row['Nom']}** — Prob: {row['probability']*100:.1f}% | Cote: {row.get('Cote', 'N/A')}")

        # persist model option
        if st.button("Save model to disk"):
            path = st.text_input("Path to save model", value='models/model.joblib')
            model.persist(path)
            st.success(f"Model saved to {path}")

###############################################################################
# End of code package
###############################################################################
