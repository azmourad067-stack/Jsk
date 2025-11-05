import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import json
from datetime import datetime
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss, precision_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==== CONFIGURATIONS MACHINE LEARNING ====
ML_CONFIG = {
    "model_type": "xgboost",
    "target_variable": "top3",
    "test_size": 0.2,
    "cross_validation": 5,
    "random_state": 42,
    "feature_importance_threshold": 0.01,
    "calibration": True
}

# ==== CONFIGURATIONS ADAPTATIVES AMÉLIORÉES ====
CONFIGS = {
    "PLAT": {
        "w_odds": 0.5,
        "w_draw": 0.3, 
        "w_weight": 0.2,
        "normalization": "zscore",
        "draw_adv_inner_is_better": True,
        "per_kg_penalty": 1.2,
        "weight_baseline": 55.0,
        "use_weight_analysis": True,
        "description": "Course de galop - Handicap poids + avantage corde intérieure"
    },
    
    "ATTELE_AUTOSTART": {
        "w_odds": 0.7,
        "w_draw": 0.25,
        "w_weight": 0.05,
        "normalization": "zscore", 
        "draw_adv_inner_is_better": False,
        "per_kg_penalty": 0.3,
        "weight_baseline": 68.0,
        "use_weight_analysis": False,
        "description": "Trot attelé autostart - Position optimale selon nombre partants"
    },
    
    "ATTELE_VOLTE": {
        "w_odds": 0.85,
        "w_draw": 0.05,
        "w_weight": 0.1,
        "normalization": "zscore",
        "draw_adv_inner_is_better": False,
        "per_kg_penalty": 0.2,
        "weight_baseline": 68.0,
        "use_weight_analysis": False,
        "description": "Trot attelé volté - Numéro sans importance"
    }
}

class HorseRacingPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.performance_metrics = {}
        
    def create_synthetic_labels(self, df, method="odds_based"):
        """Crée des labels synthétiques adaptés au nombre de partants"""
        labels = pd.Series(0, index=df.index)
        n_runners = len(df)
        
        if method == "odds_based":
            # Ajustement dynamique basé sur le nombre de partants
            df['implied_prob'] = 1 / df['odds_numeric']
            total_prob = df['implied_prob'].sum()
            
            if total_prob > 0:
                df['normalized_prob'] = df['implied_prob'] / total_prob
                
                # Facteur d'ajustement selon le nombre de partants
                adjustment_factor = self._get_adjustment_factor(n_runners)
                
                np.random.seed(ML_CONFIG["random_state"])
                for idx, row in df.iterrows():
                    if ML_CONFIG["target_variable"] == "winner":
                        prob = row['normalized_prob'] * adjustment_factor
                        if np.random.random() < prob:
                            labels.loc[idx] = 1
                    else:  # top3
                        # Plus de places dans les courses avec plus de partants
                        top3_prob = min(row['normalized_prob'] * (2.5 + (n_runners-8)*0.1), 0.95)
                        if np.random.random() < top3_prob:
                            labels.loc[idx] = 1
        return labels

    def _get_adjustment_factor(self, n_runners):
        """Facteur d'ajustement selon le nombre de partants"""
        if n_runners <= 8:
            return 0.9
        elif n_runners <= 12:
            return 0.8
        elif n_runners <= 16:
            return 0.7
        else:
            return 0.6

    def engineer_features(self, df):
        """Crée des features adaptées au nombre de partants"""
        features_df = df.copy()
        n_runners = len(df)
        
        # Features de base
        features_df['odds_reciprocal'] = 1 / features_df['odds_numeric']
        features_df['odds_log'] = np.log(features_df['odds_numeric'])
        features_df['odds_rank'] = features_df['odds_numeric'].rank()
        
        # Features de position relative ADAPTATIVES
        features_df['draw_position_ratio'] = features_df['draw_numeric'] / n_runners
        features_df['is_inner_draw'] = (features_df['draw_numeric'] <= max(3, n_runners * 0.25)).astype(int)
        features_df['is_outer_draw'] = (features_df['draw_numeric'] >= n_runners - max(2, n_runners * 0.15)).astype(int)
        features_df['draw_quartile'] = pd.qcut(features_df['draw_numeric'], 4, labels=False)
        
        # Nouvelles features basées sur le nombre de partants
        features_df['runners_count'] = n_runners
        features_df['is_small_field'] = (n_runners <= 8).astype(int)
        features_df['is_large_field'] = (n_runners > 12).astype(int)
        
        # Features de poids
        features_df['weight_deviation'] = (features_df['weight_kg'] - features_df['weight_kg'].mean()) / features_df['weight_kg'].std()
        features_df['is_light_weight'] = (features_df['weight_kg'] < features_df['weight_kg'].quantile(0.3)).astype(int)
        
        # Analyse de la "musique"
        if 'Musique' in df.columns:
            features_df['recent_perf_score'] = df['Musique'].apply(self._parse_musique_score)
        
        # Features d'interaction avec nombre de partants
        features_df['odds_draw_interaction'] = features_df['odds_reciprocal'] * (1 / features_df['draw_numeric'])
        features_df['odds_runners_interaction'] = features_df['odds_reciprocal'] * (1 / n_runners)
        features_df['draw_runners_interaction'] = features_df['draw_numeric'] / n_runners
        
        # Sélection des features finales
        feature_columns = [col for col in features_df.columns if col not in ['Nom', 'Cote', 'Numéro de corde', 'Poids', 'Musique', 'Âge/Sexe', 'Jockey', 'Entraîneur']]
        
        return features_df[feature_columns], feature_columns
    
    def _parse_musique_score(self, musique):
        """Convertit la musique en score numérique"""
        if pd.isna(musique):
            return 0.5
        try:
            positions = []
            for char in str(musique):
                if char.isdigit():
                    positions.append(int(char))
            if positions:
                avg_position = np.mean(positions)
                return 1 / avg_position
            return 0.5
        except:
            return 0.5

    def train_model(self, features, labels):
        """Entraîne le modèle avec validation adaptative"""
        if len(features) < 8:  # Minimum 8 chevaux pour un entraînement valide
            raise ValueError("Nombre insuffisant de partants pour l'entraînement ML")
            
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, 
            test_size=min(ML_CONFIG["test_size"], 0.3),  # Ajustement pour petits échantillons
            random_state=ML_CONFIG["random_state"],
            stratify=labels if len(np.unique(labels)) > 1 else None
        )
        
        # Normalisation des features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_type = ML_CONFIG["model_type"]
        
        # Ajustement des hyperparamètres selon la taille de l'échantillon
        n_estimators = min(100, len(X_train) // 2)
        max_depth = min(6, max(3, len(X_train) // 4))
        
        if model_type == "logistic":
            self.model = LogisticRegression(
                random_state=ML_CONFIG["random_state"],
                class_weight='balanced',
                max_iter=1000
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=ML_CONFIG["random_state"],
                class_weight='balanced',
                max_depth=max_depth
            )
        elif model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                random_state=ML_CONFIG["random_state"],
                max_depth=max_depth,
                learning_rate=0.1,
                scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]) if sum(y_train) > 0 else 1
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                random_state=ML_CONFIG["random_state"],
                max_depth=max_depth
            )
        
        # Entraînement du modèle
        self.model.fit(X_train_scaled, y_train)
        
        # Évaluation
        train_probs = self.model.predict_proba(X_train_scaled)[:, 1]
        test_probs = self.model.predict_proba(X_test_scaled)[:, 1]
        
        self.performance_metrics = {
            'train_auc': roc_auc_score(y_train, train_probs),
            'test_auc': roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.5,
            'train_log_loss': log_loss(y_train, train_probs),
            'test_log_loss': log_loss(y_test, test_probs) if len(np.unique(y_test)) > 1 else float('inf'),
            'feature_importance': self._get_feature_importance(features.columns),
            'n_runners': len(features)
        }
        
        return self.performance_metrics

    def _get_feature_importance(self, feature_names):
        """Extrait l'importance des features"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return {}
            
        feature_importance = dict(zip(feature_names, importances))
        return {k: v for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)}

    def predict_proba(self, features):
        """Prédit les probabilités avec calibration adaptative"""
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        features_scaled = self.scaler.transform(features)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]
        
        # Calibration adaptative selon le nombre de partants
        if ML_CONFIG["calibration"]:
            probabilities = self._calibrate_probabilities(probabilities, len(features))
            
        return probabilities

    def _calibrate_probabilities(self, probabilities, n_runners):
        """Calibre les probabilités selon le nombre de partants"""
        # Évite les probabilités extrêmes
        probabilities = np.clip(probabilities, 0.01, 0.99)
        
        # Ajustement dynamique selon le nombre de partants
        if n_runners > 1:
            target_sum = min(n_runners * 0.15, 0.95)  # Moins de concentration dans les grosses courses
            probabilities = probabilities / probabilities.sum() * target_sum
            
        return probabilities

# ==== FONCTIONS ADAPTÉES POUR TOUS NOMBRES DE PARTANTS ====
def compute_draw_score_plat(draw_series, n_runners, config):
    """Score numéro de corde adaptatif pour PLAT"""
    if n_runners <= 1:
        return pd.Series([0.0] * len(draw_series), index=draw_series.index)
    
    if config["draw_adv_inner_is_better"]:
        # Avantage cordes intérieures - ajusté selon le nombre de partants
        optimal_draws = max(1, min(4, n_runners // 4))
        scores = []
        for draw in draw_series:
            if draw <= optimal_draws:
                score = 2.0  # Bonus cordes très favorables
            elif draw <= n_runners // 2:
                score = 0.5  # Légèrement favorable
            else:
                score = -1.0  # Pénalité cordes extérieures
            scores.append(score)
        return pd.Series(scores, index=draw_series.index)
    else:
        return normalize_series(draw_series, config["normalization"])

def compute_draw_score_attele(draw_series, n_runners, config):
    """Score numéro spécialisé adaptatif pour l'attelé"""
    if n_runners <= 1:
        return pd.Series([0.0] * len(draw_series), index=draw_series.index)
    
    # Calcul dynamique des positions optimales
    if n_runners <= 9:
        # Petites courses : positions centrales favorisées
        optimal_range = list(range(max(2, n_runners//2 - 1), min(n_runners, n_runners//2 + 2)))
        penalty_threshold = n_runners
    else:
        # Grandes courses : positions 4-6 optimisées + gestion 2ème ligne
        optimal_range = [4, 5, 6]
        penalty_threshold = 9
    
    scores = []
    for draw in draw_series:
        if draw in optimal_range:
            score = 2.0  # Bonus maximal
        elif draw <= penalty_threshold:
            score = 0.0  # Neutre
        else:
            # Pénalité progressive pour 2ème ligne
            penalty = -1.0 - (draw - penalty_threshold) * 0.2
            score = max(penalty, -3.0)  # Limite la pénalité
        scores.append(score)
    
    return pd.Series(scores, index=draw_series.index)

def compute_odds_score(odds_series, n_runners, config):
    """Score basé sur les cotes avec ajustement selon nombre de partants"""
    inverse_odds = 1.0 / odds_series
    
    # Ajustement : dans les grosses courses, les cotes sont plus compressées
    if n_runners > 12:
        inverse_odds = inverse_odds ** 0.9  # Légère compression
    
    return normalize_series(inverse_odds, config["normalization"])

def compute_weight_score(weight_series, n_runners, config):
    """Score basé sur le poids avec ajustements"""
    if not config.get("use_weight_analysis", True):
        return pd.Series([0.0] * len(weight_series), index=weight_series.index)
    
    weight_penalty = (weight_series - config["weight_baseline"]) * config["per_kg_penalty"]
    
    # Ajustement : pénalité moins forte dans les grosses courses
    if n_runners > 12:
        weight_penalty = weight_penalty * 0.8
    
    return normalize_series(-weight_penalty, config["normalization"])

def analyze_race_adaptive(df, race_type="AUTO"):
    """Analyse adaptée à n'importe quel nombre de partants"""
    
    n_runners = len(df)
    print(f"🏇 Analyse d'une course à {n_runners} partants")
    
    # Détection automatique si nécessaire
    if race_type == "AUTO":
        race_type = auto_detect_race_type(df)
    
    config = CONFIGS[race_type].copy()
    
    # Ajustement dynamique des pondérations selon le nombre de partants
    if n_runners < 8:
        # Petites courses : plus de poids aux cotes
        config["w_odds"] = min(0.8, config["w_odds"] + 0.2)
        config["w_draw"] = max(0.1, config["w_draw"] - 0.1)
    elif n_runners > 16:
        # Grosses courses : plus de poids à la position
        config["w_draw"] = min(0.4, config["w_draw"] + 0.1)
        config["w_odds"] = max(0.4, config["w_odds"] - 0.1)
    
    print(f"⚙️ Configuration adaptée à {n_runners} partants")
    print(f"📊 Pondérations : Cotes {config['w_odds']:.0%} | Corde {config['w_draw']:.0%} | Poids {config['w_weight']:.0%}")
    
    # Initialisation du prédicteur ML
    predictor = HorseRacingPredictor()
    df_clean, features_df, feature_names = prepare_features_ml(df, predictor)
    
    # Méthode classique améliorée
    df_clean['score_odds'] = compute_odds_score(df_clean['odds_numeric'], n_runners, config)
    
    if race_type == "PLAT":
        df_clean['score_draw'] = compute_draw_score_plat(df_clean['draw_numeric'], n_runners, config)
    else:
        df_clean['score_draw'] = compute_draw_score_attele(df_clean['draw_numeric'], n_runners, config)
    
    df_clean['score_weight'] = compute_weight_score(df_clean['weight_kg'], n_runners, config)
    
    # Score de base avec pondérations adaptatives
    df_clean['score_base'] = (
        config["w_odds"] * df_clean['score_odds'] +
        config["w_draw"] * df_clean['score_draw'] +
        config["w_weight"] * df_clean['score_weight']
    )
    
    # Machine Learning si suffisamment de données
    use_ml = n_runners >= 6  # Minimum 6 chevaux pour ML
    
    if use_ml:
        try:
            labels = predictor.create_synthetic_labels(df_clean)
            if sum(labels) >= 2:  # Au moins 2 positifs
                metrics = predictor.train_model(features_df, labels)
                ml_probabilities = predictor.predict_proba(features_df)
                df_clean['ml_probability'] = ml_probabilities
                df_clean['score_final'] = ml_probabilities * 100
                
                print(f"✅ ML appliqué - AUC: {metrics['test_auc']:.3f}")
            else:
                use_ml = False
                df_clean['score_final'] = df_clean['score_base']
        except Exception as e:
            print(f"⚠️ ML échoué: {e}, utilisation méthode classique")
            use_ml = False
            df_clean['score_final'] = df_clean['score_base']
    else:
        df_clean['score_final'] = df_clean['score_base']
        predictor = None
    
    # Classement final
    df_ranked = df_clean.sort_values('score_final', ascending=False).reset_index(drop=True)
    df_ranked['rang'] = range(1, len(df_ranked) + 1)
    
    # Ajout de la probabilité ML ou normalisation du score classique
    if not use_ml:
        df_ranked['ml_probability'] = (
            df_ranked['score_final'] - df_ranked['score_final'].min()
        ) / (df_ranked['score_final'].max() - df_ranked['score_final'].min())
    
    return df_ranked, predictor if use_ml else None, race_type, config

def prepare_features_ml(df, predictor):
    """Préparation des données avec gestion robuste"""
    print("\n🔧 PRÉPARATION DES DONNÉES...")
    
    # Conversions sécurisées
    df['odds_numeric'] = df['Cote'].apply(safe_float_convert)
    df['draw_numeric'] = df['Numéro de corde'].apply(safe_int_convert)
    df['weight_kg'] = df['Poids'].apply(extract_weight_kg)
    
    # Nettoyage
    initial_count = len(df)
    df = df.dropna(subset=['odds_numeric', 'draw_numeric'])
    df['weight_kg'] = df['weight_kg'].fillna(df['weight_kg'].mean())
    final_count = len(df)
    
    print(f"✅ Données nettoyées : {initial_count} → {final_count} chevaux")
    
    # Engineering de features
    features_df, feature_names = predictor.engineer_features(df)
    
    return df, features_df, feature_names

def normalize_series(series, mode="zscore"):
    """Normalisation robuste"""
    if len(series) <= 1 or series.std() == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    
    if mode == "zscore":
        return (series - series.mean()) / series.std()
    elif mode == "minmax":
        min_val, max_val = series.min(), series.max()
        if max_val == min_val:
            return pd.Series([0.0] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)
    else:
        return pd.Series([0.0] * len(series), index=series.index)

# ==== FONCTIONS STREAMLIT AMÉLIORÉES ====
def main():
    st.set_page_config(
        page_title="Pronostics Hippiques ML - Tous Partants",
        page_icon="🏇",
        layout="wide"
    )
    
    st.title("🏇 Pronostics Hippiques ML - Adapté à Tous Nombre de Partants")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("Configuration ML")
    
    ml_model = st.sidebar.selectbox(
        "Modèle de prédiction",
        ["xgboost", "logistic", "random_forest", "gradient_boosting"],
        index=0
    )
    
    target_var = st.sidebar.selectbox(
        "Variable cible",
        ["top3", "winner"],
        index=0
    )
    
    ML_CONFIG.update({
        "model_type": ml_model,
        "target_variable": target_var
    })
    
    # Section URL input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        url = st.text_input("🔗 URL de la course (Geny.fr):", placeholder="https://www.geny.com/...")
    
    with col2:
        race_type = st.selectbox(
            "Type de course",
            ["AUTO", "PLAT", "ATTELE_AUTOSTART", "ATTELE_VOLTE"],
            index=0
        )
    
    # Section données manuelles pour tests
    with st.expander("🧪 Saisie manuelle de données de test"):
        st.info("Utilisez cette section pour tester avec un nombre spécifique de partants")
        n_test_runners = st.slider("Nombre de partants de test", 4, 20, 12)
        
        if st.button("Générer des données de test"):
            test_data = generate_test_data(n_test_runners)
            df_test = pd.DataFrame(test_data)
            
            with st.spinner(f"Analyse d'une course test à {n_test_runners} partants..."):
                df_ranked, predictor, detected_type, config = analyze_race_adaptive(df_test, "AUTO")
                display_results(df_ranked, predictor, detected_type, config)

    if st.button("🎯 Analyser la course URL", type="primary") and url:
        with st.spinner("Analyse en cours..."):
            try:
                # Récupération des données
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                donnees_chevaux = []
                
                # Extraction des données
                table = soup.find('table')
                if table:
                    rows = table.find_all('tr')[1:]
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 8:
                            donnees_chevaux.append({
                                "Numéro de corde": nettoyer_donnees(cols[0].text),
                                "Nom": nettoyer_donnees(cols[1].text),
                                "Musique": nettoyer_donnees(cols[5].text) if len(cols) > 5 else "",
                                "Âge/Sexe": nettoyer_donnees(cols[6].text) if len(cols) > 6 else "",
                                "Poids": nettoyer_donnees(cols[7].text) if len(cols) > 7 else "60.0",
                                "Jockey": nettoyer_donnees(cols[8].text) if len(cols) > 8 else "",
                                "Entraîneur": nettoyer_donnees(cols[9].text) if len(cols) > 9 else "",
                                "Cote": nettoyer_donnees(cols[-1].text)
                            })
                
                if donnees_chevaux:
                    df = pd.DataFrame(donnees_chevaux)
                    st.success(f"✅ {len(df)} chevaux extraits")
                    
                    # Analyse avec ML adaptatif
                    df_ranked, predictor, detected_type, config = analyze_race_adaptive(df, race_type)
                    display_results(df_ranked, predictor, detected_type, config)
                else:
                    st.error("❌ Aucune donnée extraite")
                    
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

def generate_test_data(n_runners):
    """Génère des données de test avec le nombre spécifié de partants"""
    noms = ['Star Runner', 'Thunder Bolt', 'Wind Dancer', 'Lightning Flash', 'Storm Chaser',
            'Quick Silver', 'Moon Shadow', 'Fire Blaze', 'Ice Queen', 'Desert King',
            'Ocean Wave', 'Mountain High', 'River Flow', 'Sky Rocket', 'Earth Shaker',
            'Sun Blaze', 'Night Fury', 'Day Dream', 'Winter Storm', 'Spring Bloom']
    
    test_data = {
        'Nom': noms[:n_runners],
        'Numéro de corde': [str(i+1) for i in range(n_runners)],
        'Cote': [f"{np.random.uniform(2.0, 20.0):.1f}" for _ in range(n_runners)],
        'Poids': [f"{np.random.uniform(54.0, 62.0):.1f}" for _ in range(n_runners)],
        'Musique': ['1a2a', '3a1a', '2a2a', '5a4a', '2a3a'] * (n_runners // 5 + 1)
    }
    return test_data

def display_results(df_ranked, predictor, race_type, config):
    """Affiche les résultats de l'analyse"""
    n_runners = len(df_ranked)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏇 Nombre de partants", n_runners)
    with col2:
        st.metric("📊 Type détecté", race_type)
    with col3:
        ml_used = "✅" if predictor else "⚠️"
        st.metric("🤖 ML utilisé", ml_used)
    with col4:
        top1_prob = df_ranked.iloc[0]['ml_probability'] * 100
        st.metric("🥇 Probabilité 1er", f"{top1_prob:.1f}%")
    
    # Tableau des résultats
    st.subheader("📊 Classement final")
    
    display_cols = ['rang', 'Nom', 'ml_probability', 'Cote', 'Numéro de corde', 'Poids']
    display_df = df_ranked[display_cols].copy()
    display_df['Probabilité'] = (display_df['ml_probability'] * 100).round(1).astype(str) + '%'
    display_df = display_df.rename(columns={
        'rang': 'Rang', 'Nom': 'Cheval', 'Cote': 'Cote',
        'Numéro de corde': 'Corde', 'Poids': 'Poids'
    })
    
    st.dataframe(
        display_df[['Rang', 'Cheval', 'Probabilité', 'Cote', 'Corde', 'Poids']],
        use_container_width=True
    )
    
    # Rapport détaillé
    st.subheader("📈 Analyse détaillée")
    report = generate_ml_report(df_ranked, predictor, race_type, n_runners)
    st.text_area("Rapport d'analyse", report, height=300)
    
    # Recommendations paris adaptées au nombre de partants
    st.subheader("🎯 Recommendations pour Paris")
    display_betting_recommendations(df_ranked, n_runners)

def display_betting_recommendations(df_ranked, n_runners):
    """Affiche les recommandations de paris adaptées"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top1 = df_ranked.iloc[0]
        st.metric("🥇 Premier favori", 
                 f"{top1['Nom']}", 
                 f"{top1['ml_probability']*100:.1f}%")
    
    with col2:
        # Meilleure valeur (bonne proba + cote élevée)
        value_picks = df_ranked[
            (df_ranked['ml_probability'] > df_ranked['ml_probability'].quantile(0.6)) &
            (df_ranked['odds_numeric'] > df_ranked['odds_numeric'].median())
        ]
        if len(value_picks) > 0:
            best_value = value_picks.iloc[0]
            st.metric("💎 Meilleure valeur", 
                     f"{best_value['Nom']}", 
                     f"Cote: {best_value['odds_numeric']:.1f}")
        else:
            st.metric("💎 Meilleure valeur", "Non trouvée", "")
    
    with col3:
        if predictor:
            confiance = "Élevée" if predictor.performance_metrics['test_auc'] > 0.7 else "Moyenne"
            st.metric("📈 Confiance modèle", 
                     confiance, 
                     f"AUC: {predictor.performance_metrics['test_auc']:.3f}")
        else:
            st.metric("📈 Méthode", "Classique", "ML non applicable")
    
    # Suggestions selon le nombre de partants
    st.info(f"💡 **Suggestions pour {n_runners} partants:**")
    if n_runners <= 8:
        st.write("- **Trio Ordre**: Privilégiez les 3 premiers du classement")
        st.write("- **Simple**: Les favoris sont souvent fiables")
    elif n_runners <= 12:
        st.write("- **Quinté+**: Ciblez les 5-6 premiers avec une base solide")
        st.write("- **Trio**: Bon rapport qualité/prix")
    else:
        st.write("- **Quinté+**: Élargissez à 7-8 chevaux pour la base")
        st.write("- **Super4**: Bonne alternative avec moins de risques")

def generate_ml_report(df_ranked, predictor, race_type, n_runners):
    """Génère un rapport adapté au nombre de partants"""
    report = []
    report.append(f"🤖 RAPPORT D'ANALYSE - {n_runners} PARTANTS")
    report.append("=" * 60)
    
    # Métriques de performance
    if predictor and hasattr(predictor, 'performance_metrics'):
        metrics = predictor.performance_metrics
        report.append(f"📈 PERFORMANCE DU MODÈLE")
        report.append(f"   • AUC: {metrics['test_auc']:.3f}")
        report.append(f"   • Log Loss: {metrics['test_log_loss']:.3f}")
        
        if metrics['feature_importance']:
            top_feature = list(metrics['feature_importance'].keys())[0]
            report.append(f"   • Feature principale: {top_feature}")
    
    report.append(f"\n🎯 STRATÉGIE {race_type} ({n_runners} partants):")
    
    if n_runners <= 8:
        report.append("   • Petite course: forte fiabilité des favoris")
        report.append("   • Cotes souvent représentatives de la valeur réelle")
    elif n_runners <= 12:
        report.append("   • Course standard: bon équilibre favoris/valeurs")
        report.append("   • Opportunités dans le top 5-6")
    else:
        report.append("   • Grande course: plus d'incertitudes")
        report.append("   • Valeurs potentielles dans le top 7-8")
    
    # Analyse du top
    top_count = min(5, n_runners)
    report.append(f"\n🔍 TOP {top_count} ANALYSE:")
    
    for i in range(top_count):
        cheval = df_ranked.iloc[i]
        prob_percent = cheval['ml_probability'] * 100
        
        analysis = []
        if cheval['odds_numeric'] < 4.0:
            analysis.append("cote très basse")
        elif cheval['odds_numeric'] < 8.0:
            analysis.append("cote intéressante")
            
        if cheval.get('draw_numeric', 10) <= max(3, n_runners * 0.25):
            analysis.append("bonne position")
            
        if prob_percent > 25:
            analysis.append("très haute confiance")
        elif prob_percent > 15:
            analysis.append("haute confiance")
            
        report.append(f"   {i+1}. {cheval['Nom']} → {prob_percent:.1f}% ({', '.join(analysis)})")
    
    return "\n".join(report)

# ==== FONCTIONS UTILITAIRES EXISTANTES ====
def safe_float_convert(value):
    if pd.isna(value):
        return np.nan
    try:
        cleaned = str(value).replace(',', '.').strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return np.nan

def safe_int_convert(value):
    if pd.isna(value):
        return np.nan
    try:
        cleaned = re.search(r'\d+', str(value))
        return int(cleaned.group()) if cleaned else np.nan
    except (ValueError, AttributeError):
        return np.nan

def extract_weight_kg(poids_str):
    if pd.isna(poids_str):
        return np.nan
    match = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
    if match:
        return float(match.group(1).replace(',', '.'))
    return np.nan

def nettoyer_donnees(ligne):
    ligne = ''.join(e for e in ligne if e.isalnum() or e.isspace() or e in ['.', ',', '-', '(', ')', '%'])
    return ligne.strip()

def auto_detect_race_type(df):
    """Détection automatique simplifiée"""
    weight_variation = df['weight_kg'].std() if len(df) > 1 else 0
    if weight_variation > 2.5:
        return "PLAT"
    else:
        return "ATTELE_AUTOSTART"

if __name__ == "__main__":
    main()
