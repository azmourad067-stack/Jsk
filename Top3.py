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

class HorseRacingPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.performance_metrics = {}
        
    def create_synthetic_labels(self, df, method="odds_based"):
        """Crée des labels synthétiques adaptés au nombre de partants"""
        labels = pd.Series(0, index=df.index, dtype=int)
        n_runners = len(df)
        
        if method == "odds_based":
            df['implied_prob'] = 1 / df['odds_numeric']
            total_prob = df['implied_prob'].sum()
            
            if total_prob > 0:
                df['normalized_prob'] = df['implied_prob'] / total_prob
                
                adjustment_factor = self._get_adjustment_factor(n_runners)
                
                np.random.seed(ML_CONFIG["random_state"])
                for idx, row in df.iterrows():
                    if ML_CONFIG["target_variable"] == "winner":
                        prob = row['normalized_prob'] * adjustment_factor
                        if np.random.random() < prob:
                            labels.loc[idx] = 1
                    else:
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
        
        # Features de base avec gestion de type robuste
        features_df['odds_reciprocal'] = 1 / features_df['odds_numeric']
        features_df['odds_log'] = np.log(features_df['odds_numeric'])
        features_df['odds_rank'] = features_df['odds_numeric'].rank().astype(float)
        
        # Features de position relative - TOUTES CONVERTIES EN INT/FLOAT
        features_df['draw_position_ratio'] = (features_df['draw_numeric'] / n_runners).astype(float)
        
        # Conversion EXPLICITE des booléens en int
        inner_draw_condition = (features_df['draw_numeric'] <= max(3, n_runners * 0.25))
        features_df['is_inner_draw'] = inner_draw_condition.astype(int)
        
        outer_draw_condition = (features_df['draw_numeric'] >= n_runners - max(2, n_runners * 0.15))
        features_df['is_outer_draw'] = outer_draw_condition.astype(int)
        
        # Features basées sur le nombre de partants - CONVERSION EXPLICITE
        features_df['runners_count'] = n_runners
        features_df['is_small_field'] = (n_runners <= 8).astype(int)
        features_df['is_large_field'] = (n_runners > 12).astype(int)
        
        # Features de poids
        if 'weight_kg' in df.columns:
            features_df['weight_kg'] = pd.to_numeric(features_df['weight_kg'], errors='coerce')
            features_df['weight_kg'] = features_df['weight_kg'].fillna(
                features_df['weight_kg'].mean() if not features_df['weight_kg'].isna().all() else 60.0
            )
            
            features_df['weight_deviation'] = (
                (features_df['weight_kg'] - features_df['weight_kg'].mean()) / 
                features_df['weight_kg'].std()
            ).astype(float)
            
            light_weight_condition = (features_df['weight_kg'] < features_df['weight_kg'].quantile(0.3))
            features_df['is_light_weight'] = light_weight_condition.astype(int)
        else:
            features_df['weight_deviation'] = 0.0
            features_df['is_light_weight'] = 0
        
        # Analyse de la "musique"
        if 'Musique' in df.columns:
            features_df['recent_perf_score'] = df['Musique'].apply(self._parse_musique_score).astype(float)
        else:
            features_df['recent_perf_score'] = 0.5
        
        # Features d'interaction
        features_df['odds_draw_interaction'] = (
            features_df['odds_reciprocal'] * (1 / features_df['draw_numeric'])
        ).astype(float)
        
        features_df['odds_runners_interaction'] = (
            features_df['odds_reciprocal'] * (1 / n_runners)
        ).astype(float)
        
        features_df['draw_runners_interaction'] = (
            features_df['draw_numeric'] / n_runners
        ).astype(float)
        
        # CONVERSION FINALE DE TOUTES LES COLONNES EN FLOAT
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0.0).astype(float)
        
        # Sélection des features finales
        exclude_cols = ['Nom', 'Cote', 'Numéro de corde', 'Poids', 'Musique', 'Âge/Sexe', 'Jockey', 'Entraîneur', 'weight_kg']
        feature_columns = [col for col in features_df.columns if col not in exclude_cols]
        
        return features_df[feature_columns], feature_columns
    
    def _parse_musique_score(self, musique):
        """Convertit la musique en score numérique"""
        if pd.isna(musique) or musique == '':
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
        if len(features) < 6:
            raise ValueError("Nombre insuffisant de partants pour l'entraînement ML")
        
        # S'assurer que les features sont numériques
        features = features.astype(float)
        labels = labels.astype(int)
            
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, 
            test_size=min(ML_CONFIG["test_size"], 0.3),
            random_state=ML_CONFIG["random_state"],
            stratify=labels if len(np.unique(labels)) > 1 else None
        )
        
        # Vérifier qu'il y a au moins 2 classes
        if len(np.unique(y_train)) < 2:
            raise ValueError("Données d'entraînement insuffisantes (une seule classe)")
        
        # Normalisation des features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_type = ML_CONFIG["model_type"]
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
            scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1]) if sum(y_train) > 0 else 1
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                random_state=ML_CONFIG["random_state"],
                max_depth=max_depth,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight
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
        
        # S'assurer que les features sont numériques
        features = features.astype(float)
        features_scaled = self.scaler.transform(features)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]
        
        if ML_CONFIG["calibration"]:
            probabilities = self._calibrate_probabilities(probabilities, len(features))
            
        return probabilities

    def _calibrate_probabilities(self, probabilities, n_runners):
        """Calibre les probabilités selon le nombre de partants"""
        probabilities = np.clip(probabilities, 0.01, 0.99)
        
        if n_runners > 1 and probabilities.sum() > 0:
            target_sum = min(n_runners * 0.15, 0.95)
            probabilities = probabilities / probabilities.sum() * target_sum
            
        return probabilities

# ==== FONCTIONS SIMPLIFIÉES POUR L'ANALYSE ====
def prepare_features_ml(df, predictor):
    """Préparation robuste des données avec gestion des types"""
    print("\n🔧 PRÉPARATION DES DONNÉES...")
    
    # Vérification des colonnes critiques
    required_columns = ['Cote', 'Numéro de corde', 'Nom']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Colonnes manquantes: {missing_columns}")
    
    # Conversions sécurisées avec gestion de type
    df['odds_numeric'] = df['Cote'].apply(safe_float_convert)
    df['draw_numeric'] = df['Numéro de corde'].apply(safe_int_convert)
    
    # Gestion robuste du poids
    if 'Poids' in df.columns:
        df['weight_kg'] = df['Poids'].apply(extract_weight_kg)
        df['weight_kg'] = df['weight_kg'].fillna(60.0)
    else:
        df['weight_kg'] = 60.0
        print("⚠️ Colonne 'Poids' non trouvée, utilisation de valeurs par défaut")
    
    # Conversion explicite des types
    df['odds_numeric'] = pd.to_numeric(df['odds_numeric'], errors='coerce').fillna(10.0)
    df['draw_numeric'] = pd.to_numeric(df['draw_numeric'], errors='coerce').fillna(1.0)
    df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce').fillna(60.0)
    
    # Nettoyage des données critiques
    initial_count = len(df)
    df_clean = df.dropna(subset=['odds_numeric', 'draw_numeric']).copy()
    final_count = len(df_clean)
    
    if final_count == 0:
        raise ValueError("Aucune donnée valide après nettoyage")
    
    print(f"✅ Données nettoyées : {initial_count} → {final_count} chevaux")
    
    # Engineering de features
    features_df, feature_names = predictor.engineer_features(df_clean)
    
    return df_clean, features_df, feature_names

def analyze_race_simple(df, race_type="AUTO"):
    """Version SIMPLIFIÉE de l'analyse pour éviter les erreurs"""
    
    n_runners = len(df)
    print(f"🏇 Analyse d'une course à {n_runners} partants")
    
    # Détection automatique simplifiée
    if race_type == "AUTO":
        race_type = "ATTELE_AUTOSTART"  # Par défaut
    
    # Configuration simple
    config = {
        "w_odds": 0.6, 
        "w_draw": 0.3, 
        "w_weight": 0.1
    }
    
    # Initialisation du prédicteur ML
    predictor = HorseRacingPredictor()
    df_clean, features_df, feature_names = prepare_features_ml(df, predictor)
    
    # Calcul des scores de base SIMPLES
    df_clean['score_odds'] = (1 / df_clean['odds_numeric']).astype(float)
    df_clean['score_draw'] = (1 / df_clean['draw_numeric']).astype(float)
    df_clean['score_weight'] = (1 / df_clean['weight_kg']).astype(float)
    
    # Score de base avec pondérations
    df_clean['score_base'] = (
        config["w_odds"] * df_clean['score_odds'] +
        config["w_draw"] * df_clean['score_draw'] +
        config["w_weight"] * df_clean['score_weight']
    ).astype(float)
    
    # Machine Learning seulement si conditions optimales
    use_ml = False  # Désactivé temporairement pour debug
    predictor = None
    
    if use_ml and n_runners >= 8:
        try:
            labels = predictor.create_synthetic_labels(df_clean)
            if sum(labels) >= 3:  # Au moins 3 positifs
                metrics = predictor.train_model(features_df, labels)
                ml_probabilities = predictor.predict_proba(features_df)
                df_clean['ml_probability'] = ml_probabilities.astype(float)
                df_clean['score_final'] = (ml_probabilities * 100).astype(float)
                print(f"✅ ML appliqué - AUC: {metrics['test_auc']:.3f}")
            else:
                use_ml = False
                df_clean['score_final'] = df_clean['score_base'].astype(float)
        except Exception as e:
            print(f"⚠️ ML échoué: {e}, utilisation méthode classique")
            use_ml = False
            df_clean['score_final'] = df_clean['score_base'].astype(float)
    else:
        df_clean['score_final'] = df_clean['score_base'].astype(float)
    
    # Classement final
    df_ranked = df_clean.sort_values('score_final', ascending=False).reset_index(drop=True)
    df_ranked['rang'] = range(1, len(df_ranked) + 1)
    
    # Probabilité normalisée
    min_score = df_ranked['score_final'].min()
    max_score = df_ranked['score_final'].max()
    if max_score > min_score:
        df_ranked['ml_probability'] = (
            (df_ranked['score_final'] - min_score) / (max_score - min_score)
        ).astype(float)
    else:
        df_ranked['ml_probability'] = (1.0 / len(df_ranked)).astype(float)
    
    return df_ranked, predictor, race_type, config

# ==== FONCTIONS UTILITAIRES CORRIGÉES ====
def safe_float_convert(value):
    """Conversion sécurisée vers float"""
    if pd.isna(value) or value is None:
        return np.nan
    try:
        cleaned = str(value).replace(',', '.').strip()
        return float(cleaned)
    except (ValueError, AttributeError, TypeError):
        return np.nan

def safe_int_convert(value):
    """Conversion sécurisée vers entier"""
    if pd.isna(value) or value is None:
        return np.nan
    try:
        cleaned = re.search(r'\d+', str(value))
        return int(cleaned.group()) if cleaned else np.nan
    except (ValueError, AttributeError, TypeError):
        return np.nan

def extract_weight_kg(poids_str):
    """Extrait le poids en kg depuis une chaîne"""
    if pd.isna(poids_str) or poids_str is None:
        return np.nan
    try:
        match = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
        if match:
            return float(match.group(1).replace(',', '.'))
        return np.nan
    except (ValueError, TypeError):
        return np.nan

def nettoyer_donnees(ligne):
    """Nettoyage des données texte"""
    if pd.isna(ligne) or ligne is None:
        return ""
    try:
        ligne = str(ligne)
        ligne = ''.join(e for e in ligne if e.isalnum() or e.isspace() or e in ['.', ',', '-', '(', ')', '%'])
        return ligne.strip()
    except:
        return ""

# ==== EXTRACTION DES DONNÉES SIMPLIFIÉE ====
def extract_race_data_simple(url):
    """Extraction simplifiée des données"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        donnees_chevaux = []
        
        # Recherche de tableaux
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')[1:]  # Skip header
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 4:
                    cheval_data = {}
                    
                    # Extraction basique
                    for i, col in enumerate(cols):
                        text = nettoyer_donnees(col.text)
                        if not text:
                            continue
                            
                        if i == 0 and re.match(r'^\d+$', text):
                            cheval_data['Numéro de corde'] = text
                        elif re.match(r'^\d+(?:[.,]\d+)?$', text) and 'Cote' not in cheval_data:
                            cheval_data['Cote'] = text
                        elif re.match(r'^\d+(?:[.,]\d+)?', text) and 'Poids' not in cheval_data:
                            cheval_data['Poids'] = text
                        elif len(text) > 2 and len(text) < 30 and 'Nom' not in cheval_data:
                            cheval_data['Nom'] = text
                    
                    # Validation et valeurs par défaut
                    if 'Nom' in cheval_data and 'Cote' in cheval_data and 'Numéro de corde' in cheval_data:
                        if 'Poids' not in cheval_data:
                            cheval_data['Poids'] = '60.0'
                        if 'Musique' not in cheval_data:
                            cheval_data['Musique'] = '5a5a'
                            
                        donnees_chevaux.append(cheval_data)
            
            if donnees_chevaux:  # Arrêter au premier tableau valide
                break
                
        return donnees_chevaux
        
    except Exception as e:
        st.error(f"Erreur lors de l'extraction: {e}")
        return []

# ==== INTERFACE STREAMLIT SIMPLIFIÉE ====
def main():
    st.set_page_config(
        page_title="Pronostics Hippiques",
        page_icon="🏇",
        layout="wide"
    )
    
    st.title("🏇 Pronostics Hippiques - Version Simplifiée")
    st.markdown("---")
    
    # Configuration simple
    st.sidebar.header("Configuration")
    race_type = st.sidebar.selectbox("Type de course", ["AUTO", "PLAT", "ATTELE_AUTOSTART", "ATTELE_VOLTE"], index=0)
    
    # Section URL input
    col1, col2 = st.columns([2, 1])
    with col1:
        url = st.text_input("🔗 URL de la course:", placeholder="https://www.geny.com/...")
    
    # Section données de test
    with st.expander("🧪 Tester avec des données exemple"):
        n_test_runners = st.slider("Nombre de partants de test", 4, 20, 14)
        if st.button("Générer et analyser des données test"):
            test_data = generate_test_data(n_test_runners)
            df_test = pd.DataFrame(test_data)
            
            with st.spinner(f"Analyse d'une course test à {n_test_runners} partants..."):
                try:
                    df_ranked, predictor, detected_type, config = analyze_race_simple(df_test, "AUTO")
                    display_results(df_ranked, predictor, detected_type, config, n_test_runners)
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse: {str(e)}")

    if st.button("🎯 Analyser la course URL", type="primary") and url:
        with st.spinner("Extraction et analyse en cours..."):
            try:
                # Extraction des données
                donnees_chevaux = extract_race_data_simple(url)
                
                if not donnees_chevaux:
                    st.error("❌ Aucune donnée extraite de l'URL. Vérifiez l'URL ou essayez un autre site.")
                    return
                
                st.success(f"✅ {len(donnees_chevaux)} chevaux extraits")
                
                # Affichage des données brutes
                with st.expander("📋 Données brutes extraites"):
                    st.dataframe(pd.DataFrame(donnees_chevaux))
                
                # Analyse SIMPLIFIÉE
                df = pd.DataFrame(donnees_chevaux)
                df_ranked, predictor, detected_type, config = analyze_race_simple(df, race_type)
                display_results(df_ranked, predictor, detected_type, config, len(df_ranked))
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {str(e)}")

def generate_test_data(n_runners):
    """Génère des données de test réalistes"""
    noms = ['Star Runner', 'Thunder Bolt', 'Wind Dancer', 'Lightning Flash', 'Storm Chaser',
            'Quick Silver', 'Moon Shadow', 'Fire Blaze', 'Ice Queen', 'Desert King',
            'Ocean Wave', 'Mountain High', 'River Flow', 'Sky Rocket', 'Earth Shaker',
            'Sun Blaze', 'Night Fury', 'Day Dream', 'Winter Storm', 'Spring Bloom']
    
    test_data = {
        'Nom': noms[:n_runners],
        'Numéro de corde': [str(i+1) for i in range(n_runners)],
        'Cote': [f"{max(2.0, np.random.lognormal(1.5, 0.8)):.1f}" for _ in range(n_runners)],
        'Poids': [f"{np.random.uniform(54.0, 62.0):.1f}" for _ in range(n_runners)],
        'Musique': ['1a2a', '3a1a', '2a2a', '5a4a', '2a3a'] * (n_runners // 5 + 1)
    }
    return test_data

def display_results(df_ranked, predictor, race_type, config, n_runners):
    """Affiche les résultats de l'analyse"""
    # Métriques principales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏇 Partants", n_runners)
    with col2:
        st.metric("📊 Type", race_type)
    with col3:
        top1_prob = df_ranked.iloc[0]['ml_probability'] * 100
        st.metric("🥇 Probabilité 1er", f"{top1_prob:.1f}%")
    
    # Tableau des résultats
    st.subheader("📊 Classement final")
    
    # Préparation des données d'affichage
    display_data = []
    for i, row in df_ranked.iterrows():
        display_data.append({
            'Rang': int(row['rang']),
            'Cheval': row['Nom'],
            'Probabilité': f"{row['ml_probability'] * 100:.1f}%",
            'Cote': row.get('Cote', 'N/A'),
            'Corde': row.get('Numéro de corde', 'N/A')
        })
    
    display_df = pd.DataFrame(display_data)
    st.dataframe(display_df, use_container_width=True)
    
    # Recommendations
    st.subheader("🎯 Recommendations")
    display_recommendations(df_ranked, n_runners)

def display_recommendations(df_ranked, n_runners):
    """Affiche les recommandations simplifiées"""
    st.info("💡 **Suggestions de paris:**")
    
    # Top 3
    st.write("**🥇 Top 3 recommandé:**")
    for i in range(min(3, len(df_ranked))):
        cheval = df_ranked.iloc[i]
        st.write(f"{i+1}. **{cheval['Nom']}** (Probabilité: {cheval['ml_probability']*100:.1f}%)")
    
    # Suggestions selon le nombre de partants
    st.write("**📊 Stratégie recommandée:**")
    if n_runners <= 8:
        st.write("- **Trio Ordre** avec les 3 premiers")
        st.write("- **Simple** sur le favori")
    elif n_runners <= 12:
        st.write("- **Quinté+** avec base des 5 premiers")
        st.write("- **Trio** en ciblant le top 4")
    else:
        st.write("- **Quinté+** avec base élargie (6-7 chevaux)")
        st.write("- **Super4** comme alternative")

if __name__ == "__main__":
    main()
