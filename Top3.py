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
        labels = pd.Series(0, index=df.index, dtype=int)  # Spécifier le type
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
        
        # Features de position relative ADAPTATIVES
        features_df['draw_position_ratio'] = features_df['draw_numeric'] / n_runners
        features_df['is_inner_draw'] = (features_df['draw_numeric'] <= max(3, n_runners * 0.25)).astype(int)
        features_df['is_outer_draw'] = (features_df['draw_numeric'] >= n_runners - max(2, n_runners * 0.15)).astype(int)
        
        # Nouvelles features basées sur le nombre de partants
        features_df['runners_count'] = n_runners
        features_df['is_small_field'] = (n_runners <= 8).astype(int)
        features_df['is_large_field'] = (n_runners > 12).astype(int)
        
        # Features de poids (avec gestion des valeurs manquantes)
        if 'weight_kg' in df.columns:
            # S'assurer que weight_kg est numérique
            features_df['weight_kg'] = pd.to_numeric(features_df['weight_kg'], errors='coerce')
            features_df['weight_kg'] = features_df['weight_kg'].fillna(features_df['weight_kg'].mean() if not features_df['weight_kg'].isna().all() else 60.0)
            
            features_df['weight_deviation'] = (features_df['weight_kg'] - features_df['weight_kg'].mean()) / features_df['weight_kg'].std()
            features_df['is_light_weight'] = (features_df['weight_kg'] < features_df['weight_kg'].quantile(0.3)).astype(int)
        else:
            # Valeurs par défaut si poids non disponible
            features_df['weight_deviation'] = 0.0
            features_df['is_light_weight'] = 0
        
        # Analyse de la "musique"
        if 'Musique' in df.columns:
            features_df['recent_perf_score'] = df['Musique'].apply(self._parse_musique_score)
        else:
            features_df['recent_perf_score'] = 0.5
        
        # Features d'interaction avec nombre de partants
        features_df['odds_draw_interaction'] = features_df['odds_reciprocal'] * (1 / features_df['draw_numeric'])
        features_df['odds_runners_interaction'] = features_df['odds_reciprocal'] * (1 / n_runners)
        features_df['draw_runners_interaction'] = features_df['draw_numeric'] / n_runners
        
        # Conversion de toutes les colonnes en float pour éviter les problèmes de type
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            features_df[col] = features_df[col].fillna(0.0)
        
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

# ==== FONCTIONS AMÉLIORÉES POUR L'EXTRACTION ET PRÉPARATION ====
def extract_race_data(url):
    """Fonction robuste d'extraction des données hippiques"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        donnees_chevaux = []
        
        # Méthode 1: Recherche de tableaux standards
        tables = soup.find_all('table')
        found_data = False
        
        for table in tables:
            rows = table.find_all('tr')[1:]  # Skip header
            if len(rows) > 3:  # Tableau avec au moins 3 chevaux
                found_data = True
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 6:
                        cheval_data = extract_horse_data_from_row(cols)
                        if cheval_data:
                            donnees_chevaux.append(cheval_data)
                break  # On prend le premier tableau valide
        
        # Méthode 2: Si pas de tableau trouvé, recherche de divs avec classes spécifiques
        if not found_data:
            donnees_chevaux = extract_from_divs(soup)
        
        # Méthode 3: Recherche de données JSON dans le HTML
        if not donnees_chevaux:
            donnees_chevaux = extract_from_json(soup)
            
        return donnees_chevaux
        
    except Exception as e:
        st.error(f"Erreur lors de l'extraction: {e}")
        return []

def extract_horse_data_from_row(cols):
    """Extrait les données d'un cheval depuis une ligne de tableau"""
    try:
        cheval_data = {}
        
        for i, col in enumerate(cols):
            text = nettoyer_donnees(col.text)
            if not text:
                continue
                
            # Identification du contenu par patterns
            if i == 0 and re.match(r'^\d+$', text):
                cheval_data['Numéro de corde'] = text
            elif re.match(r'^\d+(?:[.,]\d+)?$', text) and 'Cote' not in cheval_data:
                cheval_data['Cote'] = text
            elif re.match(r'^\d+(?:[.,]\d+)?\s*(kg|KG)?$', text) and 'Poids' not in cheval_data:
                cheval_data['Poids'] = text
            elif len(text) > 2 and len(text) < 30 and 'Nom' not in cheval_data:
                # Probablement le nom du cheval
                cheval_data['Nom'] = text
            elif re.match(r'^[0-9a-zA-Z]+$', text) and 'Musique' not in cheval_data:
                cheval_data['Musique'] = text
        
        # Validation des données minimales
        if 'Nom' in cheval_data and 'Cote' in cheval_data and 'Numéro de corde' in cheval_data:
            # Valeurs par défaut pour les champs manquants
            if 'Poids' not in cheval_data:
                cheval_data['Poids'] = '60.0'
            if 'Musique' not in cheval_data:
                cheval_data['Musique'] = '5a5a'
                
            return cheval_data
            
        return None
        
    except Exception as e:
        print(f"Erreur extraction ligne: {e}")
        return None

def extract_from_divs(soup):
    """Extraction depuis des divs (fallback)"""
    donnees_chevaux = []
    # Recherche de divs contenant des données de chevaux
    horse_divs = soup.find_all('div', class_=re.compile(r'horse|cheval|runner', re.I))
    
    for div in horse_divs:
        try:
            cheval_data = {}
            text_content = div.get_text(strip=True)
            
            # Extraction basique
            numero_match = re.search(r'(\d+)[\s-]', text_content)
            if numero_match:
                cheval_data['Numéro de corde'] = numero_match.group(1)
            
            nom_match = re.search(r'\d+\s+([A-Za-z\s]+?)(?=\s+\d|$)', text_content)
            if nom_match:
                cheval_data['Nom'] = nom_match.group(1).strip()
            
            cote_match = re.search(r'(\d+[.,]\d+)', text_content)
            if cote_match:
                cheval_data['Cote'] = cote_match.group(1)
            
            if 'Nom' in cheval_data and 'Cote' in cheval_data:
                cheval_data['Poids'] = '60.0'
                cheval_data['Musique'] = '5a5a'
                donnees_chevaux.append(cheval_data)
                
        except Exception:
            continue
            
    return donnees_chevaux

def extract_from_json(soup):
    """Extraction depuis des données JSON dans le HTML"""
    donnees_chevaux = []
    script_tags = soup.find_all('script', type='application/json')
    
    for script in script_tags:
        try:
            data = json.loads(script.string)
            # Parcours récursif pour trouver les données de chevaux
            horses = find_horses_in_json(data)
            if horses:
                donnees_chevaux.extend(horses)
                break
        except:
            continue
            
    return donnees_chevaux

def find_horses_in_json(data):
    """Recherche récursive des données chevaux dans JSON"""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict) and any('horse' in k.lower() for k in value[0].keys()):
                    return parse_horses_from_json(value)
            result = find_horses_in_json(value)
            if result:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_horses_in_json(item)
            if result:
                return result
    return None

def parse_horses_from_json(horses_data):
    """Parse les données chevaux depuis JSON"""
    donnees_chevaux = []
    for horse in horses_data:
        try:
            cheval_data = {}
            if 'number' in horse:
                cheval_data['Numéro de corde'] = str(horse['number'])
            if 'name' in horse:
                cheval_data['Nom'] = horse['name']
            if 'odds' in horse:
                cheval_data['Cote'] = str(horse['odds'])
            if 'weight' in horse:
                cheval_data['Poids'] = str(horse['weight'])
            
            if 'Nom' in cheval_data and 'Cote' in cheval_data:
                if 'Poids' not in cheval_data:
                    cheval_data['Poids'] = '60.0'
                cheval_data['Musique'] = '5a5a'
                donnees_chevaux.append(cheval_data)
                
        except Exception:
            continue
            
    return donnees_chevaux

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
        # Remplissage des valeurs manquantes
        if df['weight_kg'].isna().any():
            weight_mean = df['weight_kg'].mean()
            if pd.isna(weight_mean):
                weight_mean = 60.0
            df['weight_kg'] = df['weight_kg'].fillna(weight_mean)
    else:
        df['weight_kg'] = 60.0
        print("⚠️ Colonne 'Poids' non trouvée, utilisation de valeurs par défaut")
    
    # Conversion explicite des types
    df['odds_numeric'] = pd.to_numeric(df['odds_numeric'], errors='coerce')
    df['draw_numeric'] = pd.to_numeric(df['draw_numeric'], errors='coerce')
    df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
    
    # Nettoyage des données critiques
    initial_count = len(df)
    df = df.dropna(subset=['odds_numeric', 'draw_numeric'])
    final_count = len(df)
    
    if final_count == 0:
        raise ValueError("Aucune donnée valide après nettoyage")
    
    print(f"✅ Données nettoyées : {initial_count} → {final_count} chevaux")
    
    # Engineering de features
    features_df, feature_names = predictor.engineer_features(df)
    
    return df, features_df, feature_names

def analyze_race_adaptive(df, race_type="AUTO"):
    """Analyse adaptée à n'importe quel nombre de partants"""
    
    n_runners = len(df)
    print(f"🏇 Analyse d'une course à {n_runners} partants")
    
    # Détection automatique si nécessaire
    if race_type == "AUTO":
        race_type = auto_detect_race_type(df)
    
    # Configuration de base
    config = {
        "PLAT": {
            "w_odds": 0.5, "w_draw": 0.3, "w_weight": 0.2,
            "draw_adv_inner_is_better": True,
            "use_weight_analysis": True,
        },
        "ATTELE_AUTOSTART": {
            "w_odds": 0.7, "w_draw": 0.25, "w_weight": 0.05,
            "draw_adv_inner_is_better": False,
            "use_weight_analysis": False,
        },
        "ATTELE_VOLTE": {
            "w_odds": 0.85, "w_draw": 0.05, "w_weight": 0.1,
            "draw_adv_inner_is_better": False,
            "use_weight_analysis": False,
        }
    }.get(race_type, {
        "w_odds": 0.6, "w_draw": 0.3, "w_weight": 0.1,
        "draw_adv_inner_is_better": True,
        "use_weight_analysis": True,
    })
    
    # Ajustement dynamique selon le nombre de partants
    if n_runners < 8:
        config["w_odds"] = min(0.8, config["w_odds"] + 0.2)
        config["w_draw"] = max(0.1, config["w_draw"] - 0.1)
    elif n_runners > 16:
        config["w_draw"] = min(0.4, config["w_draw"] + 0.1)
        config["w_odds"] = max(0.4, config["w_odds"] - 0.1)
    
    # Initialisation du prédicteur ML
    predictor = HorseRacingPredictor()
    df_clean, features_df, feature_names = prepare_features_ml(df, predictor)
    
    # Calcul des scores de base avec gestion de type robuste
    df_clean['score_odds'] = 1 / df_clean['odds_numeric']
    df_clean['score_draw'] = 1 / df_clean['draw_numeric']
    
    if config.get("use_weight_analysis", True):
        df_clean['score_weight'] = 1 / df_clean['weight_kg']
    else:
        df_clean['score_weight'] = 0.0
    
    # Conversion en float pour éviter les problèmes
    df_clean['score_odds'] = pd.to_numeric(df_clean['score_odds'], errors='coerce').fillna(0)
    df_clean['score_draw'] = pd.to_numeric(df_clean['score_draw'], errors='coerce').fillna(0)
    df_clean['score_weight'] = pd.to_numeric(df_clean['score_weight'], errors='coerce').fillna(0)
    
    # Score de base avec pondérations
    df_clean['score_base'] = (
        config["w_odds"] * df_clean['score_odds'] +
        config["w_draw"] * df_clean['score_draw'] +
        config["w_weight"] * df_clean['score_weight']
    )
    
    # Machine Learning si suffisamment de données
    use_ml = n_runners >= 6
    
    if use_ml:
        try:
            labels = predictor.create_synthetic_labels(df_clean)
            if sum(labels) >= 2:
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
    
    # Probabilité normalisée
    if not use_ml:
        min_score = df_ranked['score_final'].min()
        max_score = df_ranked['score_final'].max()
        if max_score > min_score:
            df_ranked['ml_probability'] = (df_ranked['score_final'] - min_score) / (max_score - min_score)
        else:
            df_ranked['ml_probability'] = 1.0 / len(df_ranked)
    
    return df_ranked, predictor if use_ml else None, race_type, config

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
        # Plusieurs formats possibles
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

def auto_detect_race_type(df):
    """Détection automatique simplifiée"""
    if 'weight_kg' in df.columns:
        weight_variation = df['weight_kg'].std() if len(df) > 1 else 0
        if weight_variation > 2.5:
            return "PLAT"
    return "ATTELE_AUTOSTART"

# ==== INTERFACE STREAMLIT CORRIGÉE ====
def main():
    st.set_page_config(
        page_title="Pronostics Hippiques ML",
        page_icon="🏇",
        layout="wide"
    )
    
    st.title("🏇 Pronostics Hippiques avec Machine Learning")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("Configuration ML")
    ml_model = st.sidebar.selectbox("Modèle", ["xgboost", "logistic", "random_forest", "gradient_boosting"], index=0)
    target_var = st.sidebar.selectbox("Variable cible", ["top3", "winner"], index=0)
    
    ML_CONFIG.update({"model_type": ml_model, "target_variable": target_var})
    
    # Section URL input
    col1, col2 = st.columns([2, 1])
    with col1:
        url = st.text_input("🔗 URL de la course:", placeholder="https://www.geny.com/...")
    with col2:
        race_type = st.selectbox("Type de course", ["AUTO", "PLAT", "ATTELE_AUTOSTART", "ATTELE_VOLTE"], index=0)
    
    # Section données manuelles pour tests
    with st.expander("🧪 Saisie manuelle de données de test"):
        n_test_runners = st.slider("Nombre de partants de test", 4, 20, 14)
        if st.button("Générer des données de test"):
            test_data = generate_test_data(n_test_runners)
            df_test = pd.DataFrame(test_data)
            
            with st.spinner(f"Analyse d'une course test à {n_test_runners} partants..."):
                try:
                    df_ranked, predictor, detected_type, config = analyze_race_adaptive(df_test, "AUTO")
                    display_results(df_ranked, predictor, detected_type, config, n_test_runners)
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse: {e}")

    if st.button("🎯 Analyser la course URL", type="primary") and url:
        with st.spinner("Extraction et analyse en cours..."):
            try:
                # Extraction des données
                donnees_chevaux = extract_race_data(url)
                
                if not donnees_chevaux:
                    st.error("❌ Aucune donnée extraite de l'URL. Vérifiez l'URL ou essayez un autre site.")
                    return
                
                st.success(f"✅ {len(donnees_chevaux)} chevaux extraits")
                
                # Affichage des données brutes pour debug
                with st.expander("📋 Données brutes extraites"):
                    st.dataframe(pd.DataFrame(donnees_chevaux))
                
                # Analyse
                df = pd.DataFrame(donnees_chevaux)
                df_ranked, predictor, detected_type, config = analyze_race_adaptive(df, race_type)
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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏇 Partants", n_runners)
    with col2:
        st.metric("📊 Type", race_type)
    with col3:
        ml_used = "✅ ML" if predictor else "⚠️ Classique"
        st.metric("🤖 Méthode", ml_used)
    with col4:
        top1_prob = df_ranked.iloc[0]['ml_probability'] * 100
        st.metric("🥇 Probabilité 1er", f"{top1_prob:.1f}%")
    
    # Tableau des résultats
    st.subheader("📊 Classement final")
    
    # S'assurer que les colonnes existent
    display_cols = ['rang', 'Nom', 'ml_probability']
    if 'Cote' in df_ranked.columns:
        display_cols.append('Cote')
    if 'Numéro de corde' in df_ranked.columns:
        display_cols.append('Numéro de corde')
    
    display_df = df_ranked[display_cols].copy()
    display_df['Probabilité'] = (display_df['ml_probability'] * 100).round(1).astype(str) + '%'
    
    # Renommage des colonnes
    rename_dict = {'rang': 'Rang', 'Nom': 'Cheval'}
    if 'Cote' in display_df.columns:
        rename_dict['Cote'] = 'Cote'
    if 'Numéro de corde' in display_df.columns:
        rename_dict['Numéro de corde'] = 'Corde'
    
    display_df = display_df.rename(columns=rename_dict)
    
    # Colonnes à afficher
    show_cols = ['Rang', 'Cheval', 'Probabilité']
    if 'Cote' in display_df.columns:
        show_cols.append('Cote')
    if 'Corde' in display_df.columns:
        show_cols.append('Corde')
    
    st.dataframe(display_df[show_cols], use_container_width=True)
    
    # Recommendations paris
    st.subheader("🎯 Recommendations pour Paris")
    display_betting_recommendations(df_ranked, n_runners, predictor)

def display_betting_recommendations(df_ranked, n_runners, predictor):
    """Affiche les recommandations de paris"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top1 = df_ranked.iloc[0]
        st.metric("🥇 Premier favori", top1['Nom'], f"{top1['ml_probability']*100:.1f}%")
    
    with col2:
        # Meilleure valeur
        if 'odds_numeric' in df_ranked.columns:
            value_picks = df_ranked[
                (df_ranked['ml_probability'] > df_ranked['ml_probability'].quantile(0.6)) &
                (df_ranked['odds_numeric'] > df_ranked['odds_numeric'].median())
            ]
            if len(value_picks) > 0:
                best_value = value_picks.iloc[0]
                st.metric("💎 Meilleure valeur", best_value['Nom'], f"Cote: {best_value['odds_numeric']:.1f}")
            else:
                st.metric("💎 Meilleure valeur", "Non trouvée", "")
        else:
            st.metric("💎 Meilleure valeur", "Données manquantes", "")
    
    with col3:
        if predictor and hasattr(predictor, 'performance_metrics'):
            auc = predictor.performance_metrics.get('test_auc', 0.5)
            confiance = "Élevée" if auc > 0.7 else "Moyenne"
            st.metric("📈 Confiance modèle", confiance, f"AUC: {auc:.3f}")
        else:
            st.metric("📈 Méthode", "Classique", "Sans ML")
    
    # Suggestions adaptées
    st.info(f"💡 **Suggestions pour {n_runners} partants:**")
    if n_runners <= 8:
        st.write("- **Trio Ordre**: Privilégiez les 3 premiers du classement")
    elif n_runners <= 12:
        st.write("- **Quinté+**: Ciblez les 5-6 premiers avec une base solide")
    else:
        st.write("- **Quinté+**: Élargissez à 7-8 chevaux pour la base")

if __name__ == "__main__":
    main()
