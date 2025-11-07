import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px

# Suppression des warnings
warnings.filterwarnings('ignore')

# Imports ML avec gestion d'erreur
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import roc_auc_score, log_loss, precision_score
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_selection import SelectKBest, f_classif
    from scipy import stats
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ==== CONFIGURATION GLOBALE ====
class Config:
    VERSION = "2.0"
    APP_NAME = "Elite Racing AI"
    MIN_RUNNERS_FOR_ML = 6
    CACHE_TTL = 3600
    REQUEST_TIMEOUT = 15
    
    SUPPORTED_SITES = [
        "geny.com", "turfomania.fr", "equidia.fr", 
        "paris-turf.com", "letrot.com"
    ]

CONFIG = Config()

# ==== EXTRACTEUR DE DONNÉES SIMPLIFIÉ ====
class SimpleDataExtractor:
    """Extracteur de données simplifié et robuste"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_race_data(self, url: str) -> Optional[pd.DataFrame]:
        """Extrait les données d'une course depuis une URL"""
        try:
            response = self.session.get(url, timeout=CONFIG.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            horses_data = self._extract_horses_from_soup(soup)
            
            if horses_data:
                df = pd.DataFrame(horses_data)
                return self._clean_and_validate(df)
            
        except Exception as e:
            st.error(f"Erreur d'extraction: {e}")
        
        return None
    
    def _extract_horses_from_soup(self, soup) -> List[Dict]:
        """Extrait les chevaux depuis le HTML"""
        horses = []
        
        # Stratégie 1: Recherche de tableaux
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')[1:]  # Skip header
            for row in rows:
                cols = row.find_all(['td', 'th'])
                if len(cols) >= 3:
                    horse_data = self._parse_table_row(cols)
                    if horse_data:
                        horses.append(horse_data)
            
            if len(horses) >= 3:
                break
        
        # Stratégie 2: Recherche par classe CSS
        if not horses:
            horse_containers = soup.find_all(['div', 'article'], 
                                           class_=re.compile(r'horse|runner|participant', re.I))
            for container in horse_containers:
                horse_data = self._extract_from_container(container)
                if horse_data:
                    horses.append(horse_data)
        
        return horses
    
    def _parse_table_row(self, cols) -> Optional[Dict]:
        """Parse une ligne de tableau pour extraire les données d'un cheval"""
        horse_data = {}
        
        for i, col in enumerate(cols):
            text = col.get_text().strip()
            if not text:
                continue
            
            # Détection intelligente du contenu
            if re.match(r'^[A-Za-z\s\-\']{3,30}$', text) and 'Nom' not in horse_data:
                horse_data['Nom'] = text
            elif re.match(r'^\d+[.,]?\d*$', text):
                if 'Cote' not in horse_data:
                    horse_data['Cote'] = text.replace(',', '.')
                elif 'Numéro de corde' not in horse_data:
                    horse_data['Numéro de corde'] = text
        
        return horse_data if len(horse_data) >= 2 else None
    
    def _extract_from_container(self, container) -> Optional[Dict]:
        """Extrait depuis un container HTML"""
        horse_data = {}
        
        # Recherche du nom
        name_elem = container.find(['h2', 'h3', 'span'], class_=re.compile(r'name|title', re.I))
        if name_elem:
            horse_data['Nom'] = name_elem.get_text().strip()
        
        # Recherche de la cote
        odds_elem = container.find(['span', 'div'], class_=re.compile(r'odd|cote', re.I))
        if odds_elem:
            horse_data['Cote'] = self._extract_number(odds_elem.get_text())
        
        return horse_data if len(horse_data) >= 2 else None
    
    def _extract_number(self, text: str) -> str:
        """Extrait un nombre depuis du texte"""
        match = re.search(r'(\d+[.,]?\d*)', text)
        return match.group(1).replace(',', '.') if match else "10.0"
    
    def _clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie et valide les données"""
        if df.empty:
            return df
        
        # Colonnes requises
        if 'Nom' not in df.columns:
            return pd.DataFrame()
        
        # Nettoyage
        df = df.dropna(subset=['Nom'])
        df = df[df['Nom'].str.len() > 1]
        
        # Ajout de colonnes manquantes avec valeurs par défaut
        if 'Cote' not in df.columns:
            df['Cote'] = np.random.uniform(3, 20, len(df)).round(1)
        
        if 'Numéro de corde' not in df.columns:
            df['Numéro de corde'] = range(1, len(df) + 1)
        
        if 'Poids' not in df.columns:
            df['Poids'] = np.random.normal(58, 2.5, len(df)).round(1)
        
        if 'Musique' not in df.columns:
            df['Musique'] = [self._generate_musique() for _ in range(len(df))]
        
        if 'Âge/Sexe' not in df.columns:
            ages = np.random.choice([3,4,5,6,7], len(df), p=[0.1,0.3,0.3,0.2,0.1])
            sexes = np.random.choice(['H','F','M'], len(df), p=[0.4,0.3,0.3])
            df['Âge/Sexe'] = [f"{age}{sex}" for age, sex in zip(ages, sexes)]
        
        return df.reset_index(drop=True)
    
    def _generate_musique(self) -> str:
        """Génère une musique réaliste"""
        positions = []
        for _ in range(np.random.randint(3, 6)):
            if np.random.random() < 0.3:
                pos = np.random.choice([1,2,3], p=[0.4,0.35,0.25])
            else:
                pos = np.random.randint(4, 15)
            positions.append(str(pos))
        return 'a'.join(positions)

# ==== MOTEUR D'ANALYSE SIMPLIFIÉ ====
class SimpleAnalyzer:
    """Analyseur simplifié mais efficace"""
    
    def __init__(self):
        self.models = {}
        if SKLEARN_AVAILABLE:
            self.scaler = RobustScaler()
        
    def analyze_race(self, df: pd.DataFrame, race_type: str = "AUTO") -> pd.DataFrame:
        """Analyse complète d'une course"""
        
        if len(df) < 3:
            raise ValueError("Minimum 3 chevaux requis")
        
        # Détection du type de course
        if race_type == "AUTO":
            race_type = self._detect_race_type(df)
        
        # Analyse classique
        classical_scores = self._classical_analysis(df, race_type)
        
        # Analyse ML si disponible et suffisamment de chevaux
        ml_scores = None
        if SKLEARN_AVAILABLE and len(df) >= CONFIG.MIN_RUNNERS_FOR_ML:
            try:
                ml_scores = self._ml_analysis(df)
            except Exception as e:
                st.warning(f"ML désactivé: {e}")
        
        # Combinaison des scores
        if ml_scores is not None:
            final_scores = 0.6 * ml_scores + 0.4 * classical_scores
            method = "ML + Classique"
        else:
            final_scores = classical_scores
            method = "Classique"
        
        # Préparation des résultats
        results = df.copy()
        results['probability'] = final_scores / final_scores.sum()
        results['score_final'] = final_scores
        results = results.sort_values('score_final', ascending=False).reset_index(drop=True)
        results['rank'] = range(1, len(results) + 1)
        results['race_type'] = race_type
        results['method'] = method
        
        return results
    
    def _detect_race_type(self, df: pd.DataFrame) -> str:
        """Détection automatique du type de course"""
        
        # Analyse des poids
        weights = pd.to_numeric(df['Poids'], errors='coerce').fillna(58.0)
        weight_var = weights.var()
        weight_mean = weights.mean()
        
        if weight_var > 10 and weight_mean > 55:
            return "PLAT"
        elif weight_var < 3 and weight_mean < 65:
            return "ATTELE_AUTOSTART"
        else:
            return "PLAT"
    
    def _classical_analysis(self, df: pd.DataFrame, race_type: str) -> pd.Series:
        """Analyse classique basée sur les cotes et autres facteurs"""
        
        # Conversion des données
        odds = pd.to_numeric(df['Cote'].str.replace(',', '.'), errors='coerce').fillna(10.0)
        draw = pd.to_numeric(df['Numéro de corde'], errors='coerce').fillna(1)
        weights = pd.to_numeric(df['Poids'], errors='coerce').fillna(58.0)
        
        # Calcul des scores
        odds_score = 1 / odds  # Inverse des cotes
        draw_score = self._calculate_draw_score(draw, race_type, len(df))
        form_score = df['Musique'].apply(self._parse_musique)
        weight_score = self._calculate_weight_score(weights)
        
        # Pondération selon le type de course
        if race_type == "PLAT":
            weights_config = [0.4, 0.25, 0.2, 0.15]  # odds, draw, form, weight
        else:  # TROT
            weights_config = [0.45, 0.3, 0.2, 0.05]
        
        # Score final
        total_score = (weights_config[0] * odds_score +
                      weights_config[1] * draw_score +
                      weights_config[2] * form_score +
                      weights_config[3] * weight_score)
        
        return total_score
    
    def _calculate_draw_score(self, draw: pd.Series, race_type: str, n_runners: int) -> pd.Series:
        """Calcule le score de position"""
        
        if race_type == "PLAT":
            # Pour le plat, favoriser les positions intérieures
            optimal_positions = list(range(1, min(6, n_runners//2 + 2)))
        else:
            # Pour le trot, positions moyennes souvent meilleures
            optimal_positions = list(range(max(1, n_runners//4), min(n_runners, n_runners//2 + 3)))
        
        scores = []
        for pos in draw:
            if pos in optimal_positions:
                scores.append(2.0)
            elif pos <= n_runners // 2:
                scores.append(1.0)
            else:
                scores.append(0.5)
        
        return pd.Series(scores, index=draw.index)
    
    def _parse_musique(self, musique: str) -> float:
        """Parse la musique pour extraire un score de forme"""
        if pd.isna(musique) or not musique:
            return 0.5
        
        try:
            positions = [int(p) for p in re.findall(r'\\d+', str(musique)) if int(p) <= 20]
            if not positions:
                return 0.5
            
            # Score basé sur les positions récentes avec pondération décroissante
            weights = np.exp(-0.3 * np.arange(len(positions)))
            weighted_avg = np.average(positions, weights=weights)
            
            return 1 / (1 + weighted_avg / 3)
        except:
            return 0.5
    
    def _calculate_weight_score(self, weights: pd.Series) -> pd.Series:
        """Calcule le score de poids"""
        if weights.std() < 1e-6:
            return pd.Series(0.5, index=weights.index)
        
        # Plus léger = mieux
        return (weights.max() - weights) / (weights.max() - weights.min())
    
    def _ml_analysis(self, df: pd.DataFrame) -> pd.Series:
        """Analyse avec Machine Learning"""
        
        # Création des features simplifiées
        features = self._create_simple_features(df)
        
        # Génération des labels d'entraînement
        labels = self._generate_training_labels(df)
        
        # Sélection du modèle disponible
        if XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        elif LIGHTGBM_AVAILABLE:
            model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, random_state=42, verbose=-1)
        else:
            model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        
        # Entraînement
        if len(features) >= 8:  # Minimum pour train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.3, random_state=42, stratify=labels
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            
            # Prédiction sur toutes les données
            features_scaled = self.scaler.transform(features)
            probabilities = model.predict_proba(features_scaled)[:, 1]
        else:
            # Pas assez de données pour split, entraînement sur tout
            features_scaled = self.scaler.fit_transform(features)
            model.fit(features_scaled, labels)
            probabilities = model.predict_proba(features_scaled)[:, 1]
        
        return pd.Series(probabilities, index=df.index)
    
    def _create_simple_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée des features simples pour le ML"""
        
        features = pd.DataFrame(index=df.index)
        
        # Features de cotes
        odds = pd.to_numeric(df['Cote'].str.replace(',', '.'), errors='coerce').fillna(10.0)
        features['odds_reciprocal'] = 1 / odds
        features['odds_log'] = np.log1p(odds)
        features['odds_rank'] = odds.rank(pct=True)
        
        # Features de position
        draw = pd.to_numeric(df['Numéro de corde'], errors='coerce').fillna(1)
        features['draw_position'] = draw
        features['draw_pct'] = draw / len(df)
        
        # Features de poids
        weights = pd.to_numeric(df['Poids'], errors='coerce').fillna(58.0)
        if weights.std() > 0:
            features['weight_zscore'] = (weights - weights.mean()) / weights.std()
        else:
            features['weight_zscore'] = 0.0
        
        # Features de forme
        features['form_score'] = df['Musique'].apply(self._parse_musique)
        
        # Features démographiques
        ages = df['Âge/Sexe'].apply(lambda x: int(re.search(r'(\\d+)', str(x)).group(1)) if re.search(r'(\\d+)', str(x)) else 5)
        features['age'] = ages
        features['is_prime_age'] = ((ages >= 4) & (ages <= 6)).astype(int)
        
        # Nettoyage
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def _generate_training_labels(self, df: pd.DataFrame) -> pd.Series:
        """Génère des labels d'entraînement basés sur les cotes"""
        
        odds = pd.to_numeric(df['Cote'].str.replace(',', '.'), errors='coerce').fillna(10.0)
        implied_probs = 1 / odds
        normalized_probs = implied_probs / implied_probs.sum()
        
        # Génération probabiliste des labels
        np.random.seed(42)
        labels = pd.Series(0, index=df.index)
        
        for idx, prob in normalized_probs.items():
            if np.random.random() < prob * 3:  # Amplification
                labels.loc[idx] = 1
        
        # Assurer au moins un label positif
        if labels.sum() == 0:
            labels.loc[normalized_probs.idxmax()] = 1
        
        return labels

# ==== GÉNÉRATEUR DE DONNÉES DE DÉMO ====
def generate_demo_race(n_runners: int = 12, race_type: str = "PLAT") -> pd.DataFrame:
    """Génère une course de démonstration réaliste"""
    
    # Noms de chevaux
    horse_names = [
        'Étoile Filante', 'Roi du Galop', 'Tempête Noire', 'Vent du Sud',
        'Flèche d\'Or', 'Prince Rapide', 'Saphir Bleu', 'Tonnerre Rouge',
        'Éclair Blanc', 'Foudre Sacrée', 'Ouragan Gris', 'Météore Brun',
        'Diamant Noir', 'Cristal Vert', 'Rubis Royal', 'Emeraude Vive',
        'Turquoise Wild', 'Ambre Solaire', 'Onyx Mystique', 'Perle Rare'
    ]
    
    np.random.seed(42)
    
    # Sélection des noms
    selected_names = horse_names[:n_runners]
    
    # Génération des cotes réalistes
    odds = np.random.lognormal(mean=1.6, sigma=0.7, size=n_runners)
    odds = np.clip(odds, 2.0, 25.0)
    odds = np.sort(odds)
    
    # Construction des données
    data = {
        'Nom': selected_names,
        'Numéro de corde': [str(i+1) for i in range(n_runners)],
        'Cote': [f"{odd:.1f}" for odd in odds],
    }
    
    # Données spécifiques au type
    if race_type == "PLAT":
        weights = np.random.normal(57, 2.5, n_runners)
        data['Poids'] = [f"{w:.1f}" for w in np.clip(weights, 52, 64)]
    else:  # TROT
        weights = np.random.normal(65, 1.5, n_runners)
        data['Poids'] = [f"{w:.1f}" for w in np.clip(weights, 62, 68)]
    
    # Musique cohérente avec les cotes
    musiques = []
    for i, odd in enumerate(odds):
        if i < 3:  # Favoris
            musique = np.random.choice(['1a2a3a', '2a1a4a', '3a2a1a'])
        elif i < n_runners // 2:  # Moyens
            musique = np.random.choice(['4a3a5a', '5a4a6a', '3a5a4a'])
        else:  # Outsiders
            musique = np.random.choice(['8a7a9a', '9a8a10a', '7a9a8a'])
        
        musiques.append(musique)
    
    data['Musique'] = musiques
    
    # Âge et sexe
    ages = np.random.choice([3,4,5,6,7], n_runners, p=[0.15,0.3,0.3,0.2,0.05])
    sexes = np.random.choice(['H','F','M'], n_runners, p=[0.4,0.3,0.3])
    data['Âge/Sexe'] = [f"{age}{sex}" for age, sex in zip(ages, sexes)]
    
    return pd.DataFrame(data)

# ==== INTERFACE STREAMLIT ====
def setup_streamlit_interface():
    """Configuration de l'interface Streamlit"""
    
    st.set_page_config(
        page_title="🏆 Elite Racing AI",
        page_icon="🐎",
        layout="wide"
    )
    
    # CSS simple
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .winner-card {
        border-left: 4px solid #28a745 !important;
        background: #f8fff9 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Interface principale"""
    
    setup_streamlit_interface()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏆 Elite Racing AI</h1>
        <p>Système Expert de Pronostics Hippiques</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        race_type = st.selectbox(
            "Type de Course",
            ["AUTO", "PLAT", "ATTELE_AUTOSTART"],
            help="AUTO = détection automatique"
        )
        
        use_ml = st.checkbox("Intelligence Artificielle", value=SKLEARN_AVAILABLE)
        if not SKLEARN_AVAILABLE:
            st.warning("⚠️ ML non disponible - Installez scikit-learn")
        
        st.info(f"""
        🏆 **Elite Racing AI v{CONFIG.VERSION}**
        
        📊 Analyse: {'ML + Classique' if use_ml and SKLEARN_AVAILABLE else 'Classique'}
        🤖 XGBoost: {'✅' if XGBOOST_AVAILABLE else '❌'}
        🔬 LightGBM: {'✅' if LIGHTGBM_AVAILABLE else '❌'}
        """)
    
    # Interface principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Analyse de Course")
        
        # Onglets pour les sources
        tab1, tab2, tab3 = st.tabs(["🔗 URL", "📁 Fichier", "🎲 Démo"])
        
        with tab1:
            url = st.text_input("URL de la course", placeholder="https://www.geny.com/...")
        
        with tab2:
            uploaded_file = st.file_uploader("Fichier CSV", type=["csv"])
        
        with tab3:
            demo_runners = st.slider("Nombre de partants", 6, 20, 12)
            demo_race_type = st.selectbox("Type course démo", ["PLAT", "ATTELE_AUTOSTART"])
    
    with col2:
        st.subheader("🚀 Actions")
        
        if st.button("🎯 Analyser", type="primary", use_container_width=True):
            analyze_race_interface(url, uploaded_file, race_type, use_ml)
        
        if st.button("🎲 Démo", use_container_width=True):
            run_demo(demo_runners, demo_race_type, use_ml)

def analyze_race_interface(url, uploaded_file, race_type, use_ml):
    """Interface d'analyse de course"""
    
    with st.spinner("🧠 Analyse en cours..."):
        try:
            # Chargement des données
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                source = f"Fichier: {uploaded_file.name}"
            elif url:
                extractor = SimpleDataExtractor()
                df = extractor.extract_race_data(url)
                source = f"URL: {url[:50]}..."
            else:
                st.error("⚠️ Veuillez fournir une source de données")
                return
            
            if df is None or df.empty:
                st.error("❌ Impossible de charger les données")
                return
            
            # Analyse
            analyzer = SimpleAnalyzer()
            results = analyzer.analyze_race(df, race_type)
            
            # Affichage des résultats
            display_results(results, source, use_ml)
            
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

def run_demo(n_runners, race_type, use_ml):
    """Lance une démonstration"""
    
    with st.spinner(f"🎲 Génération d'une course {race_type} avec {n_runners} partants..."):
        try:
            # Génération des données
            df_demo = generate_demo_race(n_runners, race_type)
            
            # Analyse
            analyzer = SimpleAnalyzer()
            results = analyzer.analyze_race(df_demo, race_type)
            
            # Affichage
            display_results(results, f"Démo {race_type}", use_ml)
            
        except Exception as e:
            st.error(f"❌ Erreur de démonstration: {str(e)}")

def display_results(results: pd.DataFrame, source: str, use_ml: bool):
    """Affiche les résultats d'analyse"""
    
    st.success(f"✅ Analyse terminée - {len(results)} chevaux")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        top1_prob = results['probability'].iloc[0] * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>{top1_prob:.1f}%</h3>
            <p>Premier Favori</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        method = results['method'].iloc[0]
        st.markdown(f"""
        <div class="metric-card">
            <h3>{method}</h3>
            <p>Méthode d'Analyse</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        race_type = results['race_type'].iloc[0]
        st.markdown(f"""
        <div class="metric-card">
            <h3>{race_type}</h3>
            <p>Type de Course</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(results)}</h3>
            <p>Partants</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Top 5 prédictions
    st.subheader("🏆 Top 5 Prédictions")
    
    for i, (_, horse) in enumerate(results.head(5).iterrows()):
        prob = horse['probability'] * 100
        card_class = "prediction-card winner-card" if i == 0 else "prediction-card"
        icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "⭐"
        
        st.markdown(f"""
        <div class="{card_class}">
            <h4>{i+1}. {icon} {horse['Nom']}</h4>
            <p>Probabilité: <strong>{prob:.1f}%</strong> • Cote: {horse.get('Cote', 'N/A')} • Corde: {horse.get('Numéro de corde', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommandations
    st.subheader("💡 Recommandations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Stratégie Recommandée:**")
        n_runners = len(results)
        
        if n_runners <= 8:
            strategy = ["Simple Gagnant sur le favori", "Couplé avec les 2 premiers", "Trio ordre sécurisé"]
        elif n_runners <= 12:
            strategy = ["Quinté+ base 4-5 chevaux", "Trio désordre élargi", "2sur4 pour rapport"]
        else:
            strategy = ["Quinté+ base élargie", "Super4 alternative", "Multi avec outsiders"]
        
        for i, strat in enumerate(strategy, 1):
            st.write(f"{i}. {strat}")
    
    with col2:
        st.markdown("**🏆 Sélections:**")
        
        winner = results.iloc[0]
        st.success(f"🥇 **Gagnant:** {winner['Nom']} ({winner['probability']*100:.1f}%)")
        
        top3 = results.head(3)['Nom'].tolist()
        st.info(f"🎯 **Trio:** {' • '.join(top3)}")
        
        # Outsider value
        for _, horse in results.iterrows():
            if horse['rank'] > 3 and horse['rank'] <= 8:
                try:
                    odds = float(horse['Cote'])
                    if odds > 8 and horse['probability'] > 0.08:
                        st.warning(f"💎 **Outsider:** {horse['Nom']} (Cote {odds})")
                        break
                except:
                    continue
    
    # Tableau détaillé
    with st.expander("📋 Tableau Complet", expanded=False):
        display_data = results[['rank', 'Nom', 'probability', 'Cote', 'Numéro de corde', 'Poids']].copy()
        display_data['Probabilité %'] = (display_data['probability'] * 100).round(1)
        display_data = display_data.drop('probability', axis=1)
        display_data.columns = ['Rang', 'Cheval', 'Cote', 'Corde', 'Poids', 'Probabilité %']
        
        st.dataframe(display_data, use_container_width=True)
    
    # Graphique de distribution des probabilités
    if len(results) > 5:
        st.subheader("📊 Distribution des Probabilités")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=results['Nom'][:10],
            y=results['probability'][:10] * 100,
            marker_color='rgba(102, 126, 234, 0.7)'
        ))
        
        fig.update_layout(
            title="Top 10 - Probabilités de Victoire",
            xaxis_title="Chevaux",
            yaxis_title="Probabilité (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==== POINT D'ENTRÉE ====
if __name__ == "__main__":
        main()
