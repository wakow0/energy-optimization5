# Install required packages
import os
os.system('pip cache purge')
os.system('pip uninstall -y scikit-learn imbalanced-learn nilearn mlxtend bigframes xgboost lightgbm')
os.system('pip install scikit-learn==1.4.0 imbalanced-learn==0.12.0 xgboost==2.0.3 lightgbm==4.3.0')

# Verify versions
import sklearn
import imblearn
import xgboost
import lightgbm
print(f"scikit-learn version: {sklearn.__version__}")
print(f"imblearn version: {imblearn.__version__}")
print(f"xgboost version: {xgboost.__version__}")
print(f"lightgbm version: {lightgbm.__version__}")

import pandas as pd
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from datetime import datetime, timedelta
import joblib
from IPython.display import Image, display
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading
class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_encoder = LabelEncoder()

    def _get_sensor_columns(self):
        feature_file = os.path.join(self.data_dir, 'Wind Farm A', 'comma_feature_description.csv')
        try:
            feature_df = pd.read_csv(feature_file)
            maintenance_keywords = ['gearbox', 'bearing', 'pitch', 'generator', 'hydraulic', 'vibration',
                                   'rotor blade', 'stator winding', 'oil temperature', 'oil pressure', 'oil level']
            sensors = feature_df[
                feature_df['description'].str.contains('|'.join(maintenance_keywords), case=False, na=False)
            ]['sensor_name'].tolist()
            return sensors[:100]
        except Exception as e:
            print(f"Error reading feature description: {e}")
            datasets_path = os.path.join(self.data_dir, 'Wind Farm A', 'datasets')
            sensor_files = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]
            sample_csv = pd.read_csv(os.path.join(datasets_path, sensor_files[0]), nrows=1)
            return [col for col in sample_csv.columns if any(suffix in col for suffix in ['avg', 'max', 'std'])][:100]

    def load_and_preprocess_data(self, max_files=20):
        base_path = os.path.join(self.data_dir, 'Wind Farm A')
        datasets_path = os.path.join(base_path, 'datasets')
        event_file = os.path.join(base_path, 'comma_event_info.csv')
        
        if not os.path.exists(event_file):
            raise FileNotFoundError(f"Event file not found: {event_file}")
        
        print("Loading event data for Wind Farm A")
        event_df = pd.read_csv(event_file)
        event_df['event_start'] = pd.to_datetime(event_df['event_start'], errors='coerce')
        event_df['event_label'] = event_df['event_label'].fillna('normal')
        event_df['event_description'] = event_df['event_description'].fillna('Unspecified failure')
        print(f"Unique event descriptions:\n{event_df['event_description'].unique()}")
        
        sensor_files = sorted([os.path.join(datasets_path, f) 
                              for f in os.listdir(datasets_path) 
                              if f.endswith('.csv') and f != 'comma_feature_description.csv'])
        sensor_files = sensor_files[:max_files]
        
        if not sensor_files:
            raise ValueError("No sensor files found for Wind Farm A")
        
        print(f"Loading {len(sensor_files)} sensor files")
        chunk_size = 2
        sensor_dfs = []
        for i in range(0, len(sensor_files), chunk_size):
            chunk_files = sensor_files[i:i+chunk_size]
            try:
                if any(os.path.getsize(f) > 1000000 for f in chunk_files):
                    chunk = dd.read_csv(chunk_files).compute()
                else:
                    chunk = pd.concat([pd.read_csv(f) for f in chunk_files])
                chunk['time_stamp'] = pd.to_datetime(chunk['time_stamp'], errors='coerce')
                numeric_cols = chunk.select_dtypes(include=[np.number]).columns
                chunk[numeric_cols] = chunk[numeric_cols].fillna(chunk[numeric_cols].median())
                chunk[numeric_cols] = chunk[numeric_cols].clip(lower=-1e6, upper=1e6)
                sensor_dfs.append(chunk)
            except Exception as e:
                print(f"Error loading chunk {i//chunk_size}: {e}")
        
        if not sensor_dfs:
            raise ValueError("No sensor data loaded")
        
        sensor_data = pd.concat(sensor_dfs)
        print(f"Sensor time_stamp range: {sensor_data['time_stamp'].min()} to {sensor_data['time_stamp'].max()}")
        
        merged = pd.merge_asof(
            sensor_data.sort_values('time_stamp'),
            event_df.sort_values('event_start'),
            left_on='time_stamp',
            right_on='event_start',
            tolerance=pd.Timedelta('6h'),
            direction='nearest'
        )
        
        print(f"Merged data shape before filtering: {merged.shape}")
        merged['event_label'] = merged['event_label'].fillna('normal')
        merged = merged.drop_duplicates(subset=['time_stamp'])
        
        unmatched = sensor_data[~sensor_data['time_stamp'].isin(merged['time_stamp'])]
        if not unmatched.empty:
            unmatched_samples = min(len(unmatched), 2 * len(merged[merged['event_label'] == 'anomaly']))
            unmatched = unmatched.sample(unmatched_samples, random_state=42)
            unmatched['event_label'] = 'normal'
            unmatched['event_description'] = 'Normal operation'
            merged = pd.concat([merged, unmatched])
        
        merged = merged.reset_index(drop=True)
        print(f"Final data shape: {merged.shape}")
        merged['wind_farm'] = 'A'
        merged['label_encoded'] = self.label_encoder.fit_transform(merged['event_label'])
        print(f"Class distribution:\n{merged['event_label'].value_counts()}")
        return merged

# 2. Feature Engineering
class EnhancedFeatureEngineer:
    def __init__(self):
        self.sensor_cols = None
        self.numeric_features = None
        self.selected_features = None

    def engineer_features(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        sensor_cols = [col for col in numeric_cols if 'sensor' in col or 'power' in col or 'speed' in col]
        self.sensor_cols = sensor_cols
        
        features = pd.DataFrame(index=df.index)
        
        features['hour_of_day'] = df['time_stamp'].dt.hour
        features['day_of_week'] = df['time_stamp'].dt.dayofweek
        
        for col in self.sensor_cols:
            features[f'{col}_6h_mean'] = df[col].rolling(36, min_periods=1).mean()
            features[f'{col}_6h_std'] = df[col].rolling(36, min_periods=1).std().fillna(0)
        
        features = features.fillna(features.median(numeric_only=True))
        features = features.clip(upper=1e6)
        self.numeric_features = [col for col in features.columns if col != 'label_encoded']
        return features

    def select_features(self, X, y, n_features=20):
        X = X.fillna(X.median(numeric_only=True))
        X = X.clip(upper=1e6)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        top_indices = np.argsort(importances)[-n_features:]
        self.selected_features = [X.columns[i] for i in top_indices]
        return X[self.selected_features]

# 3. Model Training
class EnhancedTurbineFailurePredictor:
    def __init__(self):
        self.models = {
            'random_forest': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
            ]),
            'xgboost': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42))
            ]),
            'lightgbm': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, 
                                              min_data_in_leaf=1, min_child_samples=5, num_leaves=10, 
                                              force_col_wise=True, random_state=42))
            ])
        }
        self.best_model = None
        self.best_model_name = None
        self.feature_importances = {}
        self.model_results = {}

    def train_and_evaluate(self, X, y):
        print("Training and evaluating models...")
        
        results = {}
        best_f1 = -np.inf
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name.replace('_', ' ').title()}...")
            try:
                model.fit(X, y)
                y_pred = model.predict(X)
                f1 = f1_score(y, y_pred, average='weighted')
                results[name] = {'f1': f1}
                
                if f1 > best_f1:
                    best_f1 = f1
                    self.best_model = model
                    self.best_model_name = name
                
                if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                    self.feature_importances[name] = model.named_steps['classifier'].feature_importances_
                
                print(f"{name.replace('_', ' ').title()} F1: {f1:.3f}")
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
        
        print(f"\nFinal training using {self.best_model_name.replace('_', ' ').title()}...")
        self.best_model.fit(X, y)
        return results

    def predict_proba(self, X):
        if self.best_model is None:
            raise ValueError("No models trained")
        return {'probabilities': self.best_model.predict_proba(X)}

    def get_feature_importance(self, feature_names):
        importance_dict = {}
        for name, importance in self.feature_importances.items():
            indices = np.argsort(importance)[::-1][:10]
            importance_dict[name] = {
                'importance': importance[indices],
                'features': [feature_names[i] for i in indices]
            }
        return importance_dict

    def save_feature_importance_summary(self, feature_names):
        summary = []
        for name, importance in self.feature_importances.items():
            indices = np.argsort(importance)[::-1][:10]
            for i, idx in enumerate(indices):
                summary.append({
                    'model': name,
                    'feature': feature_names[idx],
                    'importance': importance[idx],
                    'rank': i + 1
                })
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('/kaggle/working/feature_importance_summary_a.csv', index=False)
        print("\nFeature Importance Summary:\n", summary_df)
        return summary_df

# 4. Maintenance Scheduler
class EnhancedMaintenanceScheduler:
    def __init__(self):
        self.risk_thresholds = {
            'critical': 0.9,
            'high': 0.75,
            'medium': 0.5
        }

    def generate_schedule(self, prediction_dict, timestamps, turbine_ids):
        probabilities = prediction_dict['probabilities'][:, 1]
        schedule = []
        for prob, ts, turbine_id in zip(probabilities, timestamps, turbine_ids):
            priority = 'low'
            for level, thresh in self.risk_thresholds.items():
                if prob >= thresh:
                    priority = level
                    break
            if prob >= 0.5:
                schedule.append({
                    'turbine_id': turbine_id,
                    'probability': prob,
                    'priority': priority,
                    'timestamp': ts
                })
        schedule_df = pd.DataFrame(schedule)
        schedule_df.to_csv('/kaggle/working/maintenance_schedule_a.csv', index=False)
        return schedule_df

# 5. Visualization
def plot_feature_importance(importance_dict, feature_names):
    for name, data in importance_dict.items():
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(data['features'])), data['importance'])
        plt.yticks(range(len(data['features'])), data['features'])
        plt.title(f'Top 10 Features - {name.replace("_", " ").title()}')
        plt.tight_layout()
        plt.savefig(f'/kaggle/working/feature_importance_{name}_a.png', dpi=300)
        plt.show()
        print(f"Chart saved: /kaggle/working/feature_importance_{name}_a.png")
        display(Image(filename=f'/kaggle/working/feature_importance_{name}_a.png'))

def plot_maintenance_timeline(schedule_df):
    plt.figure(figsize=(12, 6))
    colors = {'critical': 'red', 'high': 'orange', 'medium': 'yellow'}
    for priority in colors:
        mask = schedule_df['priority'] == priority
        plt.scatter(schedule_df[mask]['timestamp'], [1]*sum(mask), c=colors[priority], label=priority, s=100)
    plt.yticks([])
    plt.xlabel('Maintenance Date')
    plt.title('Maintenance Timeline - Wind Farm A')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/kaggle/working/maintenance_timeline_a.png', dpi=300)
    plt.show()
    print("Chart saved: /kaggle/working/maintenance_timeline_a.png")
    display(Image(filename=f'/kaggle/working/maintenance_timeline_a.png'))

def plot_confusion_matrix(y_true, y_pred, labels, target_names):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Wind Farm A')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('/kaggle/working/confusion_matrix_a.png', dpi=300)
    plt.show()
    print("Chart saved: /kaggle/working/confusion_matrix_a.png")
    display(Image(filename=f'/kaggle/working/confusion_matrix_a.png'))

# 6. Main Execution
if __name__ == "__main__":
    print("Loading and preprocessing data...")
    data_loader = DataLoader('/kaggle/input/wind-turbine-scada-data-for-early-fault-detection')
    data = data_loader.load_and_preprocess_data(max_files=20)
    
    print("Engineering features...")
    feature_engineer = EnhancedFeatureEngineer()
    features = feature_engineer.engineer_features(data)
    X = features[feature_engineer.numeric_features]
    y = data['label_encoded'].values
    
    print("Selecting top features...")
    X_selected = feature_engineer.select_features(X, y, n_features=20)
    
    print("Balancing classes with RandomOverSampler...")
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_selected, y)
    y_resampled = np.array(y_resampled)
    print(f"Class distribution after oversampling:\n{pd.Series(data_loader.label_encoder.inverse_transform(y_resampled)).value_counts()}")
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )
    
    test_timestamps = data.iloc[-len(X_test):]['time_stamp'].tolist()
    test_turbine_ids = data.iloc[-len(X_test):]['asset_id'].tolist()
    
    print("Training models...")
    predictor = EnhancedTurbineFailurePredictor()
    model_results = predictor.train_and_evaluate(X_train, y_train)
    
    print("\nFinal Evaluation:")
    predictions = predictor.predict_proba(X_test)
    y_pred = np.argmax(predictions['probabilities'], axis=1)
    labels = np.unique(y_test)
    target_names = [data_loader.label_encoder.classes_[i] for i in labels]
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))
    plot_confusion_matrix(y_test, y_pred, labels=labels, target_names=target_names)
    
    print("Generating maintenance schedule...")
    scheduler = EnhancedMaintenanceScheduler()
    schedule_df = scheduler.generate_schedule(predictions, test_timestamps, test_turbine_ids)
    print("\nMaintenance Schedule:\n", schedule_df.head())
    
    feature_importance = predictor.get_feature_importance(feature_engineer.selected_features)
    predictor.save_feature_importance_summary(feature_engineer.selected_features)
    plot_feature_importance(feature_importance, feature_engineer.selected_features)
    plot_maintenance_timeline(schedule_df)
    
    joblib.dump({
        'model': predictor.best_model,
        'feature_names': feature_engineer.selected_features,
        'label_encoder': data_loader.label_encoder
    }, '/kaggle/working/best_turbine_model_a.joblib')
    print("Model saved successfully")
