import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score

# Fonction pour parser une ligne de log
def parse_log_line(line):
    match = re.match(r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+(\d+)\s+(\d+)\s+([A-Z])\s+(.*)", line)
    if match:
        return match.groups()
    else:
        return None

# Charger les logs en utilisant le parsing personnalisé
logs = []
with open('Android_2k.log', 'r', encoding='utf-8') as file:
    for line in file:
        parsed = parse_log_line(line)
        if parsed:
            logs.append(parsed)

# Convertir les logs en DataFrame
logs_df = pd.DataFrame(logs, columns=['timestamp', 'process_id', 'thread_id', 'log_level', 'message'])

# Convertir 'timestamp' en datetime
logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'], format='%m-%d %H:%M:%S.%f')

# Extraction des caractéristiques
logs_df['message_length'] = logs_df['message'].apply(len)
logs_df['is_error'] = logs_df['log_level'].apply(lambda x: 1 if x == 'E' else 0)

# Préparer les données pour le modèle
X = logs_df[['message_length', 'is_error']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Appliquer l'algorithme Isolation Forest pour la détection d'anomalies
model = IsolationForest(contamination=0.1, random_state=42)
logs_df['anomaly_score'] = model.fit_predict(X_scaled)

# Isolation Forest attribue -1 aux points considérés comme des anomalies
anomalies = logs_df[logs_df['anomaly_score'] == -1]

# Calcul des vraies étiquettes pour les métriques (1 = normal, 0 = anomalie)
y_true = np.ones(len(X_scaled))
y_true[logs_df['anomaly_score'] == -1] = 0

# Prédictions (1 pour normal, 0 pour anomalie)
y_pred = np.where(logs_df['anomaly_score'] == 1, 1, 0)

# Calcul des métriques de performance avec zero_division pour éviter l'erreur
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

# Check if class '1' exists, otherwise default to 0.0
precision = report['1']['precision'] if '1' in report else 0.0
recall = report['1']['recall'] if '1' in report else 0.0
f1_score = report['1']['f1-score'] if '1' in report else 0.0
auc_roc = roc_auc_score(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0

# Affichage des résultats avec les métriques sur l'image
sns.set(style="darkgrid")
plt.figure(figsize=(12, 6))

# Tracer les anomalies détectées
plt.scatter(anomalies['timestamp'], [1] * len(anomalies), color='red', label=f'Anomaly (Detected: {len(anomalies)})', marker='o')

# Ajouter les métriques dans l'image
plt.text(0.05, 0.95, f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1_score:.2f}\nAUC-ROC: {auc_roc:.2f}",
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.5))

plt.title("Detection des anomalies dans les logs Android avec Isolation Forest")
plt.xlabel("Timestamp")
plt.ylabel("Anomaly Detected")
plt.legend()
plt.show()

# Générer un rapport d'anomalies détectées
anomalies.to_csv('isolation_forest_anomalies_report.csv', index=False)
