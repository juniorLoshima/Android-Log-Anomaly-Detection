import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
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

# Appliquer l'algorithme DBSCAN pour la détection d'anomalies
dbscan = DBSCAN(eps=0.5, min_samples=5)
logs_df['anomaly_score'] = dbscan.fit_predict(X_scaled)

# DBSCAN attribue -1 aux points considérés comme des anomalies (bruit)
anomalies = logs_df[logs_df['anomaly_score'] == -1]

# Calcul des vraies étiquettes (les points non-anormaux sont notés par 1)
y_true = [1 if score != -1 else 0 for score in logs_df['anomaly_score']]

# Metrics
report = classification_report(y_true, logs_df['anomaly_score'], output_dict=True)
precision = report['1']['precision']
recall = report['1']['recall']
f1_score = report['1']['f1-score']
auc_roc = roc_auc_score(y_true, logs_df['anomaly_score'])

# Affichage des résultats avec les métriques sur l'image
sns.set(style="darkgrid")
plt.figure(figsize=(12, 6))

# Tracer les anomalies détectées
plt.scatter(anomalies['timestamp'], [1] * len(anomalies), color='red', label=f'Anomaly (Detected: {len(anomalies)})', marker='o')

# Add performance metrics inside the plot
plt.text(0.05, 0.95, f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1_score:.2f}\nAUC-ROC: {auc_roc:.2f}",
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.5))

plt.title("Detection des anomalies dans les logs Android avec DBSCAN")
plt.xlabel("Timestamp")
plt.ylabel("Anomaly Detected")
plt.legend()
plt.show()

# Générer un rapport d'anomalies détectées
anomalies.to_csv('dbscan_anomalies_report.csv', index=False)
