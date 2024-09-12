import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# Créer l'Autoencoder
input_dim = X_scaled.shape[1]
autoencoder = Sequential()
autoencoder.add(Dense(64, activation='relu', input_shape=(input_dim,)))
autoencoder.add(Dense(32, activation='relu'))
autoencoder.add(Dense(64, activation='relu'))
autoencoder.add(Dense(input_dim, activation='sigmoid'))

autoencoder.compile(optimizer='adam', loss='mse')

# Entraîner l'autoencoder
autoencoder.fit(X_scaled, X_scaled, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Prédiction des anomalies
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)

# Définir le seuil pour identifier les anomalies
threshold = np.percentile(mse, 95)  # Ex: 95e percentile
anomalies = mse > threshold

# Calcul des vraies étiquettes pour les métriques (1 = normal, 0 = anomalie)
y_true = np.ones(len(X_scaled))
y_true[anomalies] = 0

# Prédictions (1 pour normal, 0 pour anomalie)
y_pred = np.where(anomalies, 0, 1)

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
anomaly_timestamps = logs_df[anomalies]['timestamp']
plt.scatter(anomaly_timestamps, [1] * len(anomaly_timestamps), color='red', label=f'Anomaly (Detected: {len(anomaly_timestamps)})', marker='o')

# Ajouter les métriques dans l'image
plt.text(0.05, 0.95, f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1_score:.2f}\nAUC-ROC: {auc_roc:.2f}",
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.5))

plt.title("Detection des anomalies dans les logs Android avec Autoencoder")
plt.xlabel("Timestamp")
plt.ylabel("Anomaly Detected")
plt.legend()
plt.show()

# Générer un rapport d'anomalies détectées
anomalies_df = logs_df[anomalies]
anomalies_df.to_csv('autoencoder_anomalies_report.csv', index=False)
