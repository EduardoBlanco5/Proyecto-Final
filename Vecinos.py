import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Cargar los conjuntos de datos
dt2 = pd.read_csv('zoo3.csv')
dt1 = pd.read_csv('zoo2.csv')

# Combinar los conjuntos de datos si es necesario
dt = dt1.merge(dt2, how='outer')

# Eliminar columnas irrelevantes
dt = dt.drop(columns=['animal_name'])

# Separar características y etiquetas
X = dt.drop(columns=['class_type'])
y = dt['class_type']

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lista para almacenar la precisión del modelo para diferentes valores de k
accuracy_scores = []

# Lista para almacenar las métricas de evaluación
metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'specificity': [],
    'f1': []
}

# Probar diferentes valores de k (número de vecinos)
for k in range(1, 20):
    # Entrenar el modelo de K-Vecinos Cercanos
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    
    # Hacer predicciones en el conjunto de prueba
    y_pred = model.predict(X_test_scaled)
    
    # Calcular las métricas de evaluación
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    
    # Almacenar métricas
    metrics['accuracy'].append(accuracy)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['specificity'].append(specificity)
    metrics['f1'].append(f1)
    
    # Almacenar la precisión del modelo
    accuracy_scores.append(accuracy)

# Graficar el número de vecinos vs. la precisión del modelo
plt.figure(figsize=(10, 6))
plt.plot(range(1, 20), accuracy_scores, marker='o', linestyle='-')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('K-Nearest Neighbors: Accuracy vs. Number of Neighbors')
plt.grid(True)
plt.xticks(np.arange(1, 20, step=1))
plt.show()

# Imprimir métricas de evaluación
print("Metrics for Different Values of k:")
for metric, values in metrics.items():
    print(metric.capitalize() + ':', values)