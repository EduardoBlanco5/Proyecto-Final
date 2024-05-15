import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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

# Entrenar el modelo SVM
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test_scaled)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Calcular las métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

# Calcular Sensitivity y Specificity
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]

sensitivity = TP / (TP + FN) if TP + FN != 0 else 0
specificity = TN / (TN + FP) if TN + FP != 0 else 0

# Imprimir resultados
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("F1 Score:", f1)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualizar los coeficientes de las características
coefs = pd.DataFrame({'feature': X.columns, 'coef': np.abs(model.coef_[0])})
coefs = coefs.sort_values(by='coef', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(coefs['feature'], coefs['coef'])
plt.xlabel('Coefficient Value')
plt.title('Feature Coefficients')
plt.show()