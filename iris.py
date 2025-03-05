import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Función sigmoide y su derivada
def sigmoid(x):
    x = np.clip(x, -500, 500)  # Evitar overflow
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sig_x):
    return sig_x * (1 - sig_x)

# Cargar dataset Iris
iris = datasets.load_iris()
X = iris.data  # 4 características (longitud y ancho de sépalo y pétalo)
y = iris.target.reshape(-1, 1)  # Etiquetas en formato columna

# Normalizar datos para que estén en rango [0,1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Convertir etiquetas en formato one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)  # Ahora tenemos 3 columnas (una por clase)

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Inicializar pesos para red con 4 entradas, 3 neuronas ocultas y 3 salidas
def initialize_weights():
    Vij = np.random.uniform(-1, 1, (5, 7))  # 4 entradas + bias -> 3 neuronas ocultas
    Wjk = np.random.uniform(-1, 1, (8, 3))  # 3 ocultas + bias -> 3 salidas
    return Vij, Wjk

def calculate_Zinj(X, Vij):
    X_bias = np.append(1, X)  # Agregar bias
    Zinj = np.dot(X_bias, Vij)
    return Zinj, X_bias

def calculate_Zj(Zinj):
    return sigmoid(Zinj)

def calculate_Yink(Zj, Wjk):
    Zj_bias = np.append(1, Zj)  # Agregar bias
    Yink = np.dot(Zj_bias, Wjk)
    return Yink, Zj_bias

def calculate_Yk(Yink):
    return sigmoid(Yink)

def calculate_dk(Tk, Yk):
    return (Tk - Yk) * sigmoid_derivative(Yk)  # Ahora Tk y Yk son vectores

def calculate_dinj(dk, Wjk):
    return np.dot(Wjk[1:], dk)  # Propagación del error sin bias

def calculate_dj(dinj, Zinj):
    return dinj * sigmoid_derivative(sigmoid(Zinj))  # Error en la capa oculta

def calculate_delta_Wjk(alpha, dk, Zj_bias):
    return alpha * np.outer(Zj_bias, dk)  # Ajuste de pesos de oculta a salida

def calculate_delta_Vij(alpha, dj, X_bias):
    return alpha * np.outer(X_bias, dj)  # Ajuste de pesos de entrada a oculta

def update_weights(Vij, delta_Vij, Wjk, delta_Wjk):
    Vij += delta_Vij
    Wjk += delta_Wjk
    return Vij, Wjk

# Parámetros de entrenamiento
alpha = 0.3
epochs = 10000

# Almacenar resultados de precisión y error para TODOS los experimentos
accuracies = []
errors = []

# Realizar varios experimentos
for experiment in range(35):
    print(f"Experimento {experiment }")
    prev_accuracy = None

    # Inicializar pesos
    Vij, Wjk = initialize_weights()

    prev_error_total = None  # Para almacenar el error total de la época anterior

    # Almacenar resultados por época
    experiment_accuracies = []
    experiment_errors = []

    # Entrenamiento
    for epoch in range(epochs):
        total_error = 0  # Error total acumulado en la época
        correct = 0  # Contador de aciertos para la precisión

        for i in range(X_train.shape[0]):
            X_sample = X_train[i]
            T_sample = y_train[i]

            # Feedforward
            Zinj, X_bias = calculate_Zinj(X_sample, Vij)
            Zj = calculate_Zj(Zinj)
            Yink, Zj_bias = calculate_Yink(Zj, Wjk)
            Yk = calculate_Yk(Yink)

            # Evitar propagación de NaN
            if np.isnan(Yk).any():
                print(f"Error en la época {epoch}: Se encontraron valores NaN. Deteniendo entrenamiento.")
                break

            # Backpropagation
            dk = calculate_dk(T_sample, Yk)
            delta_Wjk = calculate_delta_Wjk(alpha, dk, Zj_bias)
            Dinj = calculate_dinj(dk, Wjk)
            dj = calculate_dj(Dinj, Zinj)
            delta_Vij = calculate_delta_Vij(alpha, dj, X_bias)

            # Actualizar pesos
            Vij, Wjk = update_weights(Vij, delta_Vij, Wjk, delta_Wjk)

            # Sumar error cuadrático
            total_error += np.sum((T_sample - Yk) ** 2)

            # Contar aciertos para precisión
            predicted_class = np.argmax(Yk)
            actual_class = np.argmax(T_sample)
            if predicted_class == actual_class:
                correct += 1

        # Calcular precisión de la época
        accuracy = (correct / X_train.shape[0]) * 100  # Corregir a X_train.shape[0] para precisión
        experiment_accuracies.append(accuracy)

        # Mostrar precisión y error por época
        print(f"Época {epoch}: Precisión = {accuracy:.2f}%, Error total = {total_error:.4f}")

        # Condición de early stopping basada en el error total
        if prev_error_total is not None:
            error_improvement = prev_error_total - total_error
            improvement_percentage = (error_improvement / prev_error_total) * 100  # Calcular el porcentaje de mejora

            if improvement_percentage < 0.01:  # Si la mejora es menor al 0.01%
                print(f"Deteniendo entrenamiento en la época {epoch} porque la mejora en el error total es menor al 0.01%.")
                break

        prev_error_total = total_error  # Guardamos el error total para la próxima comparación

        experiment_errors.append(total_error)

    # Almacenar resultados de precisión y error por experimento
    accuracies.append(np.mean(experiment_accuracies))
    errors.append(np.mean(experiment_errors))

# === Terminaron los experimentos ===

# Graficar resultados de TODOS los experimentos
plt.figure(figsize=(10, 5))

# Gráfica de precisión (promedio por experimento)
plt.subplot(1, 2, 1)
plt.plot(accuracies, marker='o', label="Precisión promedio por experimento")  # Usamos 'o' para marcar cada experimento
plt.title(f"Precisión promedio por Experimento\nMedia: {np.mean(accuracies):.2f}% | Desviación Estándar: {np.std(accuracies):.2f}")
plt.axhline(y=95, color='orange', linestyle='--', label='Línea 95%')  # Línea del 95% en naranja
plt.axhline(y=97, color='blue', linestyle='--', label='Línea 96%')    # Línea del 96% en azul
plt.axhline(y=99, color='green', linestyle='--', label='Línea 99%')   # Línea del 99% en verde
plt.xlabel('Experimento')
plt.ylabel('Precisión (%)')
plt.legend()

# Gráfica de error (promedio por experimento)
plt.subplot(1, 2, 2)
plt.plot(errors, label="Error promedio por experimento")
plt.title(f"Error promedio (35 experimentos)\nMedia: {np.mean(errors):.2f} | Desviación Estándar: {np.std(errors):.2f}")
plt.xlabel('Experimento')
plt.ylabel('Error cuadrático medio')
plt.legend()

plt.tight_layout()
plt.show()

# Mostrar resultados finales
print(f"==== Resultados Finales sobre 35 experimentos ====")
print(f"Precisión media: {np.mean(accuracies):.2f}%")
print(f"Desviación estándar de la precisión: {np.std(accuracies):.2f}")
print(f"Error medio: {np.mean(errors):.2f}")
print(f"Desviación estándar del error: {np.std(errors):.2f}")
