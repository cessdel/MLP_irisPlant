import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
alpha = 0.5
epochs = 10000

# Inicializar pesos
Vij, Wjk = initialize_weights()

# Entrenamiento
for epoch in range(epochs):
    total_error = 0  # Error acumulado

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

        # Error cuadrático medio
        total_error += np.sum((T_sample - Yk) ** 2)

    # Mostrar error cada 1000 épocas
    if epoch % 1000 == 0:
        print(f"Época {epoch}, Error total: {total_error}")

print("Entrenamiento finalizado.")

# Pruebas después del entrenamiento
print("\nPruebas después del entrenamiento:")
correct = 0
for i in range(X_test.shape[0]):
    X_sample = X_test[i]
    T_sample = y_test[i]

    Zinj, X_bias = calculate_Zinj(X_sample, Vij)
    Zj = calculate_Zj(Zinj)
    Yink, Zj_bias = calculate_Yink(Zj, Wjk)
    Yk = calculate_Yk(Yink)

    predicted_class = np.argmax(Yk)  # Clase con la mayor probabilidad
    actual_class = np.argmax(T_sample)  # Clase real

    if predicted_class == actual_class:
        correct += 1

    print(f"Entrada: {X_sample}, Salida esperada: {actual_class}, Salida predicha: {predicted_class}")

# Mostrar precisión final
accuracy = (correct / X_test.shape[0]) * 100
print(f"\nPrecisión en datos de prueba: {accuracy:.2f}%")
