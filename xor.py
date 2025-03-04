import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights():
    Vij = np.array([[0.716, -0.536, -0.590], 
                    [0.742, -0.262, -0.170], 
                    [0.576, 0.872, 0.034]])  # Pesos entrada-oculta

    Wjk = np.array([-0.754, -0.590, 0.564, 0.744])  # Vector columna (4 elementos, 1 por cada Zj + bias)

    return Vij, Wjk

def calculate_Zinj(X, Vij):
    X_bias = np.append(1, X)
    Zinj = np.dot(X_bias, Vij)
    return Zinj

def calculate_Zj(Zinj):
    return sigmoid(Zinj)

def calculate_Yink(Zj, Wjk):
    Zj_bias = np.append(1, Zj)  # Agregar bias
    Yink = np.dot(Zj_bias, Wjk)  # Entrada a capa de salida
    return Yink, Zj_bias

def calculate_Yk(Yink):
    return sigmoid(Yink)

def calculate_dk(Tk, Yk):
    return (Tk - Yk) * sigmoid_derivative(Yk)  # Cálculo de delta k

def calculate_delta_Wjk(alpha, dk, Zj_bias):
    return alpha * dk * Zj_bias  # Ajuste de pesos de la capa oculta a salida

def calculate_dinj(dk, Wjk):
    return dk * Wjk[1:]  # Omitir el bias en la propagación del error

def calculate_dj(dinj, Zinj):
    return dinj * sigmoid_derivative(sigmoid(Zinj))  # Usar sigmoide precomputado

def calculate_delta_Vij(alpha, dj, X):
    X_bias = np.append(1, X)  # Agregar bias
    return alpha * np.outer(X_bias, dj)  # Ajuste de pesos de entrada a oculta

def update_weights(Vij, delta_Vij, Wjk, delta_Wjk):
    Vij += delta_Vij  # Actualizar pesos entrada-oculta
    Wjk += delta_Wjk  # Actualizar pesos oculta-salida
    return Vij, Wjk

# Parámetro de tasa de aprendizaje
alpha = 0.3
epochs = 10000  # Número de iteraciones

# Datos de entrada para XOR
training_data = [
    (np.array([0, 0]), 0),
    (np.array([0, 1]), 1),
    (np.array([1, 0]), 1),
    (np.array([1, 1]), 0)
]

# Inicializar pesos
Vij, Wjk = initialize_weights()

# Entrenamiento
for epoch in range(epochs):
    total_error = 0  # Para medir el error total de la red

    for X, Tk in training_data:
        Zinj = calculate_Zinj(X, Vij)
        Zj = calculate_Zj(Zinj)
        Yink, Zj_bias = calculate_Yink(Zj, Wjk)
        Yk = calculate_Yk(Yink)

        # **Evitar propagación de NaN**
        if np.isnan(Yk).any() or np.isnan(Zj).any():
            print(f"Error en la época {epoch}: Se encontraron valores NaN. Deteniendo entrenamiento.")
            break

        dk = calculate_dk(Tk, Yk)
        delta_Wjk = calculate_delta_Wjk(alpha, dk, Zj_bias)
        Dinj = calculate_dinj(dk, Wjk)
        dj = calculate_dj(Dinj, Zinj)
        delta_Vij = calculate_delta_Vij(alpha, dj, X)

        # Actualizar los pesos
        Vij, Wjk = update_weights(Vij, delta_Vij, Wjk, delta_Wjk)

        # Acumular error cuadrático medio
        total_error += (Tk - Yk) ** 2

    # Mostrar error cada 1000 épocas
    if epoch % 1000 == 0:
        print(f"Época {epoch}, Error total: {total_error}")

print("Entrenamiento finalizado.")

# Pruebas después del entrenamiento
print("\nPruebas después del entrenamiento:")
for X, Tk in training_data:
    Zinj = calculate_Zinj(X, Vij)
    Zj = calculate_Zj(Zinj)
    Yink, Zj_bias = calculate_Yink(Zj, Wjk)
    Yk = calculate_Yk(Yink)
    print(f"Entrada: {X}, Salida esperada: {Tk}, Salida predicha: {Yk:.4f}")
