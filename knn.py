import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Ler o arquivo com os dados e transformá-lo em uma lista de listas
with open('tic-tac-toe.data', 'r') as file:
    data = [line.strip().split(',') for line in file]

# Separar as características (tabuleiros) e os rótulos (positive ou negative)
X = [line[:-1] for line in data]
y = [line[-1] for line in data]

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converter as características de "x", "o" e "b" para números
label_encoder = LabelEncoder()
X_train_encoded = [label_encoder.fit_transform(row) for row in X_train]
X_test_encoded = [label_encoder.transform(row) for row in X_test]

# Converter os rótulos para números
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Criar o modelo k-NN
k = 3  # Você pode ajustar o valor de k conforme necessário
knn = KNeighborsClassifier(n_neighbors=k)

# Treinar o modelo
knn.fit(X_train_encoded, y_train_encoded)

# Fazer previsões
predictions = knn.predict(X_test_encoded)

# Avaliar o desempenho do modelo
accuracy = np.mean(predictions == y_test_encoded)
print(f'Acurácia do modelo: {accuracy:.6f}')