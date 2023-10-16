import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Lê o arquivo com os dados do jogo da velha
def ler_dataset(file):
    with open(file, 'r') as f:
        linhas = f.readlines()
    return [linha.strip().split(',') for linha in linhas]

# Converte as jogadas em vetores numéricos
def converter_dados_para_vetores(dados):
    X = []
    y = []
    for linha in dados:
        tabuleiro = linha[:-1]
        resultado = linha[-1]
        tabuleiro_numerico = [1 if s == 'x' else (-1 if s == 'o' else 0) for s in tabuleiro]
        X.append(tabuleiro_numerico)
        y.append(resultado)
    return np.array(X), np.array(y)

# Carrega o dataset e converte os dados
dados = ler_dataset('tic-tac-toe.data')
X, y = converter_dados_para_vetores(dados)

# Codifica as labels 'positive' e 'negative' em valores numéricos
le = LabelEncoder()
y = le.fit_transform(y)

# Divide o dataset em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria e treina o modelo MLP
mlp = MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Avalia o modelo
accuracy = mlp.score(X_test, y_test)
print(f'Acurácia do modelo: {accuracy}')