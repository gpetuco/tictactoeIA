import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Carregando o conjunto de dados a partir do arquivo
data = pd.read_csv('tic-tac-toe.data', header=None)

# Renomeando as colunas
data.columns = [
    'C1', 'C2', 'C3',
    'C4', 'C5', 'C6',
    'C7', 'C8', 'C9',
    'Resultado'
]

# Mapeando valores 'x' para 1, 'o' para 0 e 'b' para -1
data = data.replace({'x': 1, 'o': 0, 'b': -1})

# Separando os recursos (X) e as classes (y)
X = data.iloc[:, :-1]
y = data['Resultado']

# Definindo uma semente aleatória para garantir reproducibilidade
random_seed = 42

# Dividindo o conjunto de dados em treinamento e teste com uma semente aleatória fixa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Criando o modelo de árvore de decisão
tree = DecisionTreeClassifier(random_state=random_seed)

# Treinando o modelo
tree.fit(X_train, y_train)

# Fazendo previsões
y_pred = tree.predict(X_test)

# Avaliando a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy}')