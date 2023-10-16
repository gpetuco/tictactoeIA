import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

import argparse

# Crie um objeto ArgumentParser
parser = argparse.ArgumentParser()

# Adicione um argumento ao seu programa
parser.add_argument("valor", type=int, help="Um número inteiro")

# Parse os argumentos da linha de comando
args = parser.parse_args()

import time

total_items = 100

for i in range(total_items):
    # Calcula o progresso atual
    progress = (i / total_items) * 100

    # Cria uma barra de carregamento simples
    bar_length = 50
    block = int(round(bar_length * i / total_items))
    progress_bar = "█" * block + "-" * (bar_length - block)

    # Imprime a barra de carregamento atual
    print(f"[{progress_bar}] {progress:.1f}%", end="\r")

    # Aguarda um curto período de tempo para simular o processamento
    time.sleep(0.02)
print("\n")
if (args.valor == 1):
    algoritmo = "knn"
    print("O algoritmo selecionado foi o k-NN.\n")
    time.sleep(1)
    from knn import knn
if (args.valor == 2):
    algoritmo = "mlp"
    print("O algoritmo selecionado foi o MLP.\n")
    time.sleep(1)
    from mlp import mlp
if (args.valor == 3):
    algoritmo = "tree"
    print("O algoritmo selecionado foi o Árvore de Decisão.\n")
    time.sleep(1)
    from tree import tree

print("\n")

def exibir_tabuleiro(tabuleiro):
    print("   0   1   2")
    for i, linha in enumerate(tabuleiro):
        print(i, end="  ")
        for j, cell in enumerate(linha):
            if cell == "X":
                print("X", end=" | ")
            elif cell == "O":
                print("O", end=" | ")
            else:
                print(" ", end=" | ")
        print("\n  " + "-" * 11)
    print()

import warnings
warnings.filterwarnings("ignore")

def converter_tabuleiro(tabuleiro):
    converted_board = []
    for linha in tabuleiro:
        converted_row = []
        for cell in linha:
            if cell == "X":
                converted_row.append(1)
            elif cell == "O":
                converted_row.append(2)
            else:
                converted_row.append(0)
        converted_board.append(converted_row)
    return np.array(converted_board).flatten()

# Função para verificar se há um vencedor com base no tabuleiro atual
def prever_vencedor(tabuleiro, jogador):
    tabuleiro_numerico = converter_tabuleiro(tabuleiro)
    if(algoritmo == "knn"):
        resultado = knn.predict([tabuleiro_numerico])
    if(algoritmo == "mlp"):
        resultado = mlp.predict([tabuleiro_numerico])
    if(algoritmo == "tree"):
        resultado = tree.predict([tabuleiro_numerico])

    for linha in tabuleiro:
        if all(cell == jogador for cell in linha):
            if(resultado != [0]):
                return True

    for coluna in range(3):
        if all(tabuleiro[linha][coluna] == jogador for linha in range(3)):
            if(resultado != [0]):
                return True

    if all(tabuleiro[i][i] == jogador for i in range(3)) or all(tabuleiro[i][2 - i] == jogador for i in range(3)):
        if(resultado != [0]):
                return True
    return False

def jogo_da_velha():
    tabuleiro = [[" " for _ in range(3)] for _ in range(3)]
    jogador_atual = "X"
    jogadas = 0

    while True:
        exibir_tabuleiro(tabuleiro)
        linha, coluna = map(int, input(f"Jogador {jogador_atual}, insira a linha (0-2) e coluna (0-2) separadas por espaco: ").split())

        if tabuleiro[linha][coluna] == " ":
            tabuleiro[linha][coluna] = jogador_atual
            jogadas += 1
        else:
            print("Essa posicao se encontra ocupada. Tente novamente.")
            continue

        if prever_vencedor(tabuleiro, jogador_atual):
            exibir_tabuleiro(tabuleiro)
            print(f"Jogador {jogador_atual} venceu!")
            break
        elif jogadas == 9:
            exibir_tabuleiro(tabuleiro)
            print("O jogo empatou!")
            break

        jogador_atual = "O" if jogador_atual == "X" else "X"

if __name__ == "__main__":
    jogo_da_velha()