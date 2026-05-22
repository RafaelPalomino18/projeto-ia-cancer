"""
Arquivo principal do Trabalho Final.

Implementa: MLP para classificação, Algoritmo Genético para otimizar seus hiperparâmetros
e busca em grafos (BFS/DFS) usando correlação entre variáveis.
"""
import time
import random
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# Configuração global

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# 1. CARREGAR E PREPROCESSAR

data = load_breast_cancer()
X_df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# normalização (fit apenas uma vez)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

# divisão final: treino + teste (teste **NÃO** será usado durante o AG)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
)


# 2. GRAFO DE CORRELAÇÕES (para BFS/DFS)

corr = X_df.corr()
threshold = 0.75
graph = defaultdict(list)
for col1 in corr.columns:
    for col2 in corr.columns:
        if col1 != col2 and abs(corr.loc[col1, col2]) > threshold:
            graph[col1].append(col2)

def bfs(start):
    visited = set()
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            order.append(node)
            for neighbor in graph[node]:
                queue.append(neighbor)
    return order

def dfs(start, visited=None, order=None):
    if visited is None:
        visited = set()
        order = []
    visited.add(start)
    order.append(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(neighbor, visited, order)
    return order


# 3. ESPAÇO DE BUSCA (representação do indivíduo)

HIDDEN_OPTIONS = [(30,), (50,), (100,), (50, 30)]
ACTIVATIONS = ["relu", "tanh"]
LR_OPTIONS = [0.001, 0.01, 0.05]

def gerar_individuo():
    return {
        "hidden_layer_sizes": random.choice(HIDDEN_OPTIONS),
        "activation": random.choice(ACTIVATIONS),
        "learning_rate_init": random.choice(LR_OPTIONS),
    }


# 4. AVALIAR (fitness) — usa cross-validation no conjunto de treino (sem tocar X_test)

def avaliar(individuo, X=X_train, y=y_train, cv=3):
    # construímos um MLP com os hiperparâmetros do indivíduo
    model = MLPClassifier(
        hidden_layer_sizes=individuo["hidden_layer_sizes"],
        activation=individuo["activation"],
        learning_rate_init=individuo["learning_rate_init"],
        max_iter=300,
        random_state=RANDOM_SEED,
        early_stopping=True,
        n_iter_no_change=10
    )
    # usa cross_val_score (média das folds) para estimar generalização
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    mean_score = float(np.mean(scores))
    return mean_score


# 5. SELEÇÃO: torneio (robusto e simples)

def torneio(populacao, fitnesses, k=3):
    # retorna 1 indivíduo vencedor do torneio (maior fitness)
    aspirantes = random.sample(list(range(len(populacao))), k)
    melhor = max(aspirantes, key=lambda idx: fitnesses[idx])
    return populacao[melhor]


# 6. CROSSOVER e MUTAÇÃO

def crossover(ind1, ind2, prob_crossover=0.9):
    if random.random() > prob_crossover:
        # sem crossover: retorna cópia de um dos pais
        return random.choice([ind1.copy(), ind2.copy()])
    child = {}
    for key in ind1:
        child[key] = random.choice([ind1[key], ind2[key]])
    return child

def mutacao(individuo, taxa_mutacao=0.1):
    # mutação por gene (cada gene pode ser alterado com prob taxa_mutacao)
    novo = individuo.copy()
    if random.random() < taxa_mutacao:
        novo["hidden_layer_sizes"] = random.choice(HIDDEN_OPTIONS)
    if random.random() < taxa_mutacao:
        novo["activation"] = random.choice(ACTIVATIONS)
    if random.random() < taxa_mutacao:
        novo["learning_rate_init"] = random.choice(LR_OPTIONS)
    return novo


# 7. ALGORITMO GENÉTICO PRINCIPAL (com elitismo e controle de reprodutibilidade)

def algoritmo_genetico(pop_size=8, geracoes=6, elite_size=1, cv=3):
    populacao = [gerar_individuo() for _ in range(pop_size)]
    melhor_global = (0.0, None)  # (score, individuo)
    for g in range(geracoes):
        t0 = time.time()
        # avalia população inteira
        fitnesses = [avaliar(ind, cv=cv) for ind in populacao]
        # registra melhores locais
        ranked = sorted(list(zip(fitnesses, populacao)), key=lambda x: x[0], reverse=True)
        if ranked[0][0] > melhor_global[0]:
            melhor_global = (ranked[0][0], ranked[0][1].copy())
        print(f"\n[Geração {g+1}] melhor neste passo: {ranked[0][0]:.4f}, melhor global: {melhor_global[0]:.4f}")
        # elitismo: preserva os top elite_size indivíduos
        nova_pop = [ind.copy() for (_, ind) in ranked[:elite_size]]
        # gera restante da população
        while len(nova_pop) < pop_size:
            # seleção por torneio
            pai1 = torneio(populacao, fitnesses, k=3)
            pai2 = torneio(populacao, fitnesses, k=3)
            filho = crossover(pai1, pai2, prob_crossover=0.9)
            filho = mutacao(filho, taxa_mutacao=0.15)
            nova_pop.append(filho)
        populacao = nova_pop
        t1 = time.time()
        print(f" Geração levou {t1-t0:.1f}s")
    return melhor_global


# 8. EXECUÇÃO DO AG E SALVAR MODELO FINAL (treino no conjunto de treino completo e avaliação final no teste)

if __name__ == "__main__":
    print("Iniciando AG revisado...")
    melhor_score, melhor_params = algoritmo_genetico(pop_size=8, geracoes=6, elite_size=1, cv=3)
    print("\nMelhor (CV) encontrado pelo AG:", melhor_params, "-> CV accuracy:", melhor_score)

    # Treinar o melhor modelo no conjunto de treino completo e salvar
    melhor_model = MLPClassifier(
        hidden_layer_sizes=melhor_params["hidden_layer_sizes"],
        activation=melhor_params["activation"],
        learning_rate_init=melhor_params["learning_rate_init"],
        max_iter=500,
        random_state=RANDOM_SEED,
        early_stopping=True,
        n_iter_no_change=10
    )
    print("Treinando o melhor modelo no treino completo...")
    melhor_model.fit(X_train, y_train)
    joblib.dump(melhor_model, "mlp_melhor_ag.pkl")
    joblib.dump(scaler, "normalizador_cancer_mama.pkl")
    print("Modelo salvo em 'mlp_melhor_ag.pkl' e normalizador salvo")

    # Avaliação final no conjunto de teste (única vez)
    y_pred = melhor_model.predict(X_test)
    acc_final = accuracy_score(y_test, y_pred)
    print(f"\nAcurácia final no conjunto de teste: {acc_final:.4f}\n")
    print("Relatório detalhado (test set):")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

    # Imprimir BFS/DFS para um nó inicial
    start_node = list(graph.keys())[0]
    print("\n=== BFS a partir de:", start_node, "===")
    print(bfs(start_node))
    print("\n=== DFS a partir de:", start_node, "===")
    print(dfs(start_node))
