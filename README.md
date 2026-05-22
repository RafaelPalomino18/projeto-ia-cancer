# 🧠 Projeto de Inteligência Artificial para Diagnóstico de Câncer de Mama

Projeto desenvolvido em Python utilizando Inteligência Artificial, Redes Neurais Artificiais, Algoritmo Genético e Busca em Grafos para classificação de câncer de mama utilizando o dataset Breast Cancer Wisconsin.

O sistema foi dividido em dois trabalhos principais:

- 📁 Trabalho N1 → implementação da Rede Neural MLP
- 📁 Trabalho Final → implementação avançada utilizando:
  - Algoritmo Genético (AG)
  - Busca em Largura (BFS)
  - Busca em Profundidade (DFS)
  - Otimização automática da Rede Neural

---

# 📚 Objetivo do Projeto

O objetivo do projeto é aplicar conceitos de Inteligência Artificial e Estruturas Inteligentes para resolver problemas de classificação médica utilizando Machine Learning.

O sistema simula um ambiente de análise médica capaz de:

- Treinar uma Rede Neural Artificial
- Avaliar diagnósticos
- Otimizar hiperparâmetros automaticamente
- Utilizar técnicas de busca em grafos
- Simular extração de características de imagens médicas

---

# ⚙️ Tecnologias Utilizadas

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- OpenCV
- Joblib

---

# 🗂️ Organização do Projeto

```bash
PROJETO-IA-CANCER/
│
├── TrabalhoFinal/
│   └── BFS_DFS_AG.py
│
├── TrabalhoN1/
│   ├── avaliacao.py
│   ├── carregamentoDeDados.py
│   ├── extratorDeFeatures.py
│   ├── inferencia.py
│   └── modelo.py
│
├── venv/
├── mlp_melhor_ag.pkl
├── normalizador_cancer_mama.pkl
└── README.md
```

---

# 🧩 Trabalho N1 — Rede Neural MLP

A pasta `TrabalhoN1` contém a implementação base do sistema utilizando uma Rede Neural Artificial do tipo MLP (Multilayer Perceptron).

---

# 📥 carregamentoDeDados.py

Responsável por:

- Carregar o dataset Breast Cancer Wisconsin
- Dividir os dados entre treino e teste
- Normalizar os dados utilizando `StandardScaler`
- Salvar o normalizador

## Processo realizado:

- 80% dos dados → treino
- 20% dos dados → teste

A normalização garante que todas as variáveis tenham a mesma escala, melhorando o desempenho da rede neural.

---

# 🧠 modelo.py

Responsável pela criação e treinamento da Rede Neural Artificial.

## Configuração da Rede Neural

- MLPClassifier
- 1 camada oculta
- 100 neurônios
- Função ReLU
- Solver Adam
- 300 épocas máximas

O modelo aprende padrões capazes de diferenciar tumores benignos e malignos.

Após o treinamento:
- O modelo é salvo em:
  
```bash
mlp_cancer_mama.pkl
```

---

# 📊 avaliacao.py

Responsável pela avaliação do modelo treinado.

O sistema calcula:

- Acurácia
- Precision
- Recall
- F1-Score
- Matriz de Confusão
- Curva de perda da rede neural

A matriz de confusão mostra:
- Acertos do modelo
- Erros do modelo

A curva de perda mostra o aprendizado da rede durante o treinamento.

---

# 🖼️ extratorDeFeatures.py

Simula a extração de características de imagens médicas.

Como o projeto não utiliza imagens reais:
- O sistema seleciona amostras aleatórias do dataset
- Simula a extração das 30 features numéricas

Essa etapa representa um sistema real de visão computacional.

---

# 🩺 inferencia.py

Responsável pelas previsões finais.

O sistema:
1. Carrega o modelo treinado
2. Carrega o normalizador
3. Recebe as features extraídas
4. Normaliza os dados
5. Realiza a previsão
6. Exibe:
   - Diagnóstico final
   - Confiança da IA
   - Probabilidades

Exemplo:

```bash
DIAGNÓSTICO DA AMOSTRA: BENIGN
Confiança do Modelo: 98.75%
```

---

# 🚀 Trabalho Final — Algoritmo Genético + BFS + DFS

A pasta `TrabalhoFinal` contém a versão avançada do projeto.

Arquivo principal:

```bash
BFS_DFS_AG.py
```

Esse algoritmo implementa:

- Rede Neural Artificial (MLP)
- Algoritmo Genético
- Busca em Largura (BFS)
- Busca em Profundidade (DFS)

---

# 🧬 Algoritmo Genético (AG)

O Algoritmo Genético é utilizado para otimizar automaticamente os hiperparâmetros da Rede Neural.

O sistema testa diferentes combinações de:

- Número de neurônios
- Função de ativação
- Taxa de aprendizado

---

# 🔄 Funcionamento do AG

O algoritmo realiza:

## 1️⃣ Geração da população inicial

Cada indivíduo representa uma configuração diferente da rede neural.

Exemplo:

```python
{
    "hidden_layer_sizes": (100,),
    "activation": "relu",
    "learning_rate_init": 0.001
}
```

---

## 2️⃣ Avaliação (Fitness)

Cada indivíduo é treinado e avaliado utilizando:

- Cross Validation
- Accuracy

O fitness representa a qualidade do indivíduo.

---

## 3️⃣ Seleção por Torneio

Os melhores indivíduos possuem maior chance de reprodução.

---

## 4️⃣ Crossover

Combina características de dois indivíduos.

---

## 5️⃣ Mutação

Altera parâmetros aleatoriamente para gerar diversidade genética.

---

## 6️⃣ Elitismo

Os melhores indivíduos são preservados entre gerações.

---

# 🌐 Busca em Grafos — BFS e DFS

O projeto também implementa:

## BFS — Breadth First Search

Busca em largura.

Explora os nós vizinhos primeiro.

---

## DFS — Depth First Search

Busca em profundidade.

Explora caminhos mais profundos antes de retornar.

---

# 🔗 Grafo de Correlação

O sistema cria um grafo baseado na correlação entre as variáveis do dataset.

Se duas variáveis possuem correlação maior que:

```python
0.75
```

Elas são conectadas no grafo.

---

# 📊 Avaliação Final

Após o Algoritmo Genético encontrar os melhores hiperparâmetros:

- O melhor modelo é treinado
- O modelo é salvo
- O sistema avalia no conjunto de teste final

O sistema imprime:

- Accuracy
- Classification Report
- Matriz de Confusão

---

# 📦 Arquivos Gerados

## Modelo treinado:

```bash
mlp_melhor_ag.pkl
```

## Normalizador:

```bash
normalizador_cancer_mama.pkl
```

---

# 📊 Dataset Utilizado

## Breast Cancer Wisconsin Dataset

Dataset disponível na biblioteca Scikit-learn contendo:

- 569 amostras
- 30 características numéricas
- Classificação:
  - Benigno
  - Maligno

---

# ▶️ Como Executar

## 1️⃣ Clone o repositório

```bash
git clone https://github.com/seuusuario/projeto-ia-cancer.git
```

---

## 2️⃣ Entre na pasta

```bash
cd projeto-ia-cancer
```

---

## 3️⃣ Instale as dependências

```bash
pip install -r requirements.txt
```

---

# ▶️ Executar Trabalho N1

## Carregar dados

```bash
python TrabalhoN1/carregamentoDeDados.py
```

## Treinar rede neural

```bash
python TrabalhoN1/modelo.py
```

## Avaliar modelo

```bash
python TrabalhoN1/avaliacao.py
```

## Executar inferência

```bash
python TrabalhoN1/inferencia.py
```

---

# ▶️ Executar Trabalho Final

```bash
python TrabalhoFinal/BFS_DFS_AG.py
```

---

# 🧠 Conceitos Aplicados

O projeto utiliza conceitos de:

- Inteligência Artificial
- Machine Learning
- Redes Neurais Artificiais
- Algoritmos Genéticos
- Busca em Grafos
- BFS
- DFS
- Cross Validation
- Classificação Supervisionada
- Pré-processamento de Dados

---

# 🔮 Melhorias Futuras

Possíveis melhorias:

- Interface gráfica
- API REST
- Upload real de imagens médicas
- Deep Learning com CNN
- Dashboard Web
- TensorFlow/Keras
- Deploy em nuvem

---

# 👨‍💻 Autor

Projeto desenvolvido por Rafael Alvarenga para fins acadêmicos utilizando técnicas de Inteligência Artificial e Machine Learning aplicadas à área da saúde.

---