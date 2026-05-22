# projeto-ia-cancer
# 🧠 Projeto de IA para Diagnóstico de Câncer de Mama

Este projeto utiliza Inteligência Artificial e Redes Neurais Artificiais para realizar a classificação de câncer de mama utilizando o dataset Breast Cancer Wisconsin da biblioteca Scikit-learn.

O sistema foi desenvolvido em Python e simula um processo de análise médica utilizando Machine Learning para identificar tumores benignos e malignos a partir de características numéricas extraídas de exames.

---

# 📚 Objetivo do Projeto

O objetivo deste projeto é demonstrar na prática conceitos de:

- Inteligência Artificial
- Machine Learning
- Redes Neurais Artificiais
- Pré-processamento de dados
- Normalização
- Avaliação de modelos
- Predição de diagnósticos

Além disso, o projeto simula um fluxo semelhante ao utilizado em sistemas reais de auxílio ao diagnóstico médico.

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

# 🧩 Estrutura do Projeto

```bash
projeto-ia-cancer/
│
├── carregamentoDeDados.py
├── modelo.py
├── avaliacao.py
├── extratorDeFeatures.py
├── previsao.py
├── requirements.txt
├── README.md
├── mlp_cancer_mama.pkl
└── normalizador_cancer_mama.pkl
```

---

# 🔍 Funcionamento do Projeto

## 1️⃣ Carregamento e Pré-processamento dos Dados

O arquivo `carregamentoDeDados.py` é responsável por:

- Carregar o dataset Breast Cancer Wisconsin
- Separar os dados entre treino e teste
- Normalizar os dados usando `StandardScaler`
- Salvar o normalizador em um arquivo `.pkl`

A separação dos dados utiliza:
- 80% para treino
- 20% para teste

O modelo nunca vê os dados de teste durante o treinamento, garantindo uma avaliação mais confiável.

---

## 2️⃣ Construção e Treinamento da Rede Neural

O arquivo `modelo.py` cria e treina a rede neural utilizando o `MLPClassifier`.

### Configurações da Rede Neural

- 1 camada oculta
- 100 neurônios
- Função de ativação ReLU
- Otimizador Adam
- 300 épocas máximas

Durante o treinamento, a rede neural aprende padrões capazes de diferenciar tumores benignos e malignos.

Após o treinamento:
- O modelo é salvo em `mlp_cancer_mama.pkl`

---

# 🧪 Avaliação do Modelo

O arquivo `avaliacao.py` realiza o controle de qualidade do modelo.

Ele gera:

- Acurácia
- Precision
- Recall
- F1-Score
- Matriz de Confusão
- Curva de perda (Loss Curve)

A matriz de confusão mostra:
- Quantos diagnósticos foram corretos
- Quantos erros ocorreram

A curva de perda mostra como a rede neural aprendeu ao longo do treinamento.

---

# 🖼️ Simulação de Extração de Features

O arquivo `extratorDeFeatures.py` simula o processamento de imagens médicas.

Como imagens reais não estão sendo utilizadas, o sistema:
- Seleciona uma amostra aleatória do dataset
- Simula a extração das 30 features numéricas
- Retorna os dados para previsão

Essa etapa representa um sistema real de visão computacional.

---

# 🩺 Sistema de Predição

O arquivo `previsao.py` realiza previsões utilizando o modelo treinado.

O sistema:

1. Carrega o modelo treinado
2. Carrega o normalizador
3. Recebe as features extraídas
4. Normaliza os dados
5. Realiza a previsão
6. Exibe:
   - Diagnóstico final
   - Probabilidade
   - Confiança do modelo

Exemplo:

```bash
DIAGNÓSTICO DA AMOSTRA: BENIGN
Confiança do Modelo: 98.75%
```

---

# 📊 Dataset Utilizado

O projeto utiliza o dataset:

## Breast Cancer Wisconsin Dataset

Disponível através da biblioteca Scikit-learn.

O dataset contém:
- 569 amostras
- 30 características numéricas
- Classificação:
  - Benigno
  - Maligno

---

# 🚀 Como Executar o Projeto

## 1️⃣ Clone o repositório

```bash
git clone https://github.com/seuusuario/projeto-ia-cancer.git
```

---

## 2️⃣ Entre na pasta do projeto

```bash
cd projeto-ia-cancer
```

---

## 3️⃣ Instale as dependências

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Execute o carregamento dos dados

```bash
python carregamentoDeDados.py
```

---

## 5️⃣ Treine o modelo

```bash
python modelo.py
```

---

## 6️⃣ Avalie o modelo

```bash
python avaliacao.py
```

---

## 7️⃣ Execute previsões

```bash
python previsao.py
```

---

# 📦 Dependências

Exemplo do arquivo `requirements.txt`:

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
opencv-python
joblib
```

---

# 🧠 Conceitos Aplicados

Este projeto aplica conceitos importantes de:

- Machine Learning
- Redes Neurais Artificiais
- Classificação Supervisionada
- Pré-processamento de Dados
- Normalização
- Avaliação de Modelos
- Inteligência Artificial aplicada à saúde

---

# 🔮 Melhorias Futuras

Possíveis melhorias para o projeto:

- Interface gráfica
- API REST
- Upload real de imagens médicas
- Integração com banco de dados
- Dashboard web
- TensorFlow/Keras
- Deep Learning com CNN
- Deploy em nuvem

---

# 👨‍💻 Autor

Projeto desenvolvido por Rafael Alvarenga utilizando Python e técnicas de Inteligência Artificial para fins acadêmicos e educacionais.

---
