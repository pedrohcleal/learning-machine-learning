# 1. Dados em Machine Learning

Os **dados** são observações coletadas de algum fenômeno. Em Python, normalmente são manipulados com bibliotecas como **Pandas**, **NumPy** e **Scikit-learn**.

### Estrutura típica de dados

Normalmente um dataset tem:

* **Features (X)** → variáveis de entrada
* **Target (y)** → variável que queremos prever

Exemplo simples:

| Idade | Salário | Comprou Produto |
| ----- | ------- | --------------- |
| 25    | 3000    | 0               |
| 35    | 7000    | 1               |
| 40    | 9000    | 1               |

* **Idade e salário** → features
* **Comprou produto** → target

Em Python:

```python
import pandas as pd

df = pd.read_csv("dados.csv")

X = df[["idade", "salario"]]
y = df["comprou"]
```

---

# 2. Tipos de Dados

### 1️⃣ Dados numéricos

Valores quantitativos.

Exemplo:

* idade
* preço
* temperatura

### 2️⃣ Dados categóricos

Representam categorias.

Exemplo:

* cor (azul, vermelho)
* país
* tipo de produto

Normalmente são convertidos para números usando:

* **One-hot encoding**
* Label encoding

---

# 3. Dataset e População

Um conceito estatístico importante.

### População

Todos os dados possíveis.

Exemplo:

* todos os clientes de uma empresa

### Amostra

Subconjunto da população usado para análise.

Exemplo:

* 10 mil clientes selecionados para treinar o modelo

Treinar com toda a população quase nunca é possível (dados enormes ou impossíveis de coletar).

---

# 4. Amostragem (Sampling)

A **amostragem** consiste em selecionar um subconjunto representativo dos dados.

Isso é importante para:

* reduzir custo computacional
* evitar viés
* melhorar generalização

---

# 5. Tipos de Amostragem

### 1️⃣ Amostragem Aleatória (Random Sampling)

Seleciona dados aleatoriamente.

Exemplo:

```python
sample = df.sample(n=1000)
```

Vantagens:

* simples
* reduz viés

---

### 2️⃣ Amostragem Estratificada

Mantém a proporção das classes.

Muito usada em **problemas de classificação**.

Exemplo com **Scikit-learn**:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=y
)
```

Se o dataset tem:

* 90% classe A
* 10% classe B

A amostra mantém essa proporção.

---

### 3️⃣ Amostragem Sistemática

Seleciona dados em intervalos regulares.

Exemplo:

* pegar 1 a cada 10 registros

---

### 4️⃣ Amostragem por Cluster

Divide os dados em grupos (clusters) e seleciona alguns grupos inteiros.

Muito usada em pesquisas populacionais.

---

# 6. Divisão do Dataset

Em Machine Learning, os dados normalmente são divididos em:

### Treino

Usado para o modelo aprender.

### Teste

Usado para avaliar o modelo.

Divisão comum:

* 80% treino
* 20% teste

Exemplo:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

---

# 7. Problemas de Amostragem

### Overfitting

Modelo aprende demais os dados de treino.

Resultado:

* funciona bem no treino
* ruim no mundo real

### Underfitting

Modelo simples demais.

Resultado:

* não aprende padrões suficientes.

---

# 8. Conceitos importantes relacionados

### Balanceamento de dados

Datasets podem ser desbalanceados.

Exemplo:

* 99% fraude = não
* 1% fraude = sim

Técnicas:

* **Oversampling**
* **Undersampling**
* **SMOTE**

---

# 9. Exemplo completo simples

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dados.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

✅ **Resumo**

| Conceito   | Explicação                              |
| ---------- | --------------------------------------- |
| Dados      | Observações usadas para treinar modelos |
| Features   | Variáveis de entrada                    |
| Target     | Variável a ser prevista                 |
| População  | Conjunto completo de dados              |
| Amostra    | Subconjunto da população                |
| Amostragem | Método de seleção da amostra            |
| Train/Test | Divisão para treino e avaliação         |



A **distribuição de frequências** é um conceito fundamental da **Estatística** e muito usado em **Machine Learning** para **entender como os dados estão distribuídos** dentro de um dataset.

Basicamente, ela mostra **quantas vezes cada valor (ou intervalo de valores) aparece em um conjunto de dados**.

---

# 1. O que é Distribuição de Frequências

Uma **distribuição de frequência** organiza dados para mostrar **a quantidade de ocorrências de cada valor**.

Exemplo de dados (idades):

```
18, 19, 20, 20, 21, 21, 21, 22
```

Distribuição:

| Idade | Frequência |
| ----- | ---------- |
| 18    | 1          |
| 19    | 1          |
| 20    | 2          |
| 21    | 3          |
| 22    | 1          |

Isso permite visualizar **qual valor aparece mais**.

---

# 2. Tipos de Frequência

## Frequência Absoluta

Número de vezes que um valor aparece.

Exemplo:

| Valor | Frequência Absoluta |
| ----- | ------------------- |
| 20    | 2                   |
| 21    | 3                   |

---

## Frequência Relativa

Proporção em relação ao total.

Fórmula:

[
f_r = \frac{f_i}{n}
]

Onde:

* (f_i) = frequência absoluta
* (n) = total de observações

Exemplo:

| Valor | Frequência | Frequência Relativa |
| ----- | ---------- | ------------------- |
| 21    | 3          | 3/8 = 0.375         |

---

## Frequência Percentual

A frequência relativa em porcentagem.

[
\text{Percentual} = f_r \times 100
]

Exemplo:

```
0.375 × 100 = 37.5%
```

---

## Frequência Acumulada

Mostra **a soma progressiva das frequências**.

| Idade | Frequência | Acumulada |
| ----- | ---------- | --------- |
| 18    | 1          | 1         |
| 19    | 1          | 2         |
| 20    | 2          | 4         |
| 21    | 3          | 7         |
| 22    | 1          | 8         |

---

# 3. Distribuição de Frequência para Dados Contínuos

Quando há muitos valores diferentes (ex: salários), agrupamos em **classes ou intervalos**.

Exemplo:

| Salário   | Frequência |
| --------- | ---------- |
| 1000–2000 | 5          |
| 2000–3000 | 8          |
| 3000–4000 | 12         |
| 4000–5000 | 4          |

Esses intervalos são chamados de **classes**.

---

# 4. Distribuição de Frequências em Python

Com **Pandas** é simples calcular.

### Frequência absoluta

```python
import pandas as pd

df["idade"].value_counts()
```

Resultado:

```
21    3
20    2
18    1
19    1
22    1
```

---

### Frequência relativa

```python
df["idade"].value_counts(normalize=True)
```

---

### Frequência por intervalos

```python
pd.cut(df["salario"], bins=5).value_counts()
```

Isso cria **classes automaticamente**.

---

# 5. Visualização da Distribuição

Uma das formas mais usadas é o **histograma**.

Histograma mostra **como os dados estão espalhados**.

Exemplo com **Matplotlib**:

```python
import matplotlib.pyplot as plt

plt.hist(df["salario"], bins=10)
plt.show()
```

---

# 6. Importância em Machine Learning

A distribuição de frequência ajuda a identificar:

### 1️⃣ Desbalanceamento de classes

Exemplo:

| Classe     | Frequência |
| ---------- | ---------- |
| Não fraude | 990        |
| Fraude     | 10         |

Isso pode prejudicar o modelo.

---

### 2️⃣ Outliers

Valores muito raros ou extremos.

Exemplo:

```
[10, 11, 12, 13, 500]
```

500 é um **outlier**.

---

### 3️⃣ Distribuição dos dados

Pode revelar padrões como:

* distribuição normal
* dados enviesados
* cauda longa

Essas características influenciam algoritmos de ML.

---

# 7. Exemplo prático em Machine Learning

Antes de treinar um modelo com **Scikit-learn**, normalmente analisamos a distribuição:

```python
df["target"].value_counts()
```

Isso ajuda a decidir se precisamos:

* balancear os dados
* remover outliers
* transformar variáveis

---

✅ **Resumo**

| Conceito              | Explicação                               |
| --------------------- | ---------------------------------------- |
| Frequência            | Quantidade de vezes que um valor aparece |
| Frequência absoluta   | Número de ocorrências                    |
| Frequência relativa   | Proporção no dataset                     |
| Frequência percentual | Frequência em %                          |
| Frequência acumulada  | Soma progressiva                         |
| Classes               | Intervalos usados para dados contínuos   |

As **medidas de tendência central** são conceitos fundamentais da **Estatística** e muito usadas em **Machine Learning** para **resumir um conjunto de dados em um único valor representativo**.

Elas indicam **onde os dados tendem a se concentrar**.

As três principais são:

* **Média**
* **Mediana**
* **Moda**

---

# 1. Média (Mean)

A **média aritmética** é a soma de todos os valores dividida pela quantidade de observações.

### Fórmula

[
\bar{x} = \frac{\sum x_i}{n}
]

onde:

* (x_i) = valores do conjunto de dados
* (n) = número de observações

### Exemplo

Dados:

```
10, 15, 20, 25
```

Cálculo:

```
(10 + 15 + 20 + 25) / 4 = 17.5
```

### Em Python

Usando **NumPy**:

```python
import numpy as np

dados = [10, 15, 20, 25]
media = np.mean(dados)
print(media)
```

### Característica importante

A média **é sensível a outliers**.

Exemplo:

```
10, 15, 20, 25, 500
```

Média:

```
114
```

O valor **500 distorce totalmente a média**.

---

# 2. Mediana (Median)

A **mediana** é o **valor central** de um conjunto de dados ordenado.

### Exemplo

Dados ordenados:

```
10, 15, 20, 25, 30
```

Mediana:

```
20
```

Se houver número par de elementos:

```
10, 15, 20, 25
```

Mediana:

```
(15 + 20) / 2 = 17.5
```

### Em Python

```python
import numpy as np

np.median([10,15,20,25])
```

### Característica

A mediana **não é afetada por outliers**.

Exemplo:

```
10, 15, 20, 25, 500
```

Mediana:

```
20
```

---

# 3. Moda (Mode)

A **moda** é o valor que **aparece com maior frequência**.

### Exemplo

```
2, 3, 3, 4, 5, 5, 5
```

Moda:

```
5
```

### Em Python

Usando **Pandas**:

```python
import pandas as pd

dados = pd.Series([2,3,3,4,5,5,5])
dados.mode()
```

---

# 4. Comparação das Medidas

| Medida  | Definição                     | Sensível a Outliers |
| ------- | ----------------------------- | ------------------- |
| Média   | Soma dos valores / quantidade | Sim                 |
| Mediana | Valor central                 | Não                 |
| Moda    | Valor mais frequente          | Não                 |

---

# 5. Relação com Distribuição dos Dados

Dependendo da **distribuição dos dados**, essas medidas podem coincidir ou não.

### Distribuição simétrica

Exemplo de **Distribuição Normal**:

```
média = mediana = moda
```

---

### Distribuição assimétrica (skewed)

Quando há uma **cauda longa**:

```
moda < mediana < média
```

ou

```
média < mediana < moda
```

Isso ajuda a entender **a forma da distribuição**.

---

# 6. Uso em Machine Learning

As medidas de tendência central são usadas para:

### 1️⃣ Análise exploratória de dados

Antes de treinar modelos.

### 2️⃣ Preenchimento de dados faltantes

Exemplo:

* usar **média** para dados numéricos
* usar **mediana** quando há outliers

Exemplo com **Scikit-learn**:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
```

---

### 3️⃣ Normalização de dados

Alguns algoritmos usam a média para **centralizar os dados**.

Exemplo:

* padronização (Standardization)

---

# 7. Exemplo prático

Dataset de salários:

```
3000, 3200, 3500, 3600, 3700, 15000
```

Resultados:

| Medida  | Valor      |
| ------- | ---------- |
| Média   | 5333       |
| Mediana | 3550       |
| Moda    | não existe |

A **mediana representa melhor o salário típico** porque existe um **outlier (15000)**.

---

✅ **Resumo**

| Medida  | O que representa      |
| ------- | --------------------- |
| Média   | valor médio dos dados |
| Mediana | valor central         |
| Moda    | valor mais frequente  |

---

💡 Em projetos reais de **Machine Learning**, quase todo pipeline começa com:

1. **Exploratory Data Analysis**
2. análise de **distribuição**
3. cálculo de **média, mediana e desvio padrão**

Isso ajuda a entender **como os dados se comportam antes de treinar qualquer modelo**.

---
