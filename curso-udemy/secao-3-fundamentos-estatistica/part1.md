## 1. O que são os Dados em ML?

No contexto de Python (usando bibliotecas como Pandas e Scikit-Learn), os dados geralmente são organizados em estruturas tabulares.

* **Features (Atributos/Variáveis Independentes):** São as propriedades ou características que o modelo usa para aprender. No Python, costumamos chamar esse conjunto de $X$.
* **Target (Alvo/Variável Dependente):** É o que você quer prever. Chamamos de $y$.
* **Observações (Linhas):** Cada linha representa um exemplo individual (um cliente, um sensor, uma transação).

### Tipos de Dados

1. **Numéricos:** Podem ser contínuos (preço de uma casa) ou discretos (quantidade de filhos).
2. **Categóricos:** Representam categorias. Podem ser nominais (cores, cidades) ou ordinais (nível de escolaridade: básico, médio, superior).

---

### 2. A Importância da Amostragem (Sampling)

Dificilmente você trabalhará com a "população" inteira (todos os dados possíveis do universo). Por isso, trabalhamos com uma **amostra**. O objetivo da amostragem é garantir que essa pequena parte seja uma representação fiel do todo.

### Técnicas Comuns de Amostragem

* **Amostragem Aleatória Simples:** Cada item tem a mesma chance de ser escolhido. É o `df.sample()` do Pandas.
* **Amostragem Estratificada:** Essencial quando você tem classes desbalanceadas. Se 90% dos seus dados são de "clientes ativos" e 10% de "churn", a amostra deve manter essa mesma proporção para não enviesar o modelo.
* **Over-sampling e Under-sampling:** Técnicas usadas quando uma classe é muito rara (ex: detecção de fraudes). Você "cria" novos dados da classe minoritária ou remove dados da majoritária.

---

### 3. Divisão de Dados: Treino, Validação e Teste

Para saber se o seu modelo realmente aprendeu (e não apenas decorou), dividimos os dados em partes distintas:

1. **Conjunto de Treino:** Usado para o algoritmo aprender os padrões. Geralmente 70-80% dos dados.
2. **Conjunto de Validação:** Usado para ajustar os "hiperparâmetros" do modelo (ajuste fino).
3. **Conjunto de Teste:** O "exame final". São dados que o modelo nunca viu. Serve para medir a performance real no mundo real.

> **Dica de Python:** A função mais famosa para isso é a `train_test_split` do Scikit-Learn:
> ```python
> from sklearn.model_selection import train_test_split
> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
> 
> ```
> 
> 

---

### 4. O perigo do Viés de Amostragem

Se você treinar um modelo de reconhecimento facial usando apenas fotos de pessoas de pele clara, o modelo terá um desempenho pífio com pessoas de pele escura. Isso é um **viés de amostragem**.

Em Machine Learning, a diversidade e a qualidade da amostra são mais importantes do que a quantidade bruta de dados.

---

## Distribuição de Frequências

A **Distribuição de Frequências** é uma das ferramentas mais fundamentais da estatística descritiva e um passo crucial no **Exploratory Data Analysis (EDA)** em Machine Learning. Ela serve para organizar e resumir um conjunto de dados brutos, permitindo visualizar como os valores estão espalhados.

Imagine que você coletou a idade de 100 usuários. Olhar a lista de números soltos não ajuda muito; a distribuição de frequências agrupa esses números para revelar padrões.

---

### 1. Componentes Principais

Para construir uma distribuição, trabalhamos com quatro tipos de frequências:

* **Frequência Absoluta ($f_i$):** É o número de vezes que um valor específico (ou uma classe de valores) aparece no conjunto de dados.
* **Frequência Relativa ($f_r$):** É a proporção de cada valor em relação ao total. Calculada como $f_r = \frac{f_i}{n}$, onde $n$ é o tamanho da amostra. Geralmente expressa em porcentagem.
* **Frequência Acumulada ($F_i$):** É a soma das frequências de todos os valores anteriores ao valor atual. Ajuda a entender quantos dados estão "abaixo de" determinado ponto.
* **Frequência Relativa Acumulada ($F_r$):** A porcentagem acumulada até aquele ponto.

---

### 2. Tipos de Distribuição

A forma como você organiza a distribuição depende da natureza dos seus dados:

### A. Para Dados Discretos ou Categóricos

Se você tem poucos valores distintos (ex: "Gênero" ou "Número de Filhos"), você cria uma tabela onde cada linha é um valor único.

### B. Para Dados Contínuos (Distribuição por Classes)

Se você tem dados como "Salário" ou "Altura", onde os valores variam muito, agrupamos os dados em **intervalos (classes)**.

* **Exemplo:** De R$ 2.000 a R$ 3.000, de R$ 3.000 a R$ 4.000, e assim por diante.

---

### 3. Visualização: O Histograma

Em Python (usando `matplotlib` ou `seaborn`), a forma visual da distribuição de frequências é o **Histograma**. Ele transforma a tabela de frequências em barras onde a área de cada barra representa a frequência daquela classe.

Ao analisar o histograma em ML, você busca identificar:

* **Simetria:** Os dados estão centralizados (Distribuição Normal) ou "puxados" para um lado (Assimetria/Skewness)?
* **Outliers:** Existem valores isolados muito longe da massa principal de dados?
* **Picos (Modalidade):** Existe apenas um pico (unimodal) ou os dados se agrupam em dois lugares diferentes (bimodal)?

---

### 4. Por que isso importa para Machine Learning?

Muitos algoritmos (como Regressão Linear e Naive Bayes) assumem que seus dados seguem uma **Distribuição Normal** (em formato de sino).

Se a sua distribuição de frequências mostrar que os dados estão muito "tortos" (assimetria severa), você precisará aplicar transformações matemáticas (como Logaritmo) para que o modelo consiga aprender melhor.

> **Dica de Python:** No Pandas, você gera a frequência absoluta rapidamente com o comando:
> ```python
> df['coluna'].value_counts()
> 
> ```
> 
> 
> Ou para ver a distribuição visual:
> ```python
> df['coluna'].hist(bins=10)
> 
> ```
> 
> 

## Medidas de Tendência Central
As **Medidas de Tendência Central** são valores que resumem um conjunto de dados, encontrando um "ponto central" ou um valor típico em torno do qual os outros dados se distribuem. Em Machine Learning, elas são o primeiro passo para entender a magnitude dos seus dados e para tratar valores ausentes (*Imputation*).

As três principais medidas são a **Média**, a **Mediana** e a **Moda**.

---

### 1. Média (Mean)

É a medida mais comum, calculada somando todos os valores e dividindo pelo número total de observações ($n$).

$$\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}$$

* **Vantagem:** Utiliza todos os dados da amostra.
* **Desvantagem:** É extremamente sensível a **Outliers** (valores muito fora do padrão). Se você tem 4 pessoas que ganham R$ 2.000 e uma que ganha R$ 100.000, a média será alta, mas não representará bem o grupo.

---

### 2. Mediana (Median)

É o valor que ocupa a posição central dos dados quando eles estão ordenados (do menor para o maior).

* Se o número de observações for **ímpar**, a mediana é o termo do meio.
* Se for **par**, é a média dos dois termos centrais.

**Por que usar?** Diferente da média, a mediana é **robusta a outliers**. No exemplo dos salários acima, a mediana continuaria sendo R$ 2.000, refletindo melhor a realidade da maioria.

---

### 3. Moda (Mode)

É o valor que aparece com maior frequência no conjunto de dados.

* Um conjunto pode ser **unimodal** (uma moda), **bimodal** (duas modas) ou **amodal** (nenhuma repetição).
* **Uso em ML:** É a principal medida para dados **categóricos**. Se você precisa preencher um valor faltante em uma coluna de "Cidades", você usa a Moda (a cidade que mais aparece).

---

### Comparação Visual e Distribuição

A relação entre essas três medidas revela muito sobre a **Assimetria (Skewness)** dos seus dados:

* **Simétrica (Normal):** Média = Mediana = Moda.
* **Assimetria à Direita (Positiva):** Moda < Mediana < Média (os outliers "puxam" a média para cima).
* **Assimetria à Esquerda (Negativa):** Média < Mediana < Moda (os outliers puxam a média para baixo).

---

### Aplicação Prática em Python

No dia a dia de um desenvolvedor Python, o Pandas facilita muito esse cálculo:

```python
import pandas as pd

# Criando um exemplo rápido
data = pd.Series([10, 20, 20, 30, 1000]) # O 1000 é um outlier

print(f"Média: {data.mean()}")     # Resultado: 216.0 (distorcida pelo 1000)
print(f"Mediana: {data.median()}") # Resultado: 20.0 (mais realista)
print(f"Moda: {data.mode()[0]}")   # Resultado: 20

```

### Quando usar cada uma em Machine Learning?

1. **Média:** Quando os dados são simétricos e não possuem outliers agressivos.
2. **Mediana:** Quando há muitos outliers ou os dados são muito assimétricos.
3. **Moda:** Para variáveis qualitativas (texto/categorias).

---

### Próximo Passo

Como você está estudando para ML, gostaria de ver como as **Medidas de Dispersão** (como Desvio Padrão e Variância) complementam essas medidas de tendência central? Elas explicam o "quão longe" os dados estão desse centro.
