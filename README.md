## 📧 Classificação de SMS: Detecção de Spam com Naive Bayes

Este projeto implementa um classificador de SMS para distinguir entre mensagens **"ham"** (legítimas) e **"spam"** (indesejadas) usando o algoritmo **Naive Bayes Multinomial** e vetorização **TF-IDF**.

---

## 💻 Instalação

Para configurar o ambiente, utilize o arquivo `requirements.txt` fornecido.

1.  **Instale as dependências** usando `pip`:
    ```bash
    pip install -r requirements.txt
    ```
2.  As principais bibliotecas instaladas são: `pandas`, `scikit-learn`, `numpy`, `matplotlib`, e `seaborn`.

---

## 🛠️ Execução do Projeto (`app.py`)

O script `app.py` executa o fluxo completo do modelo de Machine Learning:

### Etapas do Script

* **Carregamento e Pré-processamento de Dados:**
    * Carrega o dataset, mapeando os rótulos **'ham'** para **0** e **'spam'** para **1**.
    * Aplica um pré-processamento simples nas mensagens (minúsculas, remoção de caracteres não alfabéticos).
* **Divisão em Treinamento e Teste:**
    * Divide o dataset em conjuntos de treino e teste (`test_size=0.2`).
* **Vetorização (TF-IDF):**
    * Utiliza `TfidfVectorizer` (com `stop_words='english'`) para converter as mensagens de texto em vetores numéricos.
* **Treinamento e Predição:**
    * Um modelo **`MultinomialNB`** é treinado e usado para fazer predições no conjunto de teste.
* **Avaliação:**
    * Calcula e exibe a **Acurácia**, **Relatório de Classificação** e **Matriz de Confusão**.

---

## 📈 Resultados e Avaliação

A matriz de confusão salva em `confusion_matrix.png` detalha o desempenho do classificador:

| Real/Predito | 0 (Ham Predito) | 1 (Spam Predito) |
| :----------: | :-------------: | :--------------: |
| **0 (Ham Real)** | 965 (**Verdadeiro Negativo**) | 0 (**Falso Positivo**) |
| **1 (Spam Real)** | 37 (**Falso Negativo**) | 113 (**Verdadeiro Positivo**) |

* O modelo alcançou 965 Verdadeiros Negativos (ham corretamente classificado) e **zero Falsos Positivos** (nenhum ham classificado como spam).
* Houve **37 Falsos Negativos** (spam classificado como ham) e 113 Verdadeiros Positivos (spam corretamente classificado).