# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    app.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mdouglas <mdouglas@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/11/12 12:46:38 by mdouglas          #+#    #+#              #
#    Updated: 2025/11/12 13:23:58 by mdouglas         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

# === 1. Carregar dados ===
df = pd.read_csv('data/spam.csv', encoding='latin-1')
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df = df[['label', 'message']]

# Converter labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# === 2. Dividir dados ===
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# === 3. Vetorização ===
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === 4. Modelo ===
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# === 5. Predição ===
y_pred = model.predict(X_test_vec)

# === 6. Avaliação ===
print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))



# # === 7. Visualização da Matriz de Confusão ===
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Classificação de SMS')
plt.savefig('confusion_matrix.png')
plt.show()
