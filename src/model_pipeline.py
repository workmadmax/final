# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    model_pipeline.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mdouglas <mdouglas@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/11/13 13:01:40 by mdouglas          #+#    #+#              #
#    Updated: 2025/11/13 14:25:17 by mdouglas         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

ENDC = '\033[0m'
BLUE = '\033[94m'

# 1️⃣ data splitting

def split_data(df, test_size=0.2, random_state=42):
	"""Split the dataset into training and testing sets."""
	X = df['message']
	y = df['label']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
	print(f"{BLUE}✅ Data split into training and testing sets!{ENDC}")
	return (X_train, X_test, y_train, y_test)

# 2️⃣ vectorization

def vectorize_text(X_train, X_test):
	"""Vectorize text data using TF-IDF."""
	vectorizer = TfidfVectorizer()
	X_train_vec = vectorizer.fit_transform(X_train)
	X_test_vec = vectorizer.transform(X_test)
	print(f"{BLUE}✅ Text data vectorized using TF-IDF!{ENDC}")
	return (vectorizer, X_train_vec, X_test_vec)

# 3️⃣ model training

def train_model(X_train_vec, y_train):
	"""Train a Naive Bayes model."""
	model = MultinomialNB()
	model.fit(X_train_vec, y_train)
	print(f"{BLUE}✅ Model trained successfully!{ENDC}")
	return (model)

# 4️⃣ save artifacts

def save_artifacts(model, vectorizer, model_dir='models'):
	"""Save the trained model and vectorizer to disk."""
	os.makedirs(model_dir, exist_ok=True)
	joblib.dump(model, os.path.join(model_dir, 'naive_bayes_model.pkl'))
	joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
	print(f"{BLUE}💾 Model and vectorizer saved to disk {model_dir}!{ENDC}")

# 5️⃣ model evaluation

def evaluate_model(model, X_test_vec, y_test):
    """Evaluate the model and return accuracy, classification report, confusion matrix, and predictions."""
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)
    return (acc, report, cm, y_pred)

# 6️⃣ plot confusion matrix

def plot_confusion_matrix(cm, save_path='./data/visualizations/confusion_matrix.png'):
    """Plot and save the confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['HAM (0)', 'SPAM (1)'],
                yticklabels=['HAM (0)', 'SPAM (1)'])
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão - Classificação de SMS')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Matriz de confusão salva em: {save_path}")

# 7️⃣ full pipeline

def train_and_evaluate_pipeline(df):
    """Orquestra todo o processo: treino, avaliação e salvamento."""
    print(f"\n{BLUE}--- 2. TREINAMENTO E AVALIAÇÃO DO MODELO ---{ENDC}")

    X_train, X_test, y_train, y_test = split_data(df)
    vectorizer, X_train_vec, X_test_vec = vectorize_text(X_train, X_test)
    model = train_model(X_train_vec, y_train)
    save_artifacts(model, vectorizer)

    acc, report, cm, _ = evaluate_model(model, X_test_vec, y_test)

    print(f"\nAcurácia: {acc:.4f}")
    print("\nRelatório de Classificação:\n", report)
    plot_confusion_matrix(cm)

    print("✅ Treinamento e avaliação concluídos.")
    return (model, X_test_vec, y_test)