# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    data_pipeline.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mdouglas <mdouglas@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/11/13 10:58:17 by mdouglas          #+#    #+#              #
#    Updated: 2025/11/13 13:52:51 by mdouglas         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import re
import os
import matplotlib.pyplot as plt

GREEN = '\033[92m'
ENDC = '\033[0m'

def	load_dataset(file_path, encoding='latin-1'):
	"""load dataset from a CSV file."""
	try:
		df = pd.read_csv(file_path, encoding=encoding)
		print(f"{GREEN}✅ Dataset load successfully!{ENDC}")
		return (df)
	except FileNotFoundError:
		print("❌ File not found. Please check the file path. {file_path}")
		return (None)
	
def	rename_and_map_labels(df):
	"""Rename columns and map labels to binary values."""
	df = df.rename(columns={'v1': 'label', 'v2': 'message'})
	df = df[['label', 'message']]
	df['label'] = df['label'].map({'ham': 0, 'spam': 1})
	return (df)

def	clean_text(text):
	"""Clean text by removing special characters and converting to lowercase."""
	text = text.lower()
	text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
	return (text)

def	process_message(df):
	"""Apply text cleaning to the 'message' column."""
	df['message'] = df['message'].apply(clean_text)
	return (df)

def plot_class_distribution(df):
    """Plot the distribution of classes in the dataset."""
    class_counts = df['label'].value_counts()
    plt.figure(figsize=(6,4))
    class_counts.plot(kind='bar', color=['blue', 'orange'])
    plt.title('Class Distribution')
    plt.xlabel('Class (0: Ham, 1: Spam)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    path = 'data/visualizations'
    os.makedirs(path, exist_ok=True) 
    file_path = os.path.join(path, 'class_distribution.png')
    plt.savefig(file_path)
    plt.close()
    print(f"{GREEN}✅ Class distribution plot saved to: {file_path}{ENDC}") 

def load_and_preprocess_data(file_path):
	"""Load and preprocess the dataset."""
	print(f"\n{GREEN}--- 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DE DADOS ---{ENDC}")
	
	df = load_dataset(file_path)
	if df is None:
		print("❌ Data loading failed. Exiting preprocessing.")
		return (None)
	
	df = rename_and_map_labels(df)
	df = process_message(df)
	plot_class_distribution(df)
	
	print(f"{GREEN}✅ Data preprocessing completed!{ENDC}")
	return (df)

