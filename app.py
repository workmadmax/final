# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    app.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mdouglas <mdouglas@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/11/12 12:46:38 by mdouglas          #+#    #+#              #
#    Updated: 2025/11/12 12:59:57 by mdouglas         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd


df = pd.read_csv('./data/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df = df.rename(columns={'v1': 'label', 'v2': 'message'})

print('\nContagem de Rótulos:')
print(df['label'].value_counts())

print('\nAmostra de Mensagens:')
print(df.sample(10))
print('\nInformações do DataFrame:')

print('--------------------------------------------')

# verifica as primeiras linhas do DataFrame
print(df.head())

# Verifica o formato das colunas
print(df.columns)