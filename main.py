# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mdouglas <mdouglas@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/11/13 13:03:49 by mdouglas          #+#    #+#              #
#    Updated: 2025/11/13 14:28:37 by mdouglas         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
from src.data_pipeline import load_and_preprocess_data
from src.model_pipeline import train_and_evaluate_pipeline, evaluate_model

DATA_PATH = './data/spam.csv' 

def run_pipeline():
    """ execute the full data and model pipeline """

    df_clean = load_and_preprocess_data(DATA_PATH)
    
    if df_clean is not None:

        model, X_test_vec, y_test = train_and_evaluate_pipeline(df_clean)
        evaluate_model(model, X_test_vec, y_test)
        print("\nProcesso de Treinamento concluído com sucesso!")

if __name__ == '__main__':
    run_pipeline()