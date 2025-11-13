# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mdouglas <mdouglas@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/11/13 13:03:49 by mdouglas          #+#    #+#              #
#    Updated: 2025/11/13 13:47:46 by mdouglas         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
from src.data_pipeline import load_and_preprocess_data
from src.model_pipeline import train_and_save_model, evaluate_and_visualize

DATA_PATH = './data/spam.csv' 

def run_pipeline():
    """Executa o pipeline completo de ML."""
    
    # 1. Prepare directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # 2. load
    df_clean = load_and_preprocess_data(DATA_PATH)
    
    if df_clean is not None:

        # Retorna X_test_vec e y_test para avaliação.
        model, X_test_vec, y_test = train_and_save_model(df_clean)
        
        # 4. Avaliar e Visualizar
        evaluate_and_visualize(model, X_test_vec, y_test)
        print("\nProcesso de Treinamento concluído com sucesso!")

if __name__ == '__main__':
    run_pipeline()