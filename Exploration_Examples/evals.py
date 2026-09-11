# EVALUATION ALGORITHMS FOR RANKING MODELS

import numpy as np
import pandas as pd
from scipy.optimize import minimize

#===============================================================================
# Calculate Positional Penalties Using GMM for Model Performance per Position
#===============================================================================

def calculate_gt_penalties(df_matches, items, true_ranking):
    n = len(items)
    item_to_idx = {item: i for i, item in enumerate(items)}
    P = np.zeros((n, n))

    for _, row in df_matches.iterrows():
        P[item_to_idx[row['Winner']], item_to_idx[row['Loser']]] += 1

    # Map the true ranking items to their matrix indices.
    true_centre = [item_to_idx[item] for item in true_ranking]

    # Optimise theta_j using the ground truth centre.
    def nll_theta(theta_array):
        nll = 0
        for i in range(n - 1):
            V_i = 0
            for j in range(i + 1, n):
                V_i += P[true_centre[j], true_centre[i]]

            theta_i = theta_array[i]

            if theta_i < 1e-5:
                log_Z_i = np.log(n - i)
            else:
                log_Z_i = np.log(1 - np.exp(-(n - i) * theta_i)) - np.log(1 - np.exp(-theta_i))

            nll += (theta_i * V_i) + log_Z_i
        return nll

    initial_guess = np.ones(n - 1)
    bounds = [(1e-3, None) for _ in range(n - 1)]

    result = minimize(nll_theta, initial_guess, bounds=bounds)
    theta_j_gt = result.x

    return theta_j_gt


#===============================================================================
# Permutation Distance Per Model from the Ground Truth Using MM and GMM
#===============================================================================

def evaluate_mm_penalty(true_ranking, model_rankings):
    """
    Calculates the total Mallows Model (MM) penalty (raw distance) 
    for each model compared to the ground truth ranking.
    """
    n = len(true_ranking)
    mm_penalties = {}
    
    for model_name, model_rank in model_rankings.items():
        # Map each item to its index in the model's ranking
        model_idx = {item: idx for idx, item in enumerate(model_rank)}
        total_distance = 0
        
        # Calculate raw errors (Kendall tau distance)
        for i in range(n - 1):
            for j in range(i + 1, n):
                item_higher_true = true_ranking[i]
                item_lower_true = true_ranking[j]
                
                # If the true lower item is ranked above the true higher item by the model
                if model_idx[item_lower_true] < model_idx[item_higher_true]:
                    total_distance += 1
                    
        mm_penalties[model_name] = total_distance
        
    for model, penalty in mm_penalties.items():
        print(f"{model}: {penalty}")
        
    return mm_penalties


def evaluate_gmm_positional_penalties(true_ranking, model_rankings, theta_j):
    """
    Calculates the vector of GMM penalties for each position 
    for each model compared to the ground truth ranking.
    """
    n = len(true_ranking)
    gmm_vectors = {}
    
    for model_name, model_rank in model_rankings.items():
        model_idx = {item: idx for idx, item in enumerate(model_rank)}
        penalties = np.zeros(n) 
        
        # Calculate errors based on the ground truth centre's positions
        for i in range(n - 1):
            v_i = 0
            for j in range(i + 1, n):
                item_higher_true = true_ranking[i]
                item_lower_true = true_ranking[j]
                
                if model_idx[item_lower_true] < model_idx[item_higher_true]:
                    v_i += 1
            
            # Multiply raw errors by theta to get the specific cost at this position
            penalties[i] = v_i * theta_j[i]
            
        # Store as a rounded list/vector
        gmm_vectors[model_name] = list(np.round(penalties, 4))
        
    # Format cleanly into a DataFrame for easy viewing
    df_gmm_positional = pd.DataFrame(gmm_vectors, index=[f"Rank_{i+1}" for i in range(n)]).T
    
    print(df_gmm_positional.to_string())
    
    return gmm_vectors, df_gmm_positional