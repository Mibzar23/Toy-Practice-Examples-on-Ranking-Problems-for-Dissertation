# SYNTHETICA DATA GENERATORS AND SUMMARY FUNCTIONS

import numpy as np
import pandas as pd

#===============================================================================
# Synthetic Data Generator with Bradley-Terry for Pairwise Matchups
#===============================================================================

def bt_syntheticdata_gen(n_items, n_matches, true_betas=None, random_seed=23):
    if random_seed is not None:
        np.random.seed(random_seed)

    # Constraint for total match slots to be even
    if (n_items * n_matches) % 2 != 0:
        raise ValueError("The total number of slots (n_items * n_matches) must be even.")

    # Setup items
    items = [f"Item_{i}" for i in range(1, n_items + 1)]

    if true_betas is None:
        true_betas = np.random.normal(0, 1, size=n_items)
    elif len(true_betas) != n_items:
        raise ValueError("The length of true_betas must match n_items.")

    str_dic = dict(zip(items, true_betas))

    # Create quotas for each item
    stubs = np.repeat(items, n_matches)
    np.random.shuffle(stubs)

    # Pair up adjacent items
    matchups = []
    for i in range(0, len(stubs), 2):
        matchups.append([stubs[i], stubs[i+1]])

    # Fix self-loops
    for i in range(len(matchups)):
        if matchups[i][0] == matchups[i][1]:
            # Find a valid past match to swap with
            for j in range(len(matchups)):
                if (matchups[j][0] != matchups[j][1] and 
                    matchups[j][0] not in matchups[i] and 
                    matchups[j][1] not in matchups[i]):
                    
                    # Swap
                    matchups[i][1], matchups[j][0] = matchups[j][0], matchups[i][1]
                    break

    records = []

    # Assign strength parameter and execute pairwise matches
    for match_id, (p1, p2) in enumerate(matchups, start=1):
        beta1 = str_dic[p1]
        beta2 = str_dic[p2]
        # Bradley-Terry outcome probability
        prob_p1_wins = np.exp(beta1) / (np.exp(beta1) + np.exp(beta2))

        # Outcome condition
        if np.random.rand() < prob_p1_wins:
            winner, loser = p1, p2
        else:
            winner, loser = p2, p1

        # Outputs
        records.append({"Match ID": match_id,
                        "P1": p1,
                        "P2": p2,
                        "Winner": winner,
                        "Loser": loser})

    df_matches = pd.DataFrame(records)
    df_rank_gt = pd.DataFrame({'Item': items, 'GT Beta': true_betas}).sort_values('GT Beta', ascending=False).reset_index(drop=True)

    return df_matches, df_rank_gt, str_dic

#===============================================================================
# Summary of Bradley-Terry Pairwise Matchups
#===============================================================================

def bt_syntheticdata_summary(df_matches, items, str_dic):
    points = df_matches['Winner'].value_counts().reset_index()
    points.columns = ['Item', 'Wins']

    df_bt_summary = pd.DataFrame({'Item': items})
    df_bt_summary = df_bt_summary.merge(points, on='Item', how='left').fillna(0)
    df_bt_summary['Wins'] = df_bt_summary['Wins'].astype(int)

    df_bt_summary['Beta'] = df_bt_summary['Item'].map(str_dic)
    df_bt_summary['Rank'] = df_bt_summary['Beta'].rank(ascending=False, method='min').astype(int)

    df_bt_summary = df_bt_summary[['Item', 'Rank', 'Wins', 'Beta']].sort_values('Rank', ascending=True).reset_index(drop=True)

    return df_bt_summary

#===============================================================================
# Synthetic Data Generator for Mallows-Type Full Rankings
#===============================================================================

def mallows_syntheticdata_gen(n_items, n_judges, model_type='MM', true_references=None, true_thetas=None, weights=None, random_seed=23):
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Log-sum-exp function
    def logsumexp(values):
        maximum = float(np.max(values))
        return maximum + float(np.log(np.exp(values - maximum).sum()))

    # Setup item strings
    items = np.array([f"Item_{i}" for i in range(1, n_items + 1)])
  
    # Defaults and validation
    if model_type in ['MM', 'GMM']:
        # Default consensus
        if true_references is None:
            true_references = [np.arange(n_items)]
        elif isinstance(true_references[0], (int, np.integer)):
            true_references = [true_references]
            
        # Default penalties
        if true_thetas is None:
            if model_type == 'MM':
                true_thetas = [2.0]                                  # strict single penalty.
            else:
                true_thetas = [np.linspace(2.0, 0.2, n_items - 1)]       # decreasing penalty.
        elif isinstance(true_thetas, (float, int)):
            true_thetas = [true_thetas]
            
        weights = [1.0]

    elif model_type == 'MMM':
        # Defaults for Mixture model (2 clusters)
        if true_references is None:
            true_references = [np.arange(n_items), np.arange(n_items)[::-1]]   # ascending vs descending
        if true_thetas is None:
            true_thetas = [1.5, 1.0]                                           # different strictness per cluster.
        if weights is None:
            weights = [0.6, 0.4]
            
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
    else:
        raise ValueError("model_type must be 'MM', 'GMM', or 'MMM'")

    # Generate synthetic rankings
    records = []
    
    for sample_id in range(1, n_judges + 1):
        # Assign cluster based on weights
        cluster_idx = np.random.choice(len(weights), p=weights)
        
        current_ref = true_references[cluster_idx]
        current_theta = true_thetas[cluster_idx]
        
        ranking = []
        
        # Repeated insertion algorithm
        for step, item_idx in enumerate(current_ref):
            inversion_count = np.arange(step + 1)
            
            # Extract penalty
            if isinstance(current_theta, (list, np.ndarray)) and len(current_theta) > 1:
                active_theta = current_theta[step - 1] if step > 0 else 0.0    # positional for GMM.
            else:
                active_theta = current_theta                                   # constant for MM.
                
            # Calculate probabilities
            log_probs = -active_theta * inversion_count
            probs = np.exp(log_probs - logsumexp(log_probs))
            
            # Insert item
            inversions = int(np.random.choice(step + 1, p=probs))
            ranking.insert(step - inversions, int(item_idx))
            
        # Map integer indices back to 'Item_X' strings
        str_ranking = items[ranking].tolist()
        
        record = {"Judge_ID": sample_id,
                  "Cluster": cluster_idx}

        # Add rank positions to the dictionary
        for rank_pos, item_str in enumerate(str_ranking, start=1):
            record[f"Rank_{rank_pos}"] = item_str
            
        records.append(record)

    # Outputs
    df_rankings = pd.DataFrame(records)
    
    # Create Ground Truth dictionary for easy reference
    gt_dic = {"Model": model_type,
              "Clusters": []}
    for c_idx in range(len(weights)):
        gt_dic["Clusters"].append({"Cluster_ID": c_idx,
                                   "Weight": weights[c_idx],
                                   "Consensus": items[true_references[c_idx]].tolist(),
                                   "Theta": true_thetas[c_idx]})

    return df_rankings, gt_dic
