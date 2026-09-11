# MODELS FOR RANKING SYNTHETIC DATA

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint
from itertools import permutations

#===============================================================================
# Ranking with Bradley-Terry Model (Pairwise Comparisons)
#===============================================================================

def fit_bt_pw(df_matches, items, stat_mthd=None, iterations=0, proposal_type=None, step_size=None, burn_in=None):
    # Pairwise preference matrix
    n = len(items)
    item_to_idx = {item: i for i, item in enumerate(items)}
    wins = np.zeros((n, n))

    for _, row in df_matches.iterrows():
        winner_idx = item_to_idx[row['Winner']]
        loser_idx = item_to_idx[row['Loser']]
        wins[winner_idx, loser_idx] += 1

    # Markov Chain Monte Carlo (Metropolis-Hastings)
    if stat_mthd == 'MCMC':
        # Apply default parameters if None are provided
        proposal_type = 'normal' if proposal_type is None else proposal_type
        step_size = 0.25 if step_size is None else step_size
        burn_in = int(iterations * 0.1) if burn_in is None else burn_in
        
        # Vectorized log-posterior
        def log_posterior(betas):
            log_prior = -0.5 * np.sum(betas**2)
            # Likelihood
            log_prob = betas[:, None] - np.logaddexp(betas[:, None], betas[None, :])
            log_lik = np.sum(wins * log_prob)
            return log_prior + log_lik

        # Initialize MCMC
        current_betas = np.zeros(n)
        current_log_post = log_posterior(current_betas)
        
        samples = np.zeros((iterations, n))
        accepted = 0
        
        # Metropolis-Hastings Loop
        for t in range(iterations):
            # Propose new state
            if proposal_type == 'normal':
                proposed_betas = current_betas + np.random.normal(0, step_size, n)
            elif proposal_type == 'uniform':
                # Fallback to uniform distribution if specified
                proposed_betas = current_betas + np.random.uniform(-step_size, step_size, n)
            else:
              print("Proposal Type: 'normal' or 'uniform'")
                
            # Recentre betas so they sum to 0
            proposed_betas = proposed_betas - np.mean(proposed_betas)
            
            # Calculate posterior of proposed state
            proposed_log_post = log_posterior(proposed_betas)
            
            # Acceptance criterion
            log_accept_ratio = proposed_log_post - current_log_post
            
            if np.log(np.random.rand()) < log_accept_ratio:
                current_betas = proposed_betas
                current_log_post = proposed_log_post
                accepted += 1
                
            samples[t] = current_betas
            
        # Discard burn-in phase and calculate the mean of the posterior distributions
        valid_samples = samples[burn_in:]
        final_betas = np.mean(valid_samples, axis=0)
        
        df_bt = pd.DataFrame({
            'Item': items,
            'Beta': final_betas
        }).sort_values('Beta', ascending=False).reset_index(drop=True)

    # Sequential Monte Carlo (Tuning-Free Adaptive)
    elif stat_mthd == 'SMC':
        n_particles = iterations if iterations > 0 else 1000
        
        # Initialise particles from prior and set uniform weights
        particles = np.random.normal(0, 1, (n_particles, n))
        particles = particles - np.mean(particles, axis=1, keepdims=True)
        weights = np.ones(n_particles) / n_particles
        
        # Process data sequentially
        for _, row in df_matches.iterrows():
            w_idx = item_to_idx[row['Winner']]
            l_idx = item_to_idx[row['Loser']]
            
            # Update weights based on likelihood of this specific match
            beta_w = particles[:, w_idx]
            beta_l = particles[:, l_idx]
            prob_win = np.exp(beta_w) / (np.exp(beta_w) + np.exp(beta_l))
            
            weights *= prob_win
            
            # Weight normalisation 
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights /= weight_sum
            else:
                weights = np.ones(n_particles) / n_particles   # fallback for degeneracy.
                
            # Effective Sample Size (ESS)
            ess = 1.0 / np.sum(weights**2)
            
            # Resample and mutate if ESS drops below 50%
            if ess < n_particles / 2:
                indices = np.random.choice(n_particles, size=n_particles, p=weights)
                particles = particles[indices]
                weights = np.ones(n_particles) / n_particles
                
                # Use empirical covariance of particles for adaptive jitter
                cov_matrix = np.cov(particles, rowvar=False) + np.eye(n) * 1e-5
                jitter = np.random.multivariate_normal(np.zeros(n), cov_matrix / n, n_particles)
                particles += jitter
                particles = particles - np.mean(particles, axis=1, keepdims=True)

        final_betas = np.average(particles, axis=0, weights=weights)
        
        df_bt = pd.DataFrame({
            'Item': items,
            'Beta': final_betas
        }).sort_values('Beta', ascending=False).reset_index(drop=True)

    # Maximum Likelihood Estimation
    else:
        # Negative log-likelihood function
        def neg_log_likelihood(betas):
            nll = 0
            for i in range(n):
                for j in range(n):
                    if wins[i, j] > 0:
                        # BT formula
                        prob_win = np.exp(betas[i]) / (np.exp(betas[i]) + np.exp(betas[j]))
                        nll -= wins[i, j] * np.log(prob_win)
            return nll

        # Parameters optimisation
        constraints = ({'type': 'eq', 'fun': lambda betas: np.sum(betas)})
        initial_betas = np.zeros(n)

        result = minimize(neg_log_likelihood, initial_betas, constraints=constraints, method='SLSQP')

        # Format output
        df_bt = pd.DataFrame({
            'Item': items,
            'Beta': result.x
        }).sort_values('Beta', ascending=False).reset_index(drop=True)

    return df_bt

#===============================================================================
# Ranking with Mallows Model (Pairwise Comparisons)
#===============================================================================
def fit_mm_pw(df_matches, items, stat_mthd=None, iterations=0, proposal_type=None, burn_in=None):
    # Pairwise preference matrix
    n = len(items)
    item_to_idx = {item: i for i, item in enumerate(items)}
    P = np.zeros((n, n))

    for _, row in df_matches.iterrows():
        win_idx = item_to_idx[row['Winner']]
        loose_idx = item_to_idx[row['Loser']]
        P[win_idx, loose_idx] += 1

    # Function to calculate Kendall's distance penalty parameter
    def calculate_penalty(ranking):
        penalty = 0
        for i in range(len(ranking)):
            for j in range(i + 1, len(ranking)):
                item_higher = ranking[i]
                item_lower = ranking[j]
                penalty += P[item_lower, item_higher]   # penalty when the lower item beat a higher one.
        return penalty

    # Markov Chain Monte Carlo (Metropolis-Hastings)
    if stat_mthd == 'MCMC':
        # Default parameters
        proposal_type = 'swap' if proposal_type is None else proposal_type
        burn_in = int(iterations * 0.1) if burn_in is None else burn_in
        
        # Initialize MCMC with a random permutation
        current_perm = list(range(n))
        np.random.shuffle(current_perm)
        current_penalty = calculate_penalty(current_perm)
        
        samples = []
        
        # Metropolis-Hastings step
        for t in range(iterations):
            proposed_perm = current_perm.copy()
            
            # Propose new state
            if proposal_type == 'swap':
                # Swap two random elements
                idx1, idx2 = np.random.choice(n, 2, replace=False)
                proposed_perm[idx1], proposed_perm[idx2] = proposed_perm[idx2], proposed_perm[idx1]
            elif proposal_type == 'insertion':
                # Select an item and insert it at a new random position
                idx_pop, idx_insert = np.random.choice(n, 2, replace=False)
                item = proposed_perm.pop(idx_pop)
                proposed_perm.insert(idx_insert, item)
            else:
              print("Proposal type: 'insertion' or 'swap'")
                
            proposed_penalty = calculate_penalty(proposed_perm)
            
            # Acceptance criterion
            log_accept_ratio = -(proposed_penalty - current_penalty)
            
            if np.log(np.random.rand()) < log_accept_ratio:
                current_perm = proposed_perm
                current_penalty = proposed_penalty
                
            samples.append((current_perm, current_penalty))
            
        # Discard burn-in phase
        valid_samples = samples[burn_in:]
        
        # Find the permutation with the absolute lowest penalty in the chain
        best_sample = min(valid_samples, key=lambda x: x[1])
        best_ranking = best_sample[0]
        min_penalty = best_sample[1]
        
        consensus_items = [items[i] for i in best_ranking]
        df_mm = pd.DataFrame({'Item': consensus_items})

        return df_mm, min_penalty

    # Sequential Monte Carlo (Tuning-Free Adaptive)
    elif stat_mthd == 'SMC':
        n_particles = iterations if iterations > 0 else 1000
        
        # Initialize particles as random permutations
        particles = [np.random.permutation(n).tolist() for _ in range(n_particles)]
        weights = np.ones(n_particles) / n_particles
        
        # Process sequentially match by match
        for _, row in df_matches.iterrows():
            w_idx = item_to_idx[row['Winner']]
            l_idx = item_to_idx[row['Loser']]
            
            for i, perm in enumerate(particles):
                # Weight increases if permutation aligns with the match outcome
                if perm.index(w_idx) < perm.index(l_idx):
                    weights[i] *= 2    # agreement reward
                else:
                    weights[i] *= 0.5  # disagreement penalisation

            # Weight normalisation        
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights /= weight_sum
            else:
                weights = np.ones(n_particles) / n_particles   # fallback for degeneracy.

            # Effective Sample Size (ESS)    
            ess = 1.0 / np.sum(weights**2)
            
            # Tuning-free mutation via discrete swaps
            if ess < n_particles / 2.0:
                indices = np.random.choice(n_particles, size=n_particles, p=weights)
                new_particles = []
                for idx in indices:
                    perm = particles[idx].copy()
                    # Add a random adjacent swap to maintain diversity
                    swap_idx = np.random.randint(0, n - 1)
                    perm[swap_idx], perm[swap_idx+1] = perm[swap_idx+1], perm[swap_idx]
                    new_particles.append(perm)
                particles = new_particles
                weights = np.ones(n_particles) / n_particles
                
        # Find the single best particle remaining
        best_particle = None
        min_pen = float('inf')
        for perm in particles:
            pen = calculate_penalty(perm)
            if pen < min_pen:
                min_pen = pen
                best_particle = perm
                
        consensus_items = [items[i] for i in best_particle]
        df_mm = pd.DataFrame({'Item': consensus_items})
        return df_mm, min_pen

    # Permutation Space Search
    else:
        # Generate the entire permutation space
        perm_space = list(permutations(range(n)))

        best_ranking = None
        min_penalty = float('inf')

        # Evaluate every permutation to find the absolute minimum penalty
        for perm in perm_space:
            current_penalty = calculate_penalty(perm)

            # Update best ranking based on lower penalty
            if current_penalty < min_penalty:
                min_penalty = current_penalty
                best_ranking = perm

        consensus_items = [items[i] for i in best_ranking]

        # Format as a table with item and rank
        df_mm = pd.DataFrame({'Item': consensus_items})

        return df_mm, min_penalty

#===============================================================================
# Ranking with GMM (Pairwise Comparisons)
#===============================================================================

def fit_gmm_pw(df_matches, items, constraint=None, stat_mthd=None, iterations=0, proposal_type=None, step_size=None, burn_in=None):
    # Pairwise preference matrix
    n = len(items)
    item_to_idx = {item: i for i, item in enumerate(items)}
    P = np.zeros((n, n))
    for _, row in df_matches.iterrows():
        P[item_to_idx[row['Winner']], item_to_idx[row['Loser']]] += 1

    def get_V(ranking):
        V = np.zeros(n - 1)
        for i in range(n - 1):
            for j in range(i + 1, n):
                V[i] += P[ranking[j], ranking[i]]
        return V

    def log_posterior_gmm(theta, V):
        if np.any(theta <= 1e-5):
            return -np.inf 
        if constraint == 'decreasing' and np.any(np.diff(theta) > 0):
            return -np.inf
            
        log_prior = -np.sum(theta)
        n_minus_i = np.arange(n, 1, -1)
        log_Z = np.log(1 - np.exp(-n_minus_i * theta)) - np.log(1 - np.exp(-theta))
        log_lik = -np.sum(theta * V) - np.sum(log_Z)
        return log_prior + log_lik

    # Markov Chain Monte Carlo (Metropolis-Hastings) - Gibbs
    if stat_mthd == 'MCMC':
        # Apply default parameters if None are provided
        proposal_type = 'normal' if proposal_type is None else proposal_type
        step_size = 0.5 if step_size is None else step_size
        burn_in = int(iterations * 0.1) if burn_in is None else burn_in
        
        # Initialise states
        current_perm = list(range(n))
        np.random.shuffle(current_perm)
        
        if constraint == 'decreasing':
            current_theta = np.linspace(2.0, 0.5, n - 1)
        else:
            current_theta = np.ones(n - 1)
            
        current_V = get_V(current_perm)
        current_log_post = log_posterior_gmm(current_theta, current_V)
        
        theta_samples = np.zeros((iterations, n - 1))
        perm_samples = []
        
        # Metropolis-Hastings step
        for t in range(iterations):
            # Update permutation supposing parameters are correct
            proposed_perm = current_perm.copy()
            idx1, idx2 = np.random.choice(n, 2, replace=False)
            proposed_perm[idx1], proposed_perm[idx2] = proposed_perm[idx2], proposed_perm[idx1]
            
            # Get posterior of this permutation
            proposed_V = get_V(proposed_perm)
            proposed_log_post_perm = log_posterior_gmm(current_theta, proposed_V)
            
            # Acceptance criterion
            if np.log(np.random.rand()) < (proposed_log_post_perm - current_log_post):
                current_perm = proposed_perm
                current_V = proposed_V
                current_log_post = proposed_log_post_perm
            
            # Update theta
            if proposal_type == 'normal':
                proposed_theta = current_theta + np.random.normal(0, step_size, n - 1)
            elif proposal_type == 'uniform':
                proposed_theta = current_theta + np.random.uniform(-step_size, step_size, n - 1)
            else:
              print("Proposal type: 'normal' or 'uniform'")

            # Evaluation    
            proposed_log_post_theta = log_posterior_gmm(proposed_theta, current_V)
            
            # Acceptance criterion
            if np.log(np.random.rand()) < (proposed_log_post_theta - current_log_post):
                current_theta = proposed_theta
                current_log_post = proposed_log_post_theta
                
            theta_samples[t] = current_theta
            perm_samples.append((current_perm, current_log_post))
            
        # Discard burn-in phase
        valid_theta = theta_samples[burn_in:]
        valid_perms = perm_samples[burn_in:]
        
        # Output
        theta_j = np.mean(valid_theta, axis=0)
        best_perm = max(valid_perms, key=lambda x: x[1])[0]
        
        consensus_items = [items[i] for i in best_perm]
        df_gmm = pd.DataFrame({'Item': consensus_items})
        
        if constraint == 'decreasing':
            print("Penalties MCMC (Constrained)")
        else:
            print("Penalties MCMC (Unconstrained)")

        return df_gmm, theta_j

    # Sequential Monte Carlo (Tuning-Free Adaptive)
    elif stat_mthd == 'SMC':
        n_particles = iterations if iterations > 0 else 1000
        
        # Initialise joint particles 
        particle_perms = [np.random.permutation(n).tolist() for _ in range(n_particles)]   # permutations are discrete
        if constraint == 'decreasing':
            particle_thetas = np.array([np.linspace(2.0, 0.5, n - 1) for _ in range(n_particles)])   # thetas are continuous
        else:
            particle_thetas = np.ones((n_particles, n - 1))
            
        weights = np.ones(n_particles) / n_particles
        
        # Process sequentially
        for _, row in df_matches.iterrows():
            w_idx = item_to_idx[row['Winner']]
            l_idx = item_to_idx[row['Loser']]
            
            for i in range(n_particles):
                perm = particle_perms[i]
                theta = particle_thetas[i]
                
                # Update weights based on positional penalty
                pos_w = perm.index(w_idx)
                pos_l = perm.index(l_idx)
                
                if pos_w < pos_l:
                    weights[i] *= 1.5
                else:
                    # Penalisation based on the active parameter of that rank
                    weights[i] *= np.exp(-theta[pos_l]) 

            # Weight normalisation
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights /= weight_sum
            else:
                weights = np.ones(n_particles) / n_particles   # fallback for degeneracy.

            # Effective Sample Size (ESS)    
            ess = 1.0 / np.sum(weights**2)
            
            # Resample and adaptive move
            if ess < n_particles / 2.0:
                indices = np.random.choice(n_particles, size=n_particles, p=weights)
                new_perms = []
                for idx in indices:
                    perm = particle_perms[idx].copy()
                    swap_idx = np.random.randint(0, n - 1)
                    perm[swap_idx], perm[swap_idx+1] = perm[swap_idx+1], perm[swap_idx]
                    new_perms.append(perm)
                
                particle_perms = new_perms
                particle_thetas = particle_thetas[indices]
                
                # Tuning-free continuous mutation
                cov_matrix = np.cov(particle_thetas, rowvar=False) + np.eye(n - 1) * 1e-5
                jitter = np.random.multivariate_normal(np.zeros(n - 1), cov_matrix / (n - 1), n_particles)
                proposed_thetas = particle_thetas + jitter
                
                # Reject invalid mutations by reverting to previous state
                for i in range(n_particles):
                    if np.any(proposed_thetas[i] <= 1e-5) or (constraint == 'decreasing' and np.any(np.diff(proposed_thetas[i]) > 0)):
                        pass
                    else:
                        particle_thetas[i] = proposed_thetas[i]
                        
                weights = np.ones(n_particles) / n_particles
                
        # Best estimation output
        theta_j = np.average(particle_thetas, axis=0, weights=weights)
        
        min_pen = float('inf')
        best_perm = None
        for perm in particle_perms:
            v_array = get_V(perm)
            post = log_posterior_gmm(theta_j, v_array)
            if -post < min_pen:
                min_pen = -post
                best_perm = perm

        consensus_items = [items[i] for i in best_perm]
        df_gmm = pd.DataFrame({'Item': consensus_items})
        
        if constraint == 'decreasing':
            print("Penalties SMC (Constrained)")
        else:
            print("Penalties SMC (Unconstrained)")
            
        return df_gmm, theta_j

    # Permutation Space Search with Maximum Likelihood Estimation
    else:
        all_perms = list(permutations(range(n)))

        # Initialize theta based on the selected constraint
        if constraint == 'decreasing':
            theta_j = np.linspace(2.0, 0.5, n - 1)
        else:
            theta_j = np.ones(n - 1)
            
        previous_centre = None
        min_weighted_penalty = float('inf')

        max_iterations = 10

        for iteration in range(max_iterations):
            # Find the exact centre using the current 'theta_j'
            current_centre = None
            current_min_penalty = float('inf')

            def calculate_weighted_penalty(ranking):
                penalty = 0
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        if P[ranking[j], ranking[i]] > 0:
                            penalty += theta_j[i] * P[ranking[j], ranking[i]]
                return penalty

            for perm in all_perms:
                perm_penalty = calculate_weighted_penalty(perm)
                if perm_penalty < current_min_penalty:
                    current_min_penalty = perm_penalty
                    current_centre = perm

            # Check for convergence
            if current_centre == previous_centre:
                break

            previous_centre = current_centre
            min_weighted_penalty = current_min_penalty

            # Optimise 'theta_j' using the new centre
            def nll_theta(theta_array):
                nll = 0
                for i in range(n - 1):
                    V_i = 0
                    for j in range(i + 1, n):
                        V_i += P[current_centre[j], current_centre[i]]

                    theta_i = theta_array[i]

                    if theta_i < 1e-5:
                        log_Z_i = np.log(n - i)
                    else:
                        log_Z_i = np.log(1 - np.exp(-(n - i) * theta_i)) - np.log(1 - np.exp(-theta_i))

                    nll += (theta_i * V_i) + log_Z_i
                return nll

            initial_guess = theta_j    # use the current theta as the starting point for optimisation.
            bounds = [(1e-3, None) for _ in range(n - 1)]

            # Conditional optimisation method based on constraints
            if constraint == 'decreasing':
                # Constraint matrix
                constraint_matrix = np.zeros((n - 2, n - 1))
                for i in range(n - 2):
                    constraint_matrix[i, i] = 1.0
                    constraint_matrix[i, i + 1] = -1.0

                decreasing_constraint = LinearConstraint(constraint_matrix, lb=0, ub=np.inf)
                result = minimize(nll_theta, initial_guess, bounds=bounds, constraints=[decreasing_constraint], method='SLSQP')
            else:
                result = minimize(nll_theta, initial_guess, bounds=bounds)
                
            theta_j = result.x

        # Output
        penalties = list(np.round(theta_j, 4)) + [0.0]
        df_penalties = pd.DataFrame([penalties], index=['Theta_Penalty'], columns=[f"Rank_{i+1}" for i in range(n)])
        
        if constraint == 'decreasing':
            print("Penalties (Constrained)")
        else:
            print("Penalties (Unconstrained)")
            
        consensus_items = [items[i] for i in current_centre]

        # Format as a table with item and rank
        df_gmm = pd.DataFrame({'Item': consensus_items})

        return df_gmm, theta_j

#===============================================================================
# Ranking with Mallows Model (Full Rankings)
#===============================================================================
def fit_mm_full(df_rankings, items, stat_mthd=None, iterations=0, proposal_type=None, burn_in=None):
    n = len(items)
    
    # Represent full ranking as matrix of pairwise preferences
    item_to_idx = {item: i for i, item in enumerate(items)}
    P = np.zeros((n, n))
    
    rank_cols = [col for col in df_rankings.columns if col.startswith('Rank_')]   # identify ranking columns.
    
    observations = np.zeros((len(df_rankings), n), dtype=int)
    
    for r_idx, row in df_rankings.iterrows():
        ranking = [item_to_idx[row[col]] for col in rank_cols]
        observations[r_idx] = ranking
        for i in range(n):
            for j in range(i + 1, n):
                P[ranking[i], ranking[j]] += 1

    # Penalty
    def calculate_penalty(ranking):
        penalty = 0
        for i in range(n):
            for j in range(i + 1, n):
                penalty += P[ranking[j], ranking[i]]
        return penalty

    # Markov Chain Monte Carlo (Metropolis-Hastings)
    if stat_mthd == 'MCMC':
        proposal_type = 'swap' if proposal_type is None else proposal_type
        burn_in = int(iterations * 0.1) if burn_in is None else burn_in
        
        current_perm = list(range(n))
        np.random.shuffle(current_perm)
        current_penalty = calculate_penalty(current_perm)
        
        samples = []
        # Permutation sampling
        for t in range(iterations):
            proposed_perm = current_perm.copy()
            # Swap two random items
            if proposal_type == 'swap':
                idx1, idx2 = np.random.choice(n, 2, replace=False)
                proposed_perm[idx1], proposed_perm[idx2] = proposed_perm[idx2], proposed_perm[idx1]
            # Select a random item and insert it in a random position
            elif proposal_type == 'insertion':
                idx_pop, idx_insert = np.random.choice(n, 2, replace=False)
                item = proposed_perm.pop(idx_pop)
                proposed_perm.insert(idx_insert, item)
            else:
              print(" Proposal type: 'insertion' or 'swap'")
                
            proposed_penalty = calculate_penalty(proposed_perm)
            log_accept_ratio = -(proposed_penalty - current_penalty)
            
            # Acceptance criterion
            if np.log(np.random.rand()) < log_accept_ratio:
                current_perm = proposed_perm
                current_penalty = proposed_penalty
                
            samples.append((current_perm, current_penalty))

        # Output    
        valid_samples = samples[burn_in:]
        best_sample = min(valid_samples, key=lambda x: x[1])
        consensus_items = [items[i] for i in best_sample[0]]
        
        df_mm = pd.DataFrame({'Item': consensus_items})
        return df_mm, best_sample[1]

    # Sequential Monte Carlo
    elif stat_mthd == 'SMC':
        n_particles = iterations if iterations > 0 else 1000
        
        # Initialise joint particles
        particle_perms = [np.random.permutation(n).tolist() for _ in range(n_particles)]   # permutations are discrete.
        particle_thetas = np.random.uniform(0.1, 2.0, n_particles)                         # theta is continuous.
        weights = np.ones(n_particles) / n_particles
        
        # Log-partition normaliser
        def calc_log_Z(theta):
            if theta < 1e-5:
                return np.sum(np.log(np.arange(1, n + 1))) # log(n!)
            j_vals = np.arange(1, n + 1)
            return np.sum(np.log(1 - np.exp(-j_vals * theta)) - np.log(1 - np.exp(-theta)))
        
        # Process data sequentially
        for obs in observations:  
            pos_in_obs = {item: idx for idx, item in enumerate(obs)}
            
            for i in range(n_particles):
                perm = particle_perms[i]
                theta = particle_thetas[i]
                
                # Calculate exact Kendall distance between particle and observation
                dist = 0
                for a in range(n):
                    for b in range(a + 1, n):
                        if pos_in_obs[perm[a]] > pos_in_obs[perm[b]]:
                            dist += 1
                            
                # Update particle weight based on Bayesian Likelihood
                log_Z = calc_log_Z(theta)
                log_lik = -theta * dist - log_Z
                
                # Exponentiate to update weight 
                weights[i] *= np.exp(log_lik)
                
            # Weight normalisation
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights /= weight_sum
            else:
                weights = np.ones(n_particles) / n_particles 

            # Effective Sample Size
            ess = 1.0 / np.sum(weights**2)

            # Resampling
            if ess < n_particles / 2.0:
                indices = np.random.choice(n_particles, size=n_particles, p=weights)
                new_perms = []
                
                # Discrete random swap to mutate permutation 
                for idx in indices:
                    p_copy = particle_perms[idx].copy()
                    idx1, idx2 = np.random.choice(n, 2, replace=False)
                    p_copy[idx1], p_copy[idx2] = p_copy[idx2], p_copy[idx1]
                    new_perms.append(p_copy)
                    
                particle_perms = new_perms
                particle_thetas = particle_thetas[indices]
                
                # Mutate theta
                jitter = np.random.normal(0, 0.1, n_particles)
                proposed_thetas = particle_thetas + jitter
                # Reject invalid negative thetas by keeping the previous value
                particle_thetas = np.where(proposed_thetas > 1e-5, proposed_thetas, particle_thetas)
                
                weights = np.ones(n_particles) / n_particles
                
        # Output
        best_idx = np.argmax(weights)
        best_perm = particle_perms[best_idx]
        best_theta = particle_thetas[best_idx]
        
        min_pen = calculate_penalty(best_perm)
        consensus_items = [items[i] for i in best_perm]
        df_mm = pd.DataFrame({'Item': consensus_items})
        
        print(f"Optimal Theta Selected by SMC: {best_theta:.4f}")
        return df_mm, min_pen

    # Permutation Space Search
    else:
        perm_space = list(permutations(range(n)))
        best_ranking = None
        min_penalty = float('inf')

        for perm in perm_space:
            current_penalty = calculate_penalty(perm)
            if current_penalty < min_penalty:
                min_penalty = current_penalty
                best_ranking = perm

        consensus_items = [items[i] for i in best_ranking]
        df_mm = pd.DataFrame({'Item': consensus_items})
        return df_mm, min_penalty

#===============================================================================
# Ranking with GMM (Full Rankings)
#===============================================================================
def fit_gmm_full(df_rankings, items, constraint=None, stat_mthd=None, iterations=0, proposal_type=None, step_size=None, burn_in=None):
    n = len(items)
    
    # Represent full ranking as matrix of pairwise preferences
    item_to_idx = {item: i for i, item in enumerate(items)}
    P = np.zeros((n, n))
    
    rank_cols = [col for col in df_rankings.columns if col.startswith('Rank_')]   # identify ranking columns.
    
    observations = np.zeros((len(df_rankings), n), dtype=int)
    
    for r_idx, row in df_rankings.iterrows():
        ranking = [item_to_idx[row[col]] for col in rank_cols]
        observations[r_idx] = ranking
        for i in range(n):
            for j in range(i + 1, n):
                P[ranking[i], ranking[j]] += 1

    def get_V(ranking):
        V = np.zeros(n - 1)
        for i in range(n - 1):
            for j in range(i + 1, n):
                V[i] += P[ranking[j], ranking[i]]
        return V

    def log_posterior_gmm(theta, V):
        if np.any(theta <= 1e-5):
            return -np.inf 
        if constraint == 'decreasing' and np.any(np.diff(theta) > 0):
            return -np.inf
            
        log_prior = -np.sum(theta)
        n_minus_i = np.arange(n, 1, -1)
        log_Z = np.log(1 - np.exp(-n_minus_i * theta)) - np.log(1 - np.exp(-theta))
        log_lik = -np.sum(theta * V) - np.sum(log_Z)
        return log_prior + log_lik

    # Markov Chain Monte Carlo (Metropolis-Hastings) - Gibbs
    if stat_mthd == 'MCMC':
        # Apply default parameters if None are provided
        proposal_type = 'normal' if proposal_type is None else proposal_type
        step_size = 0.5 if step_size is None else step_size
        burn_in = int(iterations * 0.1) if burn_in is None else burn_in
        
        # Initialise states
        current_perm = list(range(n))
        np.random.shuffle(current_perm)
        
        if constraint == 'decreasing':
            current_theta = np.linspace(2.0, 0.5, n - 1)
        else:
            current_theta = np.ones(n - 1)
            
        current_V = get_V(current_perm)
        current_log_post = log_posterior_gmm(current_theta, current_V)
        
        theta_samples = np.zeros((iterations, n - 1))
        perm_samples = []
        
        # Metropolis-Hastings step
        for t in range(iterations):
            # Update permutation supposing parameters are correct
            proposed_perm = current_perm.copy()
            idx1, idx2 = np.random.choice(n, 2, replace=False)
            proposed_perm[idx1], proposed_perm[idx2] = proposed_perm[idx2], proposed_perm[idx1]
            
            # Get posterior of this permutation
            proposed_V = get_V(proposed_perm)
            proposed_log_post_perm = log_posterior_gmm(current_theta, proposed_V)
            
            # Acceptance criterion
            if np.log(np.random.rand()) < (proposed_log_post_perm - current_log_post):
                current_perm = proposed_perm
                current_V = proposed_V
                current_log_post = proposed_log_post_perm
            
            # Update theta
            if proposal_type == 'normal':
                proposed_theta = current_theta + np.random.normal(0, step_size, n - 1)             # use normal, gausssian distribution
            elif proposal_type == 'uniform':
                proposed_theta = current_theta + np.random.uniform(-step_size, step_size, n - 1)   # use uniform distribution
            else:
                print("Proposal type: 'normal' or 'uniform'")

            # Evaluation    
            proposed_log_post_theta = log_posterior_gmm(proposed_theta, current_V)
            
            # Acceptance criterion
            if np.log(np.random.rand()) < (proposed_log_post_theta - current_log_post):
                current_theta = proposed_theta
                current_log_post = proposed_log_post_theta
                
            theta_samples[t] = current_theta
            perm_samples.append((current_perm, current_log_post))
            
        # Discard burn-in phase
        valid_theta = theta_samples[burn_in:]
        valid_perms = perm_samples[burn_in:]
        
        # Output
        theta_j = np.mean(valid_theta, axis=0)
        best_perm = max(valid_perms, key=lambda x: x[1])[0]
        
        consensus_items = [items[i] for i in best_perm]
        df_gmm = pd.DataFrame({'Item': consensus_items})
        
        if constraint == 'decreasing':
            print("Penalties MCMC (Constrained)")
        else:
            print("Penalties MCMC (Unconstrained)")

        return df_gmm, theta_j

    # Sequential Monte Carlo (Tuning-Free Adaptive)
    elif stat_mthd == 'SMC':
        n_particles = iterations if iterations > 0 else 500
        
        # Initialise joint particles 
        particle_perms = [np.random.permutation(n).tolist() for _ in range(n_particles)]   # permutations are discrete
        if constraint == 'decreasing':
            particle_thetas = np.array([np.linspace(2.0, 0.5, n - 1) for _ in range(n_particles)])   # thetas are continuous
        else:
            particle_thetas = np.ones((n_particles, n - 1))
            
        weights = np.ones(n_particles) / n_particles
        
        # Process sequentially
        for obs in observations:  
            pos_in_obs = {item: idx for idx, item in enumerate(obs)}
            
            for i in range(n_particles):
                perm = particle_perms[i]
                theta = particle_thetas[i]
                
                # Compare the proposed particle permutation against the single observation
                V_particle = np.zeros(n - 1)
                for pos in range(n - 1):
                    for j in range(pos + 1, n):
                        # If the particle says item A beats B, but observation says B beats A
                        if pos_in_obs[perm[pos]] > pos_in_obs[perm[j]]: 
                            V_particle[pos] += 1
                
                # Weight update via likelihood
                log_post = log_posterior_gmm(theta, V_particle)
                if np.isneginf(log_post):
                    weights[i] = 0
                else:
                    weights[i] *= np.exp(log_post)

            # Weight normalisation
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights /= weight_sum
            else:
                weights = np.ones(n_particles) / n_particles 

            # Effective Sample Size (ESS)    
            ess = 1.0 / np.sum(weights**2)
            
            # Resample and adaptive move
            if ess < n_particles / 2.0:
                indices = np.random.choice(n_particles, size=n_particles, p=weights)
                new_perms = []
                for idx in indices:
                    perm = particle_perms[idx].copy()
                    swap_idx = np.random.randint(0, n - 1)
                    perm[swap_idx], perm[swap_idx+1] = perm[swap_idx+1], perm[swap_idx]
                    new_perms.append(perm)
                
                particle_perms = new_perms
                particle_thetas = particle_thetas[indices]
                
                # Tuning-free continuous mutation
                cov_matrix = np.cov(particle_thetas, rowvar=False) + np.eye(n - 1) * 1e-5
                jitter = np.random.multivariate_normal(np.zeros(n - 1), cov_matrix / (n - 1), n_particles)
                proposed_thetas = particle_thetas + jitter
                
                # Reject invalid mutations by reverting to previous state
                for i in range(n_particles):
                    if not (np.any(proposed_thetas[i] <= 1e-5) or (constraint == 'decreasing' and np.any(np.diff(proposed_thetas[i]) > 0))):
                        particle_thetas[i] = proposed_thetas[i]
                        
                weights = np.ones(n_particles) / n_particles
                
        # Best estimation output
        theta_j = np.average(particle_thetas, axis=0, weights=weights)
        
        min_pen = float('inf')
        best_perm = None
        for perm in particle_perms:
            v_array = get_V(perm)
            post = log_posterior_gmm(theta_j, v_array)
            if -post < min_pen:
                min_pen = -post
                best_perm = perm

        consensus_items = [items[i] for i in best_perm]
        df_gmm = pd.DataFrame({'Item': consensus_items})
        
        if constraint == 'decreasing':
            print("Penalties SMC (Constrained)")
        else:
            print("Penalties SMC (Unconstrained)")
            
        return df_gmm, theta_j

    # Permutation Space Search with Maximum Likelihood Estimation
    else:
        all_perms = list(permutations(range(n)))

        # Initialize theta based on the selected constraint
        if constraint == 'decreasing':
            theta_j = np.linspace(2.0, 0.5, n - 1)
        else:
            theta_j = np.ones(n - 1)
            
        previous_centre = None
        min_weighted_penalty = float('inf')

        max_iterations = 10

        for iteration in range(max_iterations):
            # Find the exact centre using the current 'theta_j'
            current_centre = None
            current_min_penalty = float('inf')

            def calculate_weighted_penalty(ranking):
                penalty = 0
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        if P[ranking[j], ranking[i]] > 0:
                            penalty += theta_j[i] * P[ranking[j], ranking[i]]
                return penalty

            for perm in all_perms:
                perm_penalty = calculate_weighted_penalty(perm)
                if perm_penalty < current_min_penalty:
                    current_min_penalty = perm_penalty
                    current_centre = perm

            # Check for convergence
            if current_centre == previous_centre:
                break

            previous_centre = current_centre
            min_weighted_penalty = current_min_penalty

            # Optimise 'theta_j' using the new centre
            def nll_theta(theta_array):
                nll = 0
                for i in range(n - 1):
                    V_i = sum(P[current_centre[j], current_centre[i]] for j in range(i + 1, n))
                    theta_i = theta_array[i]
                    log_Z_i = np.log(n - i) if theta_i < 1e-5 else np.log(1 - np.exp(-(n - i) * theta_i)) - np.log(1 - np.exp(-theta_i))
                    nll += (theta_i * V_i) + log_Z_i
                return nll

            initial_guess = theta_j    # use the current theta as the starting point for optimisation.
            bounds = [(1e-3, None) for _ in range(n - 1)]

            # Conditional optimisation method based on constraints
            if constraint == 'decreasing':
                # Constraint matrix
                constraint_matrix = np.zeros((n - 2, n - 1))
                for i in range(n - 2):
                    constraint_matrix[i, i] = 1.0; constraint_matrix[i, i + 1] = -1.0
                decreasing_constraint = LinearConstraint(constraint_matrix, lb=0, ub=np.inf)
                result = minimize(nll_theta, initial_guess, bounds=bounds, constraints=[decreasing_constraint], method='SLSQP')
            else:
                result = minimize(nll_theta, initial_guess, bounds=bounds)
                
            theta_j = result.x

        # Output
        penalties = list(np.round(theta_j, 4)) + [0.0]
        df_penalties = pd.DataFrame([penalties], index=['Theta_Penalty'], columns=[f"Rank_{i+1}" for i in range(n)])
        
        if constraint == 'decreasing':
            print("Penalties (Constrained)")
        else:
            print("Penalties (Unconstrained)")
            
        consensus_items = [items[current_centre[i]] for i in range(n)]

        # Format as a table with item and rank
        df_gmm = pd.DataFrame({'Item': consensus_items})

        return df_gmm, theta_j