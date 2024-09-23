import numpy as np
import json
from sklearn.datasets import make_sparse_coded_signal
from sklearn.decomposition import DictionaryLearning, SparseCoder
from utils import reconstruction_loss_with_l1
import torch
from metrics import mcc, greedy_mcc

def run_experiment(n_components, n_features, n_nonzero_coefs):
    l1_weight = 0.05
    X, D, S = make_sparse_coded_signal(
        n_samples=1024*2, n_components=n_components, n_features=n_features, n_nonzero_coefs=n_nonzero_coefs,
        random_state=42
    )
    X_train, X_test = X[:1024], X[1024:]
    S_train, S_test = S[:1024], S[1024:]

    dict_learner = DictionaryLearning(
        n_components=n_components, transform_algorithm='lasso_lars', transform_alpha=l1_weight,
        random_state=42
    )
    dict_learner.fit(X_train)

    coder = SparseCoder(
        dictionary=dict_learner.components_, transform_algorithm='lasso_lars', transform_alpha=l1_weight
    )

    S_test_ = coder.transform(X_test)
    X_test_ = S_test_ @ dict_learner.components_

    test_loss = reconstruction_loss_with_l1(torch.tensor(X_test), torch.tensor(X_test_), torch.tensor(S_test_), l1_weight=l1_weight).item()
    mcc_test = mcc(S_test, S_test_)
    mcc_dict = mcc(D.T, dict_learner.components_.T)

    return {
        "test_loss": test_loss,
        "mcc_test": mcc_test,
        "mcc_dict": mcc_dict
    }

N_values = [16, 32, 64, 128, 256]
M_values = [8, 16, 32]
k_values = [3, 6, 9]

results = {}

for N in N_values:
    for M in M_values:
        for k in k_values:
            print(f"Running experiment with N={N}, M={M}, k={k}")
            key = f"N{N}_M{M}_k{k}"
            results[key] = run_experiment(N, M, k)

with open("dictionary_learning_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Experiment completed. Results saved to dictionary_learning_results.json")