from scipy.stats import pearsonr as _corr
from munkres import Munkres
import numpy as np


def corr(z, z_, tol=1e-12):
    if np.var(z) < tol or np.var(z_) < tol:
        return (0, 0)
    else:
        return _corr(z, z_)


def get_corr(z, z_):
    corrs = np.zeros((z.shape[1], z_.shape[1]))
    for i in range(z.shape[1]):
        for j in range(z_.shape[1]):
            corrs[i, j] = corr(z[:, i], z_[:, j])[0]
    corrs[np.isnan(corrs)] = 0
    return corrs


def match_latents(z, z_):
    matches = np.abs(get_corr(z, z_))
    indexes = Munkres().compute(-matches)
    return matches, indexes


def eval_nd(z, z_):
    matches, indexes = match_latents(z, z_)
    corrs = []
    for i in indexes:
        corrs.append(matches[i[0], i[1]])
    return corrs


def vanilla_mcc(z, z_):
    return np.mean(eval_nd(z, z_))


def greedy_mcc(z, z_):
    # find max corr(z, z_), then remove until all covered
    corrs = abs(get_corr(z, z_))
    shape = corrs.shape
    new_z, new_z_ = [], []
    D = np.min([z.shape[1], z_.shape[1]])
    for i in range(D):
        ind0, ind1 = np.unravel_index(np.argmax(corrs), shape)
        new_z.append(z[:, ind0])
        new_z_.append(z_[:, ind1])
        corrs[ind0] *= 0
        corrs[:, ind1] *= 0
    mis = vanilla_mcc(np.array(new_z).T, np.array(new_z_).T)
    return mis

def simple_mcc(z, z_):
    D = min(z.shape[1], z_.shape[1])
    return np.mean([corr(z[:, i], z_[:, i])[0] for i in range(D)])


def mcc(z, z_):
    """
    greedy mcc is much faster and gives (fairly tight) lower bound,
    so just make it default for dim mismatch.
    """
    if z_.shape[1] == z.shape[1]:
        return vanilla_mcc(z, z_)
    else:
        return simple_mcc(z, z_)