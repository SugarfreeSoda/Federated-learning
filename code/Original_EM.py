import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from scipy.cluster.vq import kmeans2


def generate_site_gmm(n, d, lamb=None, mu0=None, mu1=None, Sigma=None, seed=0):

    rng = np.random.default_rng(seed)

    if mu0 is None:
        mu0 = np.zeros(d)
    if mu1 is None:
        mu1 = np.ones(d) * 2.0
    if Sigma is None:
        Sigma = np.eye(d)
    if lamb is None:
        lamb = 0.5

    L = np.linalg.cholesky(Sigma)

    z = rng.binomial(1, lamb, size=n)
    eps = rng.normal(size=(n, d)) @ L.T
    y = np.where(z[:, None] == 1, mu1, mu0) + eps

    return y


def log_gaussian_density(Y, mu, Sigma):
    Y = np.asarray(Y)
    mu = np.asarray(mu)
    d = Y.shape[1]
    Sigma_inv = np.linalg.inv(Sigma)
    logdet = np.linalg.slogdet(Sigma)[1]
    diff = Y - mu
    quad = np.sum((diff @ Sigma_inv) * diff, axis=1)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


def responsibility(Y, mu0, mu1, lamb, Sigma):
    lamb = np.clip(lamb, 1e-8, 1 - 1e-8)

    logp1 = np.log(lamb) + log_gaussian_density(Y, mu1, Sigma)
    logp0 = np.log(1 - lamb) + log_gaussian_density(Y, mu0, Sigma)

    m = np.maximum(logp1, logp0)
    p1 = np.exp(logp1 - m)
    p0 = np.exp(logp0 - m)

    return np.clip(p1 / (p1 + p0), 1e-8, 1 - 1e-8)


def init_by_kmeans(Y, seed=0):
    rng = np.random.default_rng(seed)

    best_centers = None
    best_loss = np.inf

    for r in range(5):
        centers, labels = kmeans2(
            Y,
            k=2,
            minit="points",
            iter=50,
            seed=seed + r
        )
        loss = np.sum((Y - centers[labels]) ** 2)
        if loss < best_loss:
            best_loss = loss
            best_centers = centers.copy()

    c0, c1 = best_centers[0], best_centers[1]

    if np.mean(c0) <= np.mean(c1):
        mu0_init, mu1_init = c0, c1
    else:
        mu0_init, mu1_init = c1, c0

    return mu0_init, mu1_init


def fit_local_em(Y, Sigma, max_iter=100, tol=1e-6, seed=0):
    mu0, mu1 = init_by_kmeans(Y, seed=seed)
    lamb = 0.5

    for _ in range(max_iter):
        mu0_old = mu0.copy()
        mu1_old = mu1.copy()
        lamb_old = lamb

        w = responsibility(Y, mu0, mu1, lamb, Sigma)

        lamb = float(np.mean(w))

        den1 = np.sum(w)
        den0 = np.sum(1 - w)

        mu1 = np.sum(w[:, None] * Y, axis=0) / max(den1, 1e-12)
        mu0 = np.sum((1 - w)[:, None] * Y, axis=0) / max(den0, 1e-12)

        change = (
            np.linalg.norm(mu0 - mu0_old)
            + np.linalg.norm(mu1 - mu1_old)
            + abs(lamb - lamb_old)
        )

        if change < tol:
            break

    if np.mean(mu0) > np.mean(mu1):
        mu0, mu1 = mu1, mu0
        lamb = 1 - lamb

    return {
        "mu0": mu0,
        "mu1": mu1,
        "lambda": lamb
    }
    
    

def fit_pooled_em(y_list, Sigma, mu0_init=None, mu1_init=None,
                  lambda_init=None, max_iter=100, tol=1e-6, seed=0):
    K = len(y_list)

    if mu0_init is None or mu1_init is None:
        local_init = fit_local_em(y_list[0], Sigma, seed=seed)
        mu0 = local_init["mu0"].copy()
        mu1 = local_init["mu1"].copy()
    else:
        mu0 = mu0_init.copy()
        mu1 = mu1_init.copy()

    if lambda_init is None:
        lambdas = np.ones(K) * 0.5
    else:
        lambdas = np.asarray(lambda_init, dtype=float).copy()

    for _ in range(max_iter):
        mu0_old = mu0.copy()
        mu1_old = mu1.copy()
        lambdas_old = lambdas.copy()

        num0 = np.zeros_like(mu0)
        num1 = np.zeros_like(mu1)
        den0 = 0.0
        den1 = 0.0

        new_lambdas = np.zeros(K)

        for j, Y in enumerate(y_list):
            w = responsibility(Y, mu0, mu1, lambdas[j], Sigma)

            new_lambdas[j] = np.mean(w)

            num1 += np.sum(w[:, None] * Y, axis=0)
            den1 += np.sum(w)

            num0 += np.sum((1 - w)[:, None] * Y, axis=0)
            den0 += np.sum(1 - w)

        mu1 = num1 / max(den1, 1e-12)
        mu0 = num0 / max(den0, 1e-12)
        lambdas = new_lambdas

        change = (
            np.linalg.norm(mu0 - mu0_old)
            + np.linalg.norm(mu1 - mu1_old)
            + np.linalg.norm(lambdas - lambdas_old)
        )

        if change < tol:
            break

    if np.mean(mu0) > np.mean(mu1):
        mu0, mu1 = mu1, mu0
        lambdas = 1 - lambdas

    return {
        "mu0": mu0,
        "mu1": mu1,
        "lambda_list": lambdas
    }


# ============================================================
# 4. Average estimator: local EM + label matching
# ============================================================

def fit_average_estimator(y_list, Sigma, max_iter=100, tol=1e-6, seed=0):
    K = len(y_list)

    local_fits = []
    for j, Y in enumerate(y_list):
        fit_j = fit_local_em(
            Y,
            Sigma,
            max_iter=max_iter,
            tol=tol,
            seed=seed + 100 * j
        )
        local_fits.append(fit_j)

    lead_mu0 = local_fits[0]["mu0"]
    lead_mu1 = local_fits[0]["mu1"]

    mu0_sum = lead_mu0.copy()
    mu1_sum = lead_mu1.copy()

    for j in range(1, K):
        mu0_j = local_fits[j]["mu0"]
        mu1_j = local_fits[j]["mu1"]

        a1 = np.linalg.norm(mu1_j - lead_mu1) + np.linalg.norm(mu0_j - lead_mu0)
        a2 = np.linalg.norm(mu1_j - lead_mu0) + np.linalg.norm(mu0_j - lead_mu1)

        if a1 < a2:
            mu1_sum += mu1_j
            mu0_sum += mu0_j
        else:
            mu1_sum += mu0_j
            mu0_sum += mu1_j

    return {
        "mu0": mu0_sum / K,
        "mu1": mu1_sum / K,
        "local_fits": local_fits
    }


####### TRAINER CLASS #######

def run_one_replicate(n, K, sigma2, a, d=5, seed=0, max_iter=50):
    rng = np.random.default_rng(seed)

    mu0_true = np.ones(d) * 4.0
    mu1_true = np.ones(d) * 5.0
    Sigma = sigma2 * np.eye(d)

    lambda_true = rng.uniform(0.5 - a, 0.5 + a, size=K)

    y_list = []
    for j in range(K):
        yj = generate_site_gmm(
            n=n,
            d=d,
            lamb=lambda_true[j],
            mu0=mu0_true,
            mu1=mu1_true,
            Sigma=Sigma,
            seed=seed * 10000 + j
        )
        y_list.append(yj)

    # local initialization from lead site
    local_init = fit_local_em(y_list[0], Sigma, seed=seed)
    mu0_init = local_init["mu0"]
    mu1_init = local_init["mu1"]
    lambda_init = [0.5] * K

    # Distributed EM
    dist_trainer = Trainer(
        y_list=y_list,
        Sigma=Sigma,
        mu0_init=mu0_init,
        mu1_init=mu1_init,
        lambda_init=lambda_init,
        max_iter=max_iter,
        tol=1e-6,
        verbose=False
    )
    dist_fit = dist_trainer.fit()

    # Pooled EM
    pooled_fit = fit_pooled_em(
        y_list,
        Sigma,
        mu0_init=mu0_init,
        mu1_init=mu1_init,
        lambda_init=lambda_init,
        max_iter=max_iter,
        tol=1e-6,
        seed=seed
    )

    # Average estimator
    avg_fit = fit_average_estimator(
        y_list,
        Sigma,
        max_iter=max_iter,
        tol=1e-6,
        seed=seed
    )

    methods = {
        "Average": avg_fit,
        "Pooled": pooled_fit,
        "Distributed EM": dist_fit
    }

    records = []

    for method, fit in methods.items():
        mu0_hat = np.asarray(fit["mu0"])
        mu1_hat = np.asarray(fit["mu1"])

        # label safety
        if np.mean(mu0_hat) > np.mean(mu1_hat):
            mu0_hat, mu1_hat = mu1_hat, mu0_hat

        mu_hat = np.r_[mu0_hat, mu1_hat]
        mu_true = np.r_[mu0_true, mu1_true]

        mse = np.linalg.norm(mu_hat - mu_true) / np.sqrt(2 * d)
        bias_mu01 = mu0_hat[0] - mu0_true[0]

        records.append({
            "method": method,
            "bias_mu01": bias_mu01,
            "mse": mse,
            "mu0_hat": mu0_hat,
            "mu1_hat": mu1_hat
        })

    approx_path = compute_approximation_path(
        y_list=y_list,
        Sigma=Sigma,
        mu0_init=mu0_init,
        mu1_init=mu1_init,
        lambda_init=lambda_init,
        max_iter=max_iter
    )

    return records, approx_path


def compute_approximation_path(y_list, Sigma, mu0_init, mu1_init,
                               lambda_init, max_iter=50):
    K = len(y_list)

    dist_trainer = Trainer(
        y_list=y_list,
        Sigma=Sigma,
        mu0_init=mu0_init,
        mu1_init=mu1_init,
        lambda_init=lambda_init,
        max_iter=1,
        tol=0,
        verbose=False
    )

    pooled_mu0 = mu0_init.copy()
    pooled_mu1 = mu1_init.copy()
    pooled_lambdas = np.asarray(lambda_init, dtype=float).copy()

    approx_errors = []

    for _ in range(max_iter):
        # one distributed step
        dist_info = dist_trainer.step()
        dist_mu = np.r_[dist_info["mu0"], dist_info["mu1"]]

        # one pooled step
        pooled_next = fit_pooled_em_one_step(
            y_list,
            Sigma,
            pooled_mu0,
            pooled_mu1,
            pooled_lambdas
        )

        pooled_mu0 = pooled_next["mu0"]
        pooled_mu1 = pooled_next["mu1"]
        pooled_lambdas = pooled_next["lambda_list"]

        pooled_mu = np.r_[pooled_mu0, pooled_mu1]

        err = np.linalg.norm(dist_mu - pooled_mu) / max(np.linalg.norm(pooled_mu), 1e-12)
        approx_errors.append(err)

    return np.asarray(approx_errors)


def fit_pooled_em_one_step(y_list, Sigma, mu0, mu1, lambdas):
    K = len(y_list)

    num0 = np.zeros_like(mu0)
    num1 = np.zeros_like(mu1)
    den0 = 0.0
    den1 = 0.0

    new_lambdas = np.zeros(K)

    for j, Y in enumerate(y_list):
        w = responsibility(Y, mu0, mu1, lambdas[j], Sigma)

        new_lambdas[j] = np.mean(w)

        num1 += np.sum(w[:, None] * Y, axis=0)
        den1 += np.sum(w)

        num0 += np.sum((1 - w)[:, None] * Y, axis=0)
        den0 += np.sum(1 - w)

    return {
        "mu0": num0 / max(den0, 1e-12),
        "mu1": num1 / max(den1, 1e-12),
        "lambda_list": new_lambdas
    }



def run_section5_simulation(
    n_list=(1000,),
    K_list=(10, 30),
    sigma2_list=(2.5, 5.0),
    a_list=(0.1, 0.3),
    n_rep=200,
    d=5,
    max_iter=50,
    seed=123
):
    all_records = []
    approx_records = []

    total = (
        len(n_list)
        * len(K_list)
        * len(sigma2_list)
        * len(a_list)
        * n_rep
    )
    count = 0

    for n in n_list:
        for K in K_list:
            for sigma2 in sigma2_list:
                for a in a_list:
                    for rep in range(n_rep):
                        count += 1
                        this_seed = seed + 100000 * rep + 1000 * K + int(100 * sigma2) + int(10 * a)

                        print(
                            f"[{count}/{total}] "
                            f"n={n}, K={K}, sigma2={sigma2}, a={a}, rep={rep}"
                        )

                        records, approx_path = run_one_replicate(
                            n=n,
                            K=K,
                            sigma2=sigma2,
                            a=a,
                            d=d,
                            seed=this_seed,
                            max_iter=max_iter
                        )

                        for r in records:
                            all_records.append({
                                "n": n,
                                "K": K,
                                "sigma2": sigma2,
                                "a": a,
                                "rep": rep,
                                "method": r["method"],
                                "bias_mu01": r["bias_mu01"],
                                "mse": r["mse"]
                            })

                        for it, val in enumerate(approx_path, start=1):
                            approx_records.append({
                                "n": n,
                                "K": K,
                                "sigma2": sigma2,
                                "a": a,
                                "rep": rep,
                                "iteration": it,
                                "approx_error": val
                            })

    return pd.DataFrame(all_records), pd.DataFrame(approx_records)


##### PLOT #####
def plot_basic(df_path, n=1000, K=10):
    sub = df_path[(df_path["n"] == n) & (df_path["K"] == K)]

    scenarios = [
        (2.5, 0.1),
        (2.5, 0.3),
        (5.0, 0.1),
        (5.0, 0.3)
    ]

    plt.figure(figsize=(14, 3))

    for idx, (sigma2, a) in enumerate(scenarios, start=1):
        ax = plt.subplot(1, 4, idx)
        ss = sub[(sub["sigma2"] == sigma2) & (sub["a"] == a)]

        # 随机选一个 rep 画 path
        rep0 = ss["rep"].min()
        ss = ss[ss["rep"] == rep0]

        ax.plot(ss["iteration"], ss["approx_error"], marker="o", markersize=3)
        ax.set_title(rf"$\sigma^2={sigma2}, a={a}$")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Approximation error")

    plt.tight_layout()
    plt.show()


def plot_bias_boxplot(df_res, n=1000):
    sub = df_res[df_res["n"] == n].copy()

    scenarios = []
    for K in sorted(sub["K"].unique()):
        for sigma2 in sorted(sub["sigma2"].unique()):
            for a in sorted(sub["a"].unique()):
                scenarios.append((K, sigma2, a))

    fig, axes = plt.subplots(
        len(sorted(sub["K"].unique())),
        4,
        figsize=(16, 7),
        sharey=True
    )

    for row, K in enumerate(sorted(sub["K"].unique())):
        for col, (sigma2, a) in enumerate([(2.5,0.1),(2.5,0.3),(5.0,0.1),(5.0,0.3)]):
            ax = axes[row, col] if len(sorted(sub["K"].unique())) > 1 else axes[col]
            ss = sub[(sub["K"] == K) & (sub["sigma2"] == sigma2) & (sub["a"] == a)]

            data = [
                ss[ss["method"] == "Average"]["bias_mu01"],
                ss[ss["method"] == "Pooled"]["bias_mu01"],
                ss[ss["method"] == "Distributed EM"]["bias_mu01"]
            ]

            ax.boxplot(data, tick_labels=["Average", "Pooled", "Distributed EM"])
            ax.axhline(0, linestyle="--", linewidth=1)
            ax.set_title(rf"$K={K}, \sigma^2={sigma2}, a={a}$")
            ax.tick_params(axis="x", rotation=30)

            if col == 0:
                ax.set_ylabel("Bias")

    plt.tight_layout()
    plt.show()


def plot_mse_boxplot(df_res, n=1000):
    sub = df_res[df_res["n"] == n].copy()

    fig, axes = plt.subplots(
        len(sorted(sub["K"].unique())),
        4,
        figsize=(16, 7),
        sharey=True
    )

    for row, K in enumerate(sorted(sub["K"].unique())):
        for col, (sigma2, a) in enumerate([(2.5,0.1),(2.5,0.3),(5.0,0.1),(5.0,0.3)]):
            ax = axes[row, col] if len(sorted(sub["K"].unique())) > 1 else axes[col]
            ss = sub[(sub["K"] == K) & (sub["sigma2"] == sigma2) & (sub["a"] == a)]

            data = [
                ss[ss["method"] == "Average"]["mse"],
                ss[ss["method"] == "Pooled"]["mse"],
                ss[ss["method"] == "Distributed EM"]["mse"]
            ]

            ax.boxplot(data, tick_labels=["Average", "Pooled", "Distributed EM"])
            ax.set_title(rf"$K={K}, \sigma^2={sigma2}, a={a}$")
            ax.tick_params(axis="x", rotation=30)

            if col == 0:
                ax.set_ylabel("Mean squared error")

    plt.tight_layout()
    plt.show()