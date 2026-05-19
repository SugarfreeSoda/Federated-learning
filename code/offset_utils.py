import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import Trainer as Trainer_old, BaseSite as BaseSite_old
from utils_new import Trainer_new, BaseSite_new

def generate_shifted_sites(
    n=1000,
    K=10,
    d=5,
    sigma2=2.5,
    a=0.1,
    tau=0.0,
    seed=0,
):

    rng = np.random.default_rng(seed)

    mu0_true = np.ones(d) * 4.0
    mu1_true = np.ones(d) * 5.0
    Sigma = sigma2 * np.eye(d)

    lambdas_true = rng.uniform(0.5 - a, 0.5 + a, size=K)

    y_list = []
    delta0_list = []
    delta1_list = []

    L = np.linalg.cholesky(Sigma)

    for j in range(K):
        delta0 = rng.normal(0.0, tau, size=d)
        delta1 = rng.normal(0.0, tau, size=d)

        z = rng.binomial(1, lambdas_true[j], size=n)
        eps = rng.normal(size=(n, d)) @ L.T

        y = np.where(
            z[:, None] == 1,
            mu1_true + delta1,
            mu0_true + delta0,
        ) + eps

        y_list.append(y)
        delta0_list.append(delta0)
        delta1_list.append(delta1)

    truth = {
        "mu0": mu0_true,
        "mu1": mu1_true,
        "Sigma": Sigma,
        "lambda": lambdas_true,
        "delta0": np.vstack(delta0_list),
        "delta1": np.vstack(delta1_list),
    }

    return y_list, truth




def init_from_kmeans(y, seed=0):
    km = KMeans(n_clusters=2, n_init=5, random_state=seed)
    labels = km.fit_predict(y)
    centers = km.cluster_centers_

    order = np.argsort(centers.mean(axis=1))
    mu0_init = centers[order[0]]
    mu1_init = centers[order[1]]

    return mu0_init, mu1_init


def pooled_em(
    y_list,
    Sigma,
    mu0_init,
    mu1_init,
    lambda_init=None,
    max_iter=100,
    tol=1e-6,
):

    K = len(y_list)
    d = y_list[0].shape[1]

    sites = [BaseSite_old(y, site_id=j + 1, Sigma=Sigma) for j, y in enumerate(y_list)]

    mu0 = mu0_init.copy()
    mu1 = mu1_init.copy()

    if lambda_init is None:
        lambda_list = np.ones(K) * 0.5
    else:
        lambda_list = np.asarray(lambda_init, dtype=float).copy()

    history = []

    for _ in range(max_iter):
        mu0_old = mu0.copy()
        mu1_old = mu1.copy()
        lambda_old = lambda_list.copy()

        w_list = []
        for j, site in enumerate(sites):
            w = site.responsibility(mu0, mu1, lambda_list[j])
            w_list.append(w)
        lambda_list = np.array([np.mean(w) for w in w_list])
        lambda_list = np.clip(lambda_list, 1e-8, 1 - 1e-8)
        num0 = np.zeros(d)
        num1 = np.zeros(d)
        den0 = 0.0
        den1 = 0.0

        for y, w in zip(y_list, w_list):
            num1 += np.sum(w[:, None] * y, axis=0)
            den1 += np.sum(w)

            num0 += np.sum((1.0 - w)[:, None] * y, axis=0)
            den0 += np.sum(1.0 - w)

        mu1 = num1 / max(den1, 1e-12)
        mu0 = num0 / max(den0, 1e-12)

        change = (
            np.linalg.norm(mu0 - mu0_old)
            + np.linalg.norm(mu1 - mu1_old)
            + np.linalg.norm(lambda_list - lambda_old)
        )

        history.append({
            "mu0": mu0.copy(),
            "mu1": mu1.copy(),
            "lambda_list": lambda_list.copy(),
            "change": change,
        })

        if change < tol:
            break

    return {
        "mu0": mu0,
        "mu1": mu1,
        "lambda_list": lambda_list,
        "history": history,
    }



def local_em_single_site(
    y,
    Sigma,
    mu0_init,
    mu1_init,
    lambda_init=0.5,
    max_iter=100,
    tol=1e-6,
):
    site = BaseSite_old(y, site_id=1, Sigma=Sigma)

    mu0 = mu0_init.copy()
    mu1 = mu1_init.copy()
    lamb = float(lambda_init)

    for _ in range(max_iter):
        mu0_old = mu0.copy()
        mu1_old = mu1.copy()
        lamb_old = lamb

        w = site.responsibility(mu0, mu1, lamb)

        lamb = np.mean(w)
        lamb = float(np.clip(lamb, 1e-8, 1 - 1e-8))

        mu1 = np.sum(w[:, None] * y, axis=0) / max(np.sum(w), 1e-12)
        mu0 = np.sum((1.0 - w)[:, None] * y, axis=0) / max(np.sum(1.0 - w), 1e-12)

        change = (
            np.linalg.norm(mu0 - mu0_old)
            + np.linalg.norm(mu1 - mu1_old)
            + abs(lamb - lamb_old)
        )

        if change < tol:
            break

    if mu0.mean() > mu1.mean():
        mu0, mu1 = mu1, mu0
        lamb = 1.0 - lamb

    return {
        "mu0": mu0,
        "mu1": mu1,
        "lambda": lamb,
    }


def average_estimator(y_list, Sigma, seed=0):
    K = len(y_list)

    local_fits = []
    for j, y in enumerate(y_list):
        mu0_init, mu1_init = init_from_kmeans(y, seed=seed + j)
        fit = local_em_single_site(
            y=y,
            Sigma=Sigma,
            mu0_init=mu0_init,
            mu1_init=mu1_init,
            lambda_init=0.5,
            max_iter=100,
            tol=1e-6,
        )
        local_fits.append(fit)

    anchor_mu0 = local_fits[0]["mu0"]
    anchor_mu1 = local_fits[0]["mu1"]

    mu0_sum = anchor_mu0.copy()
    mu1_sum = anchor_mu1.copy()

    for j in range(1, K):
        mu0_j = local_fits[j]["mu0"]
        mu1_j = local_fits[j]["mu1"]

        dist_same = np.linalg.norm(mu1_j - anchor_mu1) + np.linalg.norm(mu0_j - anchor_mu0)
        dist_swap = np.linalg.norm(mu1_j - anchor_mu0) + np.linalg.norm(mu0_j - anchor_mu1)

        if dist_same <= dist_swap:
            mu0_sum += mu0_j
            mu1_sum += mu1_j
        else:
            mu0_sum += mu1_j
            mu1_sum += mu0_j

    return {
        "mu0": mu0_sum / K,
        "mu1": mu1_sum / K,
        "local_fits": local_fits,
    }



def align_mu(mu0_hat, mu1_hat, mu0_true, mu1_true):
    dist_same = np.linalg.norm(mu0_hat - mu0_true) + np.linalg.norm(mu1_hat - mu1_true)
    dist_swap = np.linalg.norm(mu0_hat - mu1_true) + np.linalg.norm(mu1_hat - mu0_true)

    if dist_same <= dist_swap:
        return mu0_hat, mu1_hat, False
    else:
        return mu1_hat, mu0_hat, True


def global_center_error(mu0_hat, mu1_hat, truth):
    mu0_true = truth["mu0"]
    mu1_true = truth["mu1"]
    d = len(mu0_true)

    mu0_hat, mu1_hat, _ = align_mu(mu0_hat, mu1_hat, mu0_true, mu1_true)

    err = np.sqrt(
        np.sum((mu0_hat - mu0_true) ** 2)
        + np.sum((mu1_hat - mu1_true) ** 2)
    ) / np.sqrt(2 * d)

    return float(err)


def site_specific_error(mu0_hat, mu1_hat, truth):
    mu0_true = truth["mu0"]
    mu1_true = truth["mu1"]
    delta0 = truth["delta0"]
    delta1 = truth["delta1"]
    K = delta0.shape[0]

    mu0_hat, mu1_hat, _ = align_mu(mu0_hat, mu1_hat, mu0_true, mu1_true)

    mse0 = np.mean([
        np.mean((mu0_hat - (mu0_true + delta0[j])) ** 2)
        for j in range(K)
    ])

    mse1 = np.mean([
        np.mean((mu1_hat - (mu1_true + delta1[j])) ** 2)
        for j in range(K)
    ])

    return float((mse0 + mse1) / 2.0)


def delta_site_specific_error(fit, truth):
    if "delta0_list" not in fit or "delta1_list" not in fit:
        return np.nan

    mu0_true = truth["mu0"]
    mu1_true = truth["mu1"]
    delta0_true = truth["delta0"]
    delta1_true = truth["delta1"]

    raw_mu0 = fit["mu0"]
    raw_mu1 = fit["mu1"]
    delta0_hat = np.asarray(fit["delta0_list"])
    delta1_hat = np.asarray(fit["delta1_list"])

    _, _, swapped = align_mu(raw_mu0, raw_mu1, mu0_true, mu1_true)

    if not swapped:
        mu0_hat, mu1_hat = raw_mu0, raw_mu1
    else:
        mu0_hat, mu1_hat = raw_mu1, raw_mu0
        delta0_hat, delta1_hat = delta1_hat, delta0_hat

    mse0 = np.mean((mu0_hat + delta0_hat - (mu0_true + delta0_true)) ** 2)
    mse1 = np.mean((mu1_hat + delta1_hat - (mu1_true + delta1_true)) ** 2)

    return float((mse0 + mse1) / 2.0)


def mu01_bias(mu0_hat, mu1_hat, truth):
    mu0_true = truth["mu0"]
    mu1_true = truth["mu1"]

    mu0_hat, mu1_hat, _ = align_mu(mu0_hat, mu1_hat, mu0_true, mu1_true)

    return float(mu0_hat[0] - mu0_true[0])


def run_one_experiment(
    n=1000,
    K=10,
    d=5,
    sigma2=2.5,
    a=0.1,
    tau=0.0,
    seed=0,
    max_iter=50,
    rho=5.0,
):
    y_list, truth = generate_shifted_sites(
        n=n,
        K=K,
        d=d,
        sigma2=sigma2,
        a=a,
        tau=tau,
        seed=seed,
    )

    Sigma = truth["Sigma"]

    mu0_init, mu1_init = init_from_kmeans(y_list[0], seed=seed)
    old_trainer = Trainer_old(
        y_list=y_list,
        Sigma=Sigma,
        mu0_init=mu0_init,
        mu1_init=mu1_init,
        lambda_init=None,
        max_iter=max_iter,
        tol=1e-6,
        inner_max_iter=50,
        inner_lr=0.1,
        verbose=False,
    )
    old_fit = old_trainer.fit()
    new_trainer = Trainer_new(
        y_list=y_list,
        Sigma=Sigma,
        mu0_init=mu0_init,
        mu1_init=mu1_init,
        lambda_init=None,
        rho=rho,
        max_iter=max_iter,
        tol=1e-6,
        inner_max_iter=50,
        inner_lr=0.1,
        verbose=False,
    )
    new_fit = new_trainer.fit()

    pooled_fit = pooled_em(
        y_list=y_list,
        Sigma=Sigma,
        mu0_init=mu0_init,
        mu1_init=mu1_init,
        lambda_init=None,
        max_iter=max_iter,
        tol=1e-6,
    )

    avg_fit = average_estimator(
        y_list=y_list,
        Sigma=Sigma,
        seed=seed,
    )

    results = []

    fits = [
        ("Average", avg_fit),
        ("Pooled EM", pooled_fit),
        ("Distributed EM", old_fit),
        ("Shift-Distributed EM", new_fit),
    ]

    for method, fit in fits:
        mu0_aligned, mu1_aligned, _ = align_mu(fit["mu0"], fit["mu1"], truth["mu0"], truth["mu1"])

        row = {
            "method": method,
            "n": n,
            "K": K,
            "d": d,
            "sigma2": sigma2,
            "a": a,
            "tau": tau,
            "rho": rho,
            "seed": seed,
            "global_error": global_center_error(fit["mu0"], fit["mu1"], truth),
            "site_specific_error": site_specific_error(fit["mu0"], fit["mu1"], truth),
            "delta_site_specific_error": delta_site_specific_error(fit, truth),
            "mu01_bias": mu01_bias(fit["mu0"], fit["mu1"], truth),
            "mu0_hat_0": mu0_aligned[0],
            "mu1_hat_0": mu1_aligned[0],
        }
        results.append(row)


    old_mu = np.concatenate([old_fit["mu0"], old_fit["mu1"]])
    new_mu = np.concatenate([new_fit["mu0"], new_fit["mu1"]])
    pool_mu = np.concatenate([pooled_fit["mu0"], pooled_fit["mu1"]])

    old_vs_pooled = np.linalg.norm(old_mu - pool_mu) / max(np.linalg.norm(pool_mu), 1e-12)
    new_vs_pooled = np.linalg.norm(new_mu - pool_mu) / max(np.linalg.norm(pool_mu), 1e-12)
    new_vs_old = np.linalg.norm(new_mu - old_mu) / max(np.linalg.norm(old_mu), 1e-12)

    for row in results:
        row["old_dist_vs_pooled_rel_error"] = old_vs_pooled
        row["new_dist_vs_pooled_rel_error"] = new_vs_pooled
        row["new_vs_old_rel_error"] = new_vs_old

    return pd.DataFrame(results)


def run_simulation(
    n=1000,
    K=10,
    d=5,
    sigma2=2.5,
    a=0.1,
    tau_grid=(0.0, 0.1, 0.25, 0.5, 1.0),
    reps=20,
    base_seed=2026,
    max_iter=50,
    rho=5.0,
):
    all_results = []

    total = len(tau_grid) * reps
    count = 0

    for tau in tau_grid:
        for r in range(reps):
            count += 1
            seed = base_seed + 1000 * int(tau * 100) + r

            print(f"[{count}/{total}] tau={tau}, rep={r+1}/{reps}")

            df_one = run_one_experiment(
                n=n,
                K=K,
                d=d,
                sigma2=sigma2,
                a=a,
                tau=tau,
                seed=seed,
                max_iter=max_iter,
                rho=rho,
            )
            df_one["rep"] = r
            all_results.append(df_one)

    return pd.concat(all_results, ignore_index=True)



def plot_metric_by_tau(df, metric="global_error"):
    summary = (
        df.groupby(["tau", "method"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(8, 4.5))

    methods = ["Average", "Pooled EM", "Distributed EM", "Shift-Distributed EM"]
    for method in methods:
        sub = summary[summary["method"] == method]
        if len(sub) == 0:
            continue
        plt.errorbar(
            sub["tau"],
            sub["mean"],
            yerr=sub["std"],
            marker="o",
            capsize=3,
            label=method,
        )

    plt.xlabel(r"Site-specific shift strength $\tau$")
    plt.ylabel(metric)
    plt.title(f"{metric} vs site-specific shift")
    plt.legend()
    plt.tight_layout()
    plt.show()


def boxplot_metric_by_tau(df, metric="global_error"):
    methods = ["Average", "Pooled EM", "Distributed EM", "Shift-Distributed EM"]
    tau_values = sorted(df["tau"].unique())

    fig, axes = plt.subplots(1, len(tau_values), figsize=(4.5 * len(tau_values), 4), sharey=True)

    if len(tau_values) == 1:
        axes = [axes]

    for ax, tau in zip(axes, tau_values):
        data = [
            df[(df["tau"] == tau) & (df["method"] == m)][metric].values
            for m in methods
        ]
        ax.boxplot(data, labels=methods)
        ax.set_title(rf"$\tau={tau}$")
        ax.tick_params(axis="x", rotation=30)

    axes[0].set_ylabel(metric)
    plt.tight_layout()
    plt.show()