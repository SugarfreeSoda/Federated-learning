import numpy as np
from numpy.linalg import inv, norm
from scipy.special import logsumexp
import matplotlib.pyplot as plt

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



class BaseSite:
    def __init__(self, y, site_id, Sigma):
        self.y = np.asarray(y, dtype=float)          # shape: (n, d)
        self.site_id = site_id
        self.n, self.d = self.y.shape

        self.Sigma = np.asarray(Sigma, dtype=float)
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.logdetSigma = np.linalg.slogdet(self.Sigma)[1]


    def log_gaussian_density(self, Y, mu):

        diff = Y - mu
        quad = np.sum((diff @ self.Sigma_inv) * diff, axis=1)
        return -0.5 * (self.d * np.log(2.0 * np.pi) + self.logdetSigma + quad)

    def gaussian_density(self, Y, mu):
        return np.exp(self.log_gaussian_density(Y, mu))


    def responsibility(self, mu0, mu1, lamb, Y=None):
        if Y is None:
            Y = self.y

        lamb = np.clip(lamb, 1e-8, 1 - 1e-8)

        logp1 = np.log(lamb) + self.log_gaussian_density(Y, mu1)
        logp0 = np.log(1.0 - lamb) + self.log_gaussian_density(Y, mu0)

        m = np.maximum(logp1, logp0)
        p1 = np.exp(logp1 - m)
        p0 = np.exp(logp0 - m)

        w = p1 / (p1 + p0)
        return np.clip(w, 1e-8, 1 - 1e-8)


    def lambda_tilde(self, mu0_t, mu1_t, lambda_t):
        w = self.responsibility(mu0_t, mu1_t, lambda_t, self.y)
        return float(np.mean(w))


    def local_Q_mu(self, mu0, mu1, mu0_t, mu1_t, lambda_t):
        w = self.responsibility(mu0_t, mu1_t, lambda_t, self.y)

        term1 = w * self.log_gaussian_density(self.y, mu1)
        term0 = (1.0 - w) * self.log_gaussian_density(self.y, mu0)

        return np.mean(term1 + term0)


    def grad_local_Q_mu_at_current(self, mu0_t, mu1_t, lambda_t):
        w = self.responsibility(mu0_t, mu1_t, lambda_t, self.y)

        diff1 = self.y - mu1_t
        diff0 = self.y - mu0_t

        grad_mu1 = np.mean((w[:, None] * diff1) @ self.Sigma_inv, axis=0)
        grad_mu0 = np.mean(((1.0 - w)[:, None] * diff0) @ self.Sigma_inv, axis=0)

        return grad_mu0, grad_mu1

    def summary_to_send(self, mu0_t, mu1_t, lambda_t):
        lambda_j_tilde = self.lambda_tilde(mu0_t, mu1_t, lambda_t)
        grad_mu0_j, grad_mu1_j = self.grad_local_Q_mu_at_current(mu0_t, mu1_t, lambda_t)

        return {
            "site_id": self.site_id,
            "lambda_tilde": lambda_j_tilde,
            "grad_mu0": grad_mu0_j,
            "grad_mu1": grad_mu1_j,
        }


class RemoteSite(BaseSite):
    pass

class LeadSite(BaseSite):

    def density_ratio_t(self, Y_lead, mu0_t, mu1_t, lambda_1_t, lambda_j_t):
        f1 = self.gaussian_density(Y_lead, mu1_t)
        f0 = self.gaussian_density(Y_lead, mu0_t)

        num = lambda_j_t * f1 + (1.0 - lambda_j_t) * f0
        den = lambda_1_t * f1 + (1.0 - lambda_1_t) * f0

        return num / np.clip(den, 1e-12, None)

    def check_Q_mu(self, mu0, mu1, mu0_t, mu1_t, lambda_list_t):
        Y = self.y
        K = len(lambda_list_t)
        lambda_1_t = lambda_list_t[0]

        total = 0.0
        for j in range(K):
            lambda_j_t = lambda_list_t[j]
            t_j = self.density_ratio_t(Y, mu0_t, mu1_t, lambda_1_t, lambda_j_t)
            w_j = self.responsibility(mu0_t, mu1_t, lambda_j_t, Y)

            term1 = w_j * self.log_gaussian_density(Y, mu1)
            term0 = (1.0 - w_j) * self.log_gaussian_density(Y, mu0)

            total += np.mean(t_j * (term1 + term0))

        return total / K

    def grad_check_Q_mu(self, mu0, mu1, mu0_t, mu1_t, lambda_list_t):

        Y = self.y
        K = len(lambda_list_t)
        lambda_1_t = lambda_list_t[0]

        grad_mu0 = np.zeros(self.d)
        grad_mu1 = np.zeros(self.d)

        for j in range(K):
            lambda_j_t = lambda_list_t[j]
            t_j = self.density_ratio_t(Y, mu0_t, mu1_t, lambda_1_t, lambda_j_t)
            w_j = self.responsibility(mu0_t, mu1_t, lambda_j_t, Y)

            diff1 = Y - mu1
            diff0 = Y - mu0

            grad_mu1_j = np.mean((t_j[:, None] * w_j[:, None] * diff1) @ self.Sigma_inv, axis=0)
            grad_mu0_j = np.mean((t_j[:, None] * (1.0 - w_j)[:, None] * diff0) @ self.Sigma_inv, axis=0)

            grad_mu1 += grad_mu1_j
            grad_mu0 += grad_mu0_j

        grad_mu0 /= K
        grad_mu1 /= K
        return grad_mu0, grad_mu1

    def grad_check_Q_mu_at_current(self, mu0_t, mu1_t, lambda_list_t):
        return self.grad_check_Q_mu(mu0_t, mu1_t, mu0_t, mu1_t, lambda_list_t)

    def grad_global_Q_mu_from_summaries(self, summaries):
        K = len(summaries)
        grad_mu0 = np.zeros(self.d)
        grad_mu1 = np.zeros(self.d)

        for s in summaries:
            grad_mu0 += s["grad_mu0"]
            grad_mu1 += s["grad_mu1"]

        grad_mu0 /= K
        grad_mu1 /= K
        return grad_mu0, grad_mu1

    def delta_Q(self, mu0, mu1, mu0_t, mu1_t, lambda_list_t, summaries):
        gradQ_mu0, gradQ_mu1 = self.grad_global_Q_mu_from_summaries(summaries)
        gradC_mu0, gradC_mu1 = self.grad_check_Q_mu_at_current(mu0_t, mu1_t, lambda_list_t)

        a0 = gradQ_mu0 - gradC_mu0
        a1 = gradQ_mu1 - gradC_mu1

        return float(np.dot(a0, mu0) + np.dot(a1, mu1))

    def surrogate_Q(self, mu0, mu1, mu0_t, mu1_t, lambda_list_t, summaries):
        q_check = self.check_Q_mu(mu0, mu1, mu0_t, mu1_t, lambda_list_t)
        dq = self.delta_Q(mu0, mu1, mu0_t, mu1_t, lambda_list_t, summaries)
        return q_check + dq

    def surrogate_grad(self, mu0, mu1, mu0_t, mu1_t, lambda_list_t, summaries):

        grad_check_mu0, grad_check_mu1 = self.grad_check_Q_mu(
            mu0, mu1, mu0_t, mu1_t, lambda_list_t
        )

        gradQ_mu0, gradQ_mu1 = self.grad_global_Q_mu_from_summaries(summaries)
        gradC0_t, gradC1_t = self.grad_check_Q_mu_at_current(mu0_t, mu1_t, lambda_list_t)

        a0 = gradQ_mu0 - gradC0_t
        a1 = gradQ_mu1 - gradC1_t

        grad_mu0 = grad_check_mu0 + a0
        grad_mu1 = grad_check_mu1 + a1

        return grad_mu0, grad_mu1
    
    
    
import numpy as np
import matplotlib.pyplot as plt


class Trainer:


    def __init__(
        self,
        y_list,
        Sigma,
        mu0_init,
        mu1_init,
        lambda_init=None,
        max_iter=50,
        tol=1e-5,
        inner_max_iter=50,
        inner_lr=0.1,
        verbose=True
    ):
        self.y_list = y_list
        self.K = len(y_list)
        self.Sigma = np.asarray(Sigma, dtype=float)

        self.mu0 = np.asarray(mu0_init, dtype=float).copy()
        self.mu1 = np.asarray(mu1_init, dtype=float).copy()

        if lambda_init is None:
            self.lambda_list = [0.5 for _ in range(self.K)]
        else:
            self.lambda_list = list(lambda_init)

        self.max_iter = max_iter
        self.tol = tol
        self.inner_max_iter = inner_max_iter
        self.inner_lr = inner_lr
        self.verbose = verbose

        # build sites
        self.lead_site = LeadSite(y_list[0], site_id=1, Sigma=self.Sigma)
        self.remote_sites = [
            RemoteSite(y_list[j], site_id=j + 1, Sigma=self.Sigma)
            for j in range(1, self.K)
        ]
        self.sites = [self.lead_site] + self.remote_sites

        self.history = []

    def collect_summaries(self, mu0_t, mu1_t, lambda_list_t):
        summaries = []
        for j, site in enumerate(self.sites):
            s = site.summary_to_send(
                mu0_t=mu0_t,
                mu1_t=mu1_t,
                lambda_t=lambda_list_t[j]
            )
            summaries.append(s)
        return summaries


    def update_lambdas(self, summaries):
        lambda_next = [float(s["lambda_tilde"]) for s in summaries]
        return lambda_next



    def update_mu(self, mu0_t, mu1_t, lambda_list_t, summaries):

        Y = self.lead_site.y
        K = len(lambda_list_t)
        lambda_1_t = lambda_list_t[0]

        gradQ_mu0, gradQ_mu1 = self.lead_site.grad_global_Q_mu_from_summaries(summaries)
        gradC_mu0, gradC_mu1 = self.lead_site.grad_check_Q_mu_at_current(
            mu0_t, mu1_t, lambda_list_t
        )

        a0 = gradQ_mu0 - gradC_mu0
        a1 = gradQ_mu1 - gradC_mu1

        num0 = np.zeros(self.lead_site.d)
        num1 = np.zeros(self.lead_site.d)
        den0 = 0.0
        den1 = 0.0

        for j in range(K):
            lambda_j_t = lambda_list_t[j]

            t_j = self.lead_site.density_ratio_t(
                Y_lead=Y,
                mu0_t=mu0_t,
                mu1_t=mu1_t,
                lambda_1_t=lambda_1_t,
                lambda_j_t=lambda_j_t
            )

            w_j = self.lead_site.responsibility(
                mu0=mu0_t,
                mu1=mu1_t,
                lamb=lambda_j_t,
                Y=Y
            )

            weight1 = t_j * w_j
            weight0 = t_j * (1.0 - w_j)

            num1 += np.mean(weight1[:, None] * Y, axis=0)
            den1 += np.mean(weight1)

            num0 += np.mean(weight0[:, None] * Y, axis=0)
            den0 += np.mean(weight0)

        num0 /= K
        num1 /= K
        den0 /= K
        den1 /= K

        eps = 1e-12
        den0 = max(den0, eps)
        den1 = max(den1, eps)

        mu0_next = (num0 + a0 @ self.Sigma) / den0
        mu1_next = (num1 + a1 @ self.Sigma) / den1

        q_value = self.lead_site.surrogate_Q(
            mu0=mu0_next,
            mu1=mu1_next,
            mu0_t=mu0_t,
            mu1_t=mu1_t,
            lambda_list_t=lambda_list_t,
            summaries=summaries
        )

        return mu0_next, mu1_next, q_value








    def step(self):
        mu0_t = self.mu0.copy()
        mu1_t = self.mu1.copy()
        lambda_list_t = self.lambda_list.copy()

        summaries = self.collect_summaries(mu0_t, mu1_t, lambda_list_t)

        lambda_next = self.update_lambdas(summaries)
        
        mu0_next, mu1_next, q_value = self.update_mu(
            mu0_t=mu0_t,
            mu1_t=mu1_t,
            lambda_list_t=lambda_list_t,
            summaries=summaries
        )


        mu_change = np.linalg.norm(mu0_next - mu0_t) + np.linalg.norm(mu1_next - mu1_t)
        lambda_change = np.linalg.norm(np.array(lambda_next) - np.array(lambda_list_t))
        total_change = mu_change + lambda_change


        self.mu0 = mu0_next
        self.mu1 = mu1_next
        self.lambda_list = lambda_next

        info = {
            "mu0": mu0_next.copy(),
            "mu1": mu1_next.copy(),
            "lambda_list": lambda_next.copy(),
            "q_value": float(q_value),
            "mu_change": float(mu_change),
            "lambda_change": float(lambda_change),
            "total_change": float(total_change),
        }
        self.history.append(info)

        return info


    def fit(self):
        for t in range(self.max_iter):
            info = self.step()

            if self.verbose:
                print(
                    f"[Iter {t+1:03d}] "
                    f"Q={info['q_value']:.6f}, "
                    f"mu_change={info['mu_change']:.6e}, "
                    f"lambda_change={info['lambda_change']:.6e}, "
                    f"total_change={info['total_change']:.6e}"
                )

            if info["total_change"] < self.tol:
                if self.verbose:
                    print("Converged.")
                break

        return {
            "mu0": self.mu0,
            "mu1": self.mu1,
            "lambda_list": self.lambda_list,
            "history": self.history,
        }


    def plot_history(self):
        if len(self.history) == 0:
            print("No history found. Run fit() first.")
            return

        q_values = [h["q_value"] for h in self.history]
        mu_changes = [h["mu_change"] for h in self.history]
        lambda_changes = [h["lambda_change"] for h in self.history]
        total_changes = [h["total_change"] for h in self.history]

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(q_values, marker="o")
        plt.title("Surrogate Q")
        plt.xlabel("Iteration")
        plt.ylabel("Q value")

        plt.subplot(1, 3, 2)
        plt.plot(mu_changes, marker="o", label="mu change")
        plt.plot(lambda_changes, marker="s", label="lambda change")
        plt.title("Parameter changes")
        plt.xlabel("Iteration")
        plt.ylabel("Change")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(total_changes, marker="o")
        plt.title("Total change")
        plt.xlabel("Iteration")
        plt.ylabel("Total change")
        plt.yscale("log")

        plt.tight_layout()
        plt.show()