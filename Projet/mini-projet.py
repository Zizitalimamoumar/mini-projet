import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# 1. Chargement et préparation des données (Titanic)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

data = data[["Survived", "Pclass", "Age", "Fare", "SibSp", "Parch"]]
data = data.dropna()

y = data["Survived"].values
y = np.where(y == 1, 1, -1)

X = data.drop(columns=["Survived"]).values
X = (X - X.mean(axis=0)) / X.std(axis=0)

n, d = X.shape
print("n =", n, ", d =", d)
# 2. Perte logistique et gradient
def logistic_loss(w, X, y):
    z = y * (X @ w)
    return np.mean(np.log(1 + np.exp(-z)))

def logistic_gradient(w, X, y):
    z = y * (X @ w)
    sigma = 1 / (1 + np.exp(z))
    return -(X.T @ (y * sigma)) / X.shape[0]
# 3. SGD
def sgd_time(X, y, lam, alpha0, n_iter):
    n, d = X.shape
    w = np.zeros(d)
    losses, times = [], []
    start = time.time()

    for k in range(n_iter):
        i = np.random.randint(n)
        grad = -(y[i] * X[i]) / (1 + np.exp(y[i] * X[i] @ w))
        w -= (alpha0 / (1 + k)) * grad

        losses.append(logistic_loss(w, X, y) + lam * np.linalg.norm(w, 1))
        times.append(time.time() - start)

    return w, times, losses
# 4. ADAM
def adam_time(X, y, lam, alpha, beta1, beta2, eps, n_iter):
    d = X.shape[1]
    w = np.zeros(d)
    m = np.zeros(d)
    v = np.zeros(d)

    losses, times = [], []
    start = time.time()

    for k in range(1, n_iter + 1):
        grad = logistic_gradient(w, X, y)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2

        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)

        w -= alpha * m_hat / (np.sqrt(v_hat) + eps)

        losses.append(logistic_loss(w, X, y) + lam * np.linalg.norm(w, 1))
        times.append(time.time() - start)

    return w, times, losses
# 5. Soft-thresholding (proximal L1)
def soft_thresholding(v, lam):
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0)
# 6. ISTA
def ista_time(X, y, lam, alpha, n_iter):
    w = np.zeros(X.shape[1])
    losses, times = [], []
    start = time.time()

    for _ in range(n_iter):
        grad = logistic_gradient(w, X, y)
        w = soft_thresholding(w - alpha * grad, alpha * lam)

        losses.append(logistic_loss(w, X, y) + lam * np.linalg.norm(w, 1))
        times.append(time.time() - start)

    return w, times, losses
# 7. FISTA (accéléré)
def fista_time(X, y, lam, alpha, n_iter):
    w = np.zeros(X.shape[1])
    z = w.copy()
    t = 1

    losses, times = [], []
    start = time.time()

    for _ in range(n_iter):
        w_old = w.copy()
        grad = logistic_gradient(z, X, y)
        w = soft_thresholding(z - alpha * grad, alpha * lam)

        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        z = w + ((t - 1) / t_new) * (w - w_old)
        t = t_new

        losses.append(logistic_loss(w, X, y) + lam * np.linalg.norm(w, 1))
        times.append(time.time() - start)

    return w, times, losses
# 8. Exécution
lam = 0.05
alpha = 0.1
n_iter = 500
w_sgd, t_sgd, l_sgd = sgd_time(X, y, lam, alpha0=0.5, n_iter=n_iter)
w_adam, t_adam, l_adam = adam_time(X, y, lam, 0.05, 0.9, 0.999, 1e-8, n_iter)
w_ista, t_ista, l_ista = ista_time(X, y, lam, alpha, n_iter)
w_fista, t_fista, l_fista = fista_time(X, y, lam, alpha, n_iter)
# 9. Comparaison des convergences
plt.figure(figsize=(8,6))
plt.plot(t_sgd, l_sgd, label="SGD")
plt.plot(t_adam, l_adam, label="ADAM")
plt.plot(t_ista, l_ista, label="ISTA")
plt.plot(t_fista, l_fista, label="FISTA")
plt.xlabel("Temps (s)")
plt.ylabel("Fonction objectif")
plt.title("SGD / ADAM / ISTA / FISTA")
plt.legend()
plt.grid(True)
plt.show()
# 10. Sparsité en fonction de λ (ISTA & FISTA)
lambdas = np.logspace(-3, 0, 10)
zeros_ista, zeros_fista = [], []

for lam in lambdas:
    w_i, _, _ = ista_time(X, y, lam, alpha, n_iter)
    w_f, _, _ = fista_time(X, y, lam, alpha, n_iter)

    zeros_ista.append(np.sum(np.abs(w_i) < 1e-4))
    zeros_fista.append(np.sum(np.abs(w_f) < 1e-4))

plt.figure(figsize=(8,6))
plt.semilogx(lambdas, zeros_ista, marker="o", label="ISTA")
plt.semilogx(lambdas, zeros_fista, marker="s", label="FISTA")
plt.xlabel("λ (régularisation L1)")
plt.ylabel("Nombre de coefficients nuls")
plt.title("Sparsité de w* en fonction de λ")
plt.legend()
plt.grid(True)
plt.show()
