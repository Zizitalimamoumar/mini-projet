import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# 1. Chargement et préparation des données
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
# 3. Soft-thresholding (L1)
def soft_thresholding(v, lam):
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0)
# 4. ISTA
def ista_time(X, y, lam, alpha, n_iter):
    w = np.zeros(X.shape[1])
    losses, times = [], []
    start = time.time()
    for _ in range(n_iter):
        grad = logistic_gradient(w, X, y)
        w = soft_thresholding(w - alpha * grad, alpha * lam)
        loss = logistic_loss(w, X, y) + lam * np.linalg.norm(w, 1)
        losses.append(loss)
        times.append(time.time() - start)
    return w, times, losses
# 5. FISTA (version accélérée)
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
        loss = logistic_loss(w, X, y) + lam * np.linalg.norm(w, 1)
        losses.append(loss)
        times.append(time.time() - start)
    return w, times, losses
# 6. Comparaison ISTA vs FISTA
lam = 0.05
alpha = 0.1
n_iter = 500
w_ista, t_ista, l_ista = ista_time(X, y, lam, alpha, n_iter)
w_fista, t_fista, l_fista = fista_time(X, y, lam, alpha, n_iter)
plt.figure()
plt.plot(t_ista, l_ista, label="ISTA")
plt.plot(t_fista, l_fista, label="FISTA")
plt.xlabel("Temps (s)")
plt.ylabel("Fonction objectif")
plt.title("ISTA vs FISTA — Convergence")
plt.legend()
plt.show()
# 7. Sparsité en fonction de lambda
lambdas = np.logspace(-3, 0, 10)
zeros_ista = []
zeros_fista = []
for lam in lambdas:
    w_i, _, _ = ista_time(X, y, lam, alpha, n_iter)
    w_f, _, _ = fista_time(X, y, lam, alpha, n_iter)
    zeros_ista.append(np.sum(np.abs(w_i) < 1e-4))
    zeros_fista.append(np.sum(np.abs(w_f) < 1e-4))
# 8. Visualisation de la sparsité
plt.figure()
plt.semilogx(lambdas, zeros_ista, marker="o", label="ISTA")
plt.semilogx(lambdas, zeros_fista, marker="s", label="FISTA")
plt.xlabel("λ (régularisation L1)")
plt.ylabel("Nombre de coefficients nuls")
plt.title("Sparsité de w* en fonction de λ")
plt.legend()
plt.show()


