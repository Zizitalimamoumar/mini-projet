# ==============================
# 1. Chargement et préparation des données
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(
    "C:/Users/zitaoumar0202/Downloads/california_housing.csv")

y_raw = data["MedHouseVal"].values
y = np.where(y_raw > np.median(y_raw), 1, -1)

X = data.drop(columns=["MedHouseVal"]).values
X = (X - X.mean(axis=0)) / X.std(axis=0)

n, d = X.shape
print(n, d)

# ==============================
# 2. Fonctions de base
# ==============================
def logistic_loss(w, X, y):
    z = y * (X @ w)
    return np.mean(np.log(1 + np.exp(-z)))

def logistic_gradient(w, X, y):
    z = y * (X @ w)
    sigma = 1 / (1 + np.exp(z))
    return -(X.T @ (y * sigma)) / X.shape[0]

def logistic_ridge_loss(w, X, y, lam):
    return logistic_loss(w, X, y) + 0.5 * lam * np.linalg.norm(w)**2

def logistic_ridge_gradient(w, X, y, lam):
    return logistic_gradient(w, X, y) + lam * w

# ==============================
# 3. Algorithmes d’optimisation
# ==============================
def gradient_descent(X, y, lam, alpha, n_iter):
    w = np.zeros(X.shape[1])
    losses = []
    for _ in range(n_iter):
        w -= alpha * logistic_ridge_gradient(w, X, y, lam)
        losses.append(logistic_ridge_loss(w, X, y, lam))
    return w, losses

def sgd(X, y, lam, alpha0, n_epochs):
    n, d = X.shape
    w = np.zeros(d)
    losses = []
    for k in range(n_epochs):
        i = np.random.randint(n)
        grad = (-(y[i] * X[i]) / (1 + np.exp(y[i] * X[i] @ w))) + lam * w
        w -= (alpha0 / (1 + k)) * grad
        losses.append(logistic_ridge_loss(w, X, y, lam))
    return w, losses

def rmsprop(X, y, lam, alpha, beta, eps, n_iter):
    w = np.zeros(X.shape[1])
    v = np.zeros_like(w)
    losses = []
    for _ in range(n_iter):
        grad = logistic_ridge_gradient(w, X, y, lam)
        v = beta * v + (1 - beta) * grad**2
        w -= alpha * grad / (np.sqrt(v) + eps)
        losses.append(logistic_ridge_loss(w, X, y, lam))
    return w, losses

def adam(X, y, lam, alpha, beta1, beta2, eps, n_iter):
    w = np.zeros(X.shape[1])
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    losses = []
    for k in range(1, n_iter + 1):
        grad = logistic_ridge_gradient(w, X, y, lam)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)
        w -= alpha * m_hat / (np.sqrt(v_hat) + eps)
        losses.append(logistic_ridge_loss(w, X, y, lam))
    return w, losses

# ==============================
# 4. Méthodes proximales L1
# ==============================
def soft_thresholding(v, lam):
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0)

def ista(X, y, lam, alpha, n_iter):
    w = np.zeros(X.shape[1])
    losses = []
    for _ in range(n_iter):
        w = soft_thresholding(w - alpha * logistic_gradient(w, X, y), alpha * lam)
        losses.append(logistic_loss(w, X, y) + lam * np.linalg.norm(w, 1))
    return w, losses

def fista(X, y, lam, alpha, n_iter):
    w = np.zeros(X.shape[1])
    z = w.copy()
    t = 1
    losses = []
    for _ in range(n_iter):
        w_new = soft_thresholding(z - alpha * logistic_gradient(z, X, y), alpha * lam)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        z = w_new + ((t - 1) / t_new) * (w_new - w)
        w, t = w_new, t_new
        losses.append(logistic_loss(w, X, y) + lam * np.linalg.norm(w, 1))
    return w, losses

# ==============================
# 5. Exécutions
# ==============================
w_gd, loss_gd = gradient_descent(X, y, 0.1, 0.1, 200)
w_sgd, loss_sgd = sgd(X, y, 0.1, 0.5, 2000)
w_rms, loss_rms = rmsprop(X, y, 0.1, 0.01, 0.9, 1e-8, 500)
w_adam, loss_adam = adam(X, y, 0.1, 0.05, 0.9, 0.999, 1e-8, 500)
w_ista, loss_ista = ista(X, y, 0.05, 0.1, 300)
w_fista, loss_fista = fista(X, y, 0.05, 0.1, 300)

# ==============================
# 6. Courbes de convergence
# ==============================
plt.figure()
plt.plot(loss_gd, label="GD")
plt.plot(loss_sgd, label="SGD")
plt.plot(loss_rms, label="RMSProp")
plt.plot(loss_adam, label="Adam")
plt.legend()
plt.xlabel("Itérations")
plt.ylabel("Fonction objectif")
plt.title("Méthodes différentiables")
plt.show()

plt.figure()
plt.plot(loss_ista, label="ISTA")
plt.plot(loss_fista, label="FISTA")
plt.legend()
plt.xlabel("Itérations")
plt.ylabel("Fonction objectif")
plt.title("Méthodes proximales L1")
plt.show()
