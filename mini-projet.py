import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Charger Titanic
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Sélection de variables numériques
data = data[["Survived", "Pclass", "Age", "Fare", "SibSp", "Parch"]]
data = data.dropna()

# Target binaire {-1, +1}
y = data["Survived"].values
y = np.where(y == 1, 1, -1)

# Features
X = data.drop(columns=["Survived"]).values

# Standardisation
X = (X - X.mean(axis=0)) / X.std(axis=0)

n, d = X.shape
print(n, d)
#Fonctions de base (logistique)
def logistic_loss(w, X, y):
    z = y * (X @ w)
    return np.mean(np.log(1 + np.exp(-z)))

def logistic_gradient(w, X, y):
    z = y * (X @ w)
    sigma = 1 / (1 + np.exp(z))
    return -(X.T @ (y * sigma)) / X.shape[0]

#4️⃣ Algorithmes d’optimisation
🔹 SGD (L2)
def sgd_time(X, y, lam, alpha0, n_iter):
    w = np.zeros(X.shape[1])
    losses, times = [], []
    start = time.time()

    for k in range(n_iter):
        i = np.random.randint(X.shape[0])
        grad = (-(y[i] * X[i]) / (1 + np.exp(y[i] * X[i] @ w))) + lam * w
        w -= (alpha0 / (1 + k)) * grad

        losses.append(logistic_loss(w, X, y) + 0.5 * lam * np.linalg.norm(w)**2)
        times.append(time.time() - start)

    return times, losses

#🔹 Adam (L2)
def adam_time(X, y, lam, alpha, beta1, beta2, eps, n_iter):
    w = np.zeros(X.shape[1])
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    losses, times = [], []
    start = time.time()

    for k in range(1, n_iter + 1):
        grad = logistic_gradient(w, X, y) + lam * w

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2

        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)

        w -= alpha * m_hat / (np.sqrt(v_hat) + eps)

        losses.append(logistic_loss(w, X, y) + 0.5 * lam * np.linalg.norm(w)**2)
        times.append(time.time() - start)

    return times, losses

#🔹 ISTA (L1)
def soft_thresholding(v, lam):
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0)

def ista_time(X, y, lam, alpha, n_iter):
    w = np.zeros(X.shape[1])
    losses, times = [], []
    start = time.time()

    for _ in range(n_iter):
        w = soft_thresholding(w - alpha * logistic_gradient(w, X, y), alpha * lam)
        losses.append(logistic_loss(w, X, y) + lam * np.linalg.norm(w, 1))
        times.append(time.time() - start)

    return times, losses

#5️⃣ Exécution & Courbes Perte vs Temps
t_sgd, l_sgd = sgd_time(X, y, lam=0.1, alpha0=0.5, n_iter=2000)
t_adam, l_adam = adam_time(X, y, lam=0.1, alpha=0.05,
                           beta1=0.9, beta2=0.999, eps=1e-8, n_iter=1000)
t_ista, l_ista = ista_time(X, y, lam=0.05, alpha=0.1, n_iter=500)

plt.figure()
plt.plot(t_sgd, l_sgd, label="SGD")
plt.plot(t_adam, l_adam, label="Adam")
plt.plot(t_ista, l_ista, label="ISTA")
plt.xlabel("Temps (s)")
plt.ylabel("Fonction objectif")
plt.title("Titanic — Perte vs Temps")
plt.legend()
plt.show()

