# ==========================================
# TP1 – Régularisation L1 & ISTA
# TP2 – Sélection de variables (Breast Cancer)
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# PARTIE 1 – SOFT-THRESHOLDING
# ==========================================

def soft_thresholding(v, gamma):
    """
    Opérateur de seuillage doux (prox norme L1)
    """
    return np.sign(v) * np.maximum(0, np.abs(v) - gamma)

# Tracé du seuillage doux
v = np.linspace(-5, 5, 1000)
gamma = 1.5

plt.figure()
plt.plot(v, soft_thresholding(v, gamma))
plt.axhline(0)
plt.axvline(0)
plt.title("Soft-Thresholding (γ = 1.5)")
plt.xlabel("v")
plt.ylabel("prox(v)")
plt.grid()
plt.show()

# ==========================================
# PARTIE 2 – ISTA (DATASET SYNTHÉTIQUE)
# ==========================================

np.random.seed(0)

n, d = 100, 50
X = np.random.randn(n, d)

w_true = np.zeros(d)
w_true[:5] = np.random.randn(5)

y = X @ w_true + 0.1 * np.random.randn(n)

# Normalisation (ESSENTIEL pour Lasso)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Constante de Lipschitz
L = np.linalg.norm(X.T @ X, 2) / n + 1e-8

def ista(X, y, lam, n_iter=100):
    """
    Algorithme ISTA pour le Lasso
    """
    n, d = X.shape
    w = np.zeros(d)
    obj = []

    for _ in range(n_iter):
        grad = (X.T @ (X @ w - y)) / n
        w = soft_thresholding(w - grad / L, lam / L)

        loss = 0.5 * np.linalg.norm(X @ w - y)**2 / n
        loss += lam * np.linalg.norm(w, 1)
        obj.append(loss)

    return w, obj

# Exécution ISTA
lam = 0.1
w_est, obj = ista(X, y, lam)

plt.figure()
plt.plot(obj)
plt.xlabel("Itérations")
plt.ylabel("Fonction objectif")
plt.title("Convergence ISTA")
plt.grid()
plt.show()

# ==========================================
# PARTIE 3 – ANALYSE DE LA PARCIMONIE
# ==========================================

alphas = np.logspace(-4, 0, 20)
non_zero = []

for a in alphas:
    lasso = Lasso(alpha=a, max_iter=10000, tol=1e-4)
    lasso.fit(X, y)
    non_zero.append(np.sum(lasso.coef_ != 0))

plt.figure()
plt.plot(np.log10(alphas), non_zero, marker='o')
plt.xlabel("log(alpha)")
plt.ylabel("Nombre de coefficients non nuls")
plt.title("Parcimonie du Lasso")
plt.grid()
plt.show()

# ==========================================
# TP2 – DATASET BREAST CANCER
# ==========================================

data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Normalisation
X = StandardScaler().fit_transform(X)

# ==========================================
# LASSO PATH
# ==========================================

alphas = np.logspace(-4, 1, 20)
coef_lasso = []

for a in alphas:
    lasso = Lasso(alpha=a, max_iter=10000, tol=1e-4)
    lasso.fit(X, y)
    coef_lasso.append(lasso.coef_)

plt.figure(figsize=(10, 6))
for i in range(X.shape[1]):
    plt.plot(np.log10(alphas), [c[i] for c in coef_lasso])

plt.xlabel("log(alpha)")
plt.ylabel("Valeur des coefficients")
plt.title("Lasso Path – Breast Cancer")
plt.grid()
plt.show()

# ==========================================
# ANALYSE POUR λ = 0.5
# ==========================================

lasso = Lasso(alpha=0.5, max_iter=10000, tol=1e-4)
lasso.fit(X, y)

coef = lasso.coef_
nb_zero = np.sum(coef == 0)

print("Nombre de variables éliminées :", nb_zero)
print("3 variables les plus prédictives :")

important_features = np.argsort(np.abs(coef))[-3:]
for i in important_features:
    print(feature_names[i], coef[i])
