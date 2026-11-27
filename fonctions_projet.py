import pandas as pd
from scipy import stats
import numpy as np




def regression_lineaire(df, a="a", b="b", afficher=True):
    """
    Régression linéaire de b en fonction de a (b = pente * a + ordonnée).
    Paramètres :
    df : pd.DataFrame
    a, b : noms des colonnes (a = variable explicative, b = variable à expliquer)
    afficher : bool, trace le graphique si True
    """
    # Régression
    res = stats.linregress(x, y)
    pente, ordonnee = res.slope, res.intercept
    r2 = res.rvalue ** 2

    resultats = {
        "pente": pente,
        "ordonnee": ordonnee,
        "r2": r2,
        "p_value": res.pvalue,
        "erreur_std": res.stderr,
    }

    if afficher:
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = pente * x_fit + ordonnee

        # Intervalle de confiance à 95 % de la droite
        n = len(x)
        y_pred = pente * x + ordonnee
        s_err = np.sqrt(np.sum((y - y_pred) ** 2) / (n - 2))
        t = stats.t.ppf(0.975, n - 2)
        ic = t * s_err * np.sqrt(1 / n + (x_fit - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, alpha=0.7, edgecolor="k", label="Données")
        ax.plot(x_fit, y_fit, color="crimson", lw=2,
                label=f"b = {pente:.2f}·a + {ordonnee:.2f}\nR² = {r2:.3f}")
        ax.fill_between(x_fit, y_fit - ic, y_fit + ic, color="crimson",
                        alpha=0.15, label="IC 95 %")

        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title(f"Régression linéaire : {b} en fonction de {a}")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return resultats