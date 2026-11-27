import pandas as pd
from scipy import stats
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from cartiflette import carti_download

def cartographier(data, variable, titre, var_dep = "dep"):
    departements = carti_download(
        values="France",
        crs=4326,
        borders="DEPARTEMENT",
        vectorfile_format="geojson",
        filter_by="FRANCE_ENTIERE",
        source="EXPRESS-COG-CARTO-TERRITOIRE",
        year=2022
    )
    departements = departements.drop(columns=["POPULATION"])

    ### On fusionne les géométries des départments Corse
    corse = departements[departements["INSEE_DEP"].isin(["2A", "2B"])].dissolve()

    ### On corrige les données de corse
    corse["INSEE_DEP"] = "20"
    corse["LIBELLE_DEPARTEMENT"] = "Corse"

    ### On retire les deux anciennes lignes
    departements = departements[~departements["INSEE_DEP"].isin(["2A", "2B"])]

    ### On intègre la Corse sous le code département 20
    departements = pd.concat(
        [departements, corse],
        ignore_index=True
    )

    departements_metropole = departements[departements["INSEE_DEP"].astype(int) <= 95]

    carte = departements_metropole.merge(
        data,
        left_on="INSEE_DEP",
        right_on= var_dep,
        how="left"

    )

    ### On change le système de coordonnées en système Lambert-93
    carte = carte.to_crs(2154)
    print(carte.info())

    carte.plot(
        column=variable,
        cmap="RdYlBu_r",
        legend=True,
        edgecolor="black",
        linewidth=0.5,
        figsize=(12, 10),
        missing_kwds={"color": "lightgrey"}
    )

    plt.title(titre)
    plt.axis("off")
    plt.show()


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


def histogramme(x, titre, xlabel, bins=30):
    plt.hist(x, bins=bins, edgecolor="white")
    plt.title(titre)
    plt.xlabel(xlabel)
    plt.ylabel("Nombre d'observations")
    plt.show()


def correlation (x, y, titre_grph, titre_axe1, titre_axe2):
    sansNa = x.notna() & y.notna()
    x = x[sansNa]
    y = y[sansNa]

    plt.scatter(x, y, s=0.3)

    pente, ordonnee = np.polyfit(x, y, 1)
    x1 = np.linspace(x.min(), x.max(), 100)
    plt.plot(x1, pente * x1 + ordonnee, color="red", label=f"y = {ordonnee:.1f} + {pente:.1f}x")

    plt.title(titre_grph)
    plt.xlabel(titre_axe1)
    plt.ylabel(titre_axe2)
    plt.legend()
    plt.show()