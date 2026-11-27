import pandas as pd
from scipy import stats
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from cartiflette import carti_download
from scipy import stats
import requests
import bs4
import re


def cartographier(data, variable, titre, var_dep = "dep"):
    """
    Prend en argument un dataframe, la variable à prendre en compte et le titre de la carte.
    Renvoie une carte de la France avec les départements colorisés selon la variable d'intérêt.
    """
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


def extraction_wikipedia(page):
    """
    Extraction des données tabulaires dans la page Wikipédia
    Renvoie un DataFrame
    """
    tableaux = page.find_all('table', {'class': 'wikitable sortable'})

    faitsDiv = list()
    for tableau in tableaux:  # On récupère les données des colonnes
        table_body = tableau.find('tbody')
        rows = table_body.find_all('tr')
        liste = list()
        for row in rows:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            if len(cols) > 0: 
                liste.append(cols)
        faitsDiv.append(liste)

    liste_dataf = list()  # On transforme les tableaux en DataFrame
    for tableau in faitsDiv:
        liste_dataf.append(pd.DataFrame(tableau))

    liste_colonnes = list()

    for tableau in tableaux:
        table_body = tableau.find('tbody')
        rows = table_body.find_all('tr')
        for row in rows:
            cols = row.find_all('th')
            if len(cols) > 0:
                cols = [ele.get_text(separator=' ').strip().title() for ele in cols]
                liste_colonnes.append(cols)

    for i in range(len(liste_dataf)):
        liste_dataf[i].columns = liste_colonnes[i]

    faitsDivDF = pd.concat(liste_dataf)  # On rassemble tous les DataFrame dans un grand 
    faitsDivDF = faitsDivDF.reset_index(drop=True)

    faitsDivDF = faitsDivDF[["Département(S) [ Note 2 ]", "Résumé"]]

    return faitsDivDF


def histogramme(x, titre, xlabel, bins=30):
    """
    Affiche l'histogramme de la variable x, sur une répartition de 30 barres par défaut.
    """
    plt.hist(x, bins=bins, edgecolor="white")
    plt.title(titre)
    plt.xlabel(xlabel)
    plt.ylabel("Nombre d'observations")
    plt.show()

def histogrammes_superposes(x, y, titre, labels, bins=30):
    """
    Superpose les histogrammes de x et y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(x, bins=bins, alpha=0.55,
            color="tab:blue", edgecolor="white", label=labels[0], density = True)
    ax.hist(y, bins=bins, alpha=0.55,
            color="tab:orange", edgecolor="white", label=labels[1], density = True)

    plt.ylabel("Nombre d'observations")
    ax.set_title(titre)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def regression(x, y, titre_grph, titre_axe1, titre_axe2):
    """
    Affiche la régression de la variable x sur la variable y.
    Imprime également le r² et la p-value de la régression.
    """
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
    res = stats.linregress(x, y)
    print(f"r = {res.rvalue:.3f}  (r² = {res.rvalue**2:.3f})")
    print(f"p-value = {res.pvalue:.3g}")


