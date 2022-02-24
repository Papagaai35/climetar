# CliMETAR
CliMETAR (spreek uit klie-*mee*-tar) is een tool om eenvoudig klimaatdata te extraheren uit METAR en SPECI berichten. METAR berichten zijn tekst-berichten die de actuele meteorologische situatie op een luchthaven aangeven, en worden periodiek uitgegeven. SPECI berichten volgen hetzelfde format, maar worden uitgegeven waneer het weer snel veranderd. Omdat METAR en SPECI berichten van over de hele wereld dezelfde standaard volgen, en beschikbaar zijn vormen deze een uitstekende bron om een eenvoudige klimaatanalyse op uit te voeren.

Daarnaast is CliMETAR in staat om klimaat- en hoogtekaarten te produceren van de omgeving van de luchthavens.

CliMETAR is ontwikkeld door de Joint Meteorologische Groep, in 2020/2021.

## Gebruik
CliMETAR kan gebruikt worden in eeh Jupyter-omgeving, zoals [mybinder.org](https://mybinder.org/) .

1. Ga naar [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Papagaai35/climetar/HEAD)
2. Wacht tot de repository is gestart, dit kan +5 minuten duren.
2b. Lees onderaan de startpagina alvast `00. Installeren & Introductie.ipynb`.
3. Open het notebook `00. Installeren & Introductie.ipynb` (voor een introductie), of start meteen met `10. Downlaoding METARs.ipynb` (als je Jupyter al kent).

In het eerste notebook `00. Installeren & Introductie.ipynb` wordt ook een korte introductie gegeven op Jupyter. Voor meer info:
* [JupyterLab Documentatie](https://jupyterlab.readthedocs.io) (via Internet; Engels)

## Databronnen
* Metar Berichten:
  ASOS-AWOS Network van Iowa Environmental Mesonet, Iowa State University ([https://mesonet.agron.iastate.edu/request/download.phtml](https://mesonet.agron.iastate.edu/request/download.phtml))
* Kaartmateriaal:
   * Basiskaartmateriaal (landen, grenzen ect.): Natural Earth [naturalearthdata.com](https://www.naturalearthdata.com/). Please note: Natural Earth Vector draws boundaries of countries according to defacto status. We show who actually controls the situation on the ground.
   * Kimaatkaart (Köppen-Geiger): Beck, H.E., N.E. Zimmermann, T.R. McVicar, N. Vergopolan, A. Berg, E.F. Wood, (2018); Present and future Köppen-Geiger climate classification maps at 1‑km resolution, Scientific Data 5:180214, [doi:10.1038/sdata.2018.214](https://www.doi.org/10.1038/sdata.2018.214).
   Data gedownload van [www.gloh2o.org/koppen/](http://www.gloh2o.org/koppen/)

CliMETAR is geschreven in `python3` en LaTeX en maakt gebruik van de packages `numpy`, `pandas`, `matplotlib` en `cartopy`.

## Vragen en Problemen
Mocht je een vraag hebben over het gebruik van dit pakket, of problemen ondervinden?

Stuur dan een mail naar [dj.gommers@mindef.nl](mailto:dj.gommers@mindef.nl) met:
 * Wat je probeert te doen;
 * Voor welke stations/landen;
 * Voeg het logbestand `climetar.log` als bijlage toe (indien mogelijk);
 * en Voeg resultaten of screenshots als bijlage toe (indien relevant);

Dan kan ik je het snelste helpen met het probleem.
