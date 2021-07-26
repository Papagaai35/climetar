# CliMETAR
CliMETAR (spreek uit klie-*mee*-tar) is een tool om eenvoudig klimaatdata te extraheren uit METAR en SPECI berichten. METAR berichten zijn tekst-berichten die de actuele meteorologische situatie op een luchthaven aangeven, en worden periodiek uitgegeven. SPECI berichten volgen hetzelfde format, maar worden uitgegeven waneer het weer snel veranderd. Omdat METAR en SPECI berichten van over de hele wereld dezelfde standaard volgen, en beschikbaar zijn vormen deze een uitstekende bron om een eenvoudige klimaatanalyse op uit te voeren.

Daarnaast is CliMETAR in staat om klimaat- en hoogtekaarten te produceren van de omgeving van de luchthavens.

CliMETAR is ontwikkeld door de Joint Meteorologische Groep, in 2020/2021.

## Onwikkeling
Ondanks dat de ontwikkeling van CliMETAR in een vergevorderd statium is, moeten er nog wel een paar dingen gebeuren:
* [ ] Spellingscontrole van de huidige documentatie
* [ ] Script om de datakwaliteit & -kwantiteit van een bepaald station inzichtelijk te maken.

## Gebruik
CliMETAR kan gebruikt worden in de JupyterHub-omgeving van het JIVC KIXS Datalab.

1. Ga naar [jupyter.mindef.nl](https://jupyter.mindef.nl/) (via mulan), en log in.
2. Kies voor de **Machine Learning**-omgeving, en klik op de 'Aanmaken'-knop onderaan de pagina.
3. Op het scherm dat verschijnt, kies Terminal
4. Tik de volgende opdracht in, en druk op <b>Enter</b>:
```
git clone https://git.mindef.nl/djgommers/climetar.git
```
5. Na enkele momenten verschijnt er links in de zijbalk een map `climetar`. Open deze map en open het notebook `00. Installeren & Introductie.ipynb` (dmv dubbelklik)

Mocht je hierin een foutje hebben gemaakt, volg dan de stappen uit [Uninstall.md](resources/Uninstall.md).

In het eerste notebook `00. Installeren & Introductie.ipynb` wordt ook een korte introductie gegeven op Jupyter. Voor meer info:
* [Jupyter binnen Defensie](https://confluence.kixs.mindef.nl/display/OP/JupyterLab) (MULAN)
* [JupyterLab Documentatie](https://jupyterlab.readthedocs.io) (via Internet op de Werkplek; Engels)

## Databronnen
* Metar Berichten:
  ASOS-AWOS Network van Iowa Environmental Mesonet, Iowa State University ([https://mesonet.agron.iastate.edu/request/download.phtml](https://mesonet.agron.iastate.edu/request/download.phtml))
* Kaartmateriaal:
   * Basiskaartmateriaal (landen, grenzen ect.): Natural Earth [naturalearthdata.com](https://www.naturalearthdata.com/).
   * Reliefkaart: DGeo_World_ShadedRelief, © [Dienst Geografie](http://dwrd.mindef.nl/sites/SWR003509/SitePages/Startpagina.aspx)
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
