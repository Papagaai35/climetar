# CliMETAR
CliMETAR (spreek uit klie-*mee*-tar) is een tool om eenvoudig klimaatdata te extraheren uit METAR en SPECI berichten. METAR berichten zijn tekst-berichten die de actuele meteorologische situatie op een luchthaven aangeven, en worden periodiek uitgegeven. SPECI berichten volgen hetzelfde format, maar worden uitgegeven waneer het weer snel veranderd. Omdat METAR en SPECI berichten van over de hele wereld dezelfde standaard volgen, en beschikbaar zijn vormen deze een uitstekende bron om een eenvoudige klimaatanalyse op uit te voeren.

CliMETAR is ontwikkeld door de Joint Meteorologische Groep, in 2020/2021.

## Onwikkeling
Ondanks dat de ontwikkeling van CliMETAR in een vergevorderd statium is, moeten er nog wel een paar dingen gebeuren:
* [ ] Spellingscontrole van de huidige documentatie
* [ ] Script om de datakwaliteit & -kwantiteit van een bepaald station inzichtelijk te maken.

## Gebruik
CliMETAR kan gebruikt worden in de JupyterHub-omgeving van het JIVC KIXS Datalab. Voor meer info:
* [Jupyter binnen Defensie](https://confluence.kixs.mindef.nl/display/OP/JupyterLab) (MULAN)
* [Jupyter](https://jupyterlab.readthedocs.io) (via Internet op de Werkplek)

1. Ga naar [jupyter.mindef.nl](https://jupyter.mindef.nl/) (via mulan), en log in.
2. Kies voor de **Machine Learning**-omgeving, en klik op de 'Aanmaken'-knop onderaan de pagina.
3. Op het scherm dat verschijnt, kies Terminal
4. Tik de volgende opdracht in, en druk op <b>Enter</b>:
```
git clone https://git.mindef.nl/djgommers/climetar.git
```
5. Na enkele momenten verschijnt er links in de zijbalk een map `climetar`. Open deze map en open het notebook `00. Installeren & Introductie.ipynb` (dmv dubbelklik)

## Databronnen
* Metar Berichten:
  ASOS-AWOS Network van Iowa Environmental Mesonet, Iowa State University ([link](https://mesonet.agron.iastate.edu/request/download.phtml))
* Kaartmateriaal:
  Natural Earth 

CliMETAR is geschreven in `python3` en LaTeX en maakt gebruik van de packages `numpy`, `pandas` `matplotlib`, `sqlite3`, en data van Natural Earth
