# CliMETAR
CliMETAR (spreek uit klie-*mee*-tar) is een tool om eenvoudig klimaatdata te extraheren uit METAR en SPECI berichten. METAR berichten zijn tekst-berichten die de actuele meteorologische situatie op een luchthaven aangeven, en worden periodiek uitgegeven. SPECI berichten volgen hetzelfde format, maar worden uitgegeven waneer het weer snel veranderd. Omdat METAR en SPECI berichten van over de hele wereld dezelfde standaard volgen, en beschikbaar zijn vormen deze een uitstekende bron om een eenvoudige klimaatanalyse op uit te voeren.

CliMETAR is ontwikkeld door de Joint Meteorologische Groep, in 2020/2021.

## Gebruik
CliMETAR kan gebruikt worden in de JupyterHub-omgeving van het JIVC KIXS Datalab. Voor meer info over Jupyter, ga naar [https://confluence.kixs.mindef.nl/display/OP/JupyterHub+ontwikkelomgevingen]()

1. Ga naar [https://jupyter.mindef.nl/](), en log in.
2. Kies voor de **Machine Learning**-omgeving.
3. Open de terminal en tik in:
```
git clone https://git.mindef.nl/djgommers/climetar.git
```
4. Open de CliMETAR map en begin bij het notebook `00. Data Retrieval.ipynb`

## Databronnen
* Metar Berichten:
  ASOS-AWOS Network van Iowa Environmental Mesonet, Iowa State University ([link](https://mesonet.agron.iastate.edu/request/download.phtml))
* Kaartmateriaal:
  Natural Earth 

CliMETAR is geschreven in `python3` en $\LaTeX$ en maakt gebruik van de packages `numpy`, `pandas` `matplotlib`, `sqlite3`, en data van Natural Earth