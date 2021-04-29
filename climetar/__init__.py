
#Prevents the 'Bad Key "text.kerning_factor" on line 4' warning when working with matplotlib
with open('/opt/conda/lib/python3.7/site-packages/matplotlib/mpl-data/stylelib/_classic_test_patch.mplstyle','r+') as fh:
    lines = fh.readlines()
    lines = ['#'+line for line in lines if line.startswith("text.kerning_factor")]
    fh.seek(0)
    fh.writelines(lines)
    fh.truncate()

from .quantities import *
from .metar import *

from .fetchmetar import MetarFetcher

from .stationrepo import StationRepo
from .metartheme import MetarTheme
from .metarfile import MetarFiles, analyse_files, format_period
from .planetary import Astro

from .metarplot import MetarPlotter

