# Imports logging posibilities
from .logger import getLogger
_log = getLogger(__name__)

#Prevents the 'Bad Key "text.kerning_factor" on line 4' warning when working with matplotlib
import os
metarfile = '/opt/conda/lib/python3.7/site-packages/matplotlib/mpl-data/stylelib/_classic_test_patch.mplstyle'
if os.path.isfile(metarfile):
    modified = False
    with open(metarfile,'r+') as fh:
        lines = fh.readlines()
        newlines = []
        for line in lines:
            if line.startswith("text.kerning_factor"):
                line = '#'+line
                modified = True
            else:
                line = line
            newlines.append(line)
        fh.seek(0)
        fh.writelines(newlines)
        fh.truncate()
    if modified:
        _log.debug('matplotlib: text.kerning_factor lines in _classic_test_patch.mplstyle ignored')


# Imports climetar modules
from .quantities import *
from .metar import *

from .fetchmetar import MetarFetcher

from .stationrepo import StationRepo
from .metartheme import MetarTheme
from .metarfile import MetarFiles, analyse_files, format_period
from .planetary import Astro

from .metarplot import MetarPlotter

_log.debug('CliMETAR imported')
