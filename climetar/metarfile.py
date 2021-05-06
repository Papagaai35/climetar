import logging
_log = logging.getLogger(__name__)

import csv
import datetime
import glob
import math
import os
import time
import pandas as pd
import itertools

from . import metar, StationRepo

def analyse_files(glob_pattern,rerun_parsed=False,output_folder=None):
    if isinstance(glob_pattern,list):
        return analyse_globs(glob_pattern,rerun_parsed,output_folder)
    else:
        return analyse_globs([glob_pattern],rerun_parsed,output_folder)

def analyse_globs(glob_patterns,rerun_parsed=False,output_folder=None):
    analysed = []
    output_files = []
    for gp in glob_patterns:
        gpa, gpo = analyse_glob(gp,rerun_parsed,output_folder)
        analysed += gpa
        output_files += gpo
    if len(analysed)==0 and not rerun_parsed:
        _log.warning('Er waren geen (nieuwe) bestanden aangetroffen, om te analyseren...')
    elif len(analysed)==0 and rerun_parsed:
        _log.warning('Er waren geen bestanden aangetroffen, om te analyseren...')
    elif len(output_files)==1:
        _log.info('Geanalyseerde metarberichten zijn opgeslagen in "%s"'%output_files[0])
    elif len(output_files)>1:
        _log.info('Geanalyseerde metarberichten zijn opgeslagen in "%s" en "%s"'%(
            '", "'.join(output_files[:-1]),output_files[-1]))
    return analysed, output_files

def analyse_glob(glob_pattern,rerun_parsed=False,output_folder=None):
    analysed = []
    output_files = set()
    for filepath in glob.iglob(glob_pattern):
        filedir, filename = os.path.split(filepath)
        filebase, fileext = os.path.splitext(str(filename))

        if 'analysed' in filebase and not rerun_parsed:
            continue
        if not os.path.isfile(filepath) or not os.access(filepath,os.R_OK):
            continue

        mfs = MetarFiles(datastore=output_folder)
        if mfs.import_raw(filepath):
            analysed.append(filepath)

            new_filename = filebase if 'analysed' in filebase else filebase+'.analysed'
            new_filename += fileext if fileext!='' else '.csv'
            new_filepath = os.path.join(filedir,new_filename)
            if new_filepath!=filepath:
                os.rename(filepath,new_filepath)
        output_files |= mfs.exported_files
    return analysed, sorted(list(output_files))

class MetarFiles(object):
    default_datastore = './data'

    def __init__(self,datastore=None):
        self.datastore = datastore if datastore is not None else self.default_datastore
        self.exported_files = set()
        self.repo = StationRepo()

    def _index_init(self):
        for filepath in glob.iglob(os.path.join(self.datastore,'*','*.metar')):
            if not os.path.isfile(filepath):
                continue
            filedir, filename = os.path.split(filepath)
            filebase, fileext = os.path.splittext(filename)
            filedirbase = os.path.basename(filedir)

            station,timespan = filedirbase.strip(), filebase.strip()
            if station in self.repo:
                if station not in self.index:
                    self.index[station] = {}
                if timespan.isnumeric() and len(timespan)==4:
                    period_from, period_to = (timespan+'-01-01',timespan+'-12-31')
                    period_from, period_to = datetime.datetime.strptime(timespan+'-01-01',' %Y-%m-%d'), datetime.datetime.strptime(timespan+'-12-31',' %Y-%m-%d')
                    self.index[station][int(timespan)] = period_from, period_to
                elif 'doy' in timespan:
                    # Partial 2011.doy001-356.metar
                    year, period = timespan.split(".")
                    period_from, period_to = period.split("-")
                    period_from, period_to = year+'-'+re.sub('\D','',period_from), year+'-'+re.sub('\D','',period_to)
                    period_from, period_to = datetime.datetime.strptime(period_from,' %Y-%j'), datetime.datetime.strptime(period_to,' %Y-%j')
                    if year in self.index[station]:
                        self.index[station][year] = min([period_from,self.index[station][year][0]]), max([period_to,self.index[station][year][1]])
                    else:
                        self.index[station][year] = period_from, period_to

    def import_chunck(self,indf,chuncknr=0):
        metar_parsed = []
        for index, row in indf.iterrows():
            try:
                mo = metar.Metar(
                    row['metar'],
                    year=row['valid'].year,month=row['valid'].month,
                    stationid=row['station'],
                    chunck=chuncknr,
                    linenr=index,
                    debug=False)
                mo.parse()
                mo.handle()
                metar_parsed.append(mo.to_dict())
            except Exception as exc:
                raise ValueError('Kon metar-bericht niet verwerken (blok %d, regel %d):\n%s' % (chuncknr,index,row['metar'])) from exc

        df = pd.DataFrame(metar_parsed)
        df['calc_color'] = metar.Metar.calc_color(df.vis,df.sky_ceiling)
        df['relh'] = metar.Metar.calc_relh(df.temp,df.dwpt)
        df = df[['type', 'station', 'stationid', 'time', 'date', 'mod',
                 'wind_dir', 'windvar_from', 'windvar_to', 'wind_spd', 'wind_gust',
                 'cavok', 'vis', 'visdir', 'vis2', 'vis2dir',
                 'rvr', 'wx', 'sky',
                 'sky_ceiling', 'sky_cover', 'sky_cover_index',
                 'sky1_height', 'sky1_cover', 'sky1_cover_index',
                 'sky2_height', 'sky2_cover', 'sky2_cover_index',
                 'sky3_height', 'sky3_cover', 'sky3_cover_index',
                 'sky4_height', 'sky4_cover', 'sky4_cover_index',
                 'temp', 'dwpt', 'relh', 'pres', 'color', 'calc_color',
                 'metar','unparsed','chunck','linenr']]
        stations = list(df.station.unique())
        for station in stations:
            export_df = df.loc[(df.station==station)]
            filename = os.path.join(self.datastore,station+'.metar')
            if os.path.isfile(filename):
                export_df.to_csv(filename,sep='\x1f',mode='a',header=False)
            else:
                export_df.to_csv(filename,sep='\x1f',mode='w',header=True)
            self.exported_files.add(filename)
    def import_raw(self,filepath,chunk=3e4):
        versiontuple = lambda v: tuple(map(int, (v.split("."))))
        num_lines = sum(1 for line in open(filepath))
        df_kwargs = {'usecols':['station','valid','metar'],'dtype':str,'parse_dates':['valid'],'dayfirst':True}
        if chunk is None:
            _log.info('Analyseren van %s, in een keer'%(filepath))
            df = pd.read_csv(filename,**df_kwargs)
            self.import_chunck(chunk_df,1)
        elif versiontuple(pd.__version__) >= versiontuple('1.2.0'):
            numchunks = num_lines//chunk+1
            times = []
            _log.info('Analyseren van %s, in %d blokken van %d METAR-berichten'%(filepath,numchunks,chunk))
            with pd.read_csv(filepath,chunksize=chunk,**df_kwargs) as reader:
                c=1
                print('Blok 1/%d wordt geanalyseerd. ETA %s'%(numchunks,format_period(num_lines/1e3)),end='\r',flush=True)
                for chunk_df in reader:
                    start = time.time()
                    self.import_chunck(chunk_df,c)
                    end = time.time()
                    duration = end-start
                    times.append(duration)
                    avgduration = mean(times[-3:] if len(times)>3 else times)
                    totduration = avgduration * ((num_lines-(c*chunk))/chunk)
                    msg = 'Blok %d/%d geanalyseerd in %s.'%(c,numchunks,format_period(duration))
                    if c!=numchunks:
                        msg += ' ETA %s.'%format_period(totduration)
                    else:
                        msg += ' Klaar.'
                    _log.debug(msg)
                    print(msg+(' '*40),end='\r',flush=True)
                    c+=1
                print("\n")
        else:
            numchunks = num_lines//chunk+1
            times = []
            _log.info('Analyseren van %s, in %d blokken van %d METAR-berichten'%(filepath,num_lines//chunk+1,chunk))
            tfr = pd.read_csv(filepath,chunksize=chunk,**df_kwargs)
            c=1
            print('Blok 1/%d wordt geanalyseerd. ETA %s'%(numchunks,format_period(num_lines/1e3)),end='\r',flush=True)
            for chunk_df in tfr:
                start = time.time()
                self.import_chunck(chunk_df,c)
                end = time.time()
                duration = end-start
                times.append(duration)
                avgduration = sum(times[-3:] if len(times)>3 else times) / min(3,len(times))
                totduration = avgduration * ((num_lines-(c*chunk))/chunk)
                msg = 'Blok %d/%d geanalyseerd in %s.'%(c,numchunks,format_period(duration))
                if c!=numchunks:
                    msg += ' ETA %s.'%format_period(totduration)
                else:
                    msg += ' Klaar.'
                _log.debug(msg)
                print(msg+(' '*40),end='\r',flush=True)
                c+=1
            print("\n")
        return True

def format_period(secs,si=False,precision=2):
    def split(value,precision):
        negative = False
        if value < 0.:
            value = -value
            negative = True
        elif value == 0.:
            return 0., 0
        expof10 = int(math.log10(value))
        if expof10>0:
            expof10 = (expof10//3)*3
        else:
            expof10 = ((-expof10+3)//3) * -3
        value *= 10 ** (-expof10)
        if value >= 500.:
            value /= 1000.
            expof10 += 3
        if negative:
            value *= -1
        return value, int(expof10)
    def prefix(expof10):
        SI_PREFIX_UNTIS = u"yzafpnμm kMGTPEZY"
        prefix_levels = (len(SI_PREFIX_UNTIS)-1)//2
        si_level = expof10//3
        if abs(si_level) > prefix_levels:
            return "e%d"%expof10
        return SI_PREFIX_UNTIS[si_level+prefix_levels]

    if secs<60. or si:
        svalue, expof10 = split(secs,precision)
        prefix = prefix(expof10).strip()
        valuestr = ('%%.%df'%precision)%svalue
        return (f'{valuestr} {prefix}s').strip().replace("  "," ")
    else:
        steps = {"sec":1,"min":60,"h":3600,"days":86400,"years":31557600}
        for i,(n,d) in enumerate(steps.items()):

            pn = list(steps.keys())[max(0,i-1)]
            nn = list(steps.keys())[min(i+1,len(steps)-1)]
            pd, nd = steps[pn], steps[nn]
            if secs/nd<1.:
                break
        if i+1<len(steps):
            return ("%.0f %s %.1f %s"%(secs//d,n,(secs-((secs//d)*d))/pd,pn)).strip().replace("  "," ")
        else:
            return ("%.0f %s %.1f %s"%(secs/d,n)).strip().replace("  "," ")
