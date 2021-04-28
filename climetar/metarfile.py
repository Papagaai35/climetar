import csv
import datetime
import glob
import os
import pandas as pd
import itertools

from . import metar, StationRepo


def import_files(glob_pattern,rerun_parsed=False,linenr=0):
    imported = []
    for filepath in glob.iglob(glob_pattern):
        filedir, filename = os.path.split(filepath)
        #print(filepath)
        filebase, fileext = os.path.splitext(str(filename))
        
        if 'parsed' in filebase and not rerun_parsed:
            continue
        if not os.path.isfile(filepath) or not os.access(filepath,os.R_OK):
            continue
        
        mfs = MetarFiles()
        if mfs.import_raw(filepath):
            imported.append(filepath)
            print('"%s" geimporteerd'%filepath)
        
            if 'parsed' in filebase:
                new_filename = filebase+'.parsed'
                new_filename += fileext if fileext!='' else '.csv'
                os.rename(filepath,os.path.join(filedir,new_filename))
        return imported

class MetarFiles(object):
    default_datastore = './data'
                      
    def __init__(self,datastore=None):
        self.datastore = datastore if datastore is not None else self.default_datastore
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
    def import_raw(self,filepath,chunk=3e4):
        versiontuple = lambda v: tuple(map(int, (v.split("."))))
        num_lines = sum(1 for line in open(filepath))
        df_kwargs = {'usecols':['station','valid','metar'],'dtype':str,'parse_dates':['valid'],'dayfirst':True}
        if chunk is None:
            print('Analyseren van %s, in een keer'%(filepath))
            df = pd.read_csv(filename,**df_kwargs)
            self.import_chunck(chunk_df,c)
        elif versiontuple(pd.__version__) >= versiontuple('1.2.0'):
            print('Analyseren van %s, in %d blokken van %d METAR-berichten'%(filepath,num_lines//chunk+1,chunk))
            with pd.read_csv(filepath,chunksize=chunk,**df_kwargs) as reader:
                c=1
                for chunk_df in reader:
                    self.import_chunck(chunk_df,c)
                    print('Blok %d geanalyseerd'%c)
                    c+=1
        else:
            print('Analyseren van %s, in %d blokken van %d METAR-berichten'%(filepath,num_lines//chunk+1,chunk))
            tfr = pd.read_csv(filepath,chunksize=chunk,**df_kwargs)
            c=1
            
            for chunk_df in tfr:
                self.import_chunck(chunk_df,c)
                print('Blok %d geanalyseerd'%c)
                c+=1
        return True
        
        