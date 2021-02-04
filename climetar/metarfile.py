import csv
import datetime
import glob
import os
import pandas as pd
import itertools

from .metar import Metar

def import_files(glob_pattern,rerun_parsed=False,linenr=0):
    imported = []
    for filepath in glob.iglob(glob_pattern):
        filedir, filename = os.path.split(filepath)
        print(filepath)
        filebase, fileext = os.path.splitext(str(filename))
        
        if 'parsed' in filebase and not rerun_parsed:
            continue
        if not os.path.isfile(filepath) or not os.access(filepath,os.R_OK):
            continue
        
        mfs = MetarFiles()
        if mfs.import_raw(filepath):
            imported.append(filepath)
            print('Imported "%s"'%filepath)
        
            if 'parsed' in filebase:
                new_filename = filebase+'.parsed'
                new_filename += fileext if fileext!='' else '.csv'
                os.rename(filepath,os.path.join(filedir,new_filename))
        return imported

class MetarFiles(object):
    default_datastore = './data'
    stations_json_file = './resources/stations.json'
                      
    def __init__(self,datastore=None):
        self.datastore = datastore if datastore is not None else self.default_datastore
        self.repo = None
                      
    def _index_init(self):
        for filepath in glob.iglob(os.path.join(self.datastore,'*','*.metar')):
            if not os.path.isfile(filepath):
                continue
            filedir, filename = os.path.split(filepath)
            filebase, fileext = os.path.splittext(filename)
            filedirbase = os.path.basename(filedir)

            station,timespan = filedirbase.strip(), filebase.strip()
            if station in self.repo['stations'] or station in self.repo['aliases']:
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
                    
    def load_station_data(self,stations_json_file=None):
        file = stations_json_file or self.stations_json_file
        with open(file,'r') as fh:
            self.repo = json.load(fh)
    
    def import_chunck(self,indf,chuncknr=0):
        metar_parsed = []
        for index, row in indf.iterrows():
            try:
                mo = Metar(row['metar'],year=row['valid'].year,month=row['valid'].month,stationid=row['station'],chunck=chuncknr,linenr=index,debug=False)
                mo.parse()
                mo.handle()
                metar_parsed.append(mo.to_dict())
            except Exception as exc:
                raise ValueError('Could not parse metar at index %d:%d:\n%s' % (chuncknr,index,line['metar'])) from exc
        
        df = pd.DataFrame(metar_parsed)
        df['calc_color'] = Metar.calc_color(df.vis,df.sky_ceiling)
        df['relh'] = Metar.calc_relh(df.temp,df.dwpt)
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
            print('Parsing %s, at once'%(filepath))
            df = pd.read_csv(filename,**df_kwargs)
            self.import_chunck(chunk_df,c)
        elif versiontuple(pd.__version__) >= versiontuple('1.2.0'):
            print('Parsing %s in %d chunks of %d lines'%(filepath,num_lines//chunk+1,chunk))
            with pd.read_csv(filepath,chunksize=chunk,**df_kwargs) as reader:
                c=1
                for chunk_df in reader:
                    self.import_chunck(chunk_df,c)
                    print('Parsed chunk %d'%c)
                    c+=1
        else:
            print('Parsing %s in %d chunks of %d lines'%(filepath,num_lines//chunk+1,chunk))
            tfr = pd.read_csv(filepath,chunksize=chunk,**df_kwargs)
            c=1
            
            for chunk_df in tfr:
                self.import_chunck(chunk_df,c)
                print('Parsed chunk %d'%c)
                c+=1
        return True
        
        