from difflib import SequenceMatcher
import json
import os
import pathlib
import string
import socket
import urllib

from IPython.core.display import display, HTML
import pandas as pd


class MetarFetcher(object):
    stations_json_file = './resources/stations.json'
    
    def __init__(self,stations,start,end,stations_json_file=None):
        self.repo = {}
        self.load_station_data(stations_json_file)
        
        self.stations = self.clean_input_stations(stations)
        self.unknown_stations = None
        self.start = self.clean_input_date(start)
        self.end = self.clean_input_date(end)
        
    @classmethod        
    def clean_input_date(cls,date_obj):
        return pd.to_datetime(date_obj,dayfirst=True).date()
        
    @classmethod
    def clean_input_stations(cls,stations):
        if isinstance(stations,tuple) and len(stations)==2:
            stations = [stations[0]]
        elif isinstance(stations,list) or (
                isinstance(stations,tuple) and len(stations)!=2):
            stations = list(stations)
        elif isinstance(stations,dict):
            stations = list(stations.keys())
        elif isinstance(stations,str):
            stations = [stations]
        else:
            raise ValueError("Stations must be a list, tuple, dict or string containing the station abbreviations")
        return stations
    
    def load_station_data(self,stations_json_file=None):
        file = stations_json_file or self.stations_json_file
        with open(file,'r') as fh:
            self.repo = json.load(fh)
    
    def all_stations_exsist(self,force=True):
        if not force and isinstance(self.unknown_stations,list):
            return len(self.unknown_stations)==0
        else:
            self.unknown_stations = []
            
        for s in self.stations:
            if s not in self.repo['stations']:
                self.unknown_stations.append(s)
        return len(self.unknown_stations)==0
    
    def show_alternatives(self,num=10):
        alternatives = self.get_alternatives(num)
        alternative_stations = list(alternatives.keys())
        if len(alternative_stations)==1:
            print("Could not find {}.".format(alternative_stations[0]))
        elif len(alternative_stations)>1:
            print("Could not find {} and {}.".format(
                ', '.join(alternative_stations[:-1]), alternative_stations[-1]))
        print(f'Please note that for American Airports FAA-abbreviations are used.')
        print(f'Did you mean:\n')
        print("{:5s} {:5s} {:30.30s} {:s}".format('','ICAO','Name','Timezone'))
        print("="*70)
        for s,alts in alternatives.items():
            sstr = s
            for alt_s,simscore in alts.items():
                alt_obj = self.repo['stations'][alt_s]
                print(f"{sstr:5s} {alt_s:5s} {alt_obj['name']:30.30s} {alt_obj['timezone']}")
                sstr = ""
            print("-"*70)            
        
    def get_alternatives(self,num=10):
        self.all_stations_exsist(force=False)
        alternatives = {}
        for s in self.unknown_stations:
            alike_stations = list(reversed(sorted(
            [(alt_s,SequenceMatcher(None, s, alt_s).ratio())
                for alt_s in self.repo['stations'].keys()],
            key=lambda item: item[1])))
            alternatives[s] = dict(alike_stations[:num])
        return alternatives
          
    def download(self,filename=None,overwrite=False,**kwargs):
        if not self.all_stations_exsist():
            errormsg = (
                "Could not find all stations.".format(self.unknown_stations[0])
                if len(self.unknown_stations)<1 else (
                "Could not find {}.".format(self.unknown_stations[0])
                if len(self.unknown_stations)==1 else
                "Could not find {} and {}.".format(
                    ', '.join(self.unknown_stations[:-1]), self.unknown_stations[-1])))
            errormsg += "\nRun show_alternatives() to show alternative locations"
            raise ValueError(errormsg)
            
        urlparams = {
            'station': self.stations,
            'year1': self.start.strftime('%Y'),
            'month1': self.start.strftime('%-m'),
            'day1': self.start.strftime('%-d'),
            'year2': self.end.strftime('%Y'),
            'month2': self.end.strftime('%-m'),
            'day2': self.end.strftime('%-d'),
            'data': 'metar',
            'tz': 'Etc/UTC',
            'format': 'onlycomma',
            'latlon': 'no',
            'elev': 'no',
            'missing': 'M',
            'trace': 'T',
            'direct': 'yes',
            'report_type': ['1','2'],
        }
        urlparams.update(kwargs)
        urlp = urllib.parse.urlencode(list(urlparams.items()),doseq=True)
        uri = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?' + urlp
        
        if self.is_connected(uri):
            filename, filemode = path_check(filename,overwrite)
            with open(filename, filemode) as fh:
                data = download_data_from_uri(uri)
                fh.write(data)
            display(HTML(f'Downloaded file to <code>{filename}</code>'))
        else:
            display(HTML(f'''
No internet access detected.<br />
Please download the file form this URL and store in <code>downloads/asos.txt</code><br />
The download could take a few seconds.<br />
<a href="{uri}">{uri}</a>'''))
        
    @classmethod
    def path_check(cls,filename=None,overwrite=False):
        filemode = 'a' if overwrite else 'w'
        if filename is None:
            filename = 'downloads/asos_{i:04d}.csv'
        dirname = os.path.dirname(filename)
        pathlib.Path(dirname).mkdir(parents=True,exist_ok=True)
        
        fields = [t[1] for t in string.Formatter().parse(filename) if t[1] is not None]
        if 'i' in fields:
            i=0
            if not overwite:
                while os.path.isfile(filename.format(i=i)):
                    i += 1
                filename = filename.format(i=i)
        return filename, filemode
    
    @classmethod
    def is_connected(cls,uri):
        hostname = urllib.parse.urlparse(uri).netloc
        try:
            host = socket.gethostbyname(hostname)
            s = socket.create_connection((host, 80), 2)
            s.close()
            return True
        except:
            pass
        return False
        
    @classmethod
    def download_data_from_uri(cls,uri):
        attempt = 0
        while attempt < 6:
            try:
                data = urllib.request.urlopen(uri, timeout=300).read().decode("utf-8")
                if data is not None and not data.startswith("ERROR"):
                    return data
            except Exception as exp:
                time.sleep(5)
            attempt += 1
        raise ValueError("Exhausted attempts to download, returning empty data")
        
        
        
    