import json
import os

class StationRepo(object):
    default_repo = './resources/stations.json'
    def __init__(self,json_or_file=None):
        self.repo = {}
        self.load_jsonstr_or_file(json_or_file if json_or_file is not None else self.default_repo)
        self.stations = self.repo.get("stations",{})
        self.networks = self.repo.get("networks",{})
        self.aliases = self.repo.get("aliases",{})
    
    def load_jsonstr_or_file(self,json_or_file):
        if json_or_file is not None:
            repo = None
            try:
                repo = json.loads(json_or_file)
            except:
                try:
                    if os.path.exists(json_or_file) and os.path.isfile(json_or_file):
                        with open(json_or_file,'r') as fh:
                            repo = json.load(fh)
                except:
                    pass
            if repo is not None:
                self.repo = repo
            else:
                raise ValueError('Invalid repo passed:\n%s'%json_or_file)
                
    def get_station(self,stationcode):
        if stationcode in self.stations:
            return stationcode, self.stations[stationcode]
        elif stationcode in self.aliases:
            alias_stationcode = self.aliases[stationcode]
            return alias_stationcode, self.stations[alias_stationcode]
        else:
            raise KeyError(f"Station '{stationcode}' could not be found")
            
    def __contains__(self,item):
        return item in self.stations or item in self.aliases