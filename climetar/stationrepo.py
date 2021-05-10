import logging
_log = logging.getLogger(__name__)

import json
import os

class StationRepo(object):
    default_repo = './resources/stations.geojson'
    def __init__(self,geojsonfile=None):
        self.repo = {}
        self.stations = {}
        self.networks = {}
        self.aliases = {}
        geojsonfile = geojsonfile if geojsonfile is not None else self.default_repo
        if geojsonfile:
            self.load(geojsonfile)

    def load(self,geojsonfile):
        repo = None
        if os.path.exists(geojsonfile) and os.path.isfile(geojsonfile):
            with open(geojsonfile,'r') as fh:
                repo = json.load(fh)
                _log.debug('Loaded StationRepo from %s'%geojsonfile)
        if repo is None:
            raise ValueError('Invalid repo passed')
        self.load_obj(repo)
    def load_obj(self,repoobj):
        self.repo = repoobj
        self.stations = {f['id']:f for f in repoobj["features"] if f['properties']['kind']=='station'}
        self.networks = {f['id']:f for f in repoobj["features"] if f['properties']['kind']=='network'}
        self.aliases = {icao:f['id'] for f in repoobj["features"] if f['properties']['kind']=='station' for icao in f['properties']['aliasses'] if icao!=f['id']}
    
    def get_station_geo(self,stationcode):
        if stationcode in self.stations:
            return stationcode, self.stations[stationcode]
        elif stationcode in self.aliases:
            alias_stationcode = self.aliases[stationcode]
            return alias_stationcode, self.stations[alias_stationcode]
        else:
            raise KeyError(f"Station '{stationcode}' could not be found")
        
    def get_station(self,stationcode):
        abbr, station_data = self.get_station_geo(stationcode)
        return abbr, {
            "abbr": station_data['id'],
            "icao": station_data['properties']['icao'],
            "network": station_data['properties']['network'],
            "name": station_data['properties']['name'],
            "timezone": station_data['properties']['timezone'],
            "latitude": station_data['geometry']['coordinates'][1],
            "longitude": station_data['geometry']['coordinates'][0],
            "elevation": station_data['properties']['elevation'],
        }

    def __contains__(self,item):
        return item in self.stations or item in self.aliases
