import os
import datetime
import json
import re
import urllib
import urllib.request

class PythonObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self,obj)

if __name__=='__main__':
    networkdata = {}
    print('Loading networks...',end='\r',flush=True)
    with urllib.request.urlopen("https://mesonet.agron.iastate.edu/geojson/networks.geojson") as response:
        networksjson = json.loads(response.read())
        for f in networksjson['features']:
            if f['id'].endswith('ASOS') or f['id'].endswith('AWOS'):
                f['properties']['kind'] = 'network'
                f['properties']['stations'] = set()
                networkdata[f['id']] = f
    networklist = list(networkdata.keys())
    print('Loading networks: Done.',flush=True)

    stationdata = {}
    props_rename = {'elevation':'elevation','sname':'name','tzname':'timezone'}
    for i, network in enumerate(networklist):
        print('Loading stations from networks: %03d/%03d %-60s'%(i+1,len(networklist),network),end="\r",flush=True)
        with urllib.request.urlopen(f"https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson") as response:
            stationsjson = json.loads(response.read())
            for f in stationsjson['features']:
                station = {k:f[k] for k in ['type','id','geometry']}
                station['properties'] = {K:f['properties'][k] for k,K in props_rename.items()}
                station['properties']['icao'] = None
                station['properties']['kind'] = 'station'
                station['properties']['network'] = network
                station['properties']['aliasses'] = {f['id'],f['properties']['sid']}
                time_domain = f['properties'].get('time_domain','')
                station['properties']['time_from'], station['properties']['time_to'] = (
                    time_domain[1:-1].split('-') 
                    if ('-' in time_domain and time_domain[0] in '({[' and time_domain[-1] in ']})')
                    else "","")
                stationdata[f['id']] = station
                networkdata[network]['properties']['stations'].add(f['id'])
    print('Loading stations from networks: %-70s'%'Done.',flush=True)

    for i, network in enumerate(networklist):
        print('Loading latest METARs (for ICAO codes): %03d/%03d %-60s'%(i+1,len(networklist),network),end="\r",flush=True)
        with urllib.request.urlopen(f"https://mesonet.agron.iastate.edu/api/1/currents.json?network={network}") as response:
            currentjson = json.loads(response.read())
            for currentrecord in currentjson['data']:
                abbr = currentrecord['station']
                metar = currentrecord.get('raw','').strip()
                if metar.startswith('SPECI ') or metar.startswith('METAR '):
                    metar = metar[6:].strip()
                if metar.startswith('COR ') or metar.startswith('ADV '):
                    metar = metar[4:].strip()
                if len(metar)>4 and metar[4]==' ':
                    icao = metar[:4].strip().upper()
                    stationdata[abbr]['properties']['icao'] = icao
                    if abbr != icao:
                        stationdata[abbr]['properties']['aliasses'].add(icao)
                if len(metar)<=4 or metar[4]!=' ':
                    print('%03d/%03d %-10s %-4s %03d %s'%(i+1,len(networklist),network,abbr,len(metar),'"'+metar[4]+'"'),flush=True)
                    print(metar,flush=True)
    print('Loading latest METARs (for ICAO codes): %-70s'%'Done.',flush=True)

    geodata = list(networkdata.values()) + list(stationdata.values())
    geojson = {"type": "FeatureCollection", "features": geodata, "generation_time": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'), "count": len(geodata)}
    with open('stations.geojson','w') as fh:
        json.dump(geojson,fh,cls=PythonObjectEncoder,indent=4)
    
    print('Written geojson to %s'%os.path.abspath(os.path.join('.','stations.geojson')))