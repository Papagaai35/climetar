#type: ignore
import datetime
import json
import os
import urllib
import sqlite3 as sql

class MetarArchive(object):
    def __init__(self,database_file):
        self.f = os.path.abspath(database_file)
        self.c = sql.connect(self.f,
            detect_types=sql.PARSE_DECLTYPES | sql.PARSE_COLNAMES)
        self.C = self.c.cursor()

        self.download_requests = []
    def get_cursor(self):
        return self.C
    def destroy_database(self):
        db = self.c.cursor()
        db.execute("""DROP TABLE IF EXISTS metarparams;""")
        db.execute("""DROP TABLE IF EXISTS metarmsg;""")
        db.execute("""DROP TABLE IF EXISTS requests;""")
        db.execute("""DROP TABLE IF EXISTS stations;""")
        db.execute("""DROP TABLE IF EXISTS networks;""")
        self.c.commit()
    def setup_networks_table(self):
        db = self.get_cursor()
        db.execute("""CREATE TABLE IF NOT EXISTS networks (
            abbr VARCHAR(16) PRIMARY KEY,
            name TEXT NOT NULL,
            polygon TEXT);""")
        self.c.commit()
    def setup_stations_table(self):
        db = self.get_cursor()
        db.execute("""CREATE TABLE IF NOT EXISTS stations (
           id INTEGER PRIMARY KEY,
           abbr VARCHAR(16) NOT NULL UNIQUE,
           network VARCHAR(16) NOT NULL,
           name TEXT NOT NULL,
           elevation NUMERIC,
           tzname TEXT,
           latitude NUMERIC,
           longitude NUMERIC,
           FOREIGN KEY (network)
               REFERENCES networks (abbr)
               ON DELETE NO ACTION
               ON UPDATE NO ACTION);""")
        self.c.commit()
    def setup_requests_table(self):
        db = self.get_cursor()
        db.execute("""CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY,
            uri TEXT NOT NULL,
            station VARCHAR(16),
            report_type INTEGER DEFAULT 0,
            valid_from DATE NOT NULL,
            valid_to DATE NOT NULL,
            dltime DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
            entries INTEGER DEFAULT NULL,
            FOREIGN KEY (station)
                REFERENCES stations (abbr)
                ON DELETE NO ACTION
                ON UPDATE NO ACTION);""")
        self.c.commit()
    def setup_metarmsg_table(self):
        db = self.get_cursor()
        db.execute("""CREATE TABLE IF NOT EXISTS metarmsg (
            id INTEGER PRIMARY KEY,
            station VARCHAR(16),
            issued DATETIME NOT NULL,
            request INTEGER,
            validtime INTEGER DEFAULT NULL,
            report_type INTEGER DEFAULT 0,
            msg TEXT NOT NULL,
            FOREIGN KEY (station)
                REFERENCES stations (abbr)
                ON DELETE NO ACTION
                ON UPDATE NO ACTION
            FOREIGN KEY (request)
                REFERENCES requests (id)
                ON DELETE NO ACTION
                ON UPDATE NO ACTION);""")
        self.c.commit()
    def setup_metarparams_table(self):
        db = self.get_cursor()
        db.execute("""CREATE TABLE IF NOT EXISTS metarparams (
            id INTEGER PRIMARY KEY,
            metarid INTEGER,
            version INTEGER,
            date DATETIME,
            station TEXT,
            mod TEXT,
            wind_dir NUMERIC,
            wind_spd NUMERIC,
            wind_gust NUMERIC,
            windvar_from NUMERIC,
            windvar_to NUMERIC,
            cavok BOOLEAN,
            vis NUMERIC,
            visdir NUMERIC,
            vis2 NUMERIC,
            vis2dir NUMERIC,
            temp NUMERIC,
            dwpt NUMERIC,
            pres NUMERIC,
            sky_ceiling NUMERIC,
            sky_cover TEXT,
            sky TEXT,
            rvr TEXT,
            wx TEXT,
            unparsed TEXT,
            FOREIGN KEY (metarid)
                REFERENCES metarmsg (id)
                ON DELETE NO ACTION
                ON UPDATE NO ACTION);""")
        self.c.commit()
    def setup_database(self):
        self.setup_networks_table()
        self.setup_stations_table()
        self.setup_requests_table()
        self.setup_metarmsg_table()
        self.setup_metarparams_table()
    def reset_database_soft(self):
        db = self.c.cursor()
        db.execute("""DROP TABLE IF EXISTS metarparams;""")
        db.execute("""DROP TABLE IF EXISTS metarmsg;""")
        db.execute("""DROP TABLE IF EXISTS requests;""")
        self.c.commit()
        self.setup_requests_table()
        self.setup_metarmsg_table()
        self.setup_metarparams_table()
    def reset_metarparams(self):
        db = self.c.cursor()
        db.execute("""DROP TABLE IF EXISTS metarparams;""")
        self.c.commit()
        self.setup_metarparams_table()
    def populate_networks(self):
        db = self.c.cursor()
        exsisting_networks = [abbr for (abbr,) in db.execute("SELECT abbr FROM networks;")]
        insert_networks = []
        with urllib.request.urlopen("https://mesonet.agron.iastate.edu/geojson/networks.geojson") as response:
            networksjson = json.loads(response.read())
            for nw in networksjson['features']:
                abbr, name, polygon = nw['id'], nw['properties']['name'], json.dumps(nw['geometry']['coordinates'])
                if abbr not in exsisting_networks:
                    insert_networks.append((abbr, name, polygon))
        db.executemany("INSERT INTO networks (abbr, name, polygon) VALUES (?,?,?);",insert_networks)
        self.c.commit()
    def populate_stations(self):
        db = self.c.cursor()
        exsisting_stations = [abbr for (abbr,) in db.execute("SELECT abbr FROM stations;")]

        insert_stations = {}
        for (abbr,) in db.execute("SELECT abbr FROM networks WHERE abbr LIKE \"%A_OS%\";"):
            with urllib.request.urlopen(f"https://mesonet.agron.iastate.edu/geojson/network/{abbr}.geojson") as response:
                networkjson = json.loads(response.read())
                for st in networkjson['features']:
                    icao, name, elev, tzname = st['properties']['sid'], st['properties']['sname'], st['properties']['elevation'], st['properties']['tzname']
                    lat, lon = st['geometry']['coordinates']
                    if icao not in exsisting_stations:
                        insert_stations[icao] = (icao, abbr, name, elev, tzname, lat, lon)
        db.executemany("INSERT INTO stations (abbr, network, name, elevation, tzname, latitude, longitude) VALUES (?,?,?,?,?,?,?);",insert_stations.values())
        self.c.commit()
    def calculate_validtimes(self):
        db = self.get_cursor()
        query = db.execute("""SELECT DISTINCT station FROM metarmsg WHERE validtime IS NULL""")
        stations = list(query)
        for s in stations:
            db.execute("""
                SELECT id, issued
                FROM metarmsg
                WHERE validtime IS NULL AND station=?
                ORDER BY issued ASC, id ASC
            """,s)
            update_list = []
            prev_ts, prev_i = None,None
            result = db.fetchone()
            while result:
                i, issued = result
                issued, = self.to_dates(issued)
                ts = int(issued.strftime('%s'))
                if prev_ts is not None:
                    update_list.append((ts-prev_ts,prev_i))
                prev_ts, prev_i = ts, i
                result = db.fetchone()

            db.executemany("UPDATE metarmsg SET validtime=? WHERE id=?;",update_list)
            self.c.commit()
            print(s[0],end=' ',flush=True)

    def check_if_in_requests(self,station,start,end):
        db = self.get_cursor()
        query = db.execute("SELECT valid_from, valid_to FROM requests WHERE station=?",(station,))
        valid_times = [self.to_dates(f,t) for f,t in query]
        return self.dates_overlap(start,end,valid_times)

    @classmethod
    def dates_overlap(cls,inner_start,inner_end,outer_list):
        for i, (outer_start, outer_end) in enumerate(outer_list):
            if outer_start <= inner_start and inner_end <= outer_end:
                return True, [] # Full overlap
            elif outer_end <= inner_start or inner_end <= outer_start:
                continue # No overlap at all
            elif i+1>=len(outer_list):
                return False, [(inner_start,inner_end)]
            elif outer_start <= inner_start and outer_end < inner_end: # Partial overlap, extends at the right
                return cls.dates_overlap(outer_end,inner_end,outer_list[i+1:])
            elif inner_start < outer_start and inner_end <= outer_end: # Partial overlap, extends at the left
                return cls.dates_overlap(inner_start,outer_start,outer_list[i+1:])
            elif inner_start < outer_start and outer_end < inner_end: # Partial overlap, extends at both sides
                before, beforelist = cls.dates_overlap(outer_end,inner_end,outer_list[i+1:])
                after, afterlist = cls.dates_overlap(inner_start,outer_start,outer_list[i+1:])
                return before and after, beforelist+afterlist
        return False, [(inner_start,inner_end)]
    @classmethod
    def download_data(cls,uri):
        attempt = 0
        while attempt < 6:
            try:
                data = urllib.request.urlopen(uri, timeout=300).read().decode("utf-8")
                if data is not None and not data.startswith("ERROR"):
                    return data
            except Exception as exp:
                print("download_data(%s) failed with %s" % (uri, exp))
                time.sleep(5)
            attempt += 1
        raise ValueError("Exhausted attempts to download, returning empty data")
    @classmethod
    def to_dates(cls,*args):
        returnlist = []
        for i,a in enumerate(args):
            if isinstance(a,datetime.datetime):
                returnlist.append(a)
            elif isinstance(a,datetime.date):
                returnlist.append(datetime.datetime.combine(a,datetime.time(0,0)))
            elif isinstance(a,int):
                returnlist.append(datetime.datetime.fromtimestamp(a))
            elif isinstance(a,str):
                if ':' in a:
                    returnlist.append(datetime.datetime.strptime(a,'%Y-%m-%d %H:%M'))
                elif '-' in a:
                    returnlist.append(datetime.datetime.strptime(a,'%Y-%m-%d'))
                elif a.isnumeric():
                    returnlist.append(datetime.datetime.fromtimestamp(int(a)))
                else:
                    raise ValueError('Could not convert ['+str(i)+']'+str(type(a))+': '+str(a))
            else:
                raise ValueError('Could not convert ['+str(i)+']'+str(type(a))+': '+str(a))
        return returnlist

    def get_metars(self,station,start,end,fields=None):
        needs_request = self.request_if_nessesary(station,start,end)
        start, end = self.to_dates(start,end)
        if needs_request:
            return False
        else:
            if isinstance(fields,list):
                dbfields = ", ".join([k for k in fields if k in ['id', 'station','issued', 'import', 'msg']])
            else:
                dbfields = 'id, issued, msg'
            db = self.get_cursor()
            query = db.execute(f"""
                SELECT {dbfields}
                FROM metarmsg
                WHERE station = ? AND issued >= ? AND issued <= ?
            """,(station,start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d')))
            if dbfields.count(',')==0:
                return [k for (k,) in query]
            return list(query)
    def query(self,querystr):
        db = self.get_cursor()
        return db.execute(querystr)

    def run_download_requests(self,**kwargs):
        for station, dllist in self.download_requests:
            for start, end in dllist:
                self.request(station,start,end,**kwargs)
    def request(self,station,start,end,**kwargs):
        self.request_metars(station,start,end,**kwargs)
        self.request_specis(station,start,end,**kwargs)
    def request_metars(self,station,start,end,**kwargs):
        start, end = self.to_dates(start,end)
        urlparams = {
            'station': station,
            'year1': start.strftime('%Y'),
            'month1': start.strftime('%-m'),
            'day1': start.strftime('%-d'),
            'year2': end.strftime('%Y'),
            'month2': end.strftime('%-m'),
            'day2': end.strftime('%-d'),
            'data': 'metar',
            'tz': 'Etc/UTC',
            'format': 'onlycomma',
            'latlon': 'no',
            'elev': 'no',
            'missing': 'M',
            'trace': 'T',
            'direct': 'no',
            'report_type': '1',
        }
        urlparams.update(kwargs)
        urlp = urllib.parse.urlencode(list(urlparams.items()),doseq=True)
        uri = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?' + urlp

        db = self.get_cursor()
        db.execute("INSERT INTO requests (uri,station,report_type,valid_from,valid_to) VALUES (?,?,?,?,?);",
            (uri,station,1,start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d')))
        request_id = db.lastrowid
        self.c.commit()

        data = self.download_data(uri)
        insert_list = []
        for line in data.split('\n'):
            elements = line.split(',',3)
            if not line.startswith('station') and len(elements)==3:
                s, v, m = elements
                insert_list.append((s, v, request_id, 1, m))
        db.executemany('INSERT INTO metarmsg (station,issued,request,report_type,msg) VALUES (?,?,?,?,?)',insert_list)
        added = db.rowcount
        self.c.commit()

        db.execute("UPDATE requests SET entries=? WHERE id=?;",(added,request_id))
        self.c.commit()

        return added
    def request_specis(self,station,start,end,**kwargs):
        start, end = self.to_dates(start,end)
        urlparams = {
            'station': station,
            'year1': start.strftime('%Y'),
            'month1': start.strftime('%-m'),
            'day1': start.strftime('%-d'),
            'year2': end.strftime('%Y'),
            'month2': end.strftime('%-m'),
            'day2': end.strftime('%-d'),
            'data': 'metar',
            'tz': 'Etc/UTC',
            'format': 'onlycomma',
            'latlon': 'no',
            'elev': 'no',
            'missing': 'M',
            'trace': 'T',
            'direct': 'no',
            'report_type': '2',
        }
        urlparams.update(kwargs)
        urlp = urllib.parse.urlencode(list(urlparams.items()),doseq=True)
        uri = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?' + urlp

        db = self.get_cursor()
        db.execute("INSERT INTO requests (uri,station,report_type,valid_from,valid_to) VALUES (?,?,?,?,?);",
            (uri,station,2,start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d')))
        request_id = db.lastrowid
        self.c.commit()

        data = self.download_data(uri)
        insert_list = []
        for line in data.split('\n'):
            elements = line.split(',',3)
            if not line.startswith('station') and len(elements)==3:
                s, v, m = elements
                insert_list.append((s, v, request_id, 2, m))
        db.executemany('INSERT INTO metarmsg (station,issued,request,report_type,msg) VALUES (?,?,?,?,?)',insert_list)
        added = db.rowcount
        self.c.commit()

        db.execute("UPDATE requests SET entries=? WHERE id=?;",(added,request_id))
        self.c.commit()

        return added
    
    def request_if_nessesary(self,station,start,end):
        start, end = self.to_dates(start,end)
        has_downloaded, download_list = self.check_if_in_requests(station,start,end)
        if not has_downloaded:
            if len(download_list)>=3:
                download_list = [start,end]
            self.download_requests.append((station,download_list))
            return True
        else:
            return False

    def import_metars(self,data,station,start,end,downloaded,**kwargs):
        start, end = self.to_dates(start,end)
        downloaded, = self.to_dates(downloaded)
        urlparams = {
            'station': station,
            'year1': start.strftime('%Y'),
            'month1': start.strftime('%-m'),
            'day1': start.strftime('%-d'),
            'year2': end.strftime('%Y'),
            'month2': end.strftime('%-m'),
            'day2': end.strftime('%-d'),
            'data': 'metar',
            'tz': 'Etc/UTC',
            'format': 'onlycomma',
            'latlon': 'no',
            'elev': 'no',
            'missing': 'M',
            'trace': 'T',
            'direct': 'no',
            'report_type': ['1','2'],
        }
        urlparams.update(kwargs)
        urlp = urllib.parse.urlencode(list(urlparams.items()),doseq=True)
        uri = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?' + urlp
        db = self.get_cursor()
        db.execute("INSERT INTO requests (uri,station,valid_from,valid_to,dltime) VALUES (?,?,?,?,?);",
            (uri,station,start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d'),downloaded.strftime('%Y-%m-%d %H:%M:%S')))
        request_id = db.lastrowid
        self.c.commit()

        insert_list = []
        for line in data.split('\n'):
            elements = line.split(',',3)
            if not line.startswith('station') and len(elements)==3:
                s, v, m = elements
                insert_list.append((s, v, request_id, m))
        db.executemany('INSERT INTO metarmsg (station,issued,request,msg) VALUES (?,?,?,?)',insert_list)
        added = db.rowcount
        self.c.commit()

        db.execute("UPDATE requests SET entries=? WHERE id=?;",(added,request_id))
        self.c.commit()
    def import_parsed(self,list_of_data):
        db = self.get_cursor()
        db.executemany("""
            INSERT INTO metarparams (
                metarid,version,date,station,mod,
                wind_dir,wind_spd,wind_gust,windvar_from,windvar_to,
                cavok,vis,visdir,vis2,vis2dir,
                temp,dwpt,pres,sky_ceiling,sky_cover,
                sky,rvr,wx,unparsed
            ) VALUES (
                ?,?,?,?,?,
                ?,?,?,?,?,
                ?,?,?,?,?,
                ?,?,?,?,?,
                ?,?,?,?)
            """,list_of_data)
        self.c.commit()
