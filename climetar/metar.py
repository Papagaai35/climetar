import logging
_log = logging.getLogger(__name__)

import collections
import datetime
import fractions
import numbers
import re

import numpy as np
import pandas as pd

from .quantities import *

class Metar(object):
    _version = 8

    _regexes = collections.OrderedDict()
    _old_regexes = collections.OrderedDict()
    _regexes['type'] = r"""^ (?:METAR|SPECI) \s+"""
    _regexes['cor'] = r"""^ CORR? \s+"""
    _regexes['station'] = r"""^ [A-Z][A-Z0-9]{3} \s+"""
    _regexes['time'] = r"""^ (?P<day>\d\d)(?P<hour>\d\d)(?P<min>\d\d)Z \s+"""
    _regexes['mod'] = r"""^ (?:AUTO|FINO|NIL|TEST|CORR?|RTD|CC[A-G]) \s+"""
    _regexes['wind'] = r"""^
        ((?P<dir>[0-3O][\dO]{2}|///|MMM|VRB))?/?
        ((?P<spd>P?[\dO]{2,3}|[/M]{2,3}|[1-9]{1}))?
        (G(?P<gust>[\dO]{1,3}|[/M]{1,3}))?
        ((?P<unit>KTS?|LT|K|T|KMH|MPS))?
        \s+"""
    _regexes['windvar'] = r"""^ (?P<from>\d\d\d)V(?P<to>\d\d\d) \s+"""
    _regexes['cavok'] = r"""^ CAVOK \s+"""
    _regexes['vis'] = r"""^
        (?:
            (?P<dists>(M|P)?[\dO]{3,4}|[/M]{3,4})
            (?P<dir>[NSEW][EW]?|NDV)?
            |
            (?P<distc>[MP]?[\dO]+|[/M]+|[MP]?\d\d?/\d\d?|\d\d?\s+\d\d?/\d\d?)
            (?P<unit>SM|KM|M|U)
        ) \s+"""
    _regexes['cavok2'] = r"""^ CAVOK \s+"""
    _regexes['rvrno'] = r"""^ RVRNO \s+"""
    _regexes['rvr'] = r"""^
        (?P<rwyname>R(?:[\dO]{2}|[/M]{2})[CLR]{0,2})
        /
        (?P<rvrlow>[MP]?(?:[\dO]{3,5}|[/M]{4}))
        (?:
            V
            (?P<rvrhigh>[MP]?(?:[\dO]{3,5}|[/M]{4}))
        )?
        (?P<rvrunit>(?:FT|[/M]{2}))?
        (?P<rvrtend>[DNU/])*
        \s+"""
    _regexes['wx'] = r"""^(
        (?P<int>(-|\+|VC)*)
        (?P<desc>(MI|PR|BC|DR|BL|SH|TS|FZ)+)?
        (?P<prec>(DZ|RA|SN|SG|IC|PL|GR|GS|UP|/)*)?
        (?P<obsc>(BR|FG|FU|VA|DU|SA|HZ|PY)+)?
        (?P<other>PO|SQ|FC|SS|DS|NSW|NULL|/+)?
        (?P<int2>[-+])?)+
        \s+"""
    _regexes['sky'] = r"""^
        (?:
            (?P<clear>CLR|SKC|SCK|NSC|NCD)
            |(?P<obscured>VV)
            |(?:
                (?P<cover>VV|BKN|SCT|FEW|[O0]VC|///)\s*?
                (?P<height>[\dO]{2,4}|///)\s*
                (?P<cloud>CB|TCU|/+)?
            )
        ) \s+"""
    _regexes['temp'] = r"""^
        (?P<temp>(M|-)?[\dO]{,2}|//|XX|MM)
        /
        (?P<dwpt>(M|-)?[\dO]{,2}|//|XX|MM)? \s+"""
    _regexes['pres'] = r"""^
        (?P<unit>A|Q|QNH|ALSTG)
        (?P<p>[\dO]{3,5}|////)
        (?P<unit2>INS)?
        \s+"""
    _regexes['rewx'] = r"""^
        RE
        (?P<int>(-|\+|VC)*)
        (?P<desc>(MI|PR|BC|DR|BL|SH|TS|FZ)+)?
        (?P<prec>(DZ|RA|SN|SG|IC|PL|GR|GS|UP|/)*)?
        (?P<obsc>(BR|FG|FU|VA|DU|SA|HZ|PY)+)?
        (?P<other>PO|SQ|FC|SS|DS|NSW|SH|TS|/+)?
        (?P<int2>[-+])?
        \s+"""
    #_regexes['shear_all'] = r'WS\s+ALL\s+RWY'
    _regexes['shear'] = r"""^
        (?:WS\s+)
        (ALL\s+RWY|RW?Y?\d\d[CLR]{0,2})?
        \s+"""
    _regexes['seatemp'] = r"""^
        W(?P<temp>(M|-)?[\dO]+|//|XX|MM)/
        (?:S(?P<state>[\dO]))?(?:H(?P<wave>[\dO]{3,4}))?
        \s+"""
    _regexes['rwy'] = r"""^
        (?P<rwyname>R?\d\d[CLR]{0,2})
        /?
        (?:
            (?P<deposit>[\d/])
            (?P<extent>[\d/])
            (?P<depth>[\d/]{2})
            |
            (?P<clrd>CLRD)
        )
        (?P<friction>[\d/]{2})
        \s+"""
    _old_regexes['rwy'] = r"""^
        (?P<rwyname>R\d\d[CLR]{0,2})
        ((?P<special>SNOCLO|CLRD[\d/]{2}))|
            (?P<deposit>[\d/])
            (?P<extent>[\d/]
            (?P<depth>[\d/]{2})
            (?P<friction>[\d/]{2})
        ) \s+"""
    _regexes['color'] = r"""^
        (BLACK)?
        (?P<code>(BLU|WHT|GRN|YLO1|YLO2|YLO|AMB|RED)\+?)
        (?:[\s/]?(BLACK)?(BLU|WHT|GRN|YLO1|YLO2|YLO|AMB|RED|/+)\+?)*
        \s+"""
    _regexes['rmk'] = r"""^ (RMKS?)(?P<remarks>.*)\Z"""
    _rmk_regexes = collections.OrderedDict()
    _rmk_regexes['auto'] = r"""^AO(?P<type>\d)\s+"""
    _rmk_regexes['slp'] = r"""^SLP(?P<p>\d\d\d)\s+"""
    _rmk_regexes['temp'] = """^
        T(?P<tsign>0|1)
        (?P<temp>\d\d\d)
        ((?P<dsign>0|1)
        (?P<dwpt>\d\d\d))?
        \s+"""

    _cloud_cover_codes = {'SKC':-3,'NCD':-2,'CLR':-1,'NSC':0,'FEW':1,'SCT':3,'BKN':5,'OVC':8,'VV':9}
    _color_codes = {'BLU':0,'WHT':1,'GRN':2,'YLO':3,'YLO1':4,'YLO2':5,'AMB':6,'RED':7}
    _regex_unparsed = r'^(?P<metar>.+?)(?P<trend>\s*(?:$|(?:[TY]EMPO|BECMG|NOSII?G|FCST|PROB\d\d).*$))'
    _multigroups = ['rvr','wx','sky','color',]
    _ignore_groups = ['M','$','=','///']

    @classmethod
    def get_regex_by_index(cls,id):
        return list(cls._regexes.keys())[id], list(cls._regexes.values())[id]
    @classmethod
    def get__regexes(cls):
        return cls._regexes
    @classmethod
    def is_match(cls,match):
        return match is not None and match.start()<match.end() and len(match.group(0).strip())>0

    def __init__(self,code,year=None,month=None,debug=None,**kwargs):
        self.data = {}
        self.year = year
        self.month = month
        self.elements = dict.fromkeys(list(self._regexes.keys()),None)
        self.orig = code
        self.code = str(code).upper()
        self.trendcode = ''
        self.debug = {}
        self.debug_for = []
        self.parsed = []
        self.unparsed = ""
        self.groupcount = dict.fromkeys(list(self._regexes.keys()),0)
        self.trends = []
        self.handled = []
        self.kwargs = kwargs
        if debug is not None:
            self.process_debug_fields(debug)
    def sanitize(self):
        self.code = self.code.strip()
        for s in '$=-':
            self.code = self.code.rstrip('-')
        self.code += ' '
        while True:
            for g in self._ignore_groups:
                if self.code.startswith(g+' '):
                    self.code = self.code[len(g)+1:]
                    continue
            break
    def process_debug_fields(self,debug=None):
        if debug is None:
            pass
        elif debug is True:
            self.debug_for = list(self._regexes.keys())
            self.debug_for.append('trendall')
        elif isinstance(debug,str):
            self.debug_for.append(debug)
        elif isinstance(debug,list):
            self.debug_for = debug
    def parse(self,trend=False):
        rxid = 0

        match = re.search(self._regex_unparsed,self.code,re.VERBOSE)
        if match is not None:
            mgd = match.groupdict()
            self.code = mgd['metar'].strip()
            self.trendcode = mgd['trend'].strip()

        while rxid<len(self._regexes):
            self.sanitize()
            rxname, rx = self.get_regex_by_index(rxid)
            rxnamef = rxname
            self.groupcount[rxname] += 1
            if self.groupcount[rxname]>1 or rxname in self._multigroups:
                rxnamef += "%d"%self.groupcount[rxname]

            match = re.search(rx,self.code,re.VERBOSE)
            if rxname in self.debug_for:
                if rxnamef not in self.debug:
                    self.debug[rxnamef] = []
                self.debug[rxnamef] += [{
                    'match': match,
                    'regex': rx,
                    'code': self.code,
                    'postcode': self.code[match.end():] if match is not None else self.code
                }]

            if not self.is_match(match):
                if self.elements[rxname] is None:
                    self.elements[rxname] = ''
                rxid += 1
                continue

            result = match.group(0).strip()
            groupresult_dict = dict([(rxnamef+'_'+k, v) for k,v in match.groupdict().items()])
            self.elements[rxnamef] = result
            self.parsed.append(match.group(0).strip())
            self.elements.update(groupresult_dict)
            self.code = self.code[match.end():]
            if rxname in self.debug_for:
                i = len(self.debug[rxnamef])-1
                self.debug[rxnamef][i]['result'] = result
                self.debug[rxnamef][i]['groups'] = groupresult_dict
        for mg in self._multigroups:
            self.elements[mg] = " ".join([self.elements[mg+str(i)] for i in range(1,self.groupcount[mg])])
        self.unparsed = self.code.strip()+' '

        if trend is True:
            if ('trend' in self.debug_for or
                    'trends' in self.debug_for or
                    'trendall' in self.debug_for):
                self.debug['trends_precode'] = self.code
                self.debug['trends_preunparsed'] = self.unparsed
            self.parse_trend()

        self.code = " ".join([pstr.strip() for pstr in self.parsed])
        self.unparsed = self.unparsed.strip()
    def parse_trend(self):
        self.unparsed += self.trendcode
        while len(self.unparsed.strip())>0:
            mt = MetarTrend(self.unparsed,debug=self.debug_for)
            mt.parse()
            if len(mt.unparsed.strip())<len(self.unparsed.strip()):
                self.trends.append(mt)
                self.unparsed = mt.unparsed.strip()+' '
                self.parsed.append(mt.code.strip())
                self.code = " ".join([pstr.strip() for pstr in self.parsed])
            else:
                break
        self.unparsed = self.unparsed.strip()
    def handle(self):
        for key in self._regexes.keys():
            fn = getattr(self,'handle_'+key,None)
            if fn is not None and key not in self.handled:
                fn()
        self.data['metar'] = self.orig
        self.data['unparsed'] = self.unparsed
    def handle_station(self):
        self.data['type'] = self.elements.get('type','')
        self.data['station'] = self.elements.get('station','ZZZZ')
        self.handled.append('station')
    def handle_mod(self):
        mod_list = [self.elements.get('cor',''),self.elements.get('mod','')]
        self.data['mod'] = " ".join([m.replace('CORR','COR').replace('FINO','NIL') for m in mod_list]).strip()
        self.handled.append('mod')
    def handle_time(self,split=False):
        self.data['time'] = np.nan
        if len(self.elements.get('time',''))>0:
            dates = self.elements.get('time')
            datef = '%d%H%MZ'
            if self.month is not None:
                datef = '%m' + datef
                dates = ('%02d'%int(self.month) ) + dates
            if self.year is not None:
                datef = '%Y' + datef
                dates = ('%04d'%int(self.year) ) + dates
            self.data['time'] = pd.to_datetime(dates,format=datef)
            self.data['date'] = self.data['time'].to_pydatetime()
            if split:
                self.data['year'] = self.data['time'].year
                self.data['month'] = self.data['time'].month
                self.data['day'] = float(self.elements['time_day'])
                self.data['hour'] = float(self.elements['time_hour'])
                self.data['min'] = float(self.elements['time_min'])
        elif split:
            for tt in ['year','month','day','hour','min']:
                self.data[tt] = np.nan
        self.handled.append('time')
    def handle_wind(self):
        for v in ['wind_dir','windvar_from','windvar_to']:
            self.data[v] = Direction(self.elements.get(v,np.nan),'deg')
        wunit = self.elements.get('wind_unit')
        for v in ['wind_spd','wind_gust']:
            self.data[v] = Speed(self.elements.get(v,np.nan),wunit)
        self.handled.append('wind')
    def handle_vis(self):
        if self.elements['cavok'] == 'CAVOK' or self.elements['cavok2'] == 'CAVOK':
            self.data['cavok'] = True
            self.data['vis'] = Distance('P9999')
            self.data['visdir'] = Direction(None,'compass')
            self.data['vis2'] = Distance(np.nan,'')
            self.data['vis2dir'] = Direction(None,'compass')
        else:
            self.data['cavok'] = False
            if self.elements.get('vis_dists') is not None:
                vis_dist = self.elements.get('vis_dists')
                self.data['vis'] = Distance(('P9999' if vis_dist=='9999' else vis_dist),'m')
                self.data['visdir'] = Direction(self.elements.get('vis_dir'),'compass')
            else: #self.elements.get('vis_distc') is not None:
                vis_dist = self.elements.get('vis_distc')
                self.data['vis'] = Distance(('P9999' if vis_dist=='9999' else vis_dist),self.elements.get('vis_unit','m'))
                self.data['visdir'] = Direction(None,'compass')
            if 'vis2' in self.elements:
                if self.elements.get('vis2_dists') is not None:
                    vis2_dist = self.elements.get('vis2_dists')
                    self.data['vis2'] = Distance(('P9999' if vis2_dist=='9999' else vis2_dist),'m')
                    self.data['vis2dir'] = Direction(self.elements.get('vis2_dir'),'compass')
                else: #self.elements.get('vis2_distc') is not None:
                    vis2_dist = self.elements.get('vis2_distc')
                    self.data['vis2'] = Distance(('P9999' if vis2_dist=='9999' else vis2_dist),
                        self.elements.get('vis2_unit',self.elements.get('vis_unit','m')))
                    self.data['vis2dir'] = Direction(None,'compass')
            else:
                self.data['vis2'] = Distance(np.nan,'')
                self.data['vis2dir'] = Direction(None,'compass')
        self.handled.append('vis')
    def handle_rvr(self):
        rvrkeys = ['rvrno'] + ['rvr%d'%i for i in range(1,self.groupcount['rvr'])]
        rvrstr = list(filter(lambda x: x != "",
            [self.elements.get(k,'') for k in rvrkeys]))
        self.data['rvr'] = " ".join(rvrstr).strip()
        self.handled.append('rvr')
    def handle_wx(self):
        wxkeys = ['wx%d'%i for i in range(1,self.groupcount['wx'])]
        wsstr = list(filter(lambda x: x != "",
            [self.elements.get(k,'') for k in wxkeys]))
        self.data['wx'] = " ".join(wsstr).strip()
        self.handled.append('wx')
    def handle_sky(self):
        self.data['sky_ceiling'] = np.nan
        self.data['sky_cover'] = ''
        self.data['sky_cover_index'] = -3
        self.data['sky'] = ''
        if self.elements['cavok'] == 'CAVOK' or self.elements['cavok2'] == 'CAVOK':
            self.data['cavok'] = True
            self.data['sky1_cover'] = 'NSC'
            self.data['sky1_cover_index'] = 0
            self.data['sky_cover_index'] = 0
            self.data['sky'] = 'CAVOK'
        else:
            self.data['sky'] = self.elements.get('sky','')
            for i in range(1,5):
                s = 'sky%d_'%i
                self.data[s+'cover'] = ''
                self.data[s+'height'] = np.nan
                self.data[s+'cover_index'] = np.nan
                if i < self.groupcount['sky']:
                    cover = self.elements.get(s+'cover','')
                    clear = self.elements.get(s+'clear','')
                    obsur = self.elements.get(s+'obscured','')
                    cover = str(cover or clear or obsur).strip()
                    height = Height.of_cloud(self.elements.get(s+'height',''))
                    if cover=='VV' and pd.isnull(self.elements.get(s+'height','')):
                        height = Height.of_cloud('000')
                    self.data[s+'cover'] = cover
                    self.data[s+'cover_index'] = self._cloud_cover_codes.get(cover,np.nan)
                    self.data['sky_cover_index'] = np.nanmax([self.data[s+'cover_index'],self.data['sky_cover_index']])
                    self.data[s+'height'] = height
                    if cover in ['SCT','BKN','OVC','VV'] and pd.isnull(self.data['sky_ceiling']):
                        self.data['sky_ceiling'] = height
        self.data['sky_cover'] = list(self._cloud_cover_codes.keys())[list(self._cloud_cover_codes.values()).index(self.data['sky_cover_index'])]
        self.handled.append('sky')
    def handle_temp(self):
        self.data['temp'] = Temperature(self.elements.get('temp_temp'),'°C',validate=lambda x: abs(x)<100)
        self.data['dwpt'] = Temperature(self.elements.get('temp_dwpt'),'°C',validate=lambda x: abs(x)<100)
        self.handled.append('temp')
    def handle_pres(self):
        self.data['pres'] = Pressure(self.elements.get('pres_p'),self.elements.get('pres_unit','Q'))
        self.handled.append('pres')
    def handle_color(self):
        self.data['color'] = self.elements.get('color1','').strip()
        self.handled.append('color')
    def handle_rmk(self):
        remarkmessage = self.elements.get('rmk_remarks','').strip()
        self.data['rmk'] = remarkmessage
        
        parsed_rkmrxs = []
        while len(remarkmessage)>0:
            found = False
            for rmkrxname,rmkrx in self._rmk_regexes.items():
                if rmkrxname in parsed_rkmrxs:
                    continue
                match = re.search(rmkrx,remarkmessage,re.VERBOSE)
                if self.is_match(match):
                    found = True
                    result_dict = {'RMK_'+rmkrxname+'_'+k:v for k,v in match.groupdict().items()}
                    result_dict['RMK_'+rmkrxname] = match.group(0)
                    self.elements.update(result_dict)
                    remarkmessage = remarkmessage[match.end():]
                    parsed_rkmrxs.append(rmkrxname)
                    break
            if found:
                continue
            if (
                len(parsed_rkmrxs)==len(list(self._rmk_regexes.keys())) or
                len(remarkmessage.strip())==0 or
                " " not in remarkmessage.strip()):
                break
            remarkmessage = remarkmessage[remarkmessage.find(" ")+1:]
        
        for key in parsed_rkmrxs:
            fn = getattr(self,'handlermk_'+key,None)
            if fn is not None and 'RMK_'+key not in self.handled:
                fn()
    def handlermk_auto(self):
        self.data['mod'] = " ".join( self.data.get('mod','').strip().split(' ') + [self.elements.get('RMK_auto','').strip()] )
        self.handled.append('RMK_auto')
    def handlermk_slp(self):
        if pd.isnull(self.data.get('pres',np.nan)) or pd.isnull(float(self.data.get('pres',np.nan))):
            self.data['pres'] = Pressure.missing_first_digit_daPa(self.elements.get('RMK_slp_p',''))
        self.handled.append('RMK_slp')
    def handlermk_temp(self):
        temp,dwpt = self.elements.get('RMK_temp_temp'), self.elements.get('RMK_temp_dwpt')
        if temp is not None and temp!='':
            tsign = self.elements.get('RMK_temp_tsign')
            tsignm = '-' if tsign is not None and tsign=='1' else ''
            self.data['temp'] = Temperature(tsignm+temp,'d°C')
        if dwpt is not None and dwpt!='':
            dsign = self.elements.get('RMK_temp_tsign')
            dsignm = '-' if dsign is not None and dsign=='1' else ''
            self.data['dwpt'] = Temperature(dsignm+dwpt,'d°C')
        self.handled.append('RMK_temp')

    def calculate_color(self):
        if ('color' not in self.data or
                self.data['color']=='') and (
                'sky' in self.handled and
                'vis' in self.handled):
            self.data['color'] = self.calc_color(self.data['vis']['m'],self.data['sky_ceiling']['ft'])
        else:
            self.data['color'] = ''
    def calculate_relh(self):
        if 'relh' not in self.data and 'temp' in self.handled:
            self.data['relh'] = self.calc_relh(self.data['temp']['°C'],self.data['dwpt']['°C'])
        else:
            self.data['relh'] = np.nan
    @classmethod
    def calc_color(cls,vis_m,sky_ceiling_ft):
        visindex = np.digitize(vis_m,         [np.inf,8000,5000,3700,2500,1600, 800])
        skyindex = np.digitize(sky_ceiling_ft,[np.inf,2500,1500, 700, 500, 300, 200])
        maxindex = np.max(np.array([skyindex,visindex]),axis=0).astype(int)
        return np.take(['','BLU','WHT','GRN','YLO1','YLO2','AMB','RED'],maxindex)
    @classmethod
    def calc_relh(cls,temp_c,dwpt_c):
        vapor_eq = lambda temp_K: 0.611 * np.exp(17.2694*(temp_K-273.16)/(temp_K-35.86))
        sat_vapor_pressure = vapor_eq(temp_c+273.15)
        vapor_pressure = vapor_eq(dwpt_c+273.15)
        return vapor_pressure/sat_vapor_pressure

    def to_tup(self,cols=None,direction='deg',distance='m',height='ft',speed='kt',temperature='°C',pressure='hPa'):
        return_tup = []
        if cols is None:
            cols = [
                'metarid', 'version', 'date', 'station', 'mod',
                'wind_dir', 'wind_spd', 'wind_gust', 'windvar_from', 'windvar_to',
                'cavok', 'vis', 'visdir', 'vis2', 'vis2dir',
                'temp', 'dwpt', 'pres', 'sky_ceiling', 'sky_cover',
                'sky', 'rvr', 'wx', 'unparsed']
        for k in cols:
            if k in self.data:
                v = self.data[k]
                if isinstance(v,Direction):
                    return_tup.append(v[direction])
                elif isinstance(v,Height):
                    return_tup.append(v[height])
                elif isinstance(v,Distance):
                    return_tup.append(v[distance])
                elif isinstance(v,Speed):
                    return_tup.append(v[speed])
                elif isinstance(v,Temperature):
                    return_tup.append(v[temperature])
                elif isinstance(v,Pressure):
                    return_tup.append(v[pressure])
                elif isinstance(v,datetime.datetime):
                    return_tup.append(v.strftime('%Y-%m-%d %H:%M:%S'))
                else:
                    return_tup.append(v)
            elif k in self.kwargs:
                return_tup.append(self.kwargs[k])
            elif k=='version':
                return_tup.append(self.version)
            else:
                return_tup.append(None)
        return tuple(None if pd.isna(v) else v for v in return_tup)

    def to_dict(self,direction='deg',distance='m',height='ft',speed='m/s',temperature='°C',pressure='hPa'):
        series_dict = {}
        for k,v in self.data.items():
            if isinstance(v,Direction):
                series_dict[k] = v[direction]
            elif isinstance(v,Height):
                series_dict[k] = v[height]
            elif isinstance(v,Distance):
                series_dict[k] = v[distance]
            elif isinstance(v,Speed):
                series_dict[k] = v[speed]
            elif isinstance(v,Temperature):
                series_dict[k] = v[temperature]
            elif isinstance(v,Pressure):
                series_dict[k] = v[pressure]
            else:
                series_dict[k] = v
        for k,v in self.kwargs.items():
            if k not in series_dict:
                series_dict[k] = v
        return series_dict
    def to_line(self,usep='\x1f',cols=None,**kwargs):
        series_dict = self.to_dict(**kwargs)
        main_list = [str(series_dict.get(k,np.nan)) for k in cols]
        return usep.join(main_list)
    def to_series(self,cols=None,**kwargs):
        series_dict = self.to_dict(**kwargs)
        if cols is not None:
            main_dict = dict([(k,series_dict.get(k,np.nan)) for k in cols])
            return pd.Series(main_dict)
        else:
            return pd.Series(series_dict)

    @classmethod
    def get_headers(cls,cols=None):
        allcols = ['type', 'station', 'mod', 'time',
        'wind_dir', 'windvar_from', 'windvar_to', 'wind_spd', 'wind_gust',
        'cavok', 'vis', 'visdir', 'vis2', 'vis2dir',
        'rvr', 'wx',
        'temp', 'dwpt', 'relh', 'pres', 'color', 'calc_color',
        'sky_ceiling', 'sky_cover', 'sky_cover_index',
        'sky1_height', 'sky1_cover', 'sky1_cover_index',
        'sky2_height', 'sky2_cover', 'sky2_cover_index',
        'sky3_height', 'sky3_cover', 'sky3_cover_index',
        'sky4_height', 'sky4_cover', 'sky4_cover_index','metar']
        COLS = []
        if isinstance(cols,list):
            for c in cols:
                if c == '_all_':
                    COLS += [ac for ac in allcols if ac not in cols]
                else:
                    COLS.append(c)
        else:
            COLS = allcols
        return COLS

class MetarTrend(Metar):
    _regexes =  collections.OrderedDict()
    _regexes['trend_type'] = r'(?:[TY]EMPO|BECMG|NOSII?G|FCST)'
    _regexes['trend_time'] = r'(?P<when>(FM|TL|AT))\s*?(?P<hour>\d\d)(?P<min>\d\d)'
    _regexes['color'] = Metar._regexes['color']
    _regexes['wind'] = Metar._regexes['wind']
    _regexes['windvar'] = Metar._regexes['windvar']
    _regexes['cavok'] = Metar._regexes['cavok']
    _regexes['vis'] = Metar._regexes['vis']
    _regexes['wx'] = Metar._regexes['wx']
    _regexes['sky'] = Metar._regexes['sky']

    _regex_unparsed = ''
    _multigroups = ['sky','color',]

    def __init__(self,code,debug=None):
        super(MetarTrend, self).__init__(code,debug)
        if 'trendall' in self.debug_for:
            for rxname in self._regexes.keys():
                if rxname not in self.debug_for:
                    self.debug_for.append(rxname)
    def parse(self):
        rxid = 0
        while rxid<len(self._regexes):
            self.sanitize()
            rxname, rx = self.get_regex_by_index(rxid)
            rxnamef = rxname
            self.groupcount[rxname] += 1
            if self.groupcount[rxname]>1 or rxname in self._multigroups:
                rxnamef += "%d"%self.groupcount[rxname]

            match = re.search(r'^'+rx+'\s+',self.code,re.VERBOSE)
            if rxname in self.debug_for:
                self.debug[rxnamef] = {
                    'match': match,
                    'regex': r'^'+rx+r'\s+',
                    'code': self.code,
                    'postcode': self.code[match.end():] if match is not None else self.code
                }

            if not self.is_match(match):
                if self.elements[rxname] is None:
                    self.elements[rxname] = ''
                rxid += 1
                continue

            self.elements[rxnamef] = match.group(0).strip()
            self.parsed.append(match.group(0).strip())
            self.elements.update(dict([(rxnamef+'_'+k, v) for k,v in match.groupdict().items()]))
            self.code = self.code[match.end():]
        for mg in self._multigroups:
            self.elements[mg] = " ".join([self.elements[mg+str(i)] for i in range(1,self.groupcount[mg])])
        self.unparsed = self.code.strip()
        self.code = " ".join([pstr.strip() for pstr in self.parsed])
    def parse_trend(self):
        raise NotImplementedError('This function is unavailable for MetarTrend')
