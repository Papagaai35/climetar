import logging
_log = logging.getLogger(__name__)

import copy
import datetime
import functools
import glob
import json
import locale
import numbers
import pathlib
import re
import shutil
import subprocess
import sys, os
import zipfile

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from .svgpath2mpl import parse_path

from . import metar, quantities, MetarTheme, StationRepo, Astro

class MetarPlotter(object):
    def __init__(self,**settings):
        self.theme = MetarTheme(settings.get('theme'))
        self.station_repo = StationRepo()
        self.style = settings.get('style','./resources/climetar.mplstyle')
        if os.path.isfile(self.style):
            plt.style.use(self.style)

        self.filepaths = {
            'data': settings.get('folder_data','./data'),
            'output': settings.get('folder_output','./results/MonthlyPDF/'),
            'tex_head': settings.get('tex_head','./resources/T0.head.tex'),
            'tex_month': settings.get('tex_month','./resources/T1.month.tex'),
            'fonts': settings.get('fonts','./resources/fonts/'),
            'logo': settings.get('logo','./resources/JMG.png'),
            'natural_earth': settings.get('natural_earth','./resources/'),
        }
        #self.locale = settings.get('lang','en_GB.utf8')
        self.locales = {}

        self.station = None
        self.station_data = None
        self.stations_on_map = []
        self.df = None
        self.pdf = None
        self.astro = None
        self.countryfinder = None
        self.filters = {}
        self.years = {}
        self.warnings = []

        self._load_locales()
        _log.debug("Loaded MetarPlotter with settings:\n"+json.dumps(settings))

    def _load_locales(self):
        #locale.setlocale(locale.LC_ALL,self.locale)
        self.locales['monthnames'] = dict([(m,datetime.datetime.strptime("%02d"%m,"%m").strftime("%B")) for m in range(1,13)])
        self.locales['monthabbr'] = dict([(m,datetime.datetime.strptime("%02d"%m,"%m").strftime("%b")) for m in range(1,13)])
        #locale.setlocale(locale.LC_ALL,locale.getdefaultlocale())
    def load_data(self,station):
        self.station, self.station_data = self.station_repo.get_station(station)
        self.icao = self.station_data['icao']

        filename = os.path.join(self.filepaths['data'],self.station_data['abbr']+'.metar')
        if (not os.path.exists(filename) or
            not os.path.isfile(filename) ):
            filename = os.path.join(self.filepaths['data'],self.station_data['icao']+'.metar')
        if (not os.path.exists(filename) or
            not os.path.isfile(filename) ):
            raise ValueError(f'Kon het databestand niet vinden "{filename}"')

        self.df = pd.read_csv(filename,
            sep='\x1f',index_col=0,parse_dates=['time'],
            dtype=dict([(k,str) for k in [
                'type','station','date','mod',
                'rvr','wx','sky','color','calc_color',
                'sky_cover','sky1_cover','sky2_cover','sky3_cover','sky4_cover',
                'metar','unparsed']])
            )
        self.df = self.df.loc[self.df.station==self.icao,:]
        self.df = self.df.sort_values(by='time')
        self.df = self.df.drop_duplicates(subset=['stationid','time','metar'])
        self.df = self.df.reset_index(drop=True)
        self.df['minutes_valid'] = (self.df.time.shift(-1)-self.df.time).dt.total_seconds()/60

        self.reset_filters()
        self.years = {}
        self.warnings = []
        _log.debug("Loaded station %s"%station)
    def get_astro(self,station=None,year=None,force=False):
        if force or self.astro is None:
            station = station if station is not None else self.station
            year = year if year is not None else int(pd.Timestamp.today().strftime('%Y'))
            self.astro = Astro(station=self.station,year=year)
        return self.astro

    def reset_filters(self):
        self.pdf = self.df.copy(deep=True)
        self.filters = dict([(k,[]) for k in ['year','month','day','hour','minutes_valid','wvctp']])
        _log.debug("Resetting all filters")
    def redo_filters(self,filters):
        for k,fltrs in filters.items():
            if hasattr(self,f'filter_{k}'):
                for fltr in fltrs:
                    getattr(self,f'filter_{k}')(*fltr)
    def apply_filters(self,filters):
        for k,v in filters.items():
            if k not in self.filters:
                raise ValueError('Kan niet filteren op "%s", kies een van year,month,day,hour,minutes_valid')
            if k=='wvctp':
                if v[0:2] in ['==','!=']:
                    self.filter_wvctp(v[0:2],v[2:])
                elif v[0] in '=':
                    self.filter_wvctp(v[0],v[1:])
                elif v in ['True',True,1,'1']:
                    self.filter_wvctp('=',True)
                elif v in ['False',False,0,'0']:
                    self.filter_wvctp('=',False)
                else:
                    raise ValueError("Een wvctp filter (Wind, Visibility, Clouds, Temp, Pressure) kan alleen =, ==, !=, 0, 1, True of False bevatten")
            elif '...' in v:
                v1,v2 = v.split('...',2)
                getattr(self,f'filter_{k}')('...',[float(v1),float(v2)])
            elif v[0:2] in ['==','>=','<=','!=']:
                getattr(self,f'filter_{k}')(v[0:2],float(v[2:]))
            elif v[0] in '=<>':
                getattr(self,f'filter_{k}')(v[0],float(v[1:]))
            else:
                raise ValueError("Filterwaarde moet beginnen met >, >=, <=, <, =, of !=  of een bereik aangeven [vanaf]...[tot]")
    def filter_wvctp(self,operator='=',value=True):
        if value in ['True',True,1,'1']:
            value = True
        if value in ['False',False,0,'0']:
            value = False
        metar_wvctp_not_ok_dict = {
            'Wind': pd.isnull(self.pdf.wind_dir) & pd.isnull(self.pdf.wind_spd),
            'Visibility': self.pdf.cavok | pd.isnull(self.pdf.vis),
            'Clouds': self.pdf.cavok | pd.isnull(self.pdf.sky),
            'Temp': pd.isnull(self.pdf.temp) | pd.isnull(self.pdf.dwpt),
            'Pressure': pd.isnull(self.pdf.pres)
        }
        metar_wvctp_not_ok = functools.reduce(lambda a,b: a|b,list(metar_wvctp_not_ok_dict.values()))
        if (operator in ['=','=='] and value is True) or (operator=='!=' and value is False):
            self.pdf = self.pdf.loc[~metar_wvctp_not_ok]
            self.store_filter('wvctp','=',True)
        elif (operator in ['=','=='] and value is False) or (operator=='!=' and value is True):
            self.pdf = self.pdf.loc[metar_wvctp_not_ok]
            self.store_filter('wvctp','=',False)
        else:
            raise ValueError("Een wvctp (Wind, Visibility, Clouds, Temp, Pressure) filter kan alleen ")
                    
    def filter_series(self,series,operator,value):
        if operator=='...':
            if isinstance(value,(list,tuple)) and len(value)>=2:
                value = [float(value[0]),float(value[1])]
                self.pdf = self.pdf.loc[series.between(value[0],value[1])]
            else:
                raise ValueError("Een filter voor een bereik '...' moet 2 waardes bevatten (in een lijst of tuple)")
        else:
            value = float(value)
            if operator=='==' or operator=='=':
                self.pdf = self.pdf.loc[series==value]
                operator='='
            elif operator=='!=':
                self.pdf = self.pdf.loc[series!=value]
            elif operator=='>':
                self.pdf = self.pdf.loc[series>value]
            elif operator=='>=':
                self.pdf = self.pdf.loc[series>=value]
            elif operator=='<=':
                self.pdf = self.pdf.loc[series<=value]
            elif operator=='<':
                self.pdf = self.pdf.loc[series<value]
            else:
                raise ValueError("Een filter kan gebruik maken van de operators [...,==,=,!=,>,>=,<=,<]")
        return operator,value

    def store_filter(self,name,operator,value):
        self.filters[name].append((operator,value))
        filterstring = f'{value[0]:.3f}...{value[1]:.3f}' if operator=='...' and isinstance(value,(list,tuple)) and len(value)>=2 else f'{operator}{value:.3f}'
        _log.debug(f"Filtering data on {name} ({filterstring})")
    def filter_year(self,operator,value):
        f = self.filter_series(self.pdf.time.dt.year,operator,value)
        self.store_filter('year',*f)
    def filter_month(self,operator,value):
        f = self.filter_series(self.pdf.time.dt.month,operator,value)
        self.store_filter('month',*f)
    def filter_day(self,operator,value):
        f = self.filter_series(self.pdf.time.dt.day,operator,value)
        self.store_filter('day',*f)
    def filter_hour(self,operator,value):
        f = self.filter_series(self.pdf.time.dt.hour,operator,value)
        self.store_filter('hour',*f)
    def filter_minutes_valid(self,operator,value):
        f = self.filter_series(self.pdf.minutes_valid,operator,value)
        self.store_filter('minutes_valid',*f)
    
    
    def get_filter_minmax(self,name):
        return self.calc_filter_minmax(self.filters[name])
    @classmethod
    def calc_filter_minmax(cls,fltrs):
        minval, maxval = None,None
        for oper, val in fltrs:
            if oper in ['>=','>','=','==']:
                minval = max(minval,val) if minval is not None else val
            if oper in ['<=','<','=','==']:
                maxval = min(maxval,val) if maxval is not None else val
            if oper == '...':
                minval = max(minval,val[0]) if minval is not None else val[0]
                maxval = min(maxval,val[1]) if maxval is not None else val[1]
        return minval, maxval
    @classmethod
    def frange(cls,fltrs,default_start,default_end):
        minval, maxval = cls.calc_filter_minmax(fltrs)
        start = (minval if minval is not None else default_start )
        end = 1+(maxval if maxval is not None else default_end )
        return range(int(start),int(end))

    @classmethod
    def convert_unit(cls,converter,data_series):
        if isinstance(converter, numbers.Real):
            return data_series / converter
        elif isinstance(converter, collections.Iterable):
            return data_series.apply(converter[1])
        else:
            raise ValueError('Kon eenheid %s niet converteren naar %s',(value,unit))
    @classmethod
    def realign_polar_xticks(cls,ax):
        for theta, label in zip(ax.get_xticks(), ax.get_xticklabels()):
            theta = theta * ax.get_theta_direction() + ax.get_theta_offset()
            theta = np.pi/2 - theta
            y, x = np.cos(theta), np.sin(theta)
            if x >= 0.1:
                label.set_horizontalalignment('left')
            if x <= -0.1:
                label.set_horizontalalignment('right')
            if y >= 0.5:
                label.set_verticalalignment('bottom')
            if y <= -0.5:
                label.set_verticalalignment('top')
    @classmethod
    def config_polar(cls,ax):
        if ax.name!='polar':
            raise ValueError('De assen voor een wind_compass_* plot, moeten polair zijn, niet %s.'%ax.name)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    @classmethod
    def prepare_maps(cls):
        global cartopy, shapely
        import cartopy
        import shapely
        import shapely.ops
    def load_map_raster(self,name):
        raise ValueError('xarray is currently not available....')
        tif_path = MapPlotHelper.search_or_extract(self.filepaths['natural_earth'],name,['tif','tiff'])
        da = xr.open_rasterio(tif_path)
        da = da.transpose('y','x','band')
        return da
    def load_map_shape(self,name):
        shp_path = MapPlotHelper.search_or_extract(self.filepaths['natural_earth'],name,['shp'])
        return cartopy.io.shapereader.Reader(shp_path)
    def map_stock_img(self,ax,zoom=.99,transform=None):
        trans = transform if transform is not None else cartopy.crs.PlateCarree()
        img_extent = np.array(ax.get_extent(crs=trans))+[-1/zoom,1/zoom,-1/zoom,1/zoom]
        if False:
            try:
                hires_available = bool(MapPlotHelper.search_files(self.filepaths['natural_earth'],'NE1_HR_LC_SR_W_DR',['tif','tiff','zip']))
                if zoom < 1 or not hires_available:
                    imgda = self.load_map_raster('NE1_LR_LC_SR_W_DR')
                    step = int(np.clip(np.ceil(.5/zoom),1,None))
                    img_extent, img_da = MapPlotHelper.slice_img(img_extent,imgda,step)
                    _log.debug('Adding NE1_LR_LC_SR_W_DR to map')
                else:
                    imgda = self.load_map_raster('NE1_HR_LC_SR_W_DR')
                    img_extent, img_da = MapPlotHelper.slice_img(img_extent,imgda,1 if zoom>=1.33 else 2)
                    _log.debug('Adding NE1_HR_LC_SR_W_DR to map')
                ax.imshow(img_da.values,
                    origin='upper',
                    transform=trans,
                    extent=img_extent,
                    zorder=-2)
            except ValueError as e:
                _log.tryexcept(repr(e),exc_info=e)
                _log.warning('De kaartachtergrond-bestanden konden niet worden gevonden. '
                    'Kaart wordt geplot zonder achtergrond.\n'
                    'Zie 00. Instaleren & Introductie, 3.1 Natural Earth, voor een oplossing')
            except Exception as err:
                _log.error(repr(err)+'\nKon achtergrond kaart niet weergeven..."',exc_info=err)

        try:
            hires_available = bool(MapPlotHelper.search_files(self.filepaths['natural_earth'],'ne_10m_admin_0_countries',['shp','zip']))
            if zoom < 1 or not hires_available:
                shp = self.load_map_shape('ne_50m_admin_0_countries')
                _log.debug('Adding ne_50m_admin_0_countries to map')
            else:
                shp = self.load_map_shape('ne_10m_admin_0_countries')
                _log.debug('Adding ne_10m_admin_0_countries to map')
            sf = cartopy.feature.ShapelyFeature(shp.geometries(),trans,
                facecolor='none',edgecolor='#666666',linewidth=.75,zorder=-1)
            ax.add_feature(sf,zorder=-1)
        except ValueError:
            _log.tryexcept(repr(e),exc_info=e)
            _log.warning('De Landgrens-bestanden konden niet worden gevonden. '
                'Kaart wordt geplot zonder grenzen.\n'
                'Zie 00. Instaleren & Introductie, 3.1 Natural Earth, voor een oplossing')
        except Exception as err:
            _log.error(repr(err)+'\nKon landgrenzen niet weergeven op kaart..."',exc_info=err)

    def categorize_wind_dirs(self):
        catborders = [0,11.25,33.75,56.25,78.75,101.25,123.75,146.25,168.75,
                      191.25,213.75,236.25,258.75,281.25,303.75,326.25,348.75,361]
        catnames   = ['','N','NNE','NE','ENE','E','ESE','SE','SSE',
                      'S','SSW','SW','WSW','W','WNW','NW','NNW','N','']
        catcenters = [np.nan,0,22.5,45.0,67.5,90.0,112.5,135.0,157.5,
                      180.0,202.5,225.0,247.5,270.0,292.5,315.0,337.5,0,np.nan]
        self.pdf['wind_dir_catindex'] = np.digitize(self.pdf.wind_dir,catborders)
        self.pdf['wind_dir_compass'] = np.take(catnames,self.pdf.wind_dir_catindex)
        self.pdf['wind_dir_catdeg'] = np.take(catcenters,self.pdf.wind_dir_catindex)
        self.pdf['wind_dir_catrad'] = np.deg2rad(self.pdf.wind_dir_catdeg)
    
    @classmethod
    def consecutive_below_threshold(cls,df,col,threshold,min_number_of_observations):
        def list_dates(s):
            return len(list(s.dt.normalize().unique()))
        assert min_number_of_observations>1
        s = df.loc[:,col]
        m = np.logical_and.reduce([s.shift(-i).le(threshold) for i in range(min_number_of_observations)])
        m = pd.Series(m,index=s.index).replace({False:np.nan}).ffill(limit=min_number_of_observations-1).fillna(False)
        gps = m.ne(m.shift(1)).cumsum().where(m)
        if gps.isnull().all():
            return pd.DataFrame({
                'timeStart': pd.Series([], dtype='datetime64[ns]'),
                'timeEnd': pd.Series([], dtype='datetime64[ns]'),
                'dates': pd.Series([], dtype='int'),
                'messages': pd.Series([], dtype='int'),
                'minutes_valid': pd.Series([], dtype='float'),
                col+'_min': pd.Series([], dtype='float'),
                col+'_max': pd.Series([], dtype='float'),
            })
        return df.groupby(gps).agg(**{
            'timeStart': ('time',min),
            'timeEnd': ('time',max),
            'dates': ('time',list_dates),
            'messages': ('station','count'),
            'minutes_valid': ('minutes_valid','sum'),
            col+'_min': (col,min),
            col+'_max': (col,max)
        }).reset_index(drop=True)
    def get_days_of_periods_below_threshold(self,col,thresholds,min_minutes_valid=30,min_number_of_observations=3):
        dft = self.pdf.copy().sort_values('time')
        belowth_dict = {'all':pd.Series(index=dft.time.dt.normalize().unique(), dtype='float')}
        for i,th in enumerate(thresholds):
            dftg = self.consecutive_below_threshold(dft,col,th,min_number_of_observations)
            dftg = dftg.loc[dftg.minutes_valid>min_minutes_valid].reset_index()
            belowth_dict[th] = dftg.groupby(dftg.timeStart.dt.normalize())[col+'_max'].min()
            for j,row in dftg.loc[dftg.dates>=2].iterrows():
                dates = pd.date_range(row.timeStart.normalize(),row.timeEnd.normalize())[1:]
                for date in dates:
                    if date in belowth_dict[th].index:
                        belowth_dict[th].at[date] = min(belowth_dict[th].at[date],row[col+'_max'])
                    else:
                        belowth_dict[th] = pd.concat([belowth_dict[th],pd.Series(row[col+'_max'],index=[date])])
            belowth_dict[th] = belowth_dict[th].sort_index()
        dfg = pd.concat(belowth_dict,axis=1,sort=True).reset_index().rename(columns={'index':'time'})
        dfg['time'] = pd.to_datetime(dfg.time)
        return dfg

    # Wind properties
    def plot_wind_compass_dir_freq(self,ax,unit='%',cat=True):
        style = self.theme.get("bar.wind")[0]
        quantity = quantities.Fraction
        unit = quantity.find_unit(unit)
        self.config_polar(ax)

        if cat and 'wind_dir_catrad' not in self.pdf:
            self.categorize_wind_dirs()
        if not cat and 'wind_dir_rad' not in self.pdf:
            self.pdf['wind_dir_rad']= np.deg2rad(self.pdf.wind_dir)
        data_gbo = self.pdf.loc[self.pdf.wind_spd>=2]
        if cat:
            gbo = data_gbo.dropna(subset=['wind_dir_catrad']).groupby('wind_dir_catrad')
        else:
            gbo = data_gbo.dropna(subset=['wind_dir_rad']).groupby('wind_dir_rad')
        data = gbo['minutes_valid'].sum()/self.pdf.minutes_valid.sum()
        index, values = data.index.values, self.convert_unit(quantity.units[unit],data.values)
        width = (2*np.pi)/len(index)

        ax.bar(index, values, width=width,**style)
        maxval = values.max()
        addval = (10**(np.log10(maxval)//1))*(.25 if np.log10(maxval)%1<.4 else (.5 if np.log10(maxval)%1 < 0.7 else 1))
        ax.set_rorigin(-.075*(maxval+addval))
        ax.grid(which='both',lw=0.4)
        ax.grid(which='minor',b=True,ls='--')
        ax.set_yticks(np.arange(0,maxval+addval,addval*2))
        ax.set_yticks(np.arange(0,maxval+addval,addval),minor=True)
        self.realign_polar_xticks(ax)
        ax.set_title('Wind Direction frequency [%s]'%unit,pad=12.5)
    def plot_wind_compass_spd(self,ax,unit='kt',cat=True):
        style = self.theme.get_ci("wind")
        quantity = quantities.Speed
        unit = quantity.find_unit(unit)
        self.config_polar(ax)

        if cat and 'wind_dir_catrad' not in self.pdf:
            self.categorize_wind_dirs()
        if not cat and 'wind_dir_rad' not in self.pdf:
            self.pdf['wind_dir_rad']= np.deg2rad(self.pdf.wind_dir)
        data_gbo = self.pdf.loc[self.pdf.wind_spd>=2]
        if cat:
            gbo = data_gbo.dropna(subset=['wind_dir_catrad']).groupby('wind_dir_catrad')
        else:
            gbo = data_gbo.dropna(subset=['wind_dir_rad']).groupby('wind_dir_rad')
        data = gbo['wind_spd'].quantile([.01,.05,.25,.5,.75,.95,.99])
        data = self.convert_unit(quantity.units[unit],data).unstack()
        data = pd.concat([data,data.loc[0:0].rename(index={0:2*np.pi})])
        ax.plot(data[.5],**style[0])
        ax.fill_between(x=data.index,y1=data[.25],y2=data[.75],zorder=-1,**style[1])
        ax.fill_between(x=data.index,y1=data[.05],y2=data[.25],zorder=-1,**style[2])
        ax.fill_between(x=data.index,y1=data[.75],y2=data[.95],zorder=-1,**style[2])
        ax.fill_between(x=data.index,y1=data[.01],y2=data[.05],zorder=-1,**style[3])
        ax.fill_between(x=data.index,y1=data[.95],y2=data[.99],zorder=-1,**style[3])
        maxval = data[.99].max()
        addval = (10**(np.log10(maxval)//1))*(.25 if np.log10(maxval)%1<.3 else (.5 if np.log10(maxval)%1 < 0.6 else 1))
        ax.set_rorigin(-.075*(maxval+addval))
        ax.grid(which='both',lw=0.4,color='k',alpha=0.38)
        ax.grid(which='minor',b=True,ls='--')
        ax.set_yticks(np.arange(0,maxval+addval,addval*2))
        ax.set_yticks(np.arange(0,maxval+addval,addval),minor=True)
        self.realign_polar_xticks(ax)
        ax.set_title('Wind Speed [%s]'%unit,pad=12.5)
    def plotset_wind(self,savefig=None):
        with plt.rc_context({'xtick.major.pad':-1}):
            fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(6.3/3*2,2.1),subplot_kw={'polar':True})
            self.plot_wind_compass_dir_freq(axs[0])
            self.plot_wind_compass_spd(axs[1])
            plt.tight_layout()
            if savefig is not None:
                plt.savefig(savefig)
                plt.close()

    # Averages per month (average daily cycles e.d)
    def _plot_dh_cycle_hoursteps(self,ax,variable,unit,quantity,title='',ylim=None,style=None):
        style = style if style is not None else self.theme.get_ci()
        unit = quantity.find_unit(unit)
        var_human = title.lower()
        
        if pd.isnull(self.pdf[variable]).all():
            if ('dh_'+variable) not in self.warnings:
                _log.warning(f"Geen data voor {var_human} in de metars")
                self.warnings.append('dh_'+variable)
            ax.text(0.5,0.5,f"{self.station} heeft\ngeen {var_human} geregistreed\nin deze maand",
                    horizontalalignment='center',verticalalignment='center',
                    transform=ax.transAxes,c='k')
            ax.set_xlim(0,24)
            if ylim is None:
                ax.set_ylim(0,10)
        else:
            gbo = self.pdf.dropna(subset=[variable]).groupby(self.pdf.time.dt.hour)
            data = gbo[variable].quantile([.01,.05,.25,.5,.75,.95,.99])
            data = self.convert_unit(quantity.units[unit],data).unstack()
            data = pd.concat([data,data.loc[0:0].rename(index={0:24})])
            ax.plot(data[.5],**style[0])
            ax.fill_between(x=data.index,y1=data[.25],y2=data[.75],zorder=-1,**style[1])
            ax.fill_between(x=data.index,y1=data[.05],y2=data[.25],zorder=-1,**style[2])
            ax.fill_between(x=data.index,y1=data[.75],y2=data[.95],zorder=-1,**style[2])
            ax.fill_between(x=data.index,y1=data[.01],y2=data[.05],zorder=-1,**style[3])
            ax.fill_between(x=data.index,y1=data[.95],y2=data[.99],zorder=-1,**style[3])
        ax.set_xticks([0,6,12,18,24]);
        ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],minor=True);
        begin, end = self.get_filter_minmax('hour')
        begin = 0 if begin is None else begin
        end = 24 if end is None else end
        ax.set_xlim(begin,end)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%dh"))
        ax.set_title(title+f" [{unit:s}]")
    def plot_dh_cycle_temp(self,ax,unit='??C'):
        style = self.theme.get_ci("temp")
        self._plot_dh_cycle_hoursteps(ax,'temp',title='Temperature',
            unit=unit,quantity=quantities.Temperature,style=style)
    def plot_dh_cycle_dwpc(self,ax,unit='??C'):
        style = self.theme.get_ci("dwpc")
        self._plot_dh_cycle_hoursteps(ax,'dwpt',title='Dew Point',
            unit=unit,quantity=quantities.Temperature,style=style)
    def plot_dh_cycle_relh(self,ax,unit='%'):
        style = self.theme.get_ci("relh")
        quantity = quantities.Fraction
        unit = quantity.find_unit(unit)
        self._plot_dh_cycle_hoursteps(ax,'relh',title='Relative Humidity',
            ylim=(0,quantities.Fraction(1,'frac')[unit]),
            unit=unit,quantity=quantity,style=style)
    def plot_dh_cycle_wspd(self,ax,unit='kt'):
        style = self.theme.get_ci("wind")
        self._plot_dh_cycle_hoursteps(ax,'wind_spd',title='Wind speed',
            ylim=(0,None),unit=unit,quantity=quantities.Speed,style=style)
    def plot_dh_cycle_vism(self,ax,unit='km'):
        style = self.theme.get_ci("vism")
        quantity = quantities.Distance
        unit = quantity.find_unit(unit)
        self._plot_dh_cycle_hoursteps(ax,'vis',title='Visibility',
            ylim=(0,quantities.Distance(10,'km')[unit]),
            unit=unit,quantity=quantity,style=style)
    def plot_dh_cycle_ceiling(self,ax,unit='ft'):
        style = self.theme.get_ci("ceiling")
        quantity = quantities.Height
        unit = quantity.find_unit(unit)
        self._plot_dh_cycle_hoursteps(ax,'sky_ceiling',title='Cloud base',
            ylim=(0,quantities.Height(5e3,'ft')[unit]),
            unit=unit,quantity=quantity,style=style)
    def plot_dh_cycle_pres(self,ax,unit='hPa'):
        style = self.theme.get_ci("pres")
        quantity = quantities.Pressure
        unit = quantity.find_unit(unit)
        self._plot_dh_cycle_hoursteps(ax,'pres',title='Surface Pressure',
            ylim=(quantity(950,'hPa')[unit],quantity(1050,'hPa')[unit]),
            unit=unit,quantity=quantity,style=style)
    def plot_frequency_gust(self,ax,unit='kt',freq_unit='%'):
        style = self.theme.get("bar.wind")[0]

        quantity = quantities.Speed
        unit = quantity.find_unit(unit)
        binval = quantity(99,'kt')[unit]
        binval = (10**(np.log10(binval)//1))*(.25 if np.log10(binval)%1<.4 else (.5 if np.log10(binval)%1 < 0.7 else 1))
        databinned = self.convert_unit(quantity.units[unit],self.pdf.wind_gust)//binval
        databinned = databinned.groupby(databinned).count()
        if len(databinned)==0:
            if 'gust' not in self.warnings:
                _log.warning("Geen windgusts in de metars")
                self.warnings.append('gust')
            ax.text(0.5,0.5,f"{self.station} heeft\ngeen windvlagen geregistreed\nin deze maand",
                    horizontalalignment='center',verticalalignment='center',
                    transform=ax.transAxes,c='k')
            ax.set_ylim(0,5)
            ax.set_xlim(0,50)
        else:
            data = dict([(i, databinned[i] if i in databinned.index else 0) for i in range(1,int(databinned.index.max())+1)])
            data = pd.Series(np.array(list(data.values())), ["{}".format(k*binval-binval) for k in data.keys()] )
            data = data / len(self.pdf)

            freq_quantity = quantities.Fraction
            freq_unit = freq_quantity.find_unit(freq_unit)
            index, values = data.index.values, self.convert_unit(freq_quantity.units[freq_unit],data.values)
            ax.bar(index,values,align='edge',width=1,**style)
            thx = values.max()*0.025
            for i in range(len(data)):
                ax.text(i+0.5,values[i]+thx,"%3.1f %s"%(values[i],freq_unit),c='k',ha='center',va='bottom')
            ax.set_ylim(0,(np.ceil((values.max()+thx*3)/0.5)*0.5))
            ax.set_xlim(*np.array(ax.get_xlim()).round())
        ax.set_ylabel('Frequency [%s]'%freq_unit)
        ax.set_title('Wind Gusts [%s]'%unit)
        
    def plot_frequency_cloud_type(self,ax,freq_unit='%'):
        freq_quantity = quantities.Fraction
        freq_unit = freq_quantity.find_unit(freq_unit)
        data = self.pdf.groupby('sky_cover')['minutes_valid'].sum().reindex(metar.Metar._cloud_cover_codes.keys()).dropna()
        data = data / self.pdf.minutes_valid.sum()
        index, values = data.index.values, self.convert_unit(freq_quantity.units[freq_unit],data.values)
        
        style = self.theme.get_setT("bar.cloud",indexes=index)
        ax.bar(index, values, **style)
        thx = values.max()*0.05
        for i in range(len(data)):
            ax.text(i,values[i]+thx,"%3.1f %s"%(values[i],freq_unit),c='k',ha='center',va='bottom')
        ax.set_ylabel('Frequency [%s]'%freq_unit)
        ax.set_title('Cloud cover type')
        ax.set_ylim(0,((values.max()+thx*3)//2+1)*2)
    def plot_frequency_color(self,ax,freq_unit='%'):
        freq_quantity = quantities.Fraction
        freq_unit = freq_quantity.find_unit(freq_unit)
        data = self.pdf.groupby('color')['minutes_valid'].sum().reindex(['BLU+','BLU','WHT','GRN','YLO','YLO1','YLO2','AMB','RED']).dropna()
        calculated = False
        if len(data)==0 and 'calc_color' in self.pdf.columns:
            data = self.pdf.groupby('calc_color')['minutes_valid'].sum().reindex(['BLU+','BLU','WHT','GRN','YLO','YLO1','YLO2','AMB','RED']).dropna()
            calculated = True
        if len(data)==0:
            ax.bar(['BLU','WHT','GRN','YLO','AMB','RED'],[0,0,0,0,0,0],color='???')
            ax.text(0.5,0.5,f"{self.station} publiceerd geen\nNATO color states",
                    horizontalalignment='center',verticalalignment='center',
                    transform=ax.transAxes,c='k')
            ax.set_ylim(0,10)
        else:
            data = data / self.pdf.minutes_valid.sum()
            index, values = data.index.values, self.convert_unit(freq_quantity.units[freq_unit],data.values)

            style = self.theme.get_setT("bar.color",indexes=index)
            ax.bar(index, values, **style)

            thx = values.max()*0.05
            for i in range(len(data)):
                ax.text(i,values[i]+thx,"%3.1f %s"%(values[i],freq_unit),c='k',ha='center',va='bottom')
        ax.set_ylabel('Frequency [%s]'%freq_unit)
        ax.set_title('NATO Color State '+('(Calculated)' if calculated else '(from METAR)'))
        ax.set_ylim(0,((values.max()+thx*3)//2+1)*2)
    def plot_frequency_precipitation(self,ax,freq_unit='%',colors=None,edgecolor=None,linewidth=None):
        preciptypes = ['RA','DZ','SN','IC','PL','GR','GS','UP','FZRA','FZDZ','FZFG']
        intensitytypes = ['+','','-','VC']

        freq_quantity = quantities.Fraction
        freq_unit = freq_quantity.find_unit(freq_unit)
        precipdf = self.pdf.iloc[:,:0]
        L = []
        for precip in preciptypes:
            for intensity in intensitytypes:
                precipdf[intensity,precip] = self.pdf.wx.str.contains(re.escape(intensity)+'(?:[A-Z]{2})*?'+re.escape(precip)+'(?:[A-Z]{2})*?')
                L.append(tuple([intensity,precip]))
        precipdf.columns = pd.MultiIndex.from_tuples(L,names=['intensity','precip'])
        precipdf = precipdf.sum().unstack().reindex(preciptypes,axis=1).reindex(intensitytypes,axis=0)
        precipdf = precipdf.loc[precipdf.sum(axis=1)>0,precipdf.sum(axis=0)>0]
        precipdf = precipdf[precipdf.sum().sort_values(ascending=False).index.values]
        barlist = {}
        norm = self.convert_unit(freq_quantity.units[freq_unit],1.)/len(self.pdf)
        if precipdf.shape[0]<1:
                ax.bar(['RA','DZ','SN','IC','PL','GR','GS','UP'],[0,0,0,0,0,0,0,0],color='#000000')
                ax.text(0.5,0.5,f"{self.station} heeft\ngeen neerslag geregistreed\nin deze maand",
                        horizontalalignment='center',verticalalignment='center',
                        transform=ax.transAxes,c='k')
                ax.set_ylim(0,10)
        else:
            styles = self.theme.get_setT("bar.precipitation_codes",indexes=precipdf.columns)
            for i in range(precipdf.shape[0]):
                heights = precipdf.iloc[i,:]*norm
                bottoms = precipdf.iloc[:i,:].sum()*norm

                barlist[i] = ax.bar(precipdf.columns,
                                    heights,
                                    bottom=bottoms,
                                    **styles[i])
                txtheights = heights/2 + bottoms
                for c in range(len(precipdf.columns)):
                    if heights[c]>5:
                        ax.text(c,txtheights[c],intensitytypes[i],
                                horizontalalignment='center',verticalalignment='center',
                                color='k',fontsize=16)

            data = precipdf.sum()
            thx = data.max()*norm*0.05
            for i in range(len(data)):
                ax.text(i,data.iloc[i]*norm+thx,"%3.1f %s"%(data.iloc[i]*norm,freq_unit),c='k',ha='center',va='bottom')
            ax.set_ylim(0,((data.max()*norm+thx*3)//2+1)*2)

            legend_elem = []
            style = self.theme.get("legend_bar.precipitation_codes")
            for i,s in enumerate(precipdf.index.values):
                text = {'-':'Light (-)','':'Normal ( )','+':'Heavy (+)','VC':'In the vicinity (VC)'}[s]
                legend_elem.append(mpl.patches.Patch(**style[i],label=text))
            ax.legend(handles=list(reversed(legend_elem)),
                      framealpha=1, frameon=False)
        ax.set_title('Significant Precipitation')
        ax.set_ylabel('Frequency [%s]'%freq_unit)
    def plot_frequency_sigwx(self,ax,freq_unit='%',colors=None,edgecolor=None,linewidth=None):
        sigwxtypes = ['TS','FZ','SH','FG','VA','BR','HZ','DU','FU','SA','PY','SQ','PO','DS','SS','FC']

        freq_quantity = quantities.Fraction
        freq_unit = freq_quantity.find_unit(freq_unit)
        sigwxdf = self.pdf.iloc[:,:0]
        for sigwx in sigwxtypes:
            sigwxdf[sigwx] = self.pdf.wx.str.contains('(?:\+|-|VC)*?(?:[A-Z]{2})*?'+re.escape(sigwx)+'(?:[A-Z]{2})*?')
        s = sigwxdf.sum()
        s = s.loc[s>0].sort_values(ascending=False)
        norm = self.convert_unit(freq_quantity.units[freq_unit],1.)/len(self.pdf)
        
        if sigwxdf.shape[0]<1:
                ax.bar(['TS','FZ','SH','FG',
                        'VA','BR','HZ','DU',
                        'FU','SA','PY','SQ',
                        'PO','DS','SS','FC'],
                       [0,0,0,0,
                        0,0,0,0,
                        0,0,0,0,
                        0,0,0,0],color='#000000')
                ax.text(0.5,0.5,f"{self.station} heeft\ngeen bijzonder weer geregistreed\nin deze maand",
                        horizontalalignment='center',verticalalignment='center',
                        transform=ax.transAxes,c='k')
                ax.set_ylim(0,10)
        else:
            style = self.theme.get_setT("bar.sigwx",indexes=s.index)

            ax.bar(s.index,s.values*norm,**style)
            thx = s.max()*norm*0.05
            for i in range(len(s)):
                ax.text(i,s.iloc[i]*norm+thx,"%3.1f %s"%(s.iloc[i]*norm,freq_unit),c='k',ha='center',va='bottom')
            ax.set_ylim(0,((s.max()*norm+thx*3)//2+1)*2)
        ax.set_title('Significant Weather');
        ax.set_ylabel('Frequency [%s]'%freq_unit)

    def plotset_daily_cycle_legend(self,savefig=None):
        with mpl.rc_context(rc={'font.size':15*1.5}):
            fig,ax = plt.subplots(1,1,figsize=(6*3,1))
            style = self.theme.get_ci("patch")
            labels = ['%02d%% Confidence interval'%i for i in [50,90,99]]
            handles = [mpl.patches.Patch(label=labels[x],**style[x+1]) for x in range(len(labels))]
            handles = [mpl.lines.Line2D([],[],linestyle='-',label='Median',**style[0])] + handles
            plt.legend(handles=handles,loc=8,ncol=len(handles),framealpha=1,frameon=False)
            plt.gca().set_axis_off()
            plt.tight_layout()
            if savefig is not None:
                plt.savefig(savefig)
                plt.close()
    def plotset_daily(self,savefig=None):
        fig,axs = plt.subplots(nrows=2,ncols=3)
        self.plot_dh_cycle_temp(axs[0][0])
        self.plot_dh_cycle_dwpc(axs[0][1])
        self.plot_dh_cycle_relh(axs[0][2])
        self.plot_dh_cycle_wspd(axs[1][0])
        self.plot_dh_cycle_vism(axs[1][1])
        self.plot_dh_cycle_ceiling(axs[1][2])
        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_gust(self,savefig=None):
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6.3/3,2.1))
        self.plot_frequency_gust(ax)
        plt.tight_layout()
        if savefig is not None:
             plt.savefig(savefig)
             plt.close()
    def plotset_wx(self,savefig=None):
        fig,axs = plt.subplots(nrows=2,ncols=2)
        self.plot_frequency_cloud_type(axs[0][0])
        self.plot_frequency_color(axs[0][1])
        self.plot_frequency_precipitation(axs[1][0])
        self.plot_frequency_sigwx(axs[1][1])
        plt.tight_layout()
        if savefig is not None:
             plt.savefig(savefig)
             plt.close()

    # Averages per year (yearly cycles) Plots
    def plot_ym_cycle_tmin_tmax(self,ax,unit='??C'):
        quantity=quantities.Temperature
        dailydata = self.pdf.groupby(self.pdf.time.dt.date).agg(tmax=('temp','max'),tmin=('temp','min')).reset_index()
        dailydata['time'] = pd.to_datetime(dailydata.time,dayfirst=True)
        gbo = dailydata.groupby(dailydata.time.dt.month)
        for var in ['tmin','tmax']:
            style = self.theme.get_ci(var)
            data = gbo[var].quantile([.01,.05,.25,.5,.75,.95,.99])
            data = self.convert_unit(quantity.units[unit],data).unstack()
            data = pd.concat([data.loc[12:12].rename(index={12:0}),data,data.loc[1:1].rename(index={1:13})])
            data = data.sort_index()
            ax.plot(data[.5],**style[0])
            ax.fill_between(x=data.index,y1=data[.25],y2=data[.75],zorder=-1,**style[1])
            ax.fill_between(x=data.index,y1=data[.05],y2=data[.25],zorder=-1,**style[2])
            ax.fill_between(x=data.index,y1=data[.75],y2=data[.95],zorder=-1,**style[2])
            ax.fill_between(x=data.index,y1=data[.01],y2=data[.05],zorder=-1,**style[3])
            ax.fill_between(x=data.index,y1=data[.95],y2=data[.99],zorder=-1,**style[3])
        xticks = list(range(1,13))
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.get_filter_minmax('month')
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        ax.set_xlim(begin-.5,end+.5)
        ax.set_title('Temperature (min/max) [%s]'%(unit))
    def plot_ym_cycle_wcet(self,ax,unit='??C',ylim=None):
        style = self.theme.get_ci('wcet')
        limit_styles = self.theme.get_set('limits.wcet')
        limit_line_styles = self.theme.get_set('line.limits.wcet')
        quantity = quantities.Temperature
        quantityWind = quantities.Speed

        for f1,ls in limit_styles.items():
            if f1==max(limit_styles.keys()):
                break
            f2 = min(list(filter(lambda x: x>f1,limit_styles.keys())))

            ax.fill_between(x=[-1,14],y1=f1,y2=f2,zorder=-5,**ls[0])
            ax.axhline(f2,zorder=-4,alpha=.8,**limit_line_styles[f1][0])

        t2m = self.convert_unit(quantity.units['??C'],self.pdf.temp)
        wind = (self.convert_unit(quantityWind.units['km/h'],self.pdf.wind_spd))**0.16
        self.pdf['wcet'] = 13.12 + 0.6215 * t2m - 11.37 * wind + 0.3965 * t2m * wind
        gbo = self.pdf.dropna(subset=['wcet']).groupby(self.pdf.time.dt.month)

        data = gbo['wcet'].quantile([.01,.05,.25,.5,.75,.95,.99])
        data = self.convert_unit(quantity.units[unit],data).unstack()
        data = pd.concat([data.loc[12:12].rename(index={12:0}),data,data.loc[1:1].rename(index={1:13})])
        data = data.sort_index()
        ax.plot(data[.5],**style[0])
        ax.fill_between(x=data.index,y1=data[.25],y2=data[.75],zorder=-1,**style[1])
        ax.fill_between(x=data.index,y1=data[.05],y2=data[.25],zorder=-1,**style[2])
        ax.fill_between(x=data.index,y1=data[.75],y2=data[.95],zorder=-1,**style[2])
        ax.fill_between(x=data.index,y1=data[.01],y2=data[.05],zorder=-1,**style[3])
        ax.fill_between(x=data.index,y1=data[.95],y2=data[.99],zorder=-1,**style[3])

        xticks = list(range(1,13))
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.get_filter_minmax('month')
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(np.floor(data[.01].min()/5)*5,
                        np.ceil(data[.99].max()/5)*5)
        ax.set_xlim(begin-.5,end+.5)
        ax.set_title('Wind Chill [%s]'%unit)
    def plot_ym_cycle_wbgt_simplified(self,ax,unit='??C',ylim=None,limit_theme='limits.wbgt'):
        style = self.theme.get_ci('wbgt')
        limit_styles = self.theme.get_set(limit_theme)
        limit_line_styles = self.theme.get_set('line.'+limit_theme)
        quantity = quantities.Temperature

        for f1,ls in limit_styles.items():
            if f1==max(limit_styles.keys()):
                break
            f2 = min(list(filter(lambda x: x>f1,limit_styles.keys())))

            ax.fill_between(x=[-1,14],y1=f1,y2=f2,zorder=-5,**ls[0])
            ax.axhline(f2,zorder=-4,**limit_line_styles[f1][0])

        t2m = self.convert_unit(quantity.units['??C'],self.pdf.temp)
        d2m = self.convert_unit(quantity.units['??C'],self.pdf.dwpt)
        vp_hPa = 6.112 * np.exp((17.67*d2m)/(d2m+243.5))
        #self.pdf['wbgt'] = 0.657 * t2m + 0.393 * vp_hPa + 3.94
        self.pdf['wbgt'] = (1.1 + 0.66 * t2m + 0.29 * vp_hPa).where(t2m>d2m)

        dailydata = self.pdf.groupby(self.pdf.time.dt.date).agg(wbgt=('wbgt','max')).reset_index()
        dailydata['time'] = pd.to_datetime(dailydata.time,dayfirst=True)
        gbo = dailydata.groupby(dailydata.time.dt.month)
        data = gbo['wbgt'].quantile([.005,.05,.25,.5,.75,.95,.995])
        data = self.convert_unit(quantity.units[unit],data).unstack()
        data = pd.concat([data.loc[12:12].rename(index={12:0}),data,data.loc[1:1].rename(index={1:13})])
        data = data.sort_index()
        ax.plot(data[.5],**style[0])
        ax.fill_between(x=data.index,y1=data[.25],y2=data[.75],zorder=-1,**style[1])
        ax.fill_between(x=data.index,y1=data[.05],y2=data[.25],zorder=-1,**style[2])
        ax.fill_between(x=data.index,y1=data[.75],y2=data[.95],zorder=-1,**style[2])
        ax.fill_between(x=data.index,y1=data[.005],y2=data[.05],zorder=-1,**style[3])
        ax.fill_between(x=data.index,y1=data[.95],y2=data[.995],zorder=-1,**style[3])

        xticks = list(range(1,13))
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.get_filter_minmax('month')
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(np.floor(data[.005].min()/5)*5,
                        np.ceil(data[.995].max()/5)*5)
        ax.set_xlim(begin-.5,end+.5)
        ax.set_title('WBGT [%s]'%unit)
    def plot_ym_cycle_relh(self,ax,unit='%'):
        style = self.theme.get_ci("relh")
        quantity = quantities.Fraction

        gbo = self.pdf.dropna(subset=['relh']).groupby(self.pdf.time.dt.month)

        data = gbo['relh'].quantile([.01,.05,.25,.5,.75,.95,.99])
        data = self.convert_unit(quantity.units[unit],data).unstack()
        data = pd.concat([data.loc[12:12].rename(index={12:0}),data,data.loc[1:1].rename(index={1:13})])
        data = data.sort_index()
        ax.plot(data[.5],**style[0])
        ax.fill_between(x=data.index,y1=data[.25],y2=data[.75],zorder=-1,**style[1])
        ax.fill_between(x=data.index,y1=data[.05],y2=data[.25],zorder=-1,**style[2])
        ax.fill_between(x=data.index,y1=data[.75],y2=data[.95],zorder=-1,**style[2])
        ax.fill_between(x=data.index,y1=data[.01],y2=data[.05],zorder=-1,**style[3])
        ax.fill_between(x=data.index,y1=data[.95],y2=data[.99],zorder=-1,**style[3])

        xticks = list(range(1,13))
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.get_filter_minmax('month')
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        ax.set_xlim(begin-.5,end+.5)
        ax.set_ylim(0,100)
        ax.set_title('Relative Humidity [%s]'%unit)
    def plot_ym_cycle_vism(self,ax,unit='km'):
        style = self.theme.get_ci("vism")
        quantity = quantities.Distance

        gbo = self.pdf.dropna(subset=['vis']).groupby(self.pdf.time.dt.month)

        data = gbo['vis'].quantile([.01,.05,.25,.5,.75,.95,.99])
        data = self.convert_unit(quantity.units[unit],data).unstack()
        data = pd.concat([data.loc[12:12].rename(index={12:0}),data,data.loc[1:1].rename(index={1:13})])
        data = data.sort_index()
        ax.plot(data[.5],**style[0])
        ax.fill_between(x=data.index,y1=data[.25],y2=data[.75],zorder=-1,**style[1])
        ax.fill_between(x=data.index,y1=data[.05],y2=data[.25],zorder=-1,**style[2])
        ax.fill_between(x=data.index,y1=data[.75],y2=data[.95],zorder=-1,**style[2])
        ax.fill_between(x=data.index,y1=data[.01],y2=data[.05],zorder=-1,**style[3])
        ax.fill_between(x=data.index,y1=data[.95],y2=data[.99],zorder=-1,**style[3])

        xticks = list(range(1,13))
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.get_filter_minmax('month')
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        ax.set_xlim(begin-.5,end+.5)
        ax.set_ylim(0,quantities.Distance(10,'km')[unit])
        ax.set_title('Visibility [%s]'%unit)
    def plot_ym_cycle_ceiling(self,ax,unit='ft'):
        style = self.theme.get_ci('ceiling')
        quantity = quantities.Height

        gbo = self.pdf.dropna(subset=['sky_ceiling']).groupby(self.pdf.time.dt.month)

        data = gbo['sky_ceiling'].quantile([.01,.05,.25,.5,.75,.95,.99])
        data = self.convert_unit(quantity.units[unit],data).unstack()
        data = pd.concat([data.loc[12:12].rename(index={12:0}),data,data.loc[1:1].rename(index={1:13})])
        data = data.sort_index()
        ax.plot(data[.5],**style[0])
        ax.fill_between(x=data.index,y1=data[.25],y2=data[.75],zorder=-1,**style[1])
        ax.fill_between(x=data.index,y1=data[.05],y2=data[.25],zorder=-1,**style[2])
        ax.fill_between(x=data.index,y1=data[.75],y2=data[.95],zorder=-1,**style[2])
        ax.fill_between(x=data.index,y1=data[.01],y2=data[.05],zorder=-1,**style[3])
        ax.fill_between(x=data.index,y1=data[.95],y2=data[.99],zorder=-1,**style[3])

        xticks = list(range(1,13))
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.get_filter_minmax('month')
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        ax.set_xlim(begin-.5,end+.5)
        ax.set_ylim(0,quantities.Distance(5e3,'ft')[unit])
        ax.set_title('Cloud base [%s]'%unit)
    def plot_ym_ceilingdays(self,ax):
        style = self.theme.get_set("bar.ceiling_aggr")
        thresholds = sorted(list(style.keys()))
        
        dfg = self.get_days_of_periods_below_threshold('sky_ceiling',thresholds)
        data = (~pd.isnull(dfg)).groupby([dfg.time.dt.year,dfg.time.dt.month]).sum().unstack().mean().unstack().T.loc[:,thresholds]
        data2 = data.iloc[:,[0]].rename(columns={data.columns[0]:str(data.columns[0])})
        for c in range(1,data.shape[1]):
            data2 = data2.assign(**{str(data.columns[c]): data.iloc[:,c] - data.iloc[:,c-1]})
            
        bottom = data2.cumsum(axis=1).shift(1,axis=1).fillna(0)
        colums = data2.index.values
        handles = []
        for c in data2.columns:
            handles.append(
                ax.bar(
                    data2.index,
                    data2[c].values,
                    bottom=bottom[c].values,
                    **style[int(c)][0]))
        xticks = list(range(1,13))
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.get_filter_minmax('month')
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        ax.set_xlim(begin-.5,end+.5)
        ax.set_ylim(0,31)
        ax.set_title('Days with low clouds')
    def plot_ym_visdays(self,ax):
        style = self.theme.get_set("bar.vis_aggr")
        thresholds = sorted(list(style.keys()))
        
        dfg = self.get_days_of_periods_below_threshold('vis',thresholds)
        data = (~pd.isnull(dfg)).groupby([dfg.time.dt.year,dfg.time.dt.month]).sum().unstack().mean().unstack().T.loc[:,thresholds]
        data2 = data.iloc[:,[0]].rename(columns={data.columns[0]:str(data.columns[0])})
        for c in range(1,data.shape[1]):
            data2 = data2.assign(**{str(data.columns[c]): data.iloc[:,c] - data.iloc[:,c-1]})
        bottom = data2.cumsum(axis=1).shift(1,axis=1).fillna(0)
        colums = data2.index.values
        handles = []
        for c in data2.columns:
            handles.append(
                ax.bar(
                    data2.index,
                    data2[c].values,
                    bottom=bottom[c].values,
                    **style[int(c)][0]))
        xticks = list(range(1,13))
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.get_filter_minmax('month')
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        ax.set_xlim(begin-.5,end+.5)
        ax.set_ylim(0,31)
        ax.set_title('Days with low visibility')
    def plot_ym_precipdays(self,ax):
        style = self.theme.get_set("bar.precipitation_aggr")
        preciptypes = {
               'Fz':['FZRA','FZDZ'],
               'Ra':['RA','DZ'],
               'Sn':['SN','IC'],
               'Gr':['PL','GR','GS'],
               'Up':['UP']}
        intensitytypes = ['+','','-','VC']
        for precipcat, precipcodes in preciptypes.items():
            self.pdf['ptype_'+precipcat] = False
            for precip in precipcodes:
                for intensity in intensitytypes:
                    precipitating = self.pdf.wx.str.contains(
                        re.escape(intensity)+'(?:[A-Z]{2})*?'+re.escape(precip)+'(?:[A-Z]{2})*?')
                    self.pdf['ptype_'+precipcat] = (
                        (self.pdf['ptype_'+precipcat].values) | (precipitating.fillna(False)))
        data = self.pdf.groupby(self.pdf.time.dt.date).agg(
            {'ptype_'+cat:any for cat in preciptypes}).reset_index()
        data['time'] = pd.to_datetime(data.time,dayfirst=True)
        data2 = data.loc[:,['time']]
        
        data2['Freezing'] = (data.ptype_Fz).astype(int)
        data2['RainSnowHail'] = (~data.ptype_Fz & ~data.ptype_Ra & data.ptype_Sn & data.ptype_Gr & ~data.ptype_Up).astype(int)
        data2['Snow'] = (~data.ptype_Fz & ~data.ptype_Ra & data.ptype_Sn & ~data.ptype_Gr & ~data.ptype_Up).astype(int)
        data2['RainSnow'] = (~data.ptype_Fz & data.ptype_Ra & data.ptype_Sn & ~data.ptype_Gr & ~data.ptype_Up).astype(int)
        data2['Rain'] = (~data.ptype_Fz & data.ptype_Ra & ~data.ptype_Sn & ~data.ptype_Gr & ~data.ptype_Up).astype(int)
        data2['RainHail'] = (~data.ptype_Fz & data.ptype_Ra & ~data.ptype_Sn & data.ptype_Gr & ~data.ptype_Up).astype(int)
        data2['Hail'] = (~data.ptype_Fz & ~data.ptype_Ra & ~data.ptype_Sn & data.ptype_Gr & ~data.ptype_Up).astype(int)
        data2['SnowHail'] = (~data.ptype_Fz & ~data.ptype_Ra & data.ptype_Sn & data.ptype_Gr & ~data.ptype_Up).astype(int)
        data2['Unknown'] = (~data.ptype_Fz & data.ptype_Up).astype(int)
        data3 = data2.groupby([data2.time.dt.year,data2.time.dt.month]).sum().unstack().mean().unstack().T

        bottom = data3.cumsum(axis=1).shift(1,axis=1).fillna(0)
        colums = data3.index.values
        handles = []
        for c in data3.columns:
            handles.append(
                ax.bar(
                    data3.index,
                    data3[c].values,
                    bottom=bottom[c].values,
                    **style[c][0]))
        xticks = list(range(1,13))
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.get_filter_minmax('month')
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        ax.set_xlim(begin-.5,end+.5)
        ax.set_ylim(0,31)
        ax.set_title('Days with Precipitation')
    def plot_ym_cycle_cloud_type(self,ax,freq_unit='%',legend=False):
        style = self.theme.get_set("bar.cloud")
        freq_quantity = quantities.Fraction
        freq_unit = freq_quantity.find_unit(freq_unit)

        data = self.pdf.groupby([self.pdf.time.dt.month,self.pdf.sky_cover])['minutes_valid'].sum().unstack()
        data = np.divide(data,data.sum(axis=1).values[:,None]).reindex(metar.Metar._cloud_cover_codes.keys(),axis=1)
        clear_index = data[['SKC','NCD','CLR','NSC']].sum().idxmax()
        obs_index = data[['OBS','VV']].sum().idxmax()
        data[clear_index] = data[['SKC','NCD','CLR','NSC']].sum(axis=1)
        data[obs_index] = data[['OBS','VV']].sum(axis=1)
        data = data[reversed([clear_index,'FEW','SCT','BKN','OVC',obs_index])]
        data = self.convert_unit(freq_quantity.units[freq_unit],data)
        begin = data.cumsum(axis=1).shift(1,axis=1).fillna(0)
        colums = data.index.values

        handles = []
        for c in data.columns:
            handles.append(
                ax.bar(
                    data.index,
                    data[c].values,
                    bottom=begin[c].values,
                    **style[c][0]))

        xticks = list(range(1,13))
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.get_filter_minmax('month')
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        ax.set_xlim(begin-.5,end+.5)
        ax.set_ylim(0,quantities.Fraction(100,'%')[freq_unit])
        ax.set_title('Cloud cover [%s]'%freq_unit)
        if legend:
            ax.legend(handles,data.columns,loc=9,ncol=3,bbox_to_anchor=(.5,-.15),
                labelspacing=.15,handlelength=1.5,handletextpad=0.4,fontsize='small',framealpha=0)
    def plot_ym_cycle_color(self,ax,freq_unit='%',ylim=None,legend=False,return_colorcodes=False):
        style = self.theme.get_set("bar.color")
        freq_quantity = quantities.Fraction
        freq_unit = freq_quantity.find_unit(freq_unit)

        colorcodes = ['BLU+','BLU','WHT','GRN','YLO','YLO1','YLO2','AMB','RED']
        data = self.pdf.groupby([self.pdf.time.dt.month,self.pdf.calc_color])['minutes_valid'].sum().unstack()
        data = np.divide(data,data.sum(axis=1).values[:,None])
        data = data.reindex(colorcodes,axis=1)

        if data['BLU+'].isnull().all():
            colorcodes.remove('BLU+')
        if data['YLO'].isnull().all() and not data[['YLO1','YLO2']].isnull().all().all():
            colorcodes.remove('YLO')
        if not data['YLO'].isnull().all() and data[['YLO1','YLO2']].isnull().all().all():
            colorcodes.remove('YLO1')
            colorcodes.remove('YLO2')
        data = data[reversed(colorcodes)].fillna(0)
        data = self.convert_unit(freq_quantity.units[freq_unit],data)
        bottom = data.cumsum(axis=1).shift(1,axis=1).fillna(0)
        colums = data.index.values

        handles = []
        for c in data.columns:
            handles.append(
                ax.bar(
                    data.index,
                    data[c].values,
                    bottom=bottom[c].values,
                    **style[c][0]))

        xticks = list(range(1,13))
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.get_filter_minmax('month')
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        ax.set_xlim(begin-.5,end+.5)
        
        if ylim is None:
            n = bottom.iloc[:,-1].max()
            order_of_magnitude = 10**np.floor(np.log10(n))
            n0 = np.ceil(n/order_of_magnitude)*order_of_magnitude
            if n0>=100:
                ax.set_ylim(0,n0)
            else:
                n1 = (1+np.ceil(n/order_of_magnitude))*order_of_magnitude
                n2 = (2+np.ceil(n/order_of_magnitude))*order_of_magnitude
                ax.set_ylim(0,n1)
                ax.set_yticks(np.arange(0,n2,order_of_magnitude))
                if n1<100:
                    labels = list(np.arange(0,n1,order_of_magnitude))+[100]
                    if order_of_magnitude>=1:
                        labels = list(map(int,labels))
                    ax.set_yticklabels(labels)
                    tearline_y = (len(labels)-1.5)/(len(labels)-1)
                    tearline_coords = [
                        (-5,-3),
                        ( 5,-3),
                        ( 5, 3),
                        (-5, 3),
                        (-5,-3),
                    ]
                    tearline_lines = mpl.path.Path(tearline_coords,[1,2,1,2,0])
                    tearline_inner = mpl.path.Path(tearline_coords,[1,2,2,2,79])
                    ax.scatter([0],[tearline_y],marker=tearline_inner,s=15,
                               c='white',linewidths=0,edgecolors='none',
                               transform=ax.transAxes,clip_on=False,zorder=99)
                    ax.scatter([0],[tearline_y],marker=tearline_lines,s=15,
                               c='white',linewidths=mpl.rcParams['axes.linewidth'],edgecolors='k',
                               transform=ax.transAxes,clip_on=False,zorder=100)
        else:
            ax.set_ylim(*ylim)
        ax.set_title('NATO Color State [%s]'%freq_unit)

        if legend:
            ax.legend(handles,data.columns,loc=9,ncol=4,bbox_to_anchor=(.5,-.15),
                  labelspacing=.15,handlelength=1.5,handletextpad=0.4,fontsize='small',framealpha=0)
        if return_colorcodes:
            return colorcodes

    def plot_ym_solar(self,ax,offsetutc=True,legend=False):
        style = self.theme.get_setT('solar')
        noonstyle = self.theme.get('scatter.solar_noon')[0]
        astro = self.get_astro()
        smx, sdf, sds = astro.solar_matrix()
        days_in_year = smx.shape[0]+1

        #cmap = plt.get_cmap('cividis')
        #twilight_keys = ['night','astronomical','nautical','civil','day']
        #colors = {k:[cmap(i/(len(twilight_keys)-1))] for i,k in enumerate(twilight_keys)}
        #noon_colors, noon_edgecolor, noon_linewidth = 'k', 'k', .5

        colors = np.array([mpl.colors.to_rgba(c) for c in style['facecolor']])
        smx_colored = colors[smx.T,:]
        ims = ax.imshow(smx_colored,
            origin='lower',aspect='auto',
            extent=(1,days_in_year,0,24))

        noon_time = sdf.noon.dt.hour + sdf.noon.dt.minute/60 + sdf.noon.dt.second/3600
        ax.scatter(noon_time.index,noon_time.values,s=1.5,marker=',',**noonstyle)

        first_day_of_month_doys = np.array([
            pd.Timestamp("{y:04d}-{m:02d}-01".format(m=m,y=astro.year)).dayofyear for m in range(1,13)]
            + [days_in_year]) #+next year 01-Jan
        middle_day_of_month_doys = (first_day_of_month_doys[1:]+first_day_of_month_doys[:-1])/2
        monthabbr = dict([(m,datetime.datetime.strptime("%02d"%m,"%m").strftime("%b")) for m in range(1,13)])
        month_markers = ax.scatter(first_day_of_month_doys,np.full(first_day_of_month_doys.shape,-0.45),
                marker='|',color='k')
        month_markers.set_clip_on(False)

        ax.set_xlim(1,days_in_year)
        ax.set_xticks(middle_day_of_month_doys)
        ax.set_xticklabels(monthabbr.values())
        ax.set_xlabel(astro.year)
        ax.set_xticks(np.arange(1,366),minor=True)

        ax.set_ylim(0,24)
        ax.set_yticks(np.arange(0,25,3))
        ax.set_yticks(np.arange(0,25),minor=True)

        if astro.tz==astro.station_data.get('timezone'):
            if offsetutc:
                tzoffset = sdf.noon.dt.tz_localize(astro.tz).dt.strftime('%z')
                tzoffset = tzoffset.str[:3] + ':' + tzoffset.str[3:]
                label = "UTC "+"/".join(list(tzoffset.unique()))
            else:
                label = astro.tz
            ax.set_ylabel(f'Lokale Tijd ({label})')
        else:
            ax.set_ylabel(f'Tijd ({astro.tz})')
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%dh"))

        if legend:
            legend_elements = {
                'Middaguur': mpl.lines.Line2D([],[],color=noon_colors[0],lw=noon_size),
                'Dag': mpl.patches.Patch(facecolor=colors['day'][0],edgecolor='k'),
                'Civiele schemering': mpl.patches.Patch(facecolor=colors['civil'][0],edgecolor='k'),
                'Nautische schemering': mpl.patches.Patch(facecolor=colors['nautical'][0],edgecolor='k'),
                'Astronomische schemering': mpl.patches.Patch(facecolor=colors['astronomical'][0],edgecolor='k'),
                'Nacht': mpl.patches.Patch(facecolor=colors['night'][0],edgecolor='k'),
            }
            ax.legend(handles=legend_elements.values(),
                      labels=legend_elements.keys(),
                      ncol=3,#len(legend_elements),
                      bbox_to_anchor=(.5,-.1),
                      loc='upper center',
            )

    def plotset_ymwide_tmin_tmax(self,savefig=None):
        fig = plt.figure(figsize=(6.3,2.1))
        width = .80 #.86
        ax = fig.add_axes([.06,.1,width,.8])
        self.plot_ym_cycle_tmin_tmax(ax)

        #ax2 = fig.add_axes([width+.02,.01,1-width-.03,.98])
        ax2 = fig.add_axes([width+.05,.1,1-width-.03,.8])
        style_tmax = self.theme.get_ci('patch.tmax')
        style_tmin = self.theme.get_ci('patch.tmin')
        legends = [
            mplLegendSubheading('Maximum'),
            mpl.lines.Line2D([],[],label='Median',**style_tmax[0]),
            mplLegendSubheading('Confidence\nintervals:',4),
            mpl.patches.Patch(label='50%',**style_tmax[1]),
            mpl.patches.Patch(label='90%',**style_tmax[2]),
            mpl.patches.Patch(label='99%',**style_tmax[3]),
            mplLegendSubheading('Minimum'),
            mpl.lines.Line2D([],[],label='Median',**style_tmin[0]),
            mplLegendSubheading('Confidence\nintervals:',4),
            mpl.patches.Patch(label='50%',**style_tmin[1]),
            mpl.patches.Patch(label='90%',**style_tmin[2]),
            mpl.patches.Patch(label='99%',**style_tmin[3]),
        ]
        ax2.legend(handles=legends,
            bbox_to_anchor=(.5,.5),
            loc='center',
            handler_map={
                mplLegendSubheading:mplLegendSubheadingHandler(),
            })
        ax2.set_axis_off()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_ymwide_wcet(self,savefig=None):
        fig = plt.figure(figsize=(6.3,2.1))
        width = .79
        ax = fig.add_axes([.06,.1,width,.8])
        self.plot_ym_cycle_wcet(ax)

        ax2 = fig.add_axes([width+.05,.1,1-width-.03,.8])
        style = self.theme.get_ci('patch.wcet')
        limit_styles = self.theme.get_set('patch.limits.wcet')

        limit_legend = []
        for f1,ls in limit_styles.items():
            if f1==max(limit_styles.keys()):
                break
            f2 = min(list(filter(lambda x: x>f1,limit_styles.keys())))
            if abs(f1)<299 and abs(f2)<299:
                f1,f2 = (f1,f2) if abs(f1)<abs(f2) else (f2,f1)
            label = (f'< {f2:.0f}' if f1<-299 else (
                     f'> {f1:.0f}' if f2>299 else (
                     f'{f1:.0f} to {f2:.0f}')))
            limit_legend.append(mpl.patches.Patch(label=label,**ls[0]))

        legends = [
            mpl.lines.Line2D([],[],label='Median',**style[0]),
            mplLegendSubheading('Confidence\nintervals:',4),
            mpl.patches.Patch(label='50%',**style[1]),
            mpl.patches.Patch(label='90%',**style[2]),
            mpl.patches.Patch(label='99%',**style[3]),
            mplLegendSpacer(),
            mplLegendSubheading('Limits'),
        ]+list(reversed(limit_legend))
        ax2.legend(handles=legends,
            bbox_to_anchor=(.5,.5),
            loc='center',
            handler_map={
                mplLegendSubheading:mplLegendSubheadingHandler(),
                mplLegendSpacer:mplLegendSpacerHandler(),
            })
        ax2.set_axis_off()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_ymwide_wbgt(self,savefig=None,limit_theme='limits.wbgt'):
        fig = plt.figure(figsize=(6.3,2.1))
        width = .80
        ax = fig.add_axes([.05,.1,width,.8])
        self.plot_ym_cycle_wbgt_simplified(ax,limit_theme=limit_theme)

        ax2 = fig.add_axes([width+.05,.1,1-width-.03,.8])
        style = self.theme.get_ci('patch.wbgt')
        limit_styles = self.theme.get_set('patch.'+limit_theme)

        limit_legend = []
        ymin, ymax = ax.get_ylim()
        for f1,ls in limit_styles.items():
            if f1==max(limit_styles.keys()):
                break
            f2 = min(list(filter(lambda x: x>f1,limit_styles.keys())))
            if abs(f1)<299 and abs(f2)<299:
                f1,f2 = (f1,f2) if abs(f1)<abs(f2) else (f2,f1)
            label = (f'< {f2:.0f}' if f1<-299 else (
                     f'> {f1:.0f}' if f2>299 else (
                     f'{f1:.0f} to {f2:.0f}')))
            if ymin <= f1 < ymax or ymin < f2 <= ymax:
                limit_legend.append(mpl.patches.Patch(label=label,**ls[0]))

        legends = [
            mpl.lines.Line2D([],[],label='Median',**style[0]),
            mplLegendSubheading('Confidence\nintervals:',4),
            mpl.patches.Patch(**style[1], label='50%'),
            mpl.patches.Patch(**style[2], label='90%'),
            mpl.patches.Patch(**style[3], label='99%')]
        if len(limit_legend)<=8:
            legends += [mplLegendSubheading('Limits')]
            legends += list(reversed(limit_legend))
        ax2.legend(handles=legends,
            bbox_to_anchor=(.5,.5),
            loc='center',
            handler_map={
                mplLegendSubheading:mplLegendSubheadingHandler(),
            })
        ax2.set_axis_off()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_ymwide_relh(self,savefig=None):
        fig = plt.figure(figsize=(6.3,2.1))
        width = .81
        ax = fig.add_axes([.05,.1,width,.8])
        self.plot_ym_cycle_relh(ax)

        ax2 = fig.add_axes([width+.05,.1,1-width-.03,.8])
        style = self.theme.get_ci('patch.relh')
        legends = [
            mpl.lines.Line2D([],[],label='Median',**style[0]),
            mplLegendSubheading('Confidence\nintervals:',4),
            mpl.patches.Patch(label='50%',**style[1]),
            mpl.patches.Patch(label='90%',**style[2]),
            mpl.patches.Patch(label='99%',**style[3]),
        ]
        ax2.legend(handles=legends,
            bbox_to_anchor=(.5,.5),
            loc='center',
            handler_map={
                mplLegendSubheading:mplLegendSubheadingHandler(),
            })
        ax2.set_axis_off()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_ymwide_vism(self,savefig=None):
        fig = plt.figure(figsize=(6.3,2.1))
        width = .81
        ax = fig.add_axes([.05,.1,width,.8])
        self.plot_ym_cycle_vism(ax)

        ax2 = fig.add_axes([width+.05,.1,1-width-.03,.8])
        style = self.theme.get_ci('patch.vism')
        legends = [
            mpl.lines.Line2D([],[],label='Median',**style[0]),
            mplLegendSubheading('Confidence\nintervals:',4),
            mpl.patches.Patch(label='50%',**style[1]),
            mpl.patches.Patch(label='90%',**style[2]),
            mpl.patches.Patch(label='99%',**style[3]),
        ]
        ax2.legend(handles=legends,
            bbox_to_anchor=(.5,.5),
            loc='center',
            handler_map={
                mplLegendSubheading:mplLegendSubheadingHandler(),
            })
        ax2.set_axis_off()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_ymwide_ceiling(self,savefig=None):
        fig = plt.figure(figsize=(6.3,2.1))
        width = .78
        ax = fig.add_axes([.07,.1,width,.8])
        self.plot_ym_cycle_ceiling(ax)

        ax2 = fig.add_axes([width+.06,.1,1-width-.03,.8])
        style = self.theme.get_ci('patch.ceiling')
        legends = [
            mpl.lines.Line2D([],[],label='Median',**style[0]),
            mplLegendSubheading('Confidence\nintervals:',4),
            mpl.patches.Patch(label='50%',**style[1]),
            mpl.patches.Patch(label='90%',**style[2]),
            mpl.patches.Patch(label='99%',**style[3]),
        ]
        ax2.legend(handles=legends,
            bbox_to_anchor=(.5,.5),
            loc='center',
            handler_map={
                mplLegendSubheading:mplLegendSubheadingHandler(),
            })
        ax2.set_axis_off()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_ymwide_ceilingdays(self,savefig=None):
        fig = plt.figure(figsize=(6.3,2.1))
        width = .80
        ax = fig.add_axes([.05,.1,width,.8])
        self.plot_ym_ceilingdays(ax)

        ax2 = fig.add_axes([width+.05,.1,1-width-.03,.8])
        style = self.theme.get_set("patch.ceiling_aggr")
        legends = [
            mpl.patches.Patch(label='??? '+quantities.Height(k).formatted_value(), **style[k][0])
            for k in reversed(sorted(list(style.keys())))
        ]
        ax2.legend(handles=legends,
            bbox_to_anchor=(.5,.5),
            loc='center',
            handler_map={
                mplLegendSubheading:mplLegendSubheadingHandler(),
            })
        ax2.set_axis_off()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_ymwide_visdays(self,savefig=None):
        fig = plt.figure(figsize=(6.3,2.1))
        width = .82
        ax = fig.add_axes([.05,.1,width,.8])
        self.plot_ym_visdays(ax)

        ax2 = fig.add_axes([width+.05,.1,1-width-.03,.8])
        style = self.theme.get_set("patch.vis_aggr")
        legends = [
            mpl.patches.Patch(label='??? '+quantities.Distance(k).formatted_value('km'), **style[k][0])
            for k in reversed(sorted(list(style.keys())))
        ]
        ax2.legend(handles=legends,
            bbox_to_anchor=(.5,.5),
            loc='center',
            handler_map={
                mplLegendSubheading:mplLegendSubheadingHandler(),
            })
        ax2.set_axis_off()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_ymwide_precipdays(self,savefig=None):
        fig = plt.figure(figsize=(6.3,2.1))
        width = .74
        ax = fig.add_axes([.05,.1,width,.8])
        self.plot_ym_precipdays(ax)

        ax2 = fig.add_axes([width+.05,.1,1-width-.03,.8])
        style = self.theme.get_set("patch.precipitation_aggr")
        legends = [
            mpl.patches.Patch(label='Freezing Rain\n(FZRA/FZDZ)', **style['Freezing'][0]),
            mpl.patches.Patch(label='Rain (RA/DZ)', **style['Rain'][0]),
            mpl.patches.Patch(label='Snow (SN/IC)', **style['Snow'][0]),
            mpl.patches.Patch(label='Hail (PL/GR/GS)', **style['Hail'][0]),
            mpl.patches.Patch(label='Rain & Snow', **style['RainSnow'][0]),
            mpl.patches.Patch(label='Rain & Hail', **style['RainHail'][0]),
            mpl.patches.Patch(label='Snow & Hail', **style['SnowHail'][0]),
            mpl.patches.Patch(label='Rain, Snow & Hail', **style['RainSnowHail'][0]),
            mpl.patches.Patch(label='Unknown', **style['Unknown'][0]),
        ]
        ax2.legend(handles=legends,
            bbox_to_anchor=(.5,.5),
            loc='center',
            handler_map={
                mplLegendSubheading:mplLegendSubheadingHandler(),
            })
        ax2.set_axis_off()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_ymwide_cloud_type(self,savefig=None):
        fig = plt.figure(figsize=(6.3,2.1))
        width = .81
        ax = fig.add_axes([.05,.1,width,.8])
        self.plot_ym_cycle_cloud_type(ax,legend=False)

        ax2 = fig.add_axes([width+.05,.1,1-width-.03,.8])
        style = self.theme.get_set('patch.cloud')
        legends = [
            mpl.patches.Patch(label=cloud,**style[cloud][0])
            for cloud in ['VV','OVC','BKN','SCT','FEW','NSC']]
        ax2.legend(handles=legends,
            bbox_to_anchor=(.5,.5),
            loc='center',
            handler_map={
                mplLegendSubheading:mplLegendSubheadingHandler(),
            })
        ax2.set_axis_off()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_ymwide_color(self,savefig=None):
        fig = plt.figure(figsize=(6.3,2.1))
        width = .81
        ax = fig.add_axes([.05,.1,width,.8])
        colorcodes = self.plot_ym_cycle_color(ax,legend=False,return_colorcodes=True)

        ax2 = fig.add_axes([width+.05,.1,1-width-.03,.8])
        styles = self.theme.get_set('patch.color')
        legends = [
            mpl.patches.Patch(label=color,**styles[color][0])
            for color in colorcodes]
        ax2.legend(handles=legends,
            bbox_to_anchor=(.5,.5),
            loc='center',
            handler_map={
                mplLegendSubheading:mplLegendSubheadingHandler(),
            })
        ax2.set_axis_off()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_ymwide_solar(self,savefig=None):
        fig = plt.figure(figsize=(6.3,2.1))
        width = .70
        ax = fig.add_axes([.1,.10,width,.85])
        self.plot_ym_solar(ax,legend=False)

        ax2 = fig.add_axes([width+.1,.1,1-width-.08,.8])
        style = self.theme.get_set('patch.solar')
        nooncolor = self.theme.get('solar_noon')[0]['facecolor']

        legends = [
            mpl.lines.Line2D([],[],label='Middaguur',color=nooncolor,lw=1.5),
            mpl.patches.Patch(label='Dag',**style['day'][0]),
            mpl.patches.Patch(label='Nacht',**style['night'][0]),
            mplLegendSubheading('Schemering'),
            mpl.patches.Patch(label='Civiel',**style['civil'][0]),
            mpl.patches.Patch(label='Nautisch',**style['nautical'][0]),
            mpl.patches.Patch(label='Astronomisch',**style['astronomical'][0]),
        ]
        ax2.legend(handles=legends,
            bbox_to_anchor=(.5,.5),
            loc='center',
            handler_map={
                mplLegendSubheading:mplLegendSubheadingHandler(),
                mplLegendSpacer:mplLegendSpacerHandler(),
            })
        ax2.set_axis_off()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()

    def plotset_monthly_cycle(self,savefig=None):
        fig,axs = plt.subplots(nrows=2,ncols=3)
        self.plot_ym_cycle_tmin_tmax(axs[0][0])
        self.plot_ym_cycle_wcet(axs[0][1])
        self.plot_ym_cycle_wbgt_simplified(axs[0][2])
        self.plot_ym_cycle_relh(axs[1][0])
        self.plot_ym_cycle_vism(axs[1][1])
        self.plot_ym_cycle_ceiling(axs[1][2])
        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_monthly_stacks(self,savefig=None):
        fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(6.3,2.1))
        self.plot_ym_cycle_cloud_type(axs[0],legend=True)
        self.plot_ym_cycle_color(axs[1],legend=True)
        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def generate_yearly_plots(self,savefig=True):
        if savefig:
            dirname_figs = os.path.join(self.filepaths['output'],self.station,'fig')
            if not os.path.exists(dirname_figs):
                pathlib.Path(dirname_figs).mkdir(parents=True, exist_ok=True)

        msg = "Figuren %s: "%self.station
        print(msg+'...',end='\r',flush=True)

        msg += "Temp "
        print(msg+'...',end='\r',flush=True)
        self.plotset_ymwide_tmin_tmax(savefig=os.path.join(dirname_figs,'temp.png') if savefig else None)
        self.plotset_ymwide_wcet(savefig=os.path.join(dirname_figs,'wcet.png') if savefig else None)
        self.plotset_ymwide_wbgt(savefig=os.path.join(dirname_figs,'wbgt.png') if savefig else None)
        self.plotset_ymwide_wbgt(savefig=os.path.join(dirname_figs,'wbgt_flag.png') if savefig else None,limit_theme='limits.wbgt_flag')

        msg += "RH "
        print(msg+'...',end='\r',flush=True)
        self.plotset_ymwide_relh(savefig=os.path.join(dirname_figs,'relh.png') if savefig else None)

        msg += "Vis "
        print(msg+'...',end='\r',flush=True)
        #self.plotset_ymwide_vism(savefig=os.path.join(dirname_figs,'vis.png') if savefig else None)
        self.plotset_ymwide_visdays(savefig=os.path.join(dirname_figs,'vis_below.png') if savefig else None)

        msg += "Wind "
        print(msg+'...',end='\r',flush=True)
        self.plotset_wind(savefig=os.path.join(dirname_figs,'wind.png') if savefig else None)

        msg += "Cloud "
        print(msg+'...',end='\r',flush=True)
        #self.plotset_ymwide_ceiling(savefig=os.path.join(dirname_figs,'ceiling.png') if savefig else None)
        self.plotset_ymwide_ceilingdays(savefig=os.path.join(dirname_figs,'ceiling_below.png') if savefig else None)
        self.plotset_ymwide_cloud_type(savefig=os.path.join(dirname_figs,'cloud_cover.png') if savefig else None)

        msg += "Precipitation "
        print(msg+'...',end='\r',flush=True)
        self.plotset_ymwide_precipdays(savefig=os.path.join(dirname_figs,'precipitation.png') if savefig else None)

        msg += "Color "
        print(msg+'...',end='\r',flush=True)
        self.plotset_ymwide_color(savefig=os.path.join(dirname_figs,'color_state.png') if savefig else None)

        msg += "Solar "
        print(msg+'...',end='\r',flush=True)
        self.plotset_ymwide_solar(savefig=os.path.join(dirname_figs,'solar.png') if savefig else None)

        msg += "Map "
        print(msg+'...',end='\r',flush=True)
        self.plotset_map(savefig=os.path.join(dirname_figs,'map.png') if savefig else None)
        self.plotset_map_outline(savefig=os.path.join(dirname_figs,'map_outline.png') if savefig else None)

        msg += "Klaar!"
        print(msg,end='\r',flush=True)
        print("\n")
        _log.info("Afbeeldingen kunnen gevonden worden in %s/"%dirname_figs)

    # Non-meteo plots
    def plotset_logo(self,savefig=None):
        if savefig is not None:
            shutil.copyfile(self.filepaths['logo'],savefig)
    def plotset_map(self,stations=None,zoom=1,clat=None,clon=None,center_station=None,figsize=None,savefig=None):
        try:
            self.prepare_maps()
        except ModuleNotFoundError as err:
            _log.error(repr(err)+'\nInstalleer opnieuw de packages via "00. Instaleren & Introductie.ipynb"',exc_info=err)
            _log.info('Kaart niet geplot...')
            return
        if isinstance(stations,str):
            stations = [stations]
        if stations is None:
            stations = self.stations_on_map
        if self.station is not None and self.station not in stations:
            stations = [self.station] + stations
        station_list = []
        for s in stations:
            if s not in self.station_repo and s in self.station_repo.networks:
                station_list += self.station_repo.get_station_codes_from_network(s)
            else:
                station_list.append(s)
        if clat is None or clon is None:
            if center_station is not None:
                station, station_data = self.station_repo.get_station(center_station)
                clat,clon = station_data['latitude'], station_data['longitude']
            else:
                lats, lons = [], []
                for s in station_list:
                    s, sd = self.station_repo.get_station(s)
                    lats.append(sd['latitude'])
                    lons.append(sd['longitude'])
                clat, clon = (np.min(lats)+np.max(lats))/2, (np.min(lons)+np.max(lons))/2
        if zoom=='auto':
            if len(station_list)>=2:
                lats, lons = [], []
                for s in station_list:
                    s, sd = self.station_repo.get_station(s)
                    lats.append(sd['latitude'])
                    lons.append(sd['longitude'])
                zoom = 0.95*np.min([
                    10/np.abs(clat-np.min(lats)),
                    10/np.abs(clat-np.max(lats)),
                    15/np.abs(clon-np.min(lons)),
                    15/np.abs(clon-np.max(lons))])
                zoom = max(min(zoom,5),.2)
            else:
                zoom = 1
            print(zoom)
        extent = [clon-(15/zoom),clon+(15/zoom),clat-(10/zoom),clat+(10/zoom)]
        proj = cartopy.crs.NearsidePerspective(
            central_longitude=clon,
            central_latitude=clat,
            satellite_height=35785831
        )
        trans = cartopy.crs.PlateCarree()

        fig,ax = plt.subplots(1,1,figsize=figsize,subplot_kw={'projection':proj})

        for s in station_list:
            station, station_data = self.station_repo.get_station(s)
            lat,lon = station_data['latitude'], station_data['longitude']
            station_text = station_data.get('icao',station)
            station_text = s if station_text is None or station_text=='' else station_text
            if extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]:
                plane_str = """m 37.398882,331.5553 195.564518,-53.33707 81.92539,81.92539 c 18.40599,18.40599 58.40702,30.50459 72.90271,27.64788 2.85671,-14.49569 -9.24189,-54.49672 -27.64788,-72.90271 L 278.21823,232.9634 331.5553,37.39888 305.29335,11.13693 216.40296,171.14812 133.86945,88.61462 142.35474,29.21765 113.13708,0 73.058272,73.05827 0,113.13708 l 29.217652,29.21766 59.39697,-8.48529 82.533498,82.53351 -160.011188,88.89039 26.26195,26.26195"""
                plane_path = parse_path(plane_str)
                plane_path.vertices -= plane_path.vertices.mean(axis=0)
                ax.scatter(lon,lat,transform=trans,
                           s=36*2,marker=plane_path,c='k',zorder=2)
                ax.annotate(station_text,xy=(lon,lat),xytext=(lon+(.4/zoom),lat+(.3/zoom)),
                            xycoords=trans._as_mpl_transform(ax),zorder=2)
            else:
                lonr, latr, clonr, clatr = np.deg2rad(np.array([lon,lat,clon,clat]))
                x = np.cos(lonr)*np.sin(clatr-latr)
                y = np.cos(clonr)*np.sin(lonr)-np.sin(clonr)*np.cos(lonr)*np.cos(clatr-latr)
                heading = np.arctan2(y,x)-.5*np.pi
                heading_deg = np.rad2deg(heading)%360
                if 45<=heading_deg<=135:
                    x = .5+(.5/np.tan(heading))
                    y = 1
                elif 135<=heading_deg<=225:
                    x = 0
                    y = .5-(.5*np.tan(heading))
                elif 225<=heading_deg<=315:
                    x = .5-(.5/np.tan(heading))
                    y = 0
                else:
                    x = 1
                    y = .5+(.5*np.tan(heading))
                x,y = np.clip(x,.02,.98),np.clip(y,.02,.98)
                ax.scatter(x,y,transform=ax.transAxes,
                           s=36*2,c='k',
                           marker=(3,0,(heading_deg-90)%360),zorder=2)
                ax.annotate(station_text,
                            xy=(x,y),
                            xytext=(x+(.01 if x<.5 else -.01),
                                    y+(.01 if y<.5 else -.01)),
                            xycoords=ax.transAxes,
                            va=('top' if y>.5 else 'bottom'),
                            ha=('right' if x>.5 else 'left'),
                            zorder=2
                           )
        ax.set_extent(extent,crs=trans)
        self.map_stock_img(ax,zoom)
        #plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    
    def plotset_map_outline(self,station=None,savefig=None,country_code=None,transparent=None):
        station = station if station is not None else self.station
        self.countryfinder = CountryFinder(**self.filepaths)
        s,sd = self.station_repo.get_station(station)
        lon,lat = sd['longitude'],sd['latitude']
        geom,attrs = self.countryfinder.find_closest_country(lon,lat,country_code)
        _log.debug("Map_outline: Plotting %s on map of %s %s"%(s,attrs['SU_A3'],attrs['NAME']))
        
        if transparent is None:
            transparent = savefig is not None
        clon,clat = geom.centroid.x,geom.centroid.y
        lonmin, latmin, lonmax, latmax = geom.bounds
        if (lonmin+360-lonmax<5):
            polygon_bounds = np.array([list(p.bounds) for p in list(geom)])
            lonmin = np.nanmin(np.where(polygon_bounds[:,0]>0,polygon_bounds[:,0],np.nan))
            lonmax = np.nanmax(np.where(polygon_bounds[:,0]<0,polygon_bounds[:,2],np.nan)+360)
        extent = min(lon,lonmin),max(lon,lonmax),min(lat,latmin),max(lat,latmax)

        proj = cartopy.crs.NearsidePerspective(central_longitude=clon,central_latitude=clat,satellite_height=35785831)
        trans = cartopy.crs.PlateCarree()
        fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})

        if transparent:
            ax.background_patch.set_facecolor('#ffffff00')
        else:
            ax.background_patch.set_facecolor(self.theme.get("map_outline.bg")[0]['facecolor'])
        ax.outline_patch.set_linewidth(0.0)
        
        ax.scatter(lon,lat,transform=trans,s=36*5,marker='.',zorder=2,**self.theme.get("map_outline.dot")[0])
        shp = cartopy.feature.ShapelyFeature([geom],trans,zorder=-1,**self.theme.get("map_outline.shp")[0])
        ax.add_feature(shp,zorder=-1)
        ax.set_extent(extent,crs=trans)

        if savefig is not None:
            plt.savefig(savefig,transparent=transparent)
            plt.close()
    
    def test_all_plotsets_to_screen(self,but_not=None):
        plotsets = sorted([name for name in self.__class__.__dict__.keys() if name.startswith('plotset_')])
        but_not = but_not if but_not is not None else []
        for pfn in plotsets:
            obj = getattr(self,pfn)
            if callable(obj) and pfn not in but_not:
                _log.info(pfn)
                obj()


    def generate_monthly_tex(self):
        with open(self.filepaths['tex_head'],'r') as fhh:
            latexhead = fhh.read()
        with open(self.filepaths['tex_month'],'r') as fhm:
            latexmonth = fhm.read()

        latexmonths = []
        months = list(self.frange(self.filters['month'],1,12))
        for m in months:
            latexmonths.append(latexmonth % {
                'years1': self.years[m][0],
                'years2': self.years[m][1],
                'monthnr': '%02d'%m,
                'monthstr': self.locales['monthnames'][m].capitalize(),
                'stationicao': self.station_data['icao'].upper(),
                'stationname': self.station_data['name'].upper(),
                'publishdate': datetime.datetime.now().strftime("%Y"),
            })
        latex = [latexhead % {
            'firstmonth': months[0],
            'fontfolder': os.path.join(os.path.abspath(self.filepaths['fonts']),'')
        }]
        latex.append("\n\\newpage\n".join(latexmonths))
        latex.append("\\end{document}")

        dirname_texs = os.path.join(self.filepaths['output'],self.station)
        if not os.path.exists(dirname_texs):
            pathlib.Path(dirname_texs).mkdir(parents=True, exist_ok=True)
        latexfile = self.station_data['icao'].upper()+"_monthly.tex"
        with open(os.path.join(dirname_texs,latexfile),'w') as fhl:
            fhl.write("\n".join(latex))
    def generate_monthly_pdf_from_tex(self):
        dirname = os.path.join(self.filepaths['output'],self.station)
        latexfile = self.station_data['icao'].upper()+"_monthly.tex"
        r = subprocess.run([
                'xelatex',
                '-synctex=1',
                '-interaction=nonstopmode',
                latexfile],
            stdout=subprocess.PIPE,
            cwd=dirname)
        for file in os.listdir(dirname):
            filepath = os.path.join(dirname,file)
            if os.path.isfile(filepath) and not filepath.endswith(".pdf") and not filepath.endswith(".png"):
                os.remove(filepath)
    def generate_monthly_pdf(self):
        msg = f'Figuren {self.station}: '
        print(msg,end='\r',flush=True)
        basefilters = copy.copy(self.filters)

        for month in self.frange(basefilters['month'],1,12):
            msg += self.locales['monthabbr'][month]+' '
            print(msg+'...',end='\r',flush=True)
            self.reset_filters()
            self.redo_filters(basefilters)
            self.filter_month('=',month)

            dirname_figs = os.path.join(self.filepaths['output'],self.station,'fig')
            if not os.path.exists(dirname_figs):
                pathlib.Path(dirname_figs).mkdir(parents=True, exist_ok=True)
            self.plotset_daily(os.path.join(dirname_figs,f'A{month:02d}.png'))
            self.plotset_wind(os.path.join(dirname_figs,f'B{month:02d}.png'))
            self.plotset_gust(os.path.join(dirname_figs,f'C{month:02d}.png'))
            self.plotset_wx(os.path.join(dirname_figs,f'D{month:02d}.png'))

            self.years[month] = self.pdf.time.dt.year.min(), self.pdf.time.dt.year.max()

        self.plotset_daily_cycle_legend(os.path.join(dirname_figs,f'LEGEND.png'))
        self.plotset_logo(os.path.join(dirname_figs,f'LOGO.png'))

        self.reset_filters()
        self.redo_filters(basefilters)

        msg += 'TEX '
        print(msg+'...',end='\r',flush=True)
        self.generate_monthly_tex()

        msg += 'PDF '
        print(msg+'...',end='\r',flush=True)
        self.generate_monthly_pdf_from_tex()

        msg += 'Klaar!'
        print(msg,end='\r',flush=True)
        print("\n")

        _log.info('De PDF kan gevonden worden in "%s"'%os.path.join(self.filepaths['output'],self.station,self.station_data['icao'].upper()+"_monthly.pdf"))

class CountryFinder(object):
    def __init__(self,**settings):
        self.filepaths = {'natural_earth': settings.get('natural_earth','./resources/')}
        
        try:
            MetarPlotter.prepare_maps()
        except ModuleNotFoundError as err:
            _log.error(repr(err)+'\nInstalleer opnieuw de packages via "00. Instaleren & Introductie.ipynb"',exc_info=err)
            _log.info('Kaart niet geplot...')
            return
        
        self.countries_with_subunits_seperate = {
            'ATF': (0,4,[],[]),
            'AUS': (0,3,['AUM'],[]),
            'CHL': (0,4,[],[]),
            'ECU': (0,4,[],[]),
            'ESP': (0,3,['ESC'],[]),
            'FRA': (3,3,[],[]),
            'IOA': (0,3,[],[]),
            'NLD': (3,3,[],[]),
            'NOR': (0,3,[],[]),
            'NZL': (0,3,[],[]),
            'PRT': (0,3,[],[]),
            'RUS': (0,3,['RUC'],[]),
            'SHN': (0,4,[],[]),
            'USA': (0,4,[],[]),
            'ZAF': (0,4,[],[])
        }
        self.alliasses = {
            'NLD': 'NLX',
            'PRT': 'PRX',
            'USA': 'USB',
            'ZAF': 'ZAX'
        }
        self.do_search_all = ['ATF','IOA','SHN']
        self.geometries = {}
        self.attributes_df = None
        self.all_attributes_df = None
        self.get_countries_from_natural_earth()
        
    def get_countries_from_natural_earth(self):
        cshpf = MapPlotHelper.search_or_extract(self.filepaths['natural_earth'],'ne_10m_admin_0_countries','shp')
        ushpf = MapPlotHelper.search_or_extract(self.filepaths['natural_earth'],'ne_10m_admin_0_map_subunits','shp')
        cgeom, cdf = MapPlotHelper.shape_to_dataframe('SU_A3',cshpf)
        ugeom, udf = MapPlotHelper.shape_to_dataframe('SU_A3',ushpf)
        cdf = cdf.assign(code=cdf.SU_A3.apply(lambda t: 'c_'+t))
        cdf = cdf.assign(geometries='')
        udf = udf.assign(code=udf.SU_A3.apply(lambda t: 'u_'+t))
        udf = udf.assign(geometries=udf.code)
        all_df = pd.concat([cdf,udf]).reset_index(drop=True)
        for i,r in all_df.loc[all_df.LEVEL==3].iterrows():
            all_df.loc[i,'geometries'] = ",".join(sorted(list(set(
                all_df.loc[(all_df.ADM0_A3==r.ADM0_A3) & (all_df.GU_A3==r.GU_A3) & (all_df.LEVEL>=3),'geometries'].to_list()
            ))))
        for i,r in all_df.loc[all_df.LEVEL==2].iterrows():
            all_df.loc[i,'geometries'] = ",".join(sorted(list(set(
                all_df.loc[(all_df.ADM0_A3==r.ADM0_A3) & (all_df.LEVEL>=3),'geometries'].to_list()
            ))))
        all_df = all_df.sort_values(['LEVEL','ADM0_A3'],ascending=False)
        
        df = cdf.loc[~cdf.ADM0_A3.isin(list(self.countries_with_subunits_seperate.keys())),:].reset_index(drop=True)
        for i,r in df.iterrows():
            df.loc[i,'geometries'] = r.code if r.geometries=='' else r.geometries
        df = df.sort_values(['LEVEL','ADM0_A3']).reset_index(drop=True)
    
        for c,(low,up,incl,excl) in self.countries_with_subunits_seperate.items():
            is_child = (all_df.ADM0_A3==c) & (all_df.LEVEL>=low) & (all_df.LEVEL<=up)
            if len(incl)>0:
                is_child = is_child | all_df.SU_A3.isin(incl)
            if len(excl)>0:
                is_child = is_child & (~all_df.SU_A3.isin(excl))
            sdf = all_df.loc[is_child]
            sdf = sdf.sort_values(['LEVEL','ADM0_A3'],ascending=False).reset_index(drop=True)
            for i,r in sdf.iterrows():
                other_row_geometries = sdf.drop(i).geometries.str.split(",").to_list()
                other_geometries = [g for gr in other_row_geometries for g in gr]
                geocodes = r.geometries.split(',')
                sdf.loc[i,'geometries'] = ",".join(list(filter(
                    lambda c: ((c not in other_geometries) or c==r.code),geocodes)))
            df = pd.concat([df,sdf],sort=False)
        df = df.loc[df.geometries!=''].reset_index(drop=True)
        df = df.sort_values(['LEVEL','ADM0_A3']).reset_index(drop=True)
        geometries = {}
        for i,r in df.iterrows():
            geo_list = [(cgeom[g[2:]].geometry if g[0]=='c' else ugeom[g[2:]].geometry) for g in r.geometries.split(',')]
            if len(geo_list)>1:
                geometries[r.key] = shapely.ops.unary_union(geo_list)
            else:
                geometries[r.key] = geo_list[0]
        self.geometries = geometries
        self.attributes_df = df
        
        all_geometries = {}
        for i,r in all_df.iterrows():
            if r.key in self.geometries:
                continue
            geo_list = [(cgeom[g[2:]].geometry if g[0]=='c' else ugeom[g[2:]].geometry) for g in r.geometries.split(',')]
            if len(geo_list)>1:
                all_geometries[r.key] = shapely.ops.unary_union(geo_list)
            else:
                all_geometries[r.key] = geo_list[0]
        self.all_geometries = all_geometries
        self.all_attributes_df = all_df
    def __contains__(self,item):
        code = None
        try:
            code = self.resolve_country_code(item)
        except ValueError:
            pass
        return code is not None
    def resolve_country_code(self,code,search_all=False):
        country_key = None
        adf = self.attributes_df
        if code in self.alliasses.keys():
            code = self.alliasses[code]
        if not search_all and code in self.do_search_all:
            return self.resolve_country_code(code,search_all=True)
        if search_all:
            adf = self.all_attributes_df
        if len(code)==2:
            if code in adf.ISO_A2.values:
                country_key = adf.loc[adf.ISO_A2==code].sort_values(['HOMEPART','MAX_LABEL'],ascending=[False,True]).key.iloc[0]
        else:
            if code in adf.key.values:
                country_key = code
            elif code in adf.ISO_A3.values:
                country_key = adf.loc[adf.ISO_A3==code].sort_values(['HOMEPART','MAX_LABEL'],ascending=[False,True]).key.iloc[0]
            elif code in adf.ADM0_A3.values:
                country_key = adf.loc[adf.ADM0_A3==code].sort_values(['HOMEPART','MAX_LABEL'],ascending=[False,True]).key.iloc[0]
        if country_key is None:
            if search_all:
                raise ValueError('Could not find any country with the code "%s"'%code)
            return self.resolve_country_code(code,search_all=True)
        return country_key
    def get_country_by_code(self,code):
        country_code = self.resolve_country_code(code)
        if country_code in self.attributes_df.key.values:
            return (
                self.geometries[country_code],
                self.attributes_df.loc[self.attributes_df.key==country_code].iloc[0].to_dict())
        return (
            self.all_geometries[country_code],
            self.all_attributes_df.loc[self.all_attributes_df.key==country_code].iloc[0].to_dict())
    def find_closest_country(self,lon,lat,country_code=None):
        p = shapely.geometry.Point(lon,lat)
        if country_code is not None:
            return self.get_country_by_code(country_code)
        else:
            distances = {}
            for k,c in self.geometries.items():
                if c.contains(p):
                    distances[k] = -1
                    break
                np1, np2 = shapely.ops.nearest_points(c,p)
                d = p.distance(np1)
                if len(distances)==0 or min(list(distances.values()))>d:
                    distances[k] = d
            closest_country = min(distances,key=distances.get)
            return (
                self.geometries[closest_country],
                self.attributes_df.loc[self.attributes_df.key==closest_country].iloc[0].to_dict())

class MapPlotHelper(object):
    system_natural_earh_folders = [
        '/home/datalab/y-schijf/Voorbeelden/NaturalEarth/physical/',
        '/home/datalab/y-schijf/Voorbeelden/NaturalEarth/cultural/'
    ]
    @classmethod
    def search_files(cls,basepath,name,exts):
        potential_locations = [
            os.path.join(basepath,'{name}.{ext}'),
            os.path.join(basepath,'natural_earth','{name}.{ext}'),
            os.path.join(basepath,'{name}','{name}.{ext}'),
            os.path.join(basepath,'natural_earth','{name}','{name}.{ext}'),
        ]
        if isinstance(exts,list):
            for ext in exts:
                for potential_location in potential_locations:
                    pl = potential_location.format(name=name,ext=ext)
                    if os.path.exists(pl) and os.path.isfile(pl):
                        return pl
        else:
            ext = exts
            for potential_location in potential_locations:
                pl = potential_location.format(name=name,ext=ext)
                if os.path.exists(pl) and os.path.isfile(pl):
                    return pl
        return False
    @classmethod
    def search_or_extract(cls,basepath,name,exts):
        data_path = cls.search_files(basepath,name,exts)
        extstr = '['+','.join(exts)+']' if isinstance(exts,list) else exts
        if data_path:
            return data_path
        zip_path = cls.search_files(basepath,name,'zip')
        zipextract_path = os.path.join(basepath,'natural_earth')
        if zip_path:
            with zipfile.ZipFile(zip_path,'r') as zipfh:
                filelist = zipfh.namelist()
                pls = ([f'{name}.{ext}' for ext in exts] + [f'{name}/{name}.{ext}' for ext in exts]
                       if isinstance(exts,list) else
                       [f'{name}.{exts}',f'{name}/{name}.{exts}'])
                if any([pl in filelist for pl in pls]):
                    _log.debug(f'Uitpakken van "{zip_path}" naar "{zipextract_path}"...')
                    zipfh.extractall(zipextract_path)
                else:
                    raise ValueError(f'Zipfile "{zip_path}" bevat niet de benodigde bestanden ({name}.{extstr})')
            data_path = cls.search_files(basepath,name,exts)
            if data_path:
                return data_path
            raise ValueError(f'Zipfile uitgepakt ({zip_path}), maar kon de bestanden niet vinden ({name}.{extstr})')
        
        for snef in cls.system_natural_earh_folders:
            sys_path = cls.search_files(snef,name,exts)
            if sys_path:
                sys_name = os.path.splittext(os.path.basename(sys_path))[0]
                sys_dir = os.path.dirname(sys_path)
                sys_glob = os.path.join(sys_dir,sys_name+'.*')
                target_dir = os.path.join(basepath,'natural_earth')
                if not os.path.exists(target_dir):
                    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
                for sys_file in glob.iglob(sys_glob):
                    new_file = os.path.join(target_dir,os.path.basename(sys_file))
                    shutil.copy2(sys_file,new_file)
                data_path = cls.search_files(basepath,name,exts)
                if data_path:
                    return data_path
        
        
        exts = exts+['zip'] if isinstance(exts,list) else [exts,'zip']
        extstr = '['+','.join(exts)+']'
        raise ValueError(f'Kon de benodigde bestanden niet vinden ({name}.{extstr})')
    @classmethod
    def slice_img(cls,img_extent_request,imgda,step=1):
        if img_extent_request[2]<-89.75:
            img_extent_request=[-180,180,-90,img_extent_request[3]]
        if img_extent_request[3]>89.75:
            img_extent_request=[-180,180,img_extent_request[2],90]
        img_idx = [[None,None],[None,None]]
        for i,eir in enumerate(img_extent_request):
            img_idx[i//2][i%2] = np.abs(imgda['x' if i//2==0 else 'y'].values - eir).argmin()
        img_slice = {'x':slice(
                        np.clip(np.min(img_idx[0])-1,0,len(imgda.x.values)),
                        np.clip(np.max(img_idx[0])-1,0,len(imgda.x.values)),
                        step),
                     'y':slice(
                        np.clip(np.min(img_idx[1])-1,0,len(imgda.y.values)),
                        np.clip(np.max(img_idx[1])-1,0,len(imgda.y.values)),
                        step)}
        img_da = imgda.isel(**img_slice)
        img_extent = np.array([[o(img_da[c].values) for o in [np.min,np.max]] for c in 'xy']).flatten()
        return img_extent,img_da
    @classmethod
    def shape_to_dataframe(cls,key,shapefile):
        MetarPlotter.prepare_maps()
        shpr = cartopy.io.shapereader.Reader(shapefile)
        data = {}
        geom = {}
        for r in shpr.records():
            data[r.attributes[key]] = r.attributes
            geom[r.attributes[key]] = r
        df = pd.DataFrame.from_dict(data,orient='index')
        return geom, df.reset_index().rename(columns={'index':'key'})

class mplLegendSubheading(object):
    def __init__(self,s,level=0):
        self.s = s
        self.level = level
        self.len = int(max([len(sp) for sp in self.s.split('\n')])-4)
        self.len *= .4 if level>3 else 1
    def get_label(self):
        return ('' if self.level>1 else '\n')+('\u2007'*int(self.len))
class mplLegendSubheadingHandler(object):
    def legend_artist(self,legend,orig_handle,fontsize,handlebox):
        x0,y0,width,height = handlebox.xdescent, handlebox.ydescent, handlebox.width, handlebox.height
        fontweight = 'normal' if orig_handle.level>0 else 'bold'
        fontsize = fontsize*.8 if orig_handle.level>3 else fontsize
        text = mpl.text.Text(x0,y0-height*.5,orig_handle.s,fontsize=fontsize,fontweight=fontweight)
        handlebox.add_artist(text)
        return text
class mplLegendSpacer(object):
    def get_label(self):
        return ''
class mplLegendSpacerHandler(object):
    def legend_artist(self,legend,orig_handle,fontsize,handlebox):
        text = mpl.text.Text(handlebox.xdescent, handlebox.ydescent,'',fontsize=fontsize)
        handlebox.add_artist(text)
        return text
