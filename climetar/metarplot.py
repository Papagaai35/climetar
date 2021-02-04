import copy
import datetime
import glob
import json
import locale
import numbers
import pathlib
import re
import shutil
import subprocess
import sys, os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import metar, quantities

class MetarTheme(object):
    def __init__(self,json_or_file=None):
        self.theme = {}
        if json_or_file is not None:
            theme = None
            try:
                theme = json.loads(json_or_file)
            except:
                try:
                    if os.path.exists(json_or_file) and os.path.isfile(json_or_file):
                        with open(json_or_file,'r') as fh:
                            theme = json.load(fh)
                except:
                    pass
            if theme is not None:
                self.theme = theme
            else:
                raise ValueError('Invalid theme passed:\n%s'%json_or_file)
    @classmethod
    def to_color(cls,color,default_alpha=None,dict_keys=None,default_color='k'):
        if default_alpha is None:
            default_alpha = [1]
        if hasattr(dict_keys,'__iter__') and not isinstance(dict_keys,list):
            dict_keys = list(dict_keys)
        
        if isinstance(color,dict):
            if dict_keys is None:
                raise ValueError('Error 1: A list of colors is expected')
            keys = sorted(list(set(list(color.keys())+dict_keys)))
            colordict = {}
            for k in keys:
                if k in color:
                    colordict[k] = cls.to_color(color[k],default_alpha)
                elif 'default' in color:
                    colordict[k] = cls.to_color(color['default'],default_alpha)
                else:
                    colordict[k] = cls.to_color(default_color,default_alpha)
            return colordict
        elif isinstance(color,list) and dict_keys is None:
            if len(color)>=len(default_alpha):
                colorlist = []
                for i, c in enumerate(color):
                    colorlist.append(mpl.colors.to_rgba(c,default_alpha[i]))
                return colorlist
            else:
                raise ValueError('Error 2: At least %d colors are necessary for this plot. %d given'%(len(default_alpha),len(color)))     
        elif isinstance(color,list):
            if len(color)>=len(dict_keys):
                colordict = {}
                for i,k in enumerate(dict_keys):
                    colordict[k] = [color[i]]
                for j in range(i,len(color)):
                    colordict[f'other_{j:d}'] = [color[j]]
                return cls.to_color(colordict,default_alpha,dict_keys)
            else:
                raise ValueError('Error 2: At least %d colors are necessary for this plot. %d given'%(len(dict_keys),len(color)))     
        elif isinstance(color,(str,tuple)) and dict_keys is None:
            colorlist = []
            for da in default_alpha:
                colorlist.append(mpl.colors.to_rgba(color,da))
            return colorlist
        elif isinstance(color,(str,tuple)):
            colordict = {}
            for k in dict_keys:
                colordict[k] = cls.to_color(color,default_alpha)
            return colordict
        else:
            raise ValueError('Error 4: Strange inputs:\n'+'\n'.join([n+': '+repr(e) for n,e in {'color':color,'default_alpha':default_alpha,'dict_keys':dict_keys,'default_color':default_color}.items()]))
    def cel(self,fnname,colors=None,edgecolor=None,linewidth=None,**kwargs):
        if fnname in self.theme:
            c,e,l = (
                colors if colors is not None else self.theme[fnname].get('colors','k'),
                edgecolor if edgecolor is not None else self.theme[fnname].get('edgecolor','none'),
                linewidth if linewidth is not None else self.theme[fnname].get('linewidth',.5),
            )
        else:
            c,e,l = (
                colors if colors is not None else 'k',
                edgecolor if edgecolor is not None else 'none',
                linewidth if linewidth is not None else .5,
            )
        c = self.to_color(c,**kwargs)
        e = self.to_color(e)[0] if e not in ['none',None] else 'none'
        l = 0 if e=='none' else l
        return c,e,l

class MetarPlotter(object):    
    def __init__(self,**settings):
        self.theme = MetarTheme(settings.get('theme'))
        self.style = settings.get('style','./resources/climetar.mplstyle')
        if os.path.isfile(self.style):
            plt.style.use(self.style)
        
        self.filepaths = {
            'data': settings.get('folder_data','./data'),
            'output': settings.get('folder_output','./results/MonthlyPDF/'),
            'stations_json': settings.get('stations_json','./resources/stations.json'),
            'tex_head': settings.get('tex_head','./resources/T0.head.tex'),
            'tex_month': settings.get('tex_month','./resources/T1.month.tex'),
            'fonts': settings.get('fonts','./resources/fonts/'),
            'logo': settings.get('logo','./resources/JMG.png'),
        }
        #self.locale = settings.get('lang','en_GB.utf8')
        self.locales = {}
        self.station_repo = {}
        
        self.station = None
        self.station_data = None
        self.df = None
        self.pdf = None
        self.filters = {}
        self.years = {}
        
        self._load_station_repo()
        self._load_locales()
    
    def _load_station_repo(self,stations_json_file=None):
        file = self.filepaths['stations_json']
        with open(file,'r') as fh:
            self.station_repo = json.load(fh)
    def _load_locales(self):
        #locale.setlocale(locale.LC_ALL,self.locale)
        self.locales['monthnames'] = dict([(m,datetime.datetime.strptime("%02d"%m,"%m").strftime("%B")) for m in range(1,13)])
        self.locales['monthabbr'] = dict([(m,datetime.datetime.strptime("%02d"%m,"%m").strftime("%b")) for m in range(1,13)])
        #locale.setlocale(locale.LC_ALL,locale.getdefaultlocale())
    def load_data(self,station):
        self.station = self.station_repo['aliases'].get(station,station)
        self.station_data = self.station_repo['stations'][self.station]
        self.icao = self.station_data['icao']
        
        filename = os.path.join(self.filepaths['data'],self.station_data['abbr']+'.metar')
        if (not os.path.exists(filename) or
            not os.path.isfile(filename) ):
            filename = os.path.join(self.filepaths['data'],self.station_data['icao']+'.metar')
        if (not os.path.exists(filename) or
            not os.path.isfile(filename) ):
            raise ValueError(f'Could not find data file in "{filename}"')
        
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
            
    def reset_filters(self):
        self.pdf = self.df.copy(deep=True)
        self.filters = dict([(k,(None,None,None)) for k in ['year','month','day','hour','minutes_valid']])
    def redo_filters(self,filters):
        for k,v in filters.items():
            if hasattr(self,f'filter_{k}'):
                getattr(self,f'filter_{k}')(*v)
    def filter_series(self,series,gte=None,lte=None,eq=None):
        if gte is not None and lte is not None:
            if gte==lte:
                gte,lte,eq = None,None,gte
            if gte>lte:
                gte,lte,eq = lte,gte,None
        if eq is not None:
            self.pdf = self.pdf.loc[series==eq]
        elif gte is not None and lte is not None:
            self.pdf = self.pdf.loc[series.between(gte,lte)]
        elif gte is not None:
            self.pdf = self.pdf.loc[series >= gte]
        elif lte is not None:
            self.pdf = self.pdf.loc[series <= lte]
        return gte,lte,eq
    def filter_year(self,gte=None,lte=None,eq=None):
        self.filters['year'] = self.filter_series(self.pdf.time.dt.year,gte,lte,eq)
    def filter_month(self,gte=None,lte=None,eq=None):
        self.filters['month'] = self.filter_series(self.pdf.time.dt.month,gte,lte,eq)
    def filter_day(self,gte=None,lte=None,eq=None):
        self.filters['day'] = self.filter_series(self.pdf.time.dt.day,gte,lte,eq)
    def filter_hour(self,gte=None,lte=None,eq=None):
        self.filters['hour'] = self.filter_series(self.pdf.time.dt.hour,gte,lte,eq)
    def filter_minutes_valid(self,gte=None,lte=None,eq=None):
        self.filters['minutes_valid'] = self.filter_series(self.pdf.minutes_valid,gte,lte,eq)
    
    @classmethod
    def frange(cls,f,default_start,default_end):
        start = (f[0] if f[0] is not None else (
            f[2] if f[2] is not None else default_start ))
        end = 1+(f[1] if f[1] is not None else (
            f[2] if f[2] is not None else default_end ))
        return range(start,end)
    @classmethod
    def convert_unit(cls,converter,data_series):
        if isinstance(converter, numbers.Real):
            return data_series / converter
        elif isinstance(converter, collections.Iterable):
            return data_series.apply(converter[1])
        else:
            raise ValueError('Could not %s convert from %s',(value,unit))
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
            raise ValueError('The axis must be polar for a wind_compass_* plot, not %s.'%ax.name)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
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
    
    def _plot_daily_cycle_hoursteps(
            self,ax,variable,unit,quantity,title='',ylim=None,
            colors=None,edgecolor=None,linewidth=None):
        
        colors, edgecolor, linewidth = self.theme.cel(
            'daily_cycle_hoursteps',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        unit = quantity.find_unit(unit)
        
        gbo = self.pdf.dropna(subset=[variable]).groupby(self.pdf.time.dt.hour)
        data = gbo[variable].quantile([.01,.05,.25,.5,.75,.95,.99])
        data = self.convert_unit(quantity.units[unit],data).unstack()
        data = data.append(data.iloc[0].rename(24))
        ax.plot(data[.5],color=colors[0])
        ax.fill_between(x=data.index,y1=data[.25],y2=data[.75],zorder=-1,color=colors[1])
        ax.fill_between(x=data.index,y1=data[.05],y2=data[.25],zorder=-1,color=colors[2])
        ax.fill_between(x=data.index,y1=data[.75],y2=data[.95],zorder=-1,color=colors[2])
        ax.fill_between(x=data.index,y1=data[.01],y2=data[.05],zorder=-1,color=colors[3])
        ax.fill_between(x=data.index,y1=data[.95],y2=data[.99],zorder=-1,color=colors[3])
        ax.set_xticks([0,6,12,18,24]);
        ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],minor=True);
        begin, end = self.filters['hour'][0:2]
        begin = 0 if begin is None else begin
        end = 24 if end is None else end
        ax.set_xlim(begin,end)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%dh"))
        ax.set_title(title)
    def plot_daily_cycle_temp(self,ax,unit='°C',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'daily_cycle_temp',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        self._plot_daily_cycle_hoursteps(ax,'temp',title='Temperature [%s]'%unit,
            unit=unit,quantity=quantities.Temperature,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def plot_daily_cycle_dwpc(self,ax,unit='°C',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'daily_cycle_dwpc',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        self._plot_daily_cycle_hoursteps(ax,'dwpt',title='Dew Point [%s]'%unit,
            unit=unit,quantity=quantities.Temperature,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def plot_daily_cycle_relh(self,ax,unit='%',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'daily_cycle_relh',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        quantity = quantities.Fraction
        unit = quantity.find_unit(unit)
        self._plot_daily_cycle_hoursteps(ax,'relh',title='Relative Humidity [%s]'%unit,
            ylim=(0,quantities.Fraction(1,'frac')[unit]),
            unit=unit,quantity=quantity,colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def plot_daily_cycle_wspd(self,ax,unit='kt',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'daily_cycle_wspd',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        self._plot_daily_cycle_hoursteps(ax,'wind_spd',title='Wind speed [%s]'%unit,
            ylim=(0,None),unit=unit,quantity=quantities.Speed,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def plot_daily_cycle_vism(self,ax,unit='km',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'daily_cycle_vism',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        quantity = quantities.Distance
        unit = quantity.find_unit(unit)
        self._plot_daily_cycle_hoursteps(ax,'vis',title='Visibility [%s]'%unit,
            ylim=(0,quantities.Distance(10,'km')[unit]),unit=unit,quantity=quantity,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def plot_daily_cycle_ceiling(self,ax,unit='ft',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'daily_cycle_ceiling',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        quantity = quantities.Height
        unit = quantity.find_unit(unit)
        self._plot_daily_cycle_hoursteps(ax,'sky_ceiling',title='Cloud base [%s]'%unit,
            ylim=(0,quantities.Distance(5e3,'ft')[unit]),unit=unit,quantity=quantity,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def plot_daily_cycle_pres(self,ax,unit='hPa',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'daily_cycle_pres',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        quantity = quantities.Pressure
        unit = quantity.find_unit(unit)
        self._plot_daily_cycle_hoursteps(ax,'spPa',title='Surface Pressure [%s]'%unit,
            ylim=(quantities.Distance(950,'hPa')[unit],quantities.Distance(1050,'hPa')[unit]),unit=unit,quantity=quantity,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    
    def plot_wind_compass_dir_freq(self,ax,unit='%',cat=True,colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'wind_compass_dir_freq',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        color = colors[0]
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
        ax.bar(index, values, width=width,color=color,edgecolor=edgecolor,linewidth=linewidth)
        maxval = values.max()
        addval = (10**(np.log10(maxval)//1))*(.25 if np.log10(maxval)%1<.4 else (.5 if np.log10(maxval)%1 < 0.7 else 1))
        ax.set_rorigin(-.075*(maxval+addval))
        ax.grid(which='both',lw=0.4)
        ax.grid(which='minor',b=True,ls='--')
        ax.set_yticks(np.arange(0,maxval+addval,addval*2))
        ax.set_yticks(np.arange(0,maxval+addval,addval),minor=True)
        self.realign_polar_xticks(ax)
        ax.set_title('Wind Direction frequency [%s]'%unit,pad=12.5)
    def plot_wind_compass_spd(self,ax,unit='kt',cat=True,colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'wind_compass_spd',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
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
        data = data.append(data.iloc[0].rename(2*np.pi))
        ax.plot(data[.5],color=colors[0])
        ax.fill_between(x=data.index,y1=data[.25],y2=data[.75],zorder=-1,color=colors[1])
        ax.fill_between(x=data.index,y1=data[.05],y2=data[.25],zorder=-1,color=colors[2])
        ax.fill_between(x=data.index,y1=data[.75],y2=data[.95],zorder=-1,color=colors[2])
        ax.fill_between(x=data.index,y1=data[.01],y2=data[.05],zorder=-1,color=colors[3])
        ax.fill_between(x=data.index,y1=data[.95],y2=data[.99],zorder=-1,color=colors[3])
        maxval = data[.99].max()
        addval = (10**(np.log10(maxval)//1))*(.25 if np.log10(maxval)%1<.3 else (.5 if np.log10(maxval)%1 < 0.6 else 1))
        ax.set_rorigin(-.075*(maxval+addval))
        ax.grid(which='both',lw=0.4,color='k',alpha=0.38)
        ax.grid(which='minor',b=True,ls='--')
        ax.set_yticks(np.arange(0,maxval+addval,addval*2))
        ax.set_yticks(np.arange(0,maxval+addval,addval),minor=True)
        self.realign_polar_xticks(ax)
        ax.set_title('Wind Speed [%s]'%unit,pad=12.5)
            
    def plot_frequency_gust(self,ax,unit='kt',freq_unit='%',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'frequency_gust',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        color = colors[0]
        
        quantity = quantities.Speed
        unit = quantity.find_unit(unit)
        binval = quantity(99,'kt')[unit]
        binval = (10**(np.log10(binval)//1))*(.25 if np.log10(binval)%1<.4 else (.5 if np.log10(binval)%1 < 0.7 else 1))
        databinned = self.convert_unit(quantity.units[unit],self.pdf.wind_gust)//binval
        databinned = databinned.groupby(databinned).count()
        data = dict([(i, databinned[i] if i in databinned.index else 0) for i in range(1,int(databinned.index.max())+1)])
        data = pd.Series(np.array(list(data.values())), ["{}".format(k*binval-binval) for k in data.keys()] )
        data = data / len(self.pdf)

        freq_quantity = quantities.Fraction
        freq_unit = freq_quantity.find_unit(freq_unit)
        index, values = data.index.values, self.convert_unit(freq_quantity.units[freq_unit],data.values)
        ax.bar(index, values, color=color,edgecolor=edgecolor,linewidth=linewidth,align='edge',width=1)
        thx = values.max()*0.025
        for i in range(len(data)):
            ax.text(i+0.5,values[i]+thx,"%3.1f %s"%(values[i],freq_unit),c='k',ha='center',va='bottom')
        ax.set_ylabel('Frequency [%s]'%freq_unit)
        ax.set_title('Wind Gusts [%s]'%unit)
        ax.set_ylim(0,(np.ceil((values.max()+thx*3)/0.5)*0.5))
        ax.set_xlim(*np.array(ax.get_xlim()).round())
    def plot_frequency_cloud_type(self,ax,freq_unit='%',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'frequency_cloud_type',colors,edgecolor,linewidth,
            default_alpha=[1],dict_keys=metar.Metar._cloud_cover_codes.keys()
        )
        
        freq_quantity = quantities.Fraction
        freq_unit = freq_quantity.find_unit(freq_unit)
        data = self.pdf.groupby('sky_cover')['minutes_valid'].sum().reindex(metar.Metar._cloud_cover_codes.keys()).dropna()
        data = data / self.pdf.minutes_valid.sum()
        index, values = data.index.values, self.convert_unit(freq_quantity.units[freq_unit],data.values)
        
        color = []
        for idx in index:
            color.append(colors[idx][0])
        
        ax.bar(index, values, color=color,edgecolor=edgecolor,linewidth=linewidth)
        thx = values.max()*0.05
        for i in range(len(data)):
            ax.text(i,values[i]+thx,"%3.1f %s"%(values[i],freq_unit),c='k',ha='center',va='bottom')
        ax.set_ylabel('Frequency [%s]'%freq_unit)
        ax.set_title('Cloud cover type')
        ax.set_ylim(0,((values.max()+thx*3)//2+1)*2)
    def plot_frequency_color(self,ax,freq_unit='%',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'frequency_color',colors,edgecolor,linewidth,
            default_alpha=[1],dict_keys=['BLU+','BLU','WHT','GRN','YLO','YLO1','YLO2','AMB','RED']
        )
        
        freq_quantity = quantities.Fraction
        freq_unit = freq_quantity.find_unit(freq_unit)
        data = self.pdf.groupby('color')['minutes_valid'].sum().reindex(['BLU+','BLU','WHT','GRN','YLO','YLO1','YLO2','AMB','RED']).dropna()
        calculated = False
        if len(data)==0 and 'calc_color' in self.pdf.columns:
            data = self.pdf.groupby('calc_color')['minutes_valid'].sum().reindex(['BLU+','BLU','WHT','GRN','YLO','YLO1','YLO2','AMB','RED']).dropna()
            calculated = True
        if len(data)==0:
            ax.bar(['BLU','WHT','GRN','YLO','AMB','RED'],[0,0,0,0,0,0],color='ḱ')
            ax.text(0.5,0.5,f"{self.STATION} does not publish\nNATO color states",
                    horizontalalignment='center',verticalalignment='center',
                    transform=ax.transAxes,c='k')
            ax.set_ylim(0,10)
        else:
            data = data / self.pdf.minutes_valid.sum()
            index, values = data.index.values, self.convert_unit(freq_quantity.units[freq_unit],data.values)
            
            color = []
            for idx in index:
                color.append(colors[idx][0])
            
            ax.bar(index, values, color=color,edgecolor=edgecolor,linewidth=linewidth)
            thx = values.max()*0.05
            for i in range(len(data)):
                ax.text(i,values[i]+thx,"%3.1f %s"%(values[i],freq_unit),c='k',ha='center',va='bottom')
        ax.set_ylabel('Frequency [%s]'%freq_unit)
        ax.set_title('NATO Color State '+('(Calculated)' if calculated else '(from METAR)'))
        ax.set_ylim(0,((values.max()+thx*3)//2+1)*2)
    def plot_frequency_percipitation(self,ax,freq_unit='%',colors=None,edgecolor=None,linewidth=None):
        preciptypes = ['RA','DZ','SN','IC','PL','GR','GS','UP','FZRA','FZDZ','FZFG']
        intensitytypes = ['+','','-','VC']
                
        colors, edgecolor, linewidth = self.theme.cel(
            'frequency_percipitation',colors,edgecolor,linewidth,
            default_alpha=[1,.7,.5,.25],dict_keys=preciptypes
        )
        
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
                ax.text(0.5,0.5,f"{self.STATION} registerd no percipitation in this month",
                        horizontalalignment='center',verticalalignment='center',
                        transform=ax.transAxes,c='k')
                ax.set_ylim(0,10)
        else:
            for i in range(precipdf.shape[0]):
                colorlist = [colors[c][i] for c in precipdf.columns]
                heights = precipdf.iloc[i,:]*norm
                bottoms = precipdf.iloc[:i,:].sum()*norm
                barlist[i] = ax.bar(precipdf.columns,
                                    heights,
                                    bottom=bottoms,
                                    color=colorlist,
                                    edgecolor=edgecolor,
                                    linewidth=linewidth)
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
            legend_colors = self.theme.to_color('k',default_alpha=[1,.7,.5,.25])
            for i,s in enumerate(precipdf.index.values):
                text = {'-':'Light (-)','':'Normal ( )','+':'Heavy (+)','VC':'In the vicinity (VC)'}[s]
                legend_elem.append(mpl.patches.Patch(facecolor=legend_colors[i],
                                                     edgecolor='k' if edgecolor=='none' else edgecolor,
                                                     linewidth=linewidth,
                                                     label=text))
            ax.legend(handles=list(reversed(legend_elem)),
                      framealpha=1, frameon=False)
        ax.set_title('Significant Precipitation')
        ax.set_ylabel('Frequency [%s]'%freq_unit)
    def plot_frequency_sigwx(self,ax,freq_unit='%',colors=None,edgecolor=None,linewidth=None):
        sigwxtypes = ['TS','FZ','SH','FG','VA','BR','HZ','DU','FU','SA','PY','SQ','PO','DS','SS','FC']
        
        colors, edgecolor, linewidth = self.theme.cel(
            'frequency_sigwx',colors,edgecolor,linewidth,
            default_alpha=[1],dict_keys=sigwxtypes
        )
                
        freq_quantity = quantities.Fraction
        freq_unit = freq_quantity.find_unit(freq_unit)
        sigwxdf = self.pdf.iloc[:,:0]
        for sigwx in sigwxtypes:
            sigwxdf[sigwx] = self.pdf.wx.str.contains('(?:\+|-|VC)*?(?:[A-Z]{2})*?'+re.escape(sigwx)+'(?:[A-Z]{2})*?')
        s = sigwxdf.sum()
        s = s.loc[s>0].sort_values(ascending=False)
        norm = self.convert_unit(freq_quantity.units[freq_unit],1.)/len(self.pdf)
        
        color = []
        for idx in s.index:
            color.append(colors[idx][0])
        
        ax.bar(s.index,s.values*norm,color=color,edgecolor=edgecolor,linewidth=linewidth)
        thx = s.max()*norm*0.05
        for i in range(len(s)):
            ax.text(i,s.iloc[i]*norm+thx,"%3.1f %s"%(s.iloc[i]*norm,freq_unit),c='k',ha='center',va='bottom')
        ax.set_ylim(0,((s.max()*norm+thx*3)//2+1)*2)
        ax.set_title('Significant Weather');
        ax.set_ylabel('Frequency [%s]'%freq_unit)
            
    def plotset_daily_cycle_legend(self,savefig=None):
        with mpl.rc_context(rc={'font.size':15*1.5}):
            fig,ax = plt.subplots(1,1,figsize=(6*3,1))
            colors, edgecolor, linewidth = self.theme.cel(
                'daily_cycle_hoursteps',default_alpha=[1,.6,.38,.15]
            )
            labels = ['%02d%% Confidence interval'%i for i in [50,90,99]]
            handles = [mpl.patches.Patch(color=colors[x+1],label=labels[x]) for x in range(len(labels))]
            handles = [mpl.lines.Line2D([],[],linestyle='-',color=colors[0],linewidth=2.5,label='Median')] + handles
            plt.legend(handles=handles,loc=8,ncol=len(handles),framealpha=1,frameon=False)
            plt.gca().set_axis_off()
            plt.tight_layout()
            if savefig is not None:
                plt.savefig(savefig)
                plt.close()
    def plotset_daily(self,savefig=None):
        fig,axs = plt.subplots(nrows=2,ncols=3)
        self.plot_daily_cycle_temp(axs[0][0])
        self.plot_daily_cycle_dwpc(axs[0][1])
        self.plot_daily_cycle_relh(axs[0][2])
        self.plot_daily_cycle_wspd(axs[1][0])
        self.plot_daily_cycle_vism(axs[1][1])
        self.plot_daily_cycle_ceiling(axs[1][2])
        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_wind(self,savefig=None):
        with plt.rc_context({'xtick.major.pad':-1}):
            fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(6.3/3*2,2.1),subplot_kw={'polar':True})
            self.plot_wind_compass_dir_freq(axs[0])
            self.plot_wind_compass_spd(axs[1])
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
        self.plot_frequency_percipitation(axs[1][0])
        self.plot_frequency_sigwx(axs[1][1])
        plt.tight_layout()
        if savefig is not None:
             plt.savefig(savefig)
             plt.close()
    def plotset_logo(self,savefig=None):
        if savefig is not None:
            shutil.copyfile(self.filepaths['logo'],savefig)
    def generate_monthly_plots(self):
        basefilters = copy.copy(self.filters)
        
        for month in self.frange(basefilters['month'],1,12):
            print(self.locales['monthabbr'][month],end=' ',flush=True)
            self.reset_filters()
            self.redo_filters(basefilters)
            self.filter_month(eq=month)
            
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
    def generate_monthly_pdf(self):
        print(f'Figures {self.station}:',end=' ',flush=True)
        self.generate_monthly_plots()
        print(f'TEX',end=' ',flush=True)
        self.generate_monthly_tex()
        print(f'PDF',end=' ',flush=True)
        self.generate_monthly_pdf_from_tex()
        print(f'Done.',flush=True)
            
        print('PDF can be found at "%s"'%os.path.join(self.filepaths['output'],self.station,self.station_data['icao'].upper()+"_monthly.pdf"))
        
    