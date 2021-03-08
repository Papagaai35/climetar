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
from .svgpath2mpl import parse_path
from .metartheme import MetarTheme
    
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
            'natural_earth': settings.get('natural_earth','./resources/'),
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
    
    @classmethod
    def prepare_maps(cls):
        global cartopy, xr, rasterio
        import cartopy
        import xarray as xr
        import rasterio    
    def load_map_raster(self,name):
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
            hires_available = bool(MapPlotHelper.search_files(self.filepaths['natural_earth'],'NE1_HR_LC_SR_W_DR',['tif','tiff','zip']))
            if zoom < 1 or not hires_available:
                imgda = self.load_map_raster('NE1_LR_LC_SR_W_DR')
                step = int(np.clip(np.ceil(.5/zoom),1,None))
                img_extent, img_da = MapPlotHelper.slice_img(img_extent,imgda,step)
            else:
                imgda = self.load_map_raster('NE1_HR_LC_SR_W_DR')
                img_extent, img_da = MapPlotHelper.slice_img(img_extent,imgda,1 if zoom>=1.33 else 2)
            ax.imshow(img_da.values,
                origin='upper',
                transform=trans,
                extent=img_extent,
                zorder=-2)
            hires_available = bool(MapPlotHelper.search_files(self.filepaths['natural_earth'],'ne_10m_admin_0_countries',['shp','zip']))
            if zoom < 1 or not hires_available:
                shp = self.load_map_shape('ne_50m_admin_0_countries')
            else:
                shp = self.load_map_shape('ne_10m_admin_0_countries')
            sf = cartopy.feature.ShapelyFeature(shp.geometries(),trans,
                facecolor='none',edgecolor='#666666',linewidth=.75,zorder=-1)
            ax.add_feature(sf,zorder=-1)
        
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
    def _plot_dh_cycle_hoursteps(
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
    def plot_dh_cycle_temp(self,ax,unit='°C',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'daily_cycle_temp',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        self._plot_dh_cycle_hoursteps(ax,'temp',title='Temperature [%s]'%unit,
            unit=unit,quantity=quantities.Temperature,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def plot_dh_cycle_dwpc(self,ax,unit='°C',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'daily_cycle_dwpc',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        self._plot_dh_cycle_hoursteps(ax,'dwpt',title='Dew Point [%s]'%unit,
            unit=unit,quantity=quantities.Temperature,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def plot_dh_cycle_relh(self,ax,unit='%',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'daily_cycle_relh',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        quantity = quantities.Fraction
        unit = quantity.find_unit(unit)
        self._plot_dh_cycle_hoursteps(ax,'relh',title='Relative Humidity [%s]'%unit,
            ylim=(0,quantities.Fraction(1,'frac')[unit]),
            unit=unit,quantity=quantity,colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def plot_dh_cycle_wspd(self,ax,unit='kt',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'daily_cycle_wspd',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        self._plot_dh_cycle_hoursteps(ax,'wind_spd',title='Wind speed [%s]'%unit,
            ylim=(0,None),unit=unit,quantity=quantities.Speed,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def plot_dh_cycle_vism(self,ax,unit='km',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'daily_cycle_vism',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        quantity = quantities.Distance
        unit = quantity.find_unit(unit)
        self._plot_dh_cycle_hoursteps(ax,'vis',title='Visibility [%s]'%unit,
            ylim=(0,quantities.Distance(10,'km')[unit]),unit=unit,quantity=quantity,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def plot_dh_cycle_ceiling(self,ax,unit='ft',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'daily_cycle_ceiling',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        quantity = quantities.Height
        unit = quantity.find_unit(unit)
        self._plot_dh_cycle_hoursteps(ax,'sky_ceiling',title='Cloud base [%s]'%unit,
            ylim=(0,quantities.Distance(5e3,'ft')[unit]),unit=unit,quantity=quantity,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def plot_dh_cycle_pres(self,ax,unit='hPa',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'daily_cycle_pres',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        quantity = quantities.Pressure
        unit = quantity.find_unit(unit)
        self._plot_dh_cycle_hoursteps(ax,'spPa',title='Surface Pressure [%s]'%unit,
            ylim=(quantities.Distance(950,'hPa')[unit],quantities.Distance(1050,'hPa')[unit]),unit=unit,quantity=quantity,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
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
        self.plot_frequency_percipitation(axs[1][0])
        self.plot_frequency_sigwx(axs[1][1])
        plt.tight_layout()
        if savefig is not None:
             plt.savefig(savefig)
             plt.close()
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
    
    # Averages per year (yearly cycles) Plots
    def plot_ym_cycle_tmin_tmax(self,ax,unit='°C',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'monthly_cycle_tmin_tmax',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15],dict_keys=['tmin','tmax'],
        )
        quantity=quantities.Temperature
        dailydata = self.pdf.groupby(self.pdf.time.dt.date).agg(tmax=('temp','max'),tmin=('temp','min')).reset_index()
        dailydata['time'] = pd.to_datetime(dailydata.time,dayfirst=True)
        gbo = dailydata.groupby(dailydata.time.dt.month)
        for var in ['tmin','tmax']:
            data = gbo[var].quantile([.01,.05,.25,.5,.75,.95,.99])
            data = self.convert_unit(quantity.units[unit],data).unstack()
            data = data.append(data.loc[1].rename(13))
            data = data.append(data.loc[12].rename(0))
            data = data.sort_index()
            ax.plot(data[.5],color=colors[var][0])
            ax.fill_between(x=data.index,y1=data[.25],y2=data[.75],zorder=-1,color=colors[var][1])
            ax.fill_between(x=data.index,y1=data[.05],y2=data[.25],zorder=-1,color=colors[var][2])
            ax.fill_between(x=data.index,y1=data[.75],y2=data[.95],zorder=-1,color=colors[var][2])
            ax.fill_between(x=data.index,y1=data[.01],y2=data[.05],zorder=-1,color=colors[var][3])
            ax.fill_between(x=data.index,y1=data[.95],y2=data[.99],zorder=-1,color=colors[var][3])
        xticks = [3,6,9,12]
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.filters['month'][0:2]
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        ax.set_xlim(begin-.5,end+.5)
        ax.set_title('Temperature (min/max) [%s]'%(unit))
    def plot_ym_cycle_wcet(self,ax,unit='°C',ylim=None,colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'monthly_cycle_wcet',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15],dict_keys=['data','limits'],deep_dict_keys={'limits':[]},
        )
        quantity = quantities.Temperature 
        quantityWind = quantities.Speed
        
        limits = dict(sorted(map(lambda x: (float(x),x),colors['limits'].keys())))
        for f1,s in limits.items():
            if f1==max(limits.keys()):
                break
            f2 = list(filter(lambda x: x>f1,limits.keys()))[0]
            ax.fill_between(x=[-1,14],y1=f1,y2=f2,zorder=-5,alpha=.6,color=colors['limits'][s][0][:3])
            ax.axhline(f2,zorder=-4,alpha=.8,color=colors['limits'][s][0][:3],linewidth=linewidth*.5)
        
        t2m = self.convert_unit(quantity.units['°C'],self.pdf.temp)
        wind = (self.convert_unit(quantityWind.units['km/h'],self.pdf.wind_spd))**0.16
        self.pdf['wcet'] = 13.12 + 0.6215 * t2m - 11.37 * wind + 0.3965 * t2m * wind
        gbo = self.pdf.dropna(subset=['wcet']).groupby(self.pdf.time.dt.month)
        
        data = gbo['wcet'].quantile([.01,.05,.25,.5,.75,.95,.99])
        data = self.convert_unit(quantity.units[unit],data).unstack()
        data = data.append(data.loc[1].rename(13))
        data = data.append(data.loc[12].rename(0))
        data = data.sort_index()
        ax.plot(data[.5],color=colors['data'][0])
        ax.fill_between(x=data.index,y1=data[.25],y2=data[.75],zorder=-1,color=colors['data'][1])
        ax.fill_between(x=data.index,y1=data[.05],y2=data[.25],zorder=-1,color=colors['data'][2])
        ax.fill_between(x=data.index,y1=data[.75],y2=data[.95],zorder=-1,color=colors['data'][2])
        ax.fill_between(x=data.index,y1=data[.01],y2=data[.05],zorder=-1,color=colors['data'][3])
        ax.fill_between(x=data.index,y1=data[.95],y2=data[.99],zorder=-1,color=colors['data'][3])
        
        xticks = [3,6,9,12]
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.filters['month'][0:2]
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(np.floor(data[.01].min()/5)*5,
                        np.ceil(data[.99].max()/5)*5)
        ax.set_xlim(begin-.5,end+.5)
        ax.set_title('Wind Chill [%s]'%unit)
    def plot_ym_cycle_wbgt_simplified(self,ax,unit='°C',ylim=None,colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'monthly_cycle_wbgt',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15],dict_keys=['data','limits'],deep_dict_keys={'limits':[]},
        )
        quantity = quantities.Temperature 
        
        limits = dict(sorted(map(lambda x: (float(x),x),colors['limits'].keys())))
        for f1,s in limits.items():
            if f1==max(limits.keys()):
                break
            f2 = list(filter(lambda x: x>f1,limits.keys()))[0]
            ax.fill_between(x=[-1,14],y1=f1,y2=f2,zorder=-5,alpha=.38,color=colors['limits'][s][0][:3])
            ax.axhline(f2,zorder=-4,alpha=.1,color=colors['limits'][s][0][:3],linewidth=linewidth*.5)
        
        t2m = self.convert_unit(quantity.units['°C'],self.pdf.temp)
        d2m = self.convert_unit(quantity.units['°C'],self.pdf.dwpt)
        vp = 6.112 * np.exp((17.67*d2m)/(d2m+243.5))
        self.pdf['wbgt'] = 0.657 * t2m + 0.393 * vp + 3.94
        
        dailydata = self.pdf.groupby(self.pdf.time.dt.date).agg(wbgt=('wbgt','max')).reset_index()
        dailydata['time'] = pd.to_datetime(dailydata.time,dayfirst=True)
        gbo = dailydata.groupby(dailydata.time.dt.month)       
        data = gbo['wbgt'].quantile([.01,.05,.25,.5,.75,.95,.99])
        data = self.convert_unit(quantity.units[unit],data).unstack()
        data = data.append(data.loc[1].rename(13))
        data = data.append(data.loc[12].rename(0))
        data = data.sort_index()
        ax.plot(data[.5],color=colors['data'][0],linewidth=linewidth*3)
        ax.fill_between(x=data.index,y1=data[.25],y2=data[.75],zorder=-1,color=colors['data'][1],edgecolor=edgecolor,linewidth=linewidth)
        ax.fill_between(x=data.index,y1=data[.05],y2=data[.25],zorder=-1,color=colors['data'][2],edgecolor=edgecolor,linewidth=linewidth)
        ax.fill_between(x=data.index,y1=data[.75],y2=data[.95],zorder=-1,color=colors['data'][2],edgecolor=edgecolor,linewidth=linewidth)
        ax.fill_between(x=data.index,y1=data[.01],y2=data[.05],zorder=-1,color=colors['data'][3],edgecolor=edgecolor,linewidth=linewidth)
        ax.fill_between(x=data.index,y1=data[.95],y2=data[.99],zorder=-1,color=colors['data'][3],edgecolor=edgecolor,linewidth=linewidth)
        
        xticks = [3,6,9,12]
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.filters['month'][0:2]
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(np.floor(data[.01].min()/5)*5,
                        np.ceil(data[.99].max()/5)*5)
        ax.set_xlim(begin-.5,end+.5)
        ax.set_title('WBGT [%s]'%unit)
    def plot_ym_cycle_relh(self,ax,unit='%',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'monthly_cycle_relh',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        quantity = quantities.Fraction
        
        gbo = self.pdf.dropna(subset=['relh']).groupby(self.pdf.time.dt.month)
        
        data = gbo['relh'].quantile([.01,.05,.25,.5,.75,.95,.99])
        data = self.convert_unit(quantity.units[unit],data).unstack()
        data = data.append(data.loc[1].rename(13))
        data = data.append(data.loc[12].rename(0))
        data = data.sort_index()
        ax.plot(data[.5],color=colors[0])
        ax.fill_between(x=data.index,y1=data[.25],y2=data[.75],zorder=-1,color=colors[1])
        ax.fill_between(x=data.index,y1=data[.05],y2=data[.25],zorder=-1,color=colors[2])
        ax.fill_between(x=data.index,y1=data[.75],y2=data[.95],zorder=-1,color=colors[2])
        ax.fill_between(x=data.index,y1=data[.01],y2=data[.05],zorder=-1,color=colors[3])
        ax.fill_between(x=data.index,y1=data[.95],y2=data[.99],zorder=-1,color=colors[3])
        
        xticks = [3,6,9,12]
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.filters['month'][0:2]
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        ax.set_xlim(begin-.5,end+.5)
        ax.set_ylim(0,100)
        ax.set_title('Relative Humidity [%s]'%unit)
    def plot_ym_cycle_vism(self,ax,unit='km',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'monthly_cycle_vism',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        quantity = quantities.Distance
        
        gbo = self.pdf.dropna(subset=['vis']).groupby(self.pdf.time.dt.month)
        
        data = gbo['vis'].quantile([.01,.05,.25,.5,.75,.95,.99])
        data = self.convert_unit(quantity.units[unit],data).unstack()
        data = data.append(data.loc[1].rename(13))
        data = data.append(data.loc[12].rename(0))
        data = data.sort_index()
        ax.plot(data[.5],color=colors[0])
        ax.fill_between(x=data.index,y1=data[.25],y2=data[.75],zorder=-1,color=colors[1])
        ax.fill_between(x=data.index,y1=data[.05],y2=data[.25],zorder=-1,color=colors[2])
        ax.fill_between(x=data.index,y1=data[.75],y2=data[.95],zorder=-1,color=colors[2])
        ax.fill_between(x=data.index,y1=data[.01],y2=data[.05],zorder=-1,color=colors[3])
        ax.fill_between(x=data.index,y1=data[.95],y2=data[.99],zorder=-1,color=colors[3])
        
        xticks = [3,6,9,12]
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.filters['month'][0:2]
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        ax.set_xlim(begin-.5,end+.5)
        ax.set_ylim(0,quantities.Distance(10,'km')[unit])
        ax.set_title('Visibility [%s]'%unit)
    def plot_ym_cycle_ceiling(self,ax,unit='ft',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'monthly_cycle_ceiling',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        quantity = quantities.Height
        
        gbo = self.pdf.dropna(subset=['sky_ceiling']).groupby(self.pdf.time.dt.month)
        
        data = gbo['sky_ceiling'].quantile([.01,.05,.25,.5,.75,.95,.99])
        data = self.convert_unit(quantity.units[unit],data).unstack()
        data = data.append(data.loc[1].rename(13))
        data = data.append(data.loc[12].rename(0))
        data = data.sort_index()
        ax.plot(data[.5],color=colors[0])
        ax.fill_between(x=data.index,y1=data[.25],y2=data[.75],zorder=-1,color=colors[1])
        ax.fill_between(x=data.index,y1=data[.05],y2=data[.25],zorder=-1,color=colors[2])
        ax.fill_between(x=data.index,y1=data[.75],y2=data[.95],zorder=-1,color=colors[2])
        ax.fill_between(x=data.index,y1=data[.01],y2=data[.05],zorder=-1,color=colors[3])
        ax.fill_between(x=data.index,y1=data[.95],y2=data[.99],zorder=-1,color=colors[3])
        
        xticks = [3,6,9,12]
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.filters['month'][0:2]
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        ax.set_xlim(begin-.5,end+.5)
        ax.set_ylim(0,quantities.Distance(5e3,'ft')[unit])
        ax.set_title('Cloud base [%s]'%unit)
    def plot_ym_raindays(self,ax,colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'frequency_gust',colors,edgecolor,linewidth,
            default_alpha=[1]
        )
        color = colors[0]
        
        preciptypes = ['RA','DZ','SN','IC','PL','GR','GS','UP']
        intensitytypes = ['+','','-','VC']
        self.pdf['percipitating'] = False
        for precip in preciptypes:
            for intensity in intensitytypes:
                percipitating = self.pdf.wx.str.contains(re.escape(intensity)+'(?:[A-Z]{2})*?'+re.escape(precip)+'(?:[A-Z]{2})*?')
                self.pdf['percipitating'] = (self.pdf['percipitating'].values) | (percipitating.fillna(False))
        data = self.pdf.groupby(self.pdf.time.dt.date).agg({'percipitating':any}).iloc[:,0].reset_index()
        data['time'] = pd.to_datetime(data.time,dayfirst=True)
        data['percipitating'] = data.percipitating.astype(int)
        data = data.groupby([data.time.dt.year,data.time.dt.month]).sum().unstack().mean().unstack().T
        index,values = data.index.values, data.values[:,0]
        
        ax.bar(index, values, color=color,edgecolor=edgecolor,linewidth=linewidth)
        ax.set_ylabel('Days with precipitation')
        ax.set_title('Precipitation')
        xticks = [3,6,9,12]
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.filters['month'][0:2]
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        ax.set_xlim(begin-.5,end+.5)
        ax.set_ylim(0,31)
    def plot_ym_cycle_cloud_type(self,ax,freq_unit='%',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.theme.cel(
            'frequency_cloud_type',colors,edgecolor,linewidth,
            default_alpha=[1],dict_keys=metar.Metar._cloud_cover_codes.keys()
        )
        freq_quantity = quantities.Fraction
        freq_unit = freq_quantity.find_unit(freq_unit)
        
        data = self.pdf.groupby([self.pdf.time.dt.month,self.pdf.sky_cover])['minutes_valid'].sum().unstack()
        data = np.divide(data,data.sum(axis=1)[:,None]).reindex(metar.Metar._cloud_cover_codes.keys(),axis=1)
        clear_index = data[['SKC','NCD','CLR','NSC']].sum().idxmax()
        data[clear_index] = data[['SKC','NCD','CLR','NSC']].sum(axis=1)
        data = data[reversed([clear_index,'FEW','SCT','BKN','OVC','VV'])]
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
                    color=colors[c][0],
                    edgecolor=edgecolor,
                    linewidth=linewidth))
        
        xticks = [3,6,9,12]
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.filters['month'][0:2]
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        ax.set_xlim(begin-.5,end+.5)
        ax.set_ylim(0,quantities.Fraction(100,'%')[freq_unit])
        ax.set_title('Cloud cover [%s]'%freq_unit)
        
        ax.legend(handles,data.columns,loc=9,ncol=3,bbox_to_anchor=(.5,-.15),
                  labelspacing=.15,handlelength=1.5,handletextpad=0.4,fontsize='small',framealpha=0)
    def plot_ym_cycle_color(self,ax,freq_unit='%',ylim=None,colors=None,edgecolor=None,linewidth=None):
        colorcodes = ['BLU+','BLU','WHT','GRN','YLO','YLO1','YLO2','AMB','RED']
        colors, edgecolor, linewidth = self.theme.cel(
            'frequency_color',colors,edgecolor,linewidth,
            default_alpha=[1],dict_keys=colorcodes
        )
        freq_quantity = quantities.Fraction
        freq_unit = freq_quantity.find_unit(freq_unit)
        
        data = self.pdf.groupby([self.pdf.time.dt.month,self.pdf.calc_color])['minutes_valid'].sum().unstack()
        data = np.divide(data,data.sum(axis=1)[:,None])
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
                    color=colors[c][0],
                    edgecolor=edgecolor,
                    linewidth=linewidth))
        
        xticks = [3,6,9,12]
        ax.set_xticks(xticks);
        ax.set_xticklabels([self.locales['monthabbr'][m] for m in xticks])
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],minor=True);
        begin, end = self.filters['month'][0:2]
        begin = 1 if begin is None else begin
        end = 12 if end is None else end
        ax.set_xlim(begin-.5,end+.5)
        if ylim is None:
            n = bottom.iloc[:,-1].max()
            ax.set_ylim(0,np.ceil(n/10**np.floor(np.log10(n)))*(10**np.floor(np.log10(n))))
        else:
            ax.set_ylim(*ylim)
        ax.set_title('NATO Color State [%s]'%freq_unit)
        
        ax.legend(handles,data.columns,loc=9,ncol=4,bbox_to_anchor=(.5,-.15),
                  labelspacing=.15,handlelength=1.5,handletextpad=0.4,fontsize='small',framealpha=0)
    
    def plotset_ymwide_tmin_tmax(self,savefig=None):
        fig = plt.figure(figsize=(6.3,2.1))
        width = .86
        ax = fig.add_axes([.01,.01,width,.98])
        self.plot_ym_cycle_tmin_tmax(ax)
        
        ax2 = fig.add_axes([width+.02,.01,1-width-.03,.98])
        colors, edgecolor, linewidth = self.theme.cel(
            'monthly_cycle_tmin_tmax',default_alpha=[1,.6,.38,.15],dict_keys=['tmin','tmax'])
        legends = [
            mplLegendSubheading('Maximum'),
            mpl.lines.Line2D([],[],color=colors['tmax'][0], label='Median'),
            mpl.patches.Patch(facecolor=colors['tmax'][1], label='50% ci'),
            mpl.patches.Patch(facecolor=colors['tmax'][2], label='90% ci'),
            mpl.patches.Patch(facecolor=colors['tmax'][3], label='99% ci'),
            mplLegendSubheading('Minimum'),
            mpl.lines.Line2D([],[],color=colors['tmin'][0], label='Median'),
            mpl.patches.Patch(facecolor=colors['tmin'][1], label='50% ci'),
            mpl.patches.Patch(facecolor=colors['tmin'][2], label='90% ci'),
            mpl.patches.Patch(facecolor=colors['tmin'][3], label='99% ci'),
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
        width = .85
        ax = fig.add_axes([.01,.01,width,.98])
        self.plot_ym_cycle_wcet(ax)
        
        ax2 = fig.add_axes([width+.02,.01,1-width-.03,.98])
        colors, edgecolor, linewidth = self.theme.cel(
            'monthly_cycle_wcet',
            default_alpha=[1,.6,.38,.15],dict_keys=['data','limits'],deep_dict_keys={'limits':[]})
          
        
        limit_legend = []
        limits = dict(sorted(map(lambda x: (float(x),x),colors['limits'].keys())))
        for f1,s in limits.items():
            if f1==max(limits.keys()):
                break
            f2 = list(filter(lambda x: x>f1,limits.keys()))[0]
            if abs(f1)<299 and abs(f2)<299:
                f1,f2 = (f1,f2) if abs(f1)<abs(f2) else (f2,f1)
            label = (f'< {f2:.0f}' if f1<-299 else (
                     f'> {f1:.0f}' if f2>299 else (
                     f'{f1:.0f} to {f2:.0f}')))
            color = tuple(list(colors['limits'][s][0][:3])+[.6])
            limit_legend.append(mpl.patches.Patch(facecolor=color, label=label))
        
        legends = [
            mpl.lines.Line2D([],[],color=colors['data'][0], label='Median'),
            mplLegendSpacer(),
            mplLegendSubheading('Confidence\n  intervals',0),
            #mplLegendSubheading('intervals',2),
            mpl.patches.Patch(facecolor=colors['data'][1], label='50%'),
            mpl.patches.Patch(facecolor=colors['data'][2], label='90%'),
            mpl.patches.Patch(facecolor=colors['data'][3], label='99%'),
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
    def plotset_ymwide_wbgt(self,savefig=None):
        fig = plt.figure(figsize=(6.3,2.1))
        width = .85
        ax = fig.add_axes([.01,.01,width,.98])
        self.plot_ym_cycle_wbgt_simplified(ax)
        
        ax2 = fig.add_axes([width+.02,.01,1-width-.03,.98])
        colors, edgecolor, linewidth = self.theme.cel(
            'monthly_cycle_wbgt',
            default_alpha=[1,.6,.38,.15],dict_keys=['data','limits'],deep_dict_keys={'limits':[]})
          
        
        limit_legend = []
        limits = dict(sorted(map(lambda x: (float(x),x),colors['limits'].keys())))
        for f1,s in limits.items():
            if f1==max(limits.keys()):
                break
            f2 = list(filter(lambda x: x>f1,limits.keys()))[0]
            label = (f'< {f2:.0f}' if f1<-299 else (
                     f'> {f1:.0f}' if f2>299 else (
                     f'{f1:.0f} to {f2:.0f}')))
            color = tuple(list(colors['limits'][s][0][:3])+[.6])
            limit_legend.append(mpl.patches.Patch(facecolor=color, label=label))
        
        legends = [
            mpl.lines.Line2D([],[],color=colors['data'][0], label='Median'),
            mpl.patches.Patch(facecolor=colors['data'][1], label='50% ci'),
            mpl.patches.Patch(facecolor=colors['data'][2], label='90% ci'),
            mpl.patches.Patch(facecolor=colors['data'][3], label='99% ci'),
            mplLegendSubheading('Limits'),
        ]+list(reversed(limit_legend))
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
        self.plot_ym_cycle_cloud_type(axs[0])
        self.plot_ym_cycle_color(axs[1])
        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_raindays(self,savefig=None):
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6.3/3,2.1))
        self.plot_ym_raindays(ax)
        plt.tight_layout()
        if savefig is not None:
             plt.savefig(savefig)
             plt.close()
    def generate_yearly_plots(self,savefig=True):
        if savefig:
            dirname_figs = os.path.join(self.filepaths['output'],self.station,'fig')
            if not os.path.exists(dirname_figs):
                pathlib.Path(dirname_figs).mkdir(parents=True, exist_ok=True)
        self.plotset_map(savefig=os.path.join(dirname_figs,'Y0.png') if savefig else None)
        self.plotset_monthly_cycle(savefig=os.path.join(dirname_figs,'Y1.png') if savefig else None)
        self.plotset_wind(savefig=os.path.join(dirname_figs,'Y2.png') if savefig else None)
        self.plotset_raindays(savefig=os.path.join(dirname_figs,'Y3.png') if savefig else None)
        self.plotset_monthly_stacks(savefig=os.path.join(dirname_figs,'Y4.png') if savefig else None)
    
    # Non-meteo plots
    def plotset_logo(self,savefig=None):
        if savefig is not None:
            shutil.copyfile(self.filepaths['logo'],savefig)
    def plotset_map(self,stations=None,zoom=1,clat=None,clon=None,figsize=None,savefig=None):
        self.prepare_maps()
        if stations is None:
            stations = []
        if self.station not in stations and self.station is not None:
            stations = [self.station] + stations
        if clat is None or clon is None:
            station_data = self.station_repo['stations'][stations[0]]
            clat,clon = station_data['latitude'], station_data['longitude']
        extent = [clat-(15/zoom),clat+(15/zoom),clon-(10/zoom),clon+(10/zoom)]
        proj = cartopy.crs.NearsidePerspective(
            central_longitude=clat,
            central_latitude=clon,
            satellite_height=35785831
        )
        trans = cartopy.crs.PlateCarree()

        fig,ax = plt.subplots(1,1,figsize=figsize,subplot_kw={'projection':proj})

        for s in stations:
            station_data = self.station_repo['stations'][s]
            lat,lon = station_data['latitude'], station_data['longitude']
            if extent[0] <= lat <= extent[1] and extent[2] <= lon <= extent[3]:
                plane_str = """m 37.398882,331.5553 195.564518,-53.33707 81.92539,81.92539 c 18.40599,18.40599 58.40702,30.50459 72.90271,27.64788 2.85671,-14.49569 -9.24189,-54.49672 -27.64788,-72.90271 L 278.21823,232.9634 331.5553,37.39888 305.29335,11.13693 216.40296,171.14812 133.86945,88.61462 142.35474,29.21765 113.13708,0 73.058272,73.05827 0,113.13708 l 29.217652,29.21766 59.39697,-8.48529 82.533498,82.53351 -160.011188,88.89039 26.26195,26.26195"""
                plane_path = parse_path(plane_str)
                plane_path.vertices -= plane_path.vertices.mean(axis=0)
                ax.scatter(lat,lon,transform=trans,
                           s=36*2,marker=plane_path,c='k',zorder=2)
                ax.annotate(station_data['icao'],xy=(lat,lon),xytext=(lat+(.4/zoom),lon+(.3/zoom)),
                            xycoords=trans._as_mpl_transform(ax),zorder=2)
            else:
                latr, lonr, clatr, clonr = np.deg2rad(np.array([lat,lon,clat,clon]))
                x = np.cos(latr)*np.sin(clonr-lonr)
                y = np.cos(clatr)*np.sin(latr)-np.sin(clatr)*np.cos(latr)*np.cos(clonr-lonr)
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
                ax.annotate(station_data['icao'],
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
        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    
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
        
class MapPlotHelper(object):
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
                    print(f'Extracting "{zip_path}" to "{zipextract_path}"...')
                    zipfh.extractall(zipextract_path)
                else:
                    raise ValueError(f'Zipfile "{zip_path}" does not contain nessesary files ({name}.{extstr})')
            data_path = cls.search_files(basepath,name,exts)
            if data_path:
                return data_path
            raise ValueError(f'Zipfile extracted ({zip_path}), but could not find datafiles ({name}.{extstr})')
        exts = exts+['zip'] if isinstance(exts,list) else [exts,'zip']
        extstr = '['+','.join(exts)+']'
        raise ValueError(f'Could not find any file ({name}.{extstr})')
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

class mplLegendSubheading(object):
    def __init__(self,s,level=0):
        self.s = s
        self.level = level
        self.len = int(max([len(sp) for sp in self.s.split('\n')])-4)
    def get_label(self):
        return ('' if self.level>1 else '\n')+('\u2007'*self.len)
class mplLegendSubheadingHandler(object):
    def legend_artist(self,legend,orig_handle,fontsize,handlebox):
        x0,y0,width,height = handlebox.xdescent, handlebox.ydescent, handlebox.width, handlebox.height
        fontweight = 'normal' if orig_handle.level>0 else 'bold'
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