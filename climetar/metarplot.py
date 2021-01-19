import json
import numbers
import re
import sys, os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
import metar
import quantities

class MetarPlotter(object):    
    @classmethod
    def define_colors(cls,color,default_alpha=None,dict_keys=None,default_color='k'):
        if default_alpha is None:
            default_alpha = [1]
        if hasattr(dict_keys,'__iter__') and not isinstance(dict_keys,list):
            dict_keys = list(dict_keys)
        
        # Input dict
        if isinstance(color,dict):
            if dict_keys is None:
                raise ValueError('Error 1: A list of colors is expected')
            keys = sorted(list(set(list(color.keys())+dict_keys)))
            colordict = {}
            for k in keys:
                if k in color:
                    colordict[k] = cls.define_colors(color[k],default_alpha)
                elif 'default' in color:
                    colordict[k] = cls.define_colors(color['default'],default_alpha)
                else:
                    colordict[k] = cls.define_colors(default_color,default_alpha)
            return colordict
        
        # Input list, output list
        elif isinstance(color,list) and dict_keys is None:
            if len(color)>=len(default_alpha):
                return color
            else:
                raise ValueError('Error 2: At least %d colors are nesseary for this plot. %d given'%(len(default_alpha),len(color)))
        
        # Input list, output dict
        elif isinstance(color,list) and dict_keys is not None:
            if len(colorlist)>=len(dict_keys):
                colordict = {}
                for i,k in enumerate(dict_keys):
                    colordict[k] = [colorlist[i]]
                for j in range(i,len(colorlist)):
                    colordict[f'other_{j:d}'] = [colorlist[j]]
                return colordict
            else:
                raise ValueError('Error 3: At least %d colors are nesseary for this plot. %d given'%(len(dict_keys),len(colorlist)))
        
        # Input single color, output list
        elif isinstance(color,(str,tuple)) and dict_keys is None:
            colorlist = []
            for da in default_alpha:
                colorlist.append(mpl.colors.to_rgba(color,da))
            return colorlist
        
        # Input single color, output dict
        elif isinstance(color,(str,tuple)) and dict_keys is not None:
            colordict = {}
            for k in dict_keys:
                colordict[k] = cls.define_colors(color,default_alpha)
            return colordict
        
        else:
            raise ValueError('Error 4: Strange inputs\n:'+'\n'.join([n+': '+repr(e) for n,e in {'color':color,'default_alpha':default_alpha,'dict_keys':dict_keys,'default_color':default_color}.items()]))
    
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

    def __init__(self,data,station):
        self.STATION = station
        self.df = data
        self.pdf = data.copy(deep=True)
        self.timelimits = [(None,None),(None,None),(None,None),(None,None)]
        self.theme = {}
    
    def use_theme(self,filename):
        with open(filename,'r') as fh:
            self.theme = json.load(fh)
    def process_colors(self,fnname,colors=None,edgecolor=None,linewidth=None,**kwargs):
        if fnname in self.theme:
            c,e,l = (
                colors if colors is not None else self.theme[fnname].get('colors','k'),
                edgecolor if edgecolor is not None else self.theme[fnname].get('edgecolor','none'),
                linewidth if linewidth is not None else self.theme[fnname].get('linewidth','.5')
            )
        else:
            c,e,l = (
                colors if colors is not None else 'k',
                edgecolor if edgecolor is not None else 'none',
                linewidth if linewidth is not None else .5,
            )
        c = self.define_colors(c,**kwargs)
        e = self.define_colors(e,default_alpha=[1])[0] if e not in ['none',None] else 'none'
        l = 0 if e=='none' else l
        return c,e,l
        
    def reset_dataset(self):
        self.pdf = self.df.copy(deep=True)
        self.wind_dir_compass = None
    def select_year(self,year=None,begin=None,end=None):
        if year is not None:
            self.pdf = self.pdf.loc[self.pdf.time.dt.year==year]
            begin, end = year, year
        elif begin is not None and end is not None:
            self.pdf = self.pdf.loc[self.pdf.time.dt.year.between(begin,end)]
        elif begin is not None and end is None:
            self.pdf = self.pdf.loc[self.pdf.time.dt.year>=begin]
        elif begin is None and end is not None:
            self.pdf = self.pdf.loc[self.pdf.time.dt.year<=end]
        self.timelimits[0] = begin, end
    def select_month(self,month=None,begin=None,end=None):
        if month is not None:
            self.pdf = self.pdf.loc[self.pdf.time.dt.month==month]
            begin, end = month, month
        elif begin is not None and end is not None:
            self.pdf = self.pdf.loc[self.pdf.time.dt.month.between(begin,end)]
        elif begin is not None and end is None:
            self.pdf = self.pdf.loc[self.pdf.time.dt.month>=begin]
        elif begin is None and end is not None:
            self.pdf = self.pdf.loc[self.pdf.time.dt.month<=end]
        self.timelimits[1] = begin, end
    def select_day(self,day=None,begin=None,end=None):
        if day is not None:
            self.pdf = self.pdf.loc[self.pdf.time.dt.day==day]
            begin, end = day, day
        elif begin is not None and end is not None:
            self.pdf = self.pdf.loc[self.pdf.time.dt.day.between(begin,end)]
        elif begin is not None and end is None:
            self.pdf = self.pdf.loc[self.pdf.time.dt.day>=begin]
        elif begin is None and end is not None:
            self.pdf = self.pdf.loc[self.pdf.time.dt.day<=end]
        self.timelimits[2] = begin, end
    def select_hour(self,hour=None,begin=None,end=None):
        if hour is not None:
            self.pdf = self.pdf.loc[self.pdf.time.dt.hour==hour]
            begin, end = hour, hour
        elif begin is not None and end is not None:
            self.pdf = self.pdf.loc[self.pdf.time.dt.hour.between(begin,end)]
        elif begin is not None and end is None:
            self.pdf = self.pdf.loc[self.pdf.time.dt.hour>=begin]
        elif begin is None and end is not None:
            self.pdf = self.pdf.loc[self.pdf.time.dt.hour<=end]
        self.timelimits[3] = begin, end
    
    def categorize_wind_dirs(self):
        self.pdf['wind_dir_catindex'] = np.digitize(self.pdf.wind_dir,[0,11.25,33.75,56.25,78.75,101.25,123.75,146.25,168.75,191.25,213.75,236.25,258.75,281.25,303.75,326.25,348.75,361])
        self.pdf['wind_dir_compass'] = np.take(['','N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW','N',''],self.pdf.wind_dir_catindex)
        self.pdf['wind_dir_catdeg'] = np.take([np.nan,0,22.5,45.0,67.5,90.0,112.5,135.0,157.5,180.0,202.5,225.0,247.5,270.0,292.5,315.0,337.5,0,np.nan],self.pdf.wind_dir_catindex)
        self.pdf['wind_dir_catrad'] = self.pdf.wind_dir_catdeg/180*np.pi
    
    def daily_cycle_hourly_plotter(
            self,ax,variable,unit,quantity,title='',
            ylim=None
            colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.process_colors(
            'daily_cycle_plotter',colors,edgecolor,linewidth,
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
        begin, end = self.timelimits[3]
        begin = 0 if begin is None else begin
        end = 24 if end is None else end
        ax.set_xlim(begin,end)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%dh"))
        ax.set_title(title)
    
    def daily_cycle_temp(self,ax,unit='°C',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.process_colors(
            'daily_cycle_temp',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        self.daily_cycle_hourly_plotter(ax,'temp',title='Temperature [%s]'%unit,
            unit=unit,quantity=quantities.Temperature,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def daily_cycle_dwpc(self,ax,unit='°C',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.process_colors(
            'daily_cycle_dwpc',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        self.daily_cycle_hourly_plotter(ax,'dwpt',title='Dew Point [%s]'%unit,
            unit=unit,quantity=quantities.Temperature,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def daily_cycle_relh(self,ax,unit='%',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.process_colors(
            'daily_cycle_relh',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        quantity = quantities.Fraction
        unit = quantity.find_unit(unit)
        self.daily_cycle_hourly_plotter(ax,'relh',title='Relative Humidity [%s]'%unit,
            ylim=(0,quantities.Fraction(1,'frac')[unit]),
            unit=unit,quantity=quantity,colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def daily_cycle_wspd(self,ax,unit='m/s',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.process_colors(
            'daily_cycle_wspd',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        self.daily_cycle_hourly_plotter(ax,'wind_spd',title='Wind speed [%s]'%unit,
            ylim=(0,None),unit=unit,quantity=quantities.Speed,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def daily_cycle_vism(self,ax,unit='km',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.process_colors(
            'daily_cycle_vism',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        quantity = quantities.Distance
        unit = quantity.find_unit(unit)
        self.daily_cycle_hourly_plotter(ax,'vis',title='Visibility [%s]'%unit,
            ylim=(0,quantities.Distance(10,'km')[unit]),unit=unit,quantity=quantity,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def daily_cycle_ceiling(self,ax,unit='ft',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.process_colors(
            'daily_cycle_ceiling',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        quantity = quantities.Height
        unit = quantity.find_unit(unit)
        self.daily_cycle_hourly_plotter(ax,'sky_ceiling',title='Cloud base [%s]'%unit,
            ylim=(0,quantities.Distance(5e3,'ft')[unit]),unit=unit,quantity=quantity,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    def daily_cycle_pres(self,ax,unit='hPa',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.process_colors(
            'daily_cycle_pres',colors,edgecolor,linewidth,
            default_alpha=[1,.6,.38,.15]
        )
        quantity = quantities.Pressure
        unit = quantity.find_unit(unit)
        self.daily_cycle_hourly_plotter(ax,'spPa',title='Surface Pressure [%s]'%unit,
            ylim=(quantities.Distance(950,'hPa')[unit],quantities.Distance(1050,'hPa')[unit]),unit=unit,quantity=quantity,
            colors=colors,edgecolor=edgecolor,linewidth=linewidth)
    
    def wind_compass_dir_freq(self,ax,unit='%',cat=True,colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.process_colors(
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
        data = gbo['validtime'].sum()/self.pdf.validtime.sum()
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
    def wind_compass_spd(self,ax,unit='kt',cat=True,colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.process_colors(
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

    def frequency_gust(self,ax,unit='m/s',freq_unit='%',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.process_colors(
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
    def frequency_cloud_type(self,ax,freq_unit='%',colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.process_colors(
            'frequency_cloud_type',colors,edgecolor,linewidth,
            default_alpha=[1],dict_keys=metar.Metar._cloud_cover_codes.keys()
        )
        
        freq_quantity = quantities.Fraction
        freq_unit = freq_quantity.find_unit(freq_unit)
        data = self.pdf.groupby('sky_cover')['validtime'].sum().reindex(metar.Metar._cloud_cover_codes.keys()).dropna()
        data = data / self.pdf.validtime.sum()
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
    def frequency_color(self,ax,freq_unit='%',label_replace_dict=None,colors=None,edgecolor=None,linewidth=None):
        colors, edgecolor, linewidth = self.process_colors(
            'frequency_color',colors,edgecolor,linewidth,
            default_alpha=[1],dict_keys=['BLU+','BLU','WHT','GRN','YLO','YLO1','YLO2','AMB','RED']
        )
        
        freq_quantity = quantities.Fraction
        freq_unit = freq_quantity.find_unit(freq_unit)
        data = self.pdf.groupby('color')['validtime'].sum().reindex(['BLU+','BLU','WHT','GRN','YLO','YLO1','YLO2','AMB','RED']).dropna()
        calculated = False
        if len(data)==0 and 'calc_color' in self.pdf.columns:
            data = self.pdf.groupby('calc_color')['validtime'].sum().reindex(['BLU+','BLU','WHT','GRN','YLO','YLO1','YLO2','AMB','RED']).dropna()
            calculated = True
        if len(data)==0:
            ax.bar(['BLU','WHT','GRN','YLO','AMB','RED'],[0,0,0,0,0,0],color='ḱ')
            ax.text(0.5,0.5,f"{self.STATION} does not publish\nNATO color states",
                    horizontalalignment='center',verticalalignment='center',
                    transform=ax.transAxes,c='k')
            ax.set_ylim(0,10)
        else:
            data = data / self.pdf.validtime.sum()
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
    def frequency_percipitation(self,ax,freq_unit='%',colors=None,edgecolor=None,linewidth=None):
        preciptypes = ['RA','DZ','SN','IC','PL','GR','GS','UP','FZRA','FZDZ','FZFG']
        intensitytypes = ['+','','-','VC']
                
        colors, edgecolor, linewidth = self.process_colors(
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
                    if heights[c]>2.5:
                        ax.text(c,txtheights[c],intensitytypes[i],
                                horizontalalignment='center',verticalalignment='center',
                                color='k',fontsize=16)
                
            data = precipdf.sum()
            thx = data.max()*norm*0.05
            for i in range(len(data)):
                ax.text(i,data.iloc[i]*norm+thx,"%3.1f %s"%(data.iloc[i]*norm,freq_unit),c='k',ha='center',va='bottom')
            ax.set_ylim(0,((data.max()*norm+thx*3)//2+1)*2)
            
            legend_elem = []
            legend_colors = self.define_colors('k',default_alpha=[1,.7,.5,.25])
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
    def frequency_sigwx(self,ax,freq_unit='%',colors=None,edgecolor=None,linewidth=None):
        sigwxtypes = ['TS','FZ','SH','FG','VA','BR','HZ','DU','FU','SA','PY','SQ','PO','DS','SS','FC']
        
        colors, edgecolor, linewidth = self.process_colors(
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

class MetarPlotterSets(MetarPlotter):
    def daily_6window_default(self):
        fig, axs = plt.subplots(nrows=2,ncols=3)
        self.daily_cycle_temp(axs[0][0])
        self.daily_cycle_dwpc(axs[0][1])
        self.daily_cycle_relh(axs[0][2])
        self.daily_cycle_wspd(axs[1][0],unit='kt')
        self.daily_cycle_vism(axs[1][1])
        self.daily_cycle_ceiling(axs[1][2])
        plt.tight_layout()
        return fig
    def wind_2window_default(self):
        with plt.rc_context({'xtick.major.pad':-10}):
            fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(6.3/3*2, 2.1),subplot_kw={'polar':True})
            self.wind_compass_dir_freq(axs[0])
            self.wind_compass_spd(axs[1],unit='kt')
            plt.tight_layout()
        return fig
    def gust_1window_default(self):
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(6.3/3, 2.1))
        self.frequency_gust(ax)
        plt.tight_layout()
        return fig
    def frequency_4window_default(self):
        fig, axs = plt.subplots(nrows=2,ncols=2)
        self.frequency_cloud_type(axs[0][0])
        self.frequency_color(axs[0][1])
        self.frequency_percipitation(axs[1][0])
        self.frequency_sigwx(axs[1][1])
        plt.tight_layout()
        return fig
