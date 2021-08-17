import logging
_log = logging.getLogger(__name__)

import collections
import os
import pathlib

import numpy as np
import pandas as pd
import PIL.Image
from PIL.TiffImagePlugin import ImageFileDirectory_v1
import matplotlib as mpl
import matplotlib.pyplot as plt

from .svgpath2mpl import parse_path
from .metarplot import MapPlotHelper, CountryFinder, mplLegendSubheading, mplLegendSubheadingHandler
from . import StationRepo


class StationMapper(object):
    @classmethod
    def import_modules(cls):
        global cartopy, shapely
        import cartopy
        import shapely
        import shapely.ops
        os.environ['REQUESTS_CA_BUNDLE'] = 'resources/mindef-ca-temp.cer' #ivm HTTPS Mapproxy
        
    @classmethod
    def marker_plane(cls):
        plane_str = """m 37.398882,331.5553 195.564518,-53.33707 81.92539,81.92539 c 18.40599,18.40599 58.40702,30.50459 72.90271,27.64788 2.85671,-14.49569 -9.24189,-54.49672 -27.64788,-72.90271 L 278.21823,232.9634 331.5553,37.39888 305.29335,11.13693 216.40296,171.14812 133.86945,88.61462 142.35474,29.21765 113.13708,0 73.058272,73.05827 0,113.13708 l 29.217652,29.21766 59.39697,-8.48529 82.533498,82.53351 -160.011188,88.89039 26.26195,26.26195"""
        plane_path = parse_path(plane_str)
        plane_path.vertices -= plane_path.vertices.mean(axis=0)
        return plane_path
    
    def __init__(self, **settings):
        self.proj = None
        self.trans = None
        
        self.mapname = ""
        self.stations_on_map = []
        self.center_lon = None
        self.center_lat = None
        self.focus_extent = None, None, None, None
        self.zoom = None
        
        self.style = settings.get('style','./resources/climetar.mplstyle')    
        self.ne_files = settings.get('natural_earth','./resources/')
        self.kg_files = settings.get('koppen','./resources/')
        self.output = settings.get('folder_output','./results/maps/')
        self.save_name = ""
        
        try:
            self.import_modules()
        except ModuleNotFoundError as err:
            _log.error(repr(err)+'\nInstalleer opnieuw de packages via "00. Instaleren & Introductie.ipynb"',exc_info=err)
            _log.info('Kaart niet geplot...')
            return
        
        self.trans = cartopy.crs.PlateCarree()
        self.station_repo = StationRepo()
        self.countryfinder = CountryFinder(natural_earth=self.ne_files)
        self.koppen = None
        if os.path.isfile(self.style):
            plt.style.use(self.style)
    
    # Setting focus and stations before creating the plots
    def focus_on(self,*args):
        self.focus_map_on(args)
    def focus_map(self,*args):
        self.focus_map_on(args)
    def focus_map_on(self,focus_points=None):
        if focus_points is None:
            focus_points = self.stations_on_map
        if len(focus_points)==0:
            self.mapname = ''
            self.zoom = .1
            self.proj = cartopy.crs.Robinson()
            return None
        if isinstance(focus_points,str):
            focus_points = [focus_points]
        
        focus_objs = []
        for fp in focus_points:
            fpobj = None
            if len(fp) in [2,3] and fp in self.countryfinder:
                geom,attrs = self.countryfinder.get_country_by_code(fp)
                name = attrs['NAME']
                focus_objs.append((geom,attrs,'country'))
                _log.debug(f'Focus on country {fp:s} {name:s}')
            elif fp in self.station_repo:
                s, sd = self.station_repo.get_station(fp)
                shp = shapely.geometry.shape(self.station_repo.stations[s]['geometry'])
                focus_objs.append((shp,sd,'station'))
                _log.debug(f'Focus on station {fp:s}')
            elif fp in self.station_repo.networks:
                n, nd = self.station_repo.get_network(fp)
                shp = shapely.geometry.shape(self.station_repo.stations[n]['geometry'])
                focus_objs.append((shp,nd,'network'))
                _log.debug(f'Focus on network {fp:s}')
            else:
                _log.warning(f'Kon geen station, netwerk of land vinden met de code {fp:s}')
        
        points = []
        focus_names = []
        for geom,attrs,ftype in focus_objs:
            names = [attrs.get(k,'').strip() for k in ['NAME_LONG','NAME','name']]
            nameid = next((i for i, n in enumerate(names) if n!=''), None)
            if nameid is not None:
                focus_names.append(names[nameid])
            
            if 'Point' == geom.geom_type:
                points.append(geom)
            else:
                lonmin, latmin, lonmax, latmax = geom.bounds
                if 'multi' in geom.geom_type.lower() and len(geom)>1 and (lonmin+360-lonmax<5):
                    polygon_bounds = np.array([list(p.bounds) for p in list(geom)])
                    lonmin = np.nanmin(np.where(polygon_bounds[:,0]>-179,polygon_bounds[:,0],np.nan))
                    lonmax = np.nanmax(np.where(polygon_bounds[:,2]<179,polygon_bounds[:,2],np.nan)+360)
                points += [
                    shapely.geometry.Point(lonmin,latmin),
                    shapely.geometry.Point(lonmin,latmax),
                    shapely.geometry.Point(lonmax,latmin),
                    shapely.geometry.Point(lonmax,latmax),
                ]
        mp = shapely.geometry.MultiPoint(points)
        lonmin, latmin, lonmax, latmax = mp.bounds
        self.center_lon,self.center_lat = (lonmin+lonmax)/2, (latmin+latmax)/2
        self.focus_extent = lonmin, lonmax, latmin, latmax
        self.calc_zoom()
        
        if len(focus_names)<3:
            self.mapname = " & ".join(focus_names)
        else:
            self.mapname = ', '.join(focus_names[:-1]) + ', and ' + focus_names[-1]
    def display_stations(self,stations):
        self.stations_on_map = stations
    def finalize_map(self):
        self.set_extent_to_map()
        if self.zoom<.2:
            self.ax.outline_patch.set_visible(True)
            self.ax.outline_patch.set_edgecolor('k')
            self.ax.outline_patch.set_linewidth(.5)
            
    # Plotsets
    def _prepare_plotset(self,legend_below_size=.1,axes_dim=None):
        if None in [self.center_lon,self.center_lat]+list(self.focus_extent):
            self.focus_map()
        if self.proj is None:
            self.proj = cartopy.crs.NearsidePerspective(
                central_longitude=self.center_lon,
                central_latitude=self.center_lat,
                satellite_height=35785831
            )
        laxes_dim = None
        if axes_dim is None:
            axes_dim = [0,0,1,1]
            if self.zoom<.2:
                figsize = (7,4)
                if legend_below_size>0:
                    legend_below_size = legend_below_size*7/4
                    axes_dim = [0,legend_below_size,1,1-legend_below_size]
                    laxes_dim = [0,0,1,legend_below_size]
            else:
                figsize = (7,7)
                if legend_below_size>0:
                    axes_dim = [0,legend_below_size,1,1-legend_below_size]
                    laxes_dim = [0,0,1,legend_below_size]
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_axes(axes_dim,projection=self.proj)
        if laxes_dim is not None:
            self.lax = self.fig.add_axes(laxes_dim)
            self.lax.set_axis_off()
        
        if self.zoom<.2:
            self.ax.set_global()
        self.set_extent_to_map()
    def plotset_white(self,savefig=None):
        self._prepare_plotset(legend_below_size=.15)
        self.plot_stations()
        self.country_borders(edgecolor='#666666')
        self.finalize_map()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_koppen(self,savefig=None):
        self._prepare_plotset(legend_below_size=.15)
        self.plot_stations()
        self.country_borders()
        
        image_data = self.get_koppen_geiger_climate()
        self.ax.imshow(**image_data,zorder=0)
        
        legend_settings = self.get_koppen_legend(all_climates=False)
        legend_settings.update({'bbox_to_anchor':(.5,.5),'loc':'center',})
        self.lax.legend(**legend_settings)
        self.finalize_map()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def plotset_relief(self,savefig=None):
        self._prepare_plotset()
        self.ax.add_wmts(
            'https://dgeo.mindef.nl/img/rest/services/Elevation/DGeo_World_ShadedRelief/ImageServer/WMTS',
            'Elevation_DGeo_World_ShadedRelief',
            zorder=0)
        self.plot_stations()
        self.country_borders(edgecolor='k')
        self.finalize_map()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    def generate_maps(self,savefig=True):
        if savefig:
            dirname_figs = self.output
            if not os.path.exists(dirname_figs):
                pathlib.Path(dirname_figs).mkdir(parents=True, exist_ok=True)
        msg = "Kaarten: "
        print(msg+'...',end='\r',flush=True)
        savename = self.save_name+'_' if len(self.save_name)>0 else ''
        msg += "Blanco "
        print(msg+'...',end='\r',flush=True)
        self.plotset_white(savefig=os.path.join(dirname_figs,f'{savename:s}white.png') if savefig else None)
        
        msg += "Klimaat "
        print(msg+'...',end='\r',flush=True)
        self.plotset_koppen(savefig=os.path.join(dirname_figs,f'{savename:s}koppen.png') if savefig else None)
        
        msg += "Relief "
        print(msg+'...',end='\r',flush=True)
        self.plotset_relief(savefig=os.path.join(dirname_figs,f'{savename:s}relief.png') if savefig else None)
        
        msg += "Klaar!"
        print(msg,end='\r',flush=True)
        print("\n")
        if savefig:
            _log.info("Kaarten kunnen gevonden worden in %s/%s*.png"%(dirname_figs,savename))
                
    # Zoom and Extent
    def calc_zoom(self):
        lonmin, lonmax, latmin, latmax = self.focus_extent
        dlon = np.max([.1,np.abs(self.center_lon-lonmin),np.abs(self.center_lon-lonmax)])
        dlat = np.max([.1,np.abs(self.center_lat-latmin),np.abs(self.center_lat-latmax)])
        self.zoom = np.clip(0.95*np.min([15/dlon,10/dlat]),.2,6)
    def get_extent(self):
        if self.zoom<.2:
            return [-180,180,90,-90]
        return [
            self.center_lon-(15/self.zoom),
            self.center_lon+(15/self.zoom),
            self.center_lat+(10/self.zoom),
            self.center_lat-(10/self.zoom)
        ]
    def get_outer_extent(self):
        geom = self.ax._get_extent_geom(crs=self.trans)
        lonmin, latmin, lonmax, latmax = geom.bounds
        if (lonmin+360-lonmax<5) and geom.type=='MultiPolygon' and len(geom)>1:
            polygon_bounds = np.array([list(p.bounds) for p in list(geom)])
            lonmin = np.nanmin(np.where(polygon_bounds[:,0]>-179,polygon_bounds[:,0],np.nan))
            lonmax = np.nanmax(np.where(polygon_bounds[:,2]<179,polygon_bounds[:,2],np.nan)+360)
        return lonmin, lonmax, latmin, latmax
    def set_extent_to_map(self):
        if self.zoom>=.2:
            self.ax.set_extent(self.get_extent(),crs=self.trans)
    
    # Map Features (borders e.d.)
    def country_borders(self,zorder=2,**kwargs):
        file = self.ne_files,'ne_10m_admin_0_countries','shp'
        altf = self.ne_files,'ne_50m_admin_0_countries','shp'
        if self.zoom<1:
            file = self.ne_files,'ne_50m_admin_0_countries','shp'
            altf = self.ne_files,'ne_10m_admin_0_countries','shp'
        
        if bool(MapPlotHelper.search_files(*file)):
            shp_path = MapPlotHelper.search_or_extract(*file)
        elif bool(MapPlotHelper.search_files(*altf)):
            shp_path = MapPlotHelper.search_or_extract(*altf)
        else:
            _log.tryexcept(repr(e),exc_info=e)
            _log.warning('De Landgrens-bestanden konden niet worden gevonden. '
                'Kaart wordt geplot zonder grenzen.\n'
                'Zie 00. Instaleren & Introductie, 3.1 Natural Earth, voor een oplossing')
            return
        shp = cartopy.io.shapereader.Reader(shp_path)
        shapekwargs = {'facecolor':'none','edgecolor':'k','linewidth':.33 if self.zoom<=1.5 else .75,'zorder':zorder}
        shapekwargs.update(kwargs)
        sf = cartopy.feature.ShapelyFeature(
            shp.geometries(),
            self.trans,
            **shapekwargs)
        self.ax.add_feature(sf,zorder=shapekwargs['zorder'],linewidth=shapekwargs['linewidth'])
    
    # Plotting stations
    def plot_stations(self,stations=None,marker='plane',color='k',plot_ooa=True,
                      station_text_fmt='{icao}',format_elements=None,
                      scatter_kwargs=None,text_kwargs=None):
        if scatter_kwargs is None:
            scatter_kwargs = {}
        if text_kwargs is None:
            text_kwargs = {}
        if format_elements is None:
            format_elements = {}
        if stations is None:
            stations = self.stations_on_map
        if isinstance(stations,str):
            stations = [stations]
        assert self.ax is not None, 'Please run init_map before plotting'
            
        self.set_extent_to_map()
        if marker=='plane':
            marker = self.marker_plane()
        skwargs = {'s': 36*2,'c': color,'zorder': 3}
        skwargs.update(scatter_kwargs)
        tkwargs = {'color': color, 'zorder': 3}
        tkwargs.update(text_kwargs)
        
        station_list = []
        for s in stations:
            if s not in self.station_repo and s in self.station_repo.networks:
                station_list += self.station_repo.get_station_codes_from_network(s)
            else:
                station_list.append(s)
        
        for s in station_list:
            station, station_data = self.station_repo.get_station(s)
            lon,lat = station_data['longitude'], station_data['latitude']
            geom = self.ax._get_extent_geom(crs=self.trans)
            p = shapely.geometry.Point(lon,lat)
            inside_area = geom.contains(p)
            
            if (not plot_ooa) and (not inside_area):
                continue
            
            format_dict = {} if s not in format_elements else dict(format_elements[s])
            station_data.update(format_dict)
            format_dict = collections.defaultdict(str,**format_dict)
            station_text = station_text_fmt.format_map(format_dict).strip()
            station_text = s if station_text=='' else station_text
            
            if inside_area:
                self.ax.scatter(x=lon,y=lat,
                    marker=marker,
                    transform=self.trans,
                    **skwargs)
                self.ax.annotate(
                    station_text,
                    xy=(lon,lat),
                    xytext=(lon+(.4/self.zoom),lat+(.3/self.zoom)),
                    xycoords=self.trans._as_mpl_transform(self.ax),
                    **tkwargs
                )
                #self.ax.annotate(
                #    station_text,
                #    xy=(lon,lat),
                #    xytext=(lon+(.4/self.zoom),lat+(.3/self.zoom)),
                #    xycoords=self.trans._as_mpl_transform(self.ax),
                #    **tkwargs)
            else:
                x,y,heading = self.calculate_xy_ooa(lon,lat,geom)
                self.ax.scatter(
                    x,y,
                    marker=(3,0,(-heading)%360),
                    transform=self.ax.transAxes,
                    **skwargs)
                self.ax.annotate(
                    station_text,
                    xy=(x,y),
                    xytext=(x+(.01 if x<.5 else -.01),y+(.01 if y<.5 else -.01)),
                    xycoords=self.ax.transAxes,
                    va=('top' if y>.5 else 'bottom'),
                    ha=('right' if x>.5 else 'left'),
                    **tkwargs)
    def calculate_xy_ooa(self,lon,lat,geom):
        px,py = self.proj.transform_point(lon,lat,self.trans)
        gx,gy = self.proj.transform_point(geom.centroid.x,geom.centroid.y,self.trans)
        dy,dx = py-gy, px-gy
        if np.isnan(dy) or np.isnan(dx):
            dy,dx = lat-geom.centroid.y, lon-geom.centroid.x
        heading_rad = np.arctan2(dy,dx)
        heading_math = np.rad2deg(heading_rad)
        heading_deg = (90-heading_math)%360
        heading = np.deg2rad(heading_deg)
        if 45<=heading_deg<=135:
            x,y = 1,.5+(.5/np.tan(heading))
        elif 135<=heading_deg<=225:
            x,y = .5-(.5*np.tan(heading)),0
        elif 225<=heading_deg<=315:
            x,y = 0,.5-(.5/np.tan(heading))
        else:
            x,y = .5+(.5*np.tan(heading)),1
        x,y = np.clip(x,.02,.98),np.clip(y,.02,.98)
        return x,y,heading_deg
    
    # Koppen Climate
    @classmethod
    def crop_geotiff(cls,im,extent=None,margin=0):
        if extent is None:
            return im
        def find_nearest(array, value,updown='a',margin=0):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            margin = int(margin)
            idx += -1 if array[idx]-value>0 and updown=='d' else (
                1 if array[idx]-value<0 and updown=='u' else 0)
            idx += margin
            #idx = np.clip(idx,0,2*len(array)-1)
            return idx, array[idx%len(array)]

        lonmin,lonmax,latmin,latmax = extent
        Tiepoint, PixelScale = im.tag[33922], im.tag[33550]
        left,upper = Tiepoint[3] - PixelScale[0]*Tiepoint[0], Tiepoint[4] + PixelScale[1]*Tiepoint[1]
        right,bottom = Tiepoint[3] - PixelScale[0]*(Tiepoint[0]-im.size[0]), Tiepoint[4] + PixelScale[1]*(Tiepoint[1]-im.size[1])
        imlons = np.linspace(left,right,im.size[0]+1)
        imlats = np.linspace(upper,bottom,im.size[1]+1)
        lidx, llon = find_nearest(imlons,lonmin,'d',-margin)
        ridx, rlon = find_nearest(imlons,lonmax,'u',margin)
        bidx, blat = find_nearest(imlats,latmin,'u',margin)
        uidx, ulat = find_nearest(imlats,latmax,'d',-margin)
        if uidx<0:
            lidx, ridx = 0, im.width
            uidx, bidx = 0, max(abs(uidx),abs(bidx))
            imc = im.crop((lidx,uidx,ridx,bidx))
        elif bidx>=im.height:
            lidx, ridx = 0, im.width
            uidx, bidx = min(abs(uidx),abs(bidx)), im.height
            imc = im.crop((lidx,uidx,ridx,bidx))
        elif lidx<0 or ridx>=im.width:
            if lidx<0:
                lidx += im.width
                ridx += im.width 
            dst = PIL.Image.new('P', (ridx, im.height))
            dst.paste(im, (0, 0))
            dst.paste(im.crop((0,0,ridx-im.width,im.height)),(im.width, 0))
            imc = dst.crop((lidx,uidx,ridx,bidx))
        else:
            imc = im.crop((lidx,uidx,ridx,bidx)) #(left, upper, right, lower)
        llon = Tiepoint[3] - PixelScale[0]*(Tiepoint[0]-lidx) #+ 180)% 360) - 180
        ulat = Tiepoint[4] + PixelScale[1]*(Tiepoint[1]-uidx) #+ 90) % 180) - 90
        imc.tag = ImageFileDirectory_v1()
        for k,v in im.tag.items():
            imc.tag[k] = v
        imc.tag[33922] = (0.,0.,im.tag[33922][2],llon,ulat,im.tag[33922][5])
        
        return imc
    def get_koppen_geiger_climate(self,margin=None):
        PIL.Image.MAX_IMAGE_PIXELS = None
        filename = self.kg_files + ('Beck_KG_V1_present_0p083.tif' if self.zoom<=1.5 else f'Beck_KG_V1_present_0p0083.tif')
        self.set_extent_to_map()
        extent = self.get_outer_extent()
        if margin is None:
            margin = np.ceil(np.exp(.5/self.zoom)*30)
        
        im = PIL.Image.open(filename)
        img = self.crop_geotiff(im,extent,margin)
        
        clim_data = np.array(img.getdata(),dtype=np.uint8).reshape(img.size,order='F').T

        Tiepoint, PixelScale = img.tag[33922], img.tag[33550]
        left,upper = Tiepoint[3] - PixelScale[0]*Tiepoint[0], Tiepoint[4] - PixelScale[1]*Tiepoint[1]
        right,bottom = Tiepoint[3] + PixelScale[0]*(img.size[0]-Tiepoint[0]), Tiepoint[4] - PixelScale[1]*(img.size[1]-Tiepoint[1])
    
        colorcount = len(img.tag[320])//3
        colors = (np.array(img.tag[320]).reshape(3,colorcount).T) / 65535
        cmap = mpl.colors.ListedColormap(colors,'Köppen-Geiger')
        cmap.set_bad('w');cmap.set_under('w');cmap.set_over('w');
        #cmap.set_extremes(bad='k',under='w',over='w')
        norm = mpl.colors.Normalize(0,colorcount-1)
        
        labels = ['No data',
                  'Af','Am','Aw',
                  'BWh','BWk','BSh','BSk',
                  'Csa','Csb','Csc',
                  'Cwa','Cwb','Cwc',
                  'Cfa','Cfb','Cfc',
                  'Dsa','Dsb','Dsc','Dsd',
                  'Dwa','Dwb','Dwc','Dwd',
                  'Dfa','Dfb','Dfc','Dfd',
                  'ET','EF']
        unique_clim_id_counts = np.unique(clim_data,return_counts=True)
        unique_clim_ids = list(unique_clim_id_counts[0])
        legend_handles = {labels[clim_id]: colors[clim_id] for clim_id in sorted(unique_clim_ids)}
        
        self.koppen = {
            'labels': labels,
            'colors': colors,
            'colors_in_img': legend_handles
        }
    
        return {'img':clim_data,
                'extent':(left,right,bottom,upper),'origin':'upper',
                'transform':cartopy.crs.PlateCarree(),
                'interpolation': 'nearest',
                'cmap':cmap,'norm':norm}
    def get_koppen_legend(self,all_climates=False):
        
        if self.koppen is None:
            raise ValueError('Please run get_koppen_geiger_climate before generating the legend.')
        max_group_size = 4
        climate_groups = {}
        colors = {}
        columns = ['A','B','Cs','Cw','Cf','Ds','Dw','Df','E','N']
        if all_climates:
            colors = {l:self.koppen['colors'] for i,l in enumerate(self.koppen['labels'])}
        else:
            colors = self.koppen['colors_in_img']
            group_size = {cg:sum([1 if k.startswith(cg) else 0 for k in colors.keys()])
                         for cg in columns}
            if group_size['Cs']+group_size['Cw']+group_size['Cf']<=4:
                columns = ['A','B','C','Ds','Dw','Df','E','N']
                group_size['C'] = group_size['Cs']+group_size['Cw']+group_size['Cf']
                group_size.pop('Cs');group_size.pop('Cw');group_size.pop('Cf');
                if group_size['Ds']+group_size['Dw']+group_size['Df']<=4:
                    columns = ['A','B','C','D','E','N']
                    group_size['D'] = group_size['Ds']+group_size['Dw']+group_size['Df']
                    group_size.pop('Ds');group_size.pop('Dw');group_size.pop('Df');
            elif group_size['Ds']+group_size['Dw']+group_size['Df']<=4:
                columns = ['A','B','Cs','Cw','Cf','D','E','N']
                group_size['D'] = group_size['Ds']+group_size['Dw']+group_size['Df']
                group_size.pop('Ds');group_size.pop('Dw');group_size.pop('Df');
        legend_columns = {}
        for cg in columns:
            climate_groups[cg] = {k:v for k,v in sorted(colors.items(),key=lambda i:i[0])
                                       if k.startswith(cg)}
            if len(climate_groups[cg])>0:
                def first_that_starts_with(list,start):
                    return next(x for x in list if x.startswith(start))
                heading = {
                    first_that_starts_with(columns,'A'): 'Tropical',
                    first_that_starts_with(columns,'B'): 'Arid',
                    first_that_starts_with(columns,'C'): 'Temperate',
                    first_that_starts_with(columns,'D'): 'Cold',
                    first_that_starts_with(columns,'E'): 'Polar'
                }
                legend_columns[cg] = {-1:mplLegendSubheading(heading.get(cg,""))}
                for i in range(max_group_size):
                    if i<len(climate_groups[cg]):
                        climate = list(climate_groups[cg].keys())[i]
                        legend_columns[cg][i] = mpl.patches.Patch(
                            facecolor = climate_groups[cg][climate],
                            label = climate,
                            edgecolor = 'k',
                            linewidth = .5)
                    else:
                        legend_columns[cg][i] = mpl.lines.Line2D([],[],color="none")
        legend_handles = []
        for column in legend_columns.values():
            legend_handles += list(column.values())
        return {
            'handles':legend_handles,
            'ncol':len(legend_columns),
            'handler_map':{
                mplLegendSubheading:mplLegendSubheadingHandler(),
            }
        }
        
