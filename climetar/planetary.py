import logging
_log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import StationRepo

sqrt = np.sqrt
sin = lambda x: np.sin(np.deg2rad(x))
cos = lambda x: np.cos(np.deg2rad(x))
tan = lambda x: np.tan(np.deg2rad(x))
asin = arcsin = lambda x: np.rad2deg(np.arcsin(x))
acos = arccos = lambda x: np.rad2deg(np.arccos(x))
atan = arctan = lambda x: np.rad2deg(np.arctan(x))
atan2 = arctan2 = lambda y,x: np.rad2deg(np.arctan2(y,x))
pi = np.pi

def arccosinf(x):
    out = np.where(x>1,np.inf,np.where(x<-1,-np.inf,np.nan))
    return np.rad2deg(np.arccos(x,out=out,where=np.abs(x)<=1))
def fixdeg360(angle):
    return (angle + 360) % 360
def fixdeg180(angle):
    angle = fixdeg360(angle)
    return np.where(angle<180,angle,angle-360)
def solar_calculations(lat_deg,lon_deg,julian2000,time_frac):
    solar = PlannetaryDataset()
    solar['lat'] = fixdeg180(lat_deg)
    solar['lon'] = fixdeg180(lon_deg)
    solar['JD'] = julian2000

    solar['N_longitude_ascending_node'] = 0.
    solar['i_inclination_eliptic'] = 0.
    solar['w_argument_perihelion'] = fixdeg360(282.9404 + 4.70935e-5 * julian2000)
    solar['a_semimajor_axis'] = 1.
    solar['e_eccentricity'] = 0.016709 - 1.151e-9 * julian2000
    solar['M_mean_anomaly'] = fixdeg360(356.0470 + 0.9856002585 * julian2000)
    solar['w1_longitude_perihelion'] = solar.N + solar.w
    solar['L_mean_longitude'] = solar.M + solar.w1
    solar['q_perihelion_distance'] = solar.a * (1-solar.e)
    solar['Q_aphelion_distance'] = solar.a * (1+solar.e)
    solar['P_orbital_period'] = solar.a ** 1.5

    solar['E_eccentric_anomaly'] = (solar.M + solar.e*(180/pi) * sin(solar.M)
        * (1. + solar.e * cos(solar.M)))
    solar['xv'] = cos(solar.E) - solar.e
    solar['yv'] = sqrt(1. - solar.e**2) * sin(solar.E)
    solar['v_true_anomaly'] = arctan2(solar.yv,solar.xv)
    solar['r_distance'] = sqrt(solar.xv**2+solar.yv**2)

    solar['lonsun_true_longitude'] = solar.v + solar.w
    solar['xs'] = solar.r * cos(solar.lonsun)
    solar['ys'] = solar.r * sin(solar.lonsun)

    earth_ecl_obliquity_ecliptic = 23.4393 - 3.563e-7 * julian2000
    solar['xe'] = solar.xs
    solar['ye'] = solar.ys * cos(earth_ecl_obliquity_ecliptic)
    solar['ze'] = solar.ys * sin(earth_ecl_obliquity_ecliptic)
    solar['RA_right_ascension'] = arctan2(solar.ye,solar.xe)
    solar['Dec_declination'] = arctan2(solar.ze,sqrt(solar.xe**2+solar.ye**2))

    time_frac_noon = .5 + solar.lon/360
    JDnoon = julian2000//1 + time_frac_noon
    solar['Lnoon_mean_longitude'] = fixdeg360(356.0470 + 0.9856002585 * JDnoon) + fixdeg360(282.9404 + 4.70935e-5 * JDnoon)
    solar['GMST0noon'] = (fixdeg360(solar.Lnoon)/15 + 12)%24
    solar['GMSTnoon'] = (solar.GMST0noon + (time_frac_noon*24))%24
    solar['LSTnoon'] = (solar.GMSTnoon + lon_deg/15)%24

    solar['GMST0'] = (fixdeg180(solar.L + 180)/15)%24
    solar['GMST'] = (solar.GMST0 + (time_frac*24))%24
    solar['LST'] = (solar.GMST + lon_deg/15)%24

    solar['HA_hour_angle'] = (solar.LST*15) - solar.RA
    solar['x'] = cos(solar.HA) * cos(solar.Dec)
    solar['y'] = sin(solar.HA) * cos(solar.Dec)
    solar['z'] = sin(solar.Dec)
    solar['xhor'] = solar.x * sin(lat_deg) - solar.z * cos(lat_deg)
    solar['yhor'] = solar.y
    solar['zhor'] = solar.x * cos(lat_deg) + solar.z * sin(lat_deg)
    solar['az_azimuth'] = arctan2(solar.yhor,solar.xhor)+180
    solar['alt_altitude'] = arctan2(solar.zhor,sqrt(solar.xhor**2+solar.yhor**2))

    solar['h_altitude_above_horizon'] = arcsin(sin(lat_deg)*sin(solar.Dec)
        + cos(lat_deg)*cos(solar.Dec)*cos(solar.HA))
    solar['noondeg_UT_sun_in_south_deg'] = fixdeg180(solar.RA-solar.GMST0*15-lon_deg)
    solar['noon_JD_sun_in_south'] = solar.JD//1 + solar.noondeg/360

    return solar
def lunar_calculations(lat_deg,lon_deg,julian2000,time_frac,solar=None):
    if solar is None:
        solar = solar_calculations(lat_deg,lon_deg,julian2000,time_frac)
    lunar = PlannetaryDataset()
    lunar['lat'] = fixdeg180(lat_deg)
    lunar['lon'] = fixdeg180(lon_deg)
    lunar['JD'] = julian2000

    lunar['N_longitude_ascending_node'] = fixdeg180(125.1228 - 0.0529538083 * julian2000)
    lunar['i_inclination_eliptic'] = 5.1454
    lunar['w_argument_perihelion'] = fixdeg180(318.0634 + 0.1643573223 * julian2000)
    lunar['a_semimajor_axis'] = 60.2666 # Earth radii
    lunar['e_eccentricity'] = 0.054900
    lunar['M_mean_anomaly'] = fixdeg180(115.3654 + 13.0649929509 * julian2000)
    lunar['w1_longitude_perihelion'] = lunar.N + lunar.w
    lunar['L_mean_longitude'] = lunar.M + lunar.w1
    lunar['q_perihelion_distance'] = lunar.a * (1-lunar.e)
    lunar['Q_aphelion_distance'] = lunar.a * (1+lunar.e)
    lunar['P_orbital_period'] = lunar.a ** 1.5

    lunar['D_mean_elongation'] = lunar.L - solar.L
    lunar['F_argument_latitude'] = lunar.L - lunar.N

    lunar['E_eccentric_anomaly'] = np.rad2deg(solve_kepler_eq_rad(np.deg2rad(lunar.M),lunar.e))
    lunar['xv'] = lunar.a * (cos(lunar.E) - lunar.e)
    lunar['yv'] = lunar.a * (sqrt(1.-lunar.e**2) * sin(lunar.E))
    lunar['v_true_anomaly'] = arctan2(lunar.yv,lunar.xv)
    lunar['ruc_distance_uncorrected'] = sqrt(lunar.xv**2+lunar.yv**2)

    lunar['vw_true_longitude'] = lunar.v + lunar.w
    lunar['xh'] = lunar.ruc * (cos(lunar.N)*cos(lunar.vw) - sin(lunar.N)*sin(lunar.vw)*cos(lunar.i))
    lunar['yh'] = lunar.ruc * (sin(lunar.N)*cos(lunar.vw) + cos(lunar.N)*sin(lunar.vw)*cos(lunar.i))
    lunar['zh'] = lunar.ruc * (sin(lunar.vw)*sin(lunar.i))
    lunar['lonecluc_ecliptic_longitude_uncorrected'] = arctan2(lunar.yh,lunar.xh)
    lunar['latecluc_ecliptic_latitude_uncorrected'] = arctan2(lunar.zh,sqrt(lunar.xh**2+lunar.yh**2))

    lunar['loneclcorr_lonecl_correction'] = (
        -1.274 * sin(lunar.M - 2*lunar.D)             # (the Evection)
        +0.658 * sin(2*lunar.D)                       # (the Variation)
        -0.186 * sin(solar.M)                         # (the Yearly Equation)
        -0.059 * sin(2*lunar.M - 2*lunar.D)           #
        -0.057 * sin(lunar.M - 2*lunar.D + solar.M)   #
        +0.053 * sin(lunar.M + 2*lunar.D)             #
        +0.046 * sin(2*lunar.D - solar.M)             #
        +0.041 * sin(lunar.M - solar.M)               #
        -0.035 * sin(lunar.D)                         # (the Parallactic Equation)
        -0.031 * sin(lunar.M + solar.M)               #
        -0.015 * sin(2*lunar.F - 2*lunar.D)           #
        +0.011 * sin(lunar.M - 4*lunar.D)             #
    )
    lunar['lateclcorr_latecl_correction'] = (
        -0.173 * sin(lunar.F - 2*lunar.D)
        -0.055 * sin(lunar.M - lunar.F - 2*lunar.D)
        -0.046 * sin(lunar.M + lunar.F - 2*lunar.D)
        +0.033 * sin(lunar.F + 2*lunar.D)
        +0.017 * sin(2*lunar.M + lunar.F)
    )
    lunar['rcorr_distance_correction'] = (
        -0.58 * cos(lunar.M - 2*lunar.D)
        -0.46 * cos(2*lunar.D)
    )
    lunar['lonecl_ecliptic_longitude'] = lunar.lonecluc + lunar.loneclcorr
    lunar['latecl_ecliptic_latitude'] = lunar.latecluc + lunar.lateclcorr
    lunar['r_distance'] = lunar.ruc + lunar.rcorr

    lunar['xg'] = lunar.r * cos(lunar.lonecl) * cos(lunar.latecl)
    lunar['yg'] = lunar.r * sin(lunar.lonecl) * cos(lunar.latecl)
    lunar['zg'] = lunar.r                     * sin(lunar.latecl)

    earth_ecl_obliquity_ecliptic = 23.4393 - 3.563e-7 * julian2000
    lunar['xe'] = lunar.xg
    lunar['ye'] = lunar.yg * cos(earth_ecl_obliquity_ecliptic) - lunar.zg * sin(earth_ecl_obliquity_ecliptic)
    lunar['ze'] = lunar.yg * sin(earth_ecl_obliquity_ecliptic) + lunar.zg * cos(earth_ecl_obliquity_ecliptic)
    lunar['RA_right_ascension'] = arctan2(lunar.ye,lunar.xe)
    lunar['Dec_declination'] = arctan2(lunar.ze,sqrt(lunar.xe**2+lunar.ye**2))
    lunar['rg_geocentric_distance']= sqrt(lunar.xe**2+lunar.ye**2+lunar.ze**2)

    lunar['HA_hour_angle'] = solar.LST-lunar.RA
    lunar['x'] = cos(lunar.HA) * cos(lunar.Dec)
    lunar['y'] = sin(lunar.HA) * cos(lunar.Dec)
    lunar['z'] = sin(lunar.Dec)
    lunar['xhor'] = lunar.x * sin(lat_deg) - lunar.z * cos(lat_deg)
    lunar['yhor'] = lunar.y
    lunar['zhor'] = lunar.x * cos(lat_deg) + lunar.z * sin(lat_deg)
    lunar['az_azimuth'] = arctan2(lunar.yhor,lunar.xhor)+180
    lunar['alt_altitude'] = arctan2(lunar.zhor,sqrt(lunar.xhor**2+lunar.yhor**2))

    gclat_geocentric_latitude = lat_deg - 0.1924 * sin(2*lat_deg)
    rho_distance_center_earth = 0.99833 + 0.00167 * cos(2*lat_deg)
    lunar['mpar_parallax'] = arcsin(1/lunar.r)
    lunar['topalt_topocentric_altitude'] = lunar.alt - lunar.mpar * cos(lunar.alt)
    lunar['g_auxiliary_angle'] = arctan2(tan(gclat_geocentric_latitude),lunar.HA)
    lunar['topRA_topocentric_right_ascension'] = lunar.RA - np.divide(
            lunar.mpar * rho_distance_center_earth * cos(gclat_geocentric_latitude) * sin(lunar.HA),
            cos(lunar.Dec),
        out=np.full(lunar.RA.shape,np.nan),
        where=np.abs(cos(lunar.Dec))>=1e-4)
    lunar_topDec_step1 = lunar.Dec - (lunar.mpar * rho_distance_center_earth)
    lunar_topDec_step2 = np.divide(
            sin(gclat_geocentric_latitude) * sin(lunar.g - lunar.Dec),
            sin(lunar.g),
        out=np.full(lunar.g.shape,np.nan),
        where=np.abs(lunar.g)>=1e-4)
    lunar_topDec_step3 = np.nan_to_num(
        lunar_topDec_step2,
        nan=sin(-lunar.Dec)*cos(lunar.HA),
        posinf=np.inf,neginf=-np.inf)
    lunar['topDec_topocentric_declination'] = lunar_topDec_step1 * lunar_topDec_step3

    lunar['d_apparent_diameter'] = (1873.7*(60/lunar.r))/60/60
    lunar['elong_elongation'] = arccos(cos(solar.lonsun-lunar.lonecl) * cos(lunar.latecl))
    lunar['FV_phase_angle'] = 180 - lunar.elong
    lunar['phase'] = (1+cos(lunar.FV)) / 2
    lunar['magnitude'] = -21.62 + 5 * np.log10(solar.r*lunar.rg) + 0.026 * lunar.FV + 4.0e-9 * lunar.FV**4

    lunar['noondeg_UT_moon_in_south_deg'] = fixdeg180(lunar.RA-solar.GMST0*15-lon_deg)
    lunar['noon_JD_moon_in_south'] = lunar.JD//1 + lunar.noondeg/360

    return lunar

def solve_kepler_eq_rad(M,e,max_iterations=20):
    E0 = M + e * np.sin(M) * (1. + e * np.cos(M))
    E1 = E0 - (E0 - e * np.sin(E0) - M) / (1 - e*np.cos(E0))
    iterations = 1
    while np.abs(E0-E1).max()>np.deg2rad(.001):
        E0 = np.copy(E1)
        E1 = E0 - (E0 - e * np.sin(E0) - M) / (1 - e*np.cos(E0))
        iterations += 1
        if iterations > max_iterations:
            print(np.abs(E0-E1).max())
            diff = np.abs(E0-E1).max()
            raise ValueError(f'Did not converge in {iterations:d} iterations')
    return E1

class Astro(object):
    solar_twilights = {'astronomical':-18.,'nautical':-12.,'civil':-6.,'sun':-.833,}

    def __init__(self,lon=None,lat=None,station=None,unix=None,year=None,tz=None):
        self.lon = lon
        self.lat = lat
        self.tz = tz
        self.unix = unix
        self.station_data = {}

        if station is not None and (self.lon is None or self.lat is None):
            sr = StationRepo()
            s,sd = sr.get_station(station)
            self.station_data = sd
            self.lon, self.lat = sd['latitude'], sd['longitude']
            self.tz = tz if tz is not None else sd['timezone']
        if self.lon is None and self.lat is None:
            raise ValueError()
        self.tz = self.tz if self.tz is not None else 'UTC'

        if self.unix is None:
            if year is None:
                raise ValueError()
            self.unix = self.year_to_unix(year,'H')
        self.year = pd.to_datetime(self.unix,unit='s',origin='unix').strftime('%Y').astype(int).value_counts().idxmax()

        #[print(getattr(self,attr)) for attr in 'lon,lat,tz,station_data,year'.split(',')]

    @classmethod
    def year_to_unix(cls,year,freq=None,add_time=None):
        start = pd.to_datetime(f'{year:04d}-001',format='%Y-%j')
        end_doy = 366 if start.is_leap_year else 365
        end = pd.to_datetime(f'{year+1:04d}-001',format='%Y-%j')
        dates = pd.date_range(start,end,freq=freq)
        if add_time is not None:
            if isinstance(add_time,(int,float)):
                add_time = pd.Timedelta(add_time,freq)
            dates = dates + add_time
        return cls.pdtimestamp_to_unix(dates)

    @classmethod
    def unix_to_julian(cls,unix_ts):
        julian_day = unix_ts/86400.0 + 2440587.5
        julian2000 = julian_day - 2451543.5 #0:00 0Jan2000 = 31Dec1999
        time_frac = unix_ts/86400%1
        return julian2000,time_frac

    @classmethod
    def pdtimestamp_to_unix(cls,pd_timestamps):
        return ((pd_timestamps - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')).values

    @classmethod
    def pdtimestamp_to_julian(cls,pd_timestamps):
        return cls.unix_to_julian(cls.pdtimestamp_to_unix(pd_timestamps))

    @classmethod
    def julian_to_pdtimestamp(cls,julian2000,precision_s=1):
        julian2000s = np.round((julian2000+10956)*86400*precision_s)/precision_s
        dates = pd.to_datetime(julian2000s,unit='s',origin='unix') #unix_to_julian(0)=-10956
        dates = dates.to_series() if 'index' in str(type(dates)) else dates
        return dates.astype('datetime64[s]')

    @classmethod
    def convert_tz(cls,series,to_tz,from_tz=None):
        if series.dt.tz is None:
            from_tz = 'Etc/UTC' if from_tz is None else from_tz
            series = series.dt.tz_localize(from_tz)
        return series.dt.tz_convert(to_tz).dt.tz_localize(None)
    def solar_calculate_dawndusk(self,opt_itermax=10,opt_diffmax_s=1):
        opt_diffmax = opt_diffmax_s/86400
        dates = pd.date_range(
            pd.Timestamp(min(self.unix),unit='s'),
            pd.Timestamp(max(self.unix)+1,unit='s'),
            freq='H')
        julian2000, time_frac = self.pdtimestamp_to_julian(dates)
        JDmin, JDmax = self.pdtimestamp_to_julian(dates.to_series().iloc[[0,-1]])[0]
        sds = solar_calculations(self.lat, self.lon, julian2000, time_frac)
        sdf = {'date': self.julian_to_pdtimestamp(sds.JD,opt_diffmax_s).values}

        for twilight, angle in self.solar_twilights.items():
            sds_dawn = sds.copy()
            iter_dawn = 0
            UTdawn_old, JDdawn_old = np.zeros(julian2000.shape), np.zeros(julian2000.shape)
            while True:
                iter_dawn += 1
                LHA = arccosinf(sin(angle)-sin(self.lat)*sin(sds_dawn.Dec)/cos(self.lat)*cos(sds_dawn.Dec))
                #UTdawn_new = fixdeg180(sds_dawn.noondeg - LHA)/15.04107
                #UTdawn_new = np.where(UTdawn_new>24.,np.nan,UTdawn_new)
                #UTdawn_new = np.where(UTdawn_new<0.,np.nan,UTdawn_new)
                #JDdawn_new = julian2000//1 + UTdawn_new/24.0

                UTdawn_new = (sds_dawn.noondeg - LHA)
                TFdawn_new = UTdawn_new/15.04107/24.0
                JDdawn_new = (julian2000//1) + (TFdawn_new)
                JDdawn_new = np.where(JDdawn_new>julian2000+.5,JDdawn_new-1,JDdawn_new)
                JDdawn_new = np.where(JDdawn_new<julian2000-.5,JDdawn_new+1,JDdawn_new)

                if iter_dawn>1 and np.nanmax(np.abs(JDdawn_new-JDdawn_old))<opt_diffmax:
                    break
                if iter_dawn>opt_itermax:
                    JDdawn_new = np.where(np.abs(JDdawn_new-JDdawn_old)>opt_diffmax,np.nan,JDdawn_new)
                    break
                UTdawn_old, JDdawn_old = UTdawn_new, JDdawn_new
                sds_dawn = solar_calculations(self.lat,self.lon,JDdawn_new,TFdawn_new)
            sds['%sdawn_%s'%(twilight[0],twilight)] = JDdawn_new
            name = 'sunrise' if twilight=='sun' else f'{twilight}_dawn'
            sdf[name] = self.julian_to_pdtimestamp(JDdawn_new,opt_diffmax_s).values
        sdf['noon'] = self.julian_to_pdtimestamp(sds.noon,opt_diffmax_s).values
        for twilight, angle in reversed(list(self.solar_twilights.items())):
            sds_dusk = sds.copy()
            iter_dusk = 0
            UTdusk_old, JDdusk_old = np.zeros(julian2000.shape), np.zeros(julian2000.shape)
            while True:
                iter_dusk += 1
                LHA = arccosinf(sin(angle)-sin(self.lat)*sin(sds_dusk.Dec)/cos(self.lat)*cos(sds_dusk.Dec))
                #UTdusk_new = fixdeg180(sds_dusk.noondeg + LHA)/15.04107
                #UTdusk_new = np.where(UTdusk_new>24.,np.nan,UTdusk_new)
                #UTdusk_new = np.where(UTdusk_new<0.,np.nan,UTdusk_new)
                #JDdusk_new = julian2000//1 + UTdusk_new/24.0

                UTdusk_new = (sds_dusk.noondeg + LHA)
                TFdusk_new = UTdusk_new/15.04107/24.0
                JDdusk_new = (julian2000//1) + TFdusk_new
                JDdusk_new = np.where(JDdusk_new>julian2000+.5,JDdusk_new-1,JDdusk_new)
                JDdusk_new = np.where(JDdusk_new<julian2000-.5,JDdusk_new+1,JDdusk_new)

                if iter_dusk>1 and np.nanmax(np.abs(JDdusk_new-JDdusk_old))<opt_diffmax:
                    break
                if iter_dusk>opt_itermax:
                    JDdusk_new = np.where(np.abs(JDdusk_new-JDdusk_old)>opt_diffmax,np.nan,JDdusk_new)
                    break
                UTdusk_old, JDdusk_old = UTdusk_new, JDdusk_new
                sds_dusk = solar_calculations(self.lat,self.lon,JDdusk_new,TFdusk_new)
            sds['%sdusk_%s'%(twilight[0],twilight)] = JDdusk_new
            name = 'sunset' if twilight=='sun' else f'{twilight}_dusk'
            sdf[name] = self.julian_to_pdtimestamp(JDdusk_new,opt_diffmax_s).values
        return pd.DataFrame(sdf), sds

    def solar_dawndusk(self):
        sdf, sds = self.solar_calculate_dawndusk()

        for col in sdf.columns:
            sdf[col] = self.convert_tz(sdf[col],self.tz)

        idxs = []
        for date in sdf.date.dt.strftime('%Y-%m-%d').unique():
            date = pd.to_datetime(date,format='%Y-%m-%d')
            dates = sdf.date.loc[sdf.date.between(date,date+pd.Timedelta('1 day'))]
            idxs.append((sdf.noon-dates).dt.total_seconds().abs().idxmin())
        df = sdf.loc[idxs,:].reset_index(drop=True)
        df['date'] = pd.to_datetime(df.date.dt.strftime('%Y-%m-%d'),format='%Y-%m-%d')
        for col in ['astronomical_dawn','nautical_dawn','civil_dawn','sunrise','noon','sunset','civil_dusk','nautical_dusk','astronomical_dusk']:
            #df.loc[~df[col].between(df.date.min(),df.date.max()),col] = np.nan
            pass
        return df.iloc[:-1,:], sds

    def solar_matrix(self):
        sdf, sds = self.solar_dawndusk()
        div = pd.date_range(sdf.date.min(),sdf.date.min()+pd.Timedelta('1 day'),freq='min')[:-1].shape[0]
        dates = pd.date_range(sdf.date.min(),sdf.date.max(),freq='min')[:-1].values[:,None]
        target_shape = dates.shape[0]//div,div

        start_data = solar_calculations(
            self.lat,self.lon,*self.pdtimestamp_to_julian(
                self.convert_tz(sdf.date.iloc[[0]],'UTC',self.tz)
            ))
        start_val = np.digitize(start_data['RA_right_ascension'],
            np.array([-np.inf]+list(self.solar_twilights.values())+[np.inf])
        )-1
        udall = np.full(target_shape,start_val)

        for p, twilight in enumerate(self.solar_twilights.keys()):
            dawn = sdf['sunrise' if twilight=='sun' else f'{twilight}_dawn'].values[None,:]
            dusk = sdf['sunset' if twilight=='sun' else f'{twilight}_dusk'].values[None,:]
            after_dawn = np.sum(dates>dawn,axis=1)
            after_dusk = np.sum(dates>dusk,axis=1)
            dusk_before_dawn = (dusk<dawn).astype(int)
            updown = (after_dawn-after_dusk).reshape(*target_shape)
            udall += updown
        return  udall, sdf, sds

    def lunar_calculate_dawndusk(self,opt_itermax=100,opt_diffmax_s=1):
        opt_diffmax = opt_diffmax_s/86400
        dates = pd.date_range(pd.Timestamp(min(self.unix),unit='s'),pd.Timestamp(max(self.unix)+1,unit='s'),freq='H')
        julian2000, time_frac = self.pdtimestamp_to_julian(dates)
        JDmin, JDmax = self.pdtimestamp_to_julian(dates.to_series().iloc[[0,-1]])[0]
        lds = lunar_calculations(self.lat, self.lon, julian2000, time_frac)
        ldf = {'date': self.julian_to_pdtimestamp(lds.JD,opt_diffmax_s).values}

        lds_dawn = lds.copy()
        iter_dawn = 0
        UTdawn_old, JDdawn_old = np.zeros(julian2000.shape), np.zeros(julian2000.shape)
        while True:
            iter_dawn += 1
            angle = -0.583 -(lds_dawn.mpar+lds_dawn.d)
            LHA = arccosinf(sin(angle)-sin(self.lat)*sin(lds_dawn.Dec)/cos(self.lat)*cos(lds_dawn.Dec))
            UTdawn_new = (lds_dawn.noondeg - LHA)
            TFdawn_new = UTdawn_new/15.04107/24.0
            JDdawn_new = (julian2000//1) + (TFdawn_new)
            JDdawn_new = np.where(JDdawn_new>julian2000+.5,JDdawn_new-1,JDdawn_new)
            JDdawn_new = np.where(JDdawn_new<julian2000-.5,JDdawn_new+1,JDdawn_new)

            if iter_dawn>1 and np.nanmax(np.abs(JDdawn_new-JDdawn_old))<opt_diffmax:
                break
            if iter_dawn>opt_itermax:
                break
            if np.isnan(JDdawn_new).all():
                break
            UTdawn_old, JDdawn_old = UTdawn_new, JDdawn_new
            lds_dawn = lunar_calculations(self.lat,self.lon,JDdawn_new,TFdawn_new)

        lds['dawn'] = JDdawn_new
        ldf['moonrise'] = self.julian_to_pdtimestamp(JDdawn_new,opt_diffmax_s).values

        ldf['noon'] = self.julian_to_pdtimestamp(lds.noon,opt_diffmax_s).values

        lds_dusk = lds.copy()
        iter_dusk = 0
        UTdusk_old, JDdusk_old = np.zeros(julian2000.shape), np.zeros(julian2000.shape)
        while True:
            iter_dusk += 1
            angle = -0.583 -(lds_dawn.mpar+lds_dawn.d)
            LHA = arccosinf(sin(angle)-sin(self.lat)*sin(lds_dusk.Dec)/cos(self.lat)*cos(lds_dusk.Dec))
            UTdusk_new = (lds_dusk.noondeg + LHA)
            TFdusk_new = UTdusk_new/15.04107/24.0
            JDdusk_new = (julian2000//1) + TFdusk_new
            JDdusk_new = np.where(JDdusk_new>julian2000+.5,JDdusk_new-1,JDdusk_new)
            JDdusk_new = np.where(JDdusk_new<julian2000-.5,JDdusk_new+1,JDdusk_new)

            if iter_dusk>1 and np.nanmax(np.abs(JDdusk_new-JDdusk_old))<opt_diffmax:
                break
            if iter_dusk>opt_itermax:
                break
            if np.isnan(JDdusk_new).all():
                break
            UTdusk_old, JDdusk_old = UTdusk_new, JDdusk_new
            lds_dusk = lunar_calculations(self.lat,self.lon,JDdusk_new,TFdusk_new)
        lds['dusk'] = JDdusk_new
        ldf['moonset'] = self.julian_to_pdtimestamp(JDdusk_new,opt_diffmax_s).values

        return pd.DataFrame(ldf), lds

    def lunar_dawndusk(self,window_h=3):
        ldf, lds = self.lunar_calculate_dawndusk()

        for col in ldf.columns:
            ldf[col] = self.convert_tz(ldf[col],self.tz)

        series = {}
        for col in ['moonrise','noon','moonset']:
            idxs = []
            parsed = []
            unique_dates = list(ldf[col].unique())
            for date in unique_dates:
                if pd.isnull(date):
                    continue
                if any([(s<date and date<e) for s,e in parsed]):
                    continue
                start = date - pd.Timedelta(f'{window_h:d} hours')
                end = date + pd.Timedelta(f'{window_h:d} hours')
                mean_date = ldf.loc[ldf[col].between(start,end),col].mean()
                idx = (ldf.date-mean_date).dt.total_seconds().abs().idxmin()
                idxs.append(idx)
                parsed.append((start,end))
            idxs = list(sorted(list(set(idxs))))
            series[col] = ldf.loc[idxs,col]
        df = series['moonrise'].to_frame()
        df = df.join(series['noon'],how='outer')
        df = df.join(series['moonset'],how='outer')
        df = df.join(ldf.date,how='outer')
        for col in ['moonrise','noon','moonset']:
            df.loc[~df[col].between(df.date.min(),df.date.max()),col] = np.nan
        return df.loc[:,['date','moonrise','noon','moonset']], lds

    def lunar_matrix(self):
        ldf, lds = self.lunar_dawndusk()
        div = pd.date_range(ldf.date.min(),ldf.date.min()+pd.Timedelta('1 day'),freq='min')[:-1].shape[0]
        dates = pd.date_range(ldf.date.min(),ldf.date.max(),freq='min')[:-1].values[:,None]
        target_shape = dates.shape[0]//div,div

        start_data = lunar_calculations(
            self.lat,self.lon,*self.pdtimestamp_to_julian(
                self.convert_tz(ldf.date.iloc[[0]],'UTC',self.tz)
            ))
        start_val = (start_data.RA>-(start_data.mpar+start_data.d)).astype(float)
        udall = np.full(target_shape,start_val)

        dawn = ldf['moonrise'].dropna().values[None,:]
        dusk = ldf['moonset'].dropna().values[None,:]
        after_dawn = np.sum(dates>dawn,axis=1)
        after_dusk = np.sum(dates>dusk,axis=1)
        updown = (after_dawn-after_dusk)

        updownmx = updown.reshape(*target_shape)
        udall += updownmx
        return udall, ldf, lds
    def lunar_phase_matrix(self):
        lmx, ldf, lds = self.lunar_matrix()

        div = pd.date_range(ldf.date.min(),ldf.date.min()+pd.Timedelta('1 day'),freq='min')[:-1].shape[0]
        dates = pd.date_range(ldf.date.min(),ldf.date.max(),freq='min')[:-1].values[:,None]
        target_shape = dates.shape[0]//div,div

        phase = np.roll(np.repeat(lds.phase[:-1],div//24),div//48)
        phase[0:1+div//48] = phase[1+div//48]
        phasemx = phase.reshape(*target_shape)
        lunarmx = np.where(lmx,phasemx,np.nan)
        return lunarmx, ldf, lds

class PlannetaryDataset(object):
    def __init__(self,d=None):
        self._dict = {}
        self._abbr = {}
        if d is not None:
            self._dict = dict(d)
            self.regenerate_abbr()

    def regenerate_abbr(self):
        self._abbr = {k.split("_",2)[0]:k for k in self._dict.keys()}

    def __setitem__(self, key, item):
        abbr = key.split("_",2)[0]
        if abbr in self._abbr and key != self._abbr[abbr]:
            raise KeyError('A key with the same abbr already exsists (%s)'%self._abbr[abbr])
        if hasattr(item,'size') and item.size==1:
            item = np.atleast_1d(item)[0]
        self._dict[key] = item
        self._abbr[abbr] = key

    def __getitem__(self, key):
        return self._dict[key]

    def __getattr__(self,name):
        if name in self._abbr:
            return self._dict[self._abbr[name]]
        elif name in self._dict:
            return self._dict[name]
        else:
            raise AttributeError('Please set a variable as a key,value pair first, before attribute [%s]'%name)

    def __repr__(self):
        return repr(self._dict)

    def __len__(self):
        return len(self._dict)

    def __delitem__(self, key):
        del self._dict[key]
        self.regenerate_abbr()

    def clear(self):
        self._abbr.clear()
        return self._dict.clear()

    def to_dict(self):
        return self._dict.copy()

    def copy(self):
        return self.__class__(self._dict.copy())

    def has_key(self, k):
        return k in self._dict

    def update(self, *args, **kwargs):
        r = self._dict.update(*args, **kwargs)
        self.regenerate_abbr()
        return r

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def pop(self, *args):
        r = self._dict.pop(*args)
        self.regenerate_abbr()
        return r

    def to_frame(self):
        shape = None
        data = {}
        for k,v in list(self._dict.items()):
            if hasattr(v,'shape') and len(v.shape)>0:
                if shape is None:
                    shape = v.shape
                elif v.shape != shape:
                    print(k,v.shape,shape)
                    raise ValueError(f'{k} has another shape than other vars in this dataset')
                data[k] = v
            elif hasattr(v,'shape') and len(v.shape)==0:
                data[k] = np.atleast_1d(v)[0]
            else:
                data[k] = v
        datadict = {k:(v if hasattr(v,'shape') else np.full(shape,v)) for k,v in data.items()}
        return pd.DataFrame(datadict)

    def __cmp__(self, dict_):
        return self.__cmp__(self._dict, dict_)

    def __contains__(self, item):
        return item in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __unicode__(self):
        return unicode(repr(self._dict))
