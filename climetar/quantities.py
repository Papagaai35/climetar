"""Module for storing and parsing values with their units.

ToDo:
* Rewrite to use `pint` [https://pint.readthedocs.io/en/stable/]()
"""
import logging
_log = logging.getLogger(__name__)

import collections
import fractions
import numbers
import re
import unicodedata

import numpy as np
import pandas as pd

class Quantity(object):
    """Root object to store values with their unit.

    Attributes
    - units (dict):
        keys (str): unit name,
        value (int|list[callable,callable]): conversion from unit to storageunit
            if int: conversion factor (mutiplication; 1 unit = x storage_units)
            if list[callable,callable]: conversion to at i=0 and from i=1 the
                storage_unit
    - storage_unit (str): Unit in which the value is stored
    - default_unit (str): Unit in which the value is exported, by default
    - nanvalues (list[str]): List of values which should result in np.nan
    - nonnumeric (list[str]): List of units which do not use numbers, but
        strings to represent a value
    - alias_units (dict): a translation dict for units
        keys (str): unit alias
        value (str): unit as in cls.units
    - unitmods (list[callable]): list of modification-functions, used when
        searching for a specific unit, in order of priority.
    """
    units = { }
    storage_unit = ''
    default_unit = ''
    nanvalues = []
    nonnumeric = []
    alias_units = {}
    unitmods = [
        lambda x: x,
        lambda x: x.lower(),
        lambda x: re.sub('[^a-zA-Z]','',x),
        lambda x: unicodedata.normalize('NFD', x).encode(
            'ascii', 'ignore').decode("utf-8"),
        lambda x: re.sub('[^a-zA-Z]','',x).lower(),
        lambda x: unicodedata.normalize('NFD', x).encode(
            'ascii', 'ignore').decode("utf-8").lower(),
    ]

    def __init__(self,value,unit=None,default_unit=None):
        self.orig = value,unit
        self.value = np.nan
        self.greater = 0

        unit = None if unit=='' else unit
        self.default_unit = None if default_unit=='' else default_unit

        if self.no_default_unit() and not self.no_storage_unit():
            self.default_unit = self.storage_unit
        if unit is None and not self.no_default_unit():
            unit = self.default_unit
        elif unit is None and not self.no_storage_unit():
            unit = self.storage_unit
        assert unit is not None and isinstance(unit,str)

        if not (isinstance(value, numbers.Real) or unit in self.nonnumeric):
            value, self.greater = self.number_from_str(value)
        self.value = self.convert_from(value,unit)

    def __float__(self):
        val = self.value + self.greater
        if self.default_unit!=self.storage_unit:
            val = self.convert_to(val,self.default_unit)
        return float(val)
    def formatted_value(self,display_unit=None):
        val, unit = self.value, self.storage_unit
        if display_unit is not None:
            val, unit = self.convert_to(val,display_unit), display_unit
        elif (not self.no_default_unit()) and self.default_unit!=self.storage_unit:
            val, unit = self.convert_to(val,self.default_unit), self.default_unit
        return "%s%g %s"%(
            ['','>','<'][self.greater],
            val,
            '[-]' if unit=='' else unit)
    def __repr__(self):
        return "%s(%s)"%(self.__class__.__name__,self.formatted_value())
    def __getitem__(self,key):
        return self.convert_to(self.value+self.greater,key)
    
    def no_default_unit(self):
        return self.default_unit is None or self.default_unit==''
    def no_storage_unit(self):
        return self.storage_unit is None or self.storage_unit==''

    @classmethod
    def convert_from(cls,value,unit):
        conversion = cls.units[cls.find_unit(unit)]
        if (    isinstance(conversion, numbers.Real) and
                isinstance(value, numbers.Real)):
            return value * conversion
        elif isinstance(conversion, collections.Iterable):
            return conversion[0](value)
        else:
            raise ValueError('Could not %s convert from %s',(value,unit))

    @classmethod
    def convert_to(cls,value,unit):
        conversion = cls.units[cls.find_unit(unit)]
        if (    isinstance(conversion, numbers.Real) and
                isinstance(value, numbers.Real)):
            return value / conversion
        elif isinstance(conversion, collections.Iterable):
            return conversion[1](value)
        else:
            raise ValueError('Could not %s convert to %s',(value,unit))

    @classmethod
    def number_from_str(cls,number):
        if number is None or pd.isnull(number):
            return np.nan, 0
        lennum, sign, greater = len(number), 1, 0
        if (number in ['',('/'*lennum),('M'*lennum),('X'*lennum)] or
            number in cls.nanvalues):
            return np.nan, 0
        number = number.replace('O','0')
        if number.startswith('M'):
            sign = -1
            number = number[1:]
        if number.startswith('P'):
            greater = 1
            number = number[1:]
        if '/' in number:
            if ' ' in number:
                numbersum = fractions.Fraction('0')
                for numberpart in number.split():
                    numbersum += fractions.Fraction(numberpart)
                number = numbersum
            else:
                number = fractions.Fraction(number)
        return sign*float(number), greater

    @classmethod
    def get_unit_translation_dict(cls):
        unittrans = {}
        ignorelistall = []
        for mfn in cls.unitmods:
            ignorelistlevel = []
            for v in cls.units.keys():
                k = mfn(str(v))
                if k in ignorelistlevel and k in unittrans:
                    del unittrans[k]
                elif k not in ignorelistall:
                    unittrans[k] = v
                    ignorelistlevel.append(k)
            ignorelistall = ignorelistall + ignorelistlevel[:]
            ignorelistlevel = []
            for k, v in cls.alias_units.items():
                k = mfn(k)
                if k in ignorelistlevel and k in unittrans:
                    del unittrans[k]
                elif k not in ignorelistall:
                    unittrans[k] = v
                    ignorelistlevel.append(k)
            ignorelistall = ignorelistall + ignorelistlevel[:]
        return unittrans

    @classmethod
    def find_unit(cls,unit):
        if unit in cls.units:
            return unit
        if unit in cls.alias_units:
            return cls.alias_units[unit]
        unittrans = cls.get_unit_translation_dict()
        for mfn in cls.unitmods:
            k = mfn(str(unit))
            if k in unittrans:
                return unittrans[k]
        raise KeyError('The unit %s is not defined in %s'%(unit,cls.__name__))

class Direction(Quantity):
    compass_dirs = {
        'N':  0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,
        'E': 90.0, 'ESE':112.5, 'SE':135.0, 'SSE':157.5,
        'S':180.0, 'SSW':202.5, 'SW':225.0, 'WSW':247.5,
        'W':270.0, 'WNW':292.5, 'NW':315.0, 'WNW':337.5
    }

    units = {
        'deg': 1,
        'rad': np.pi/180,
        'compass': [
            lambda cmps: {
        'N':  0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,
        'E': 90.0, 'ESE':112.5, 'SE':135.0, 'SSE':157.5,
        'S':180.0, 'SSW':202.5, 'SW':225.0, 'WSW':247.5,
        'W':270.0, 'WNW':292.5, 'NW':315.0, 'WNW':337.5
    }.get(cmps,np.nan),
            lambda dirdeg: min({
        'N':  0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,
        'E': 90.0, 'ESE':112.5, 'SE':135.0, 'SSE':157.5,
        'S':180.0, 'SSW':202.5, 'SW':225.0, 'WSW':247.5,
        'W':270.0, 'WNW':292.5, 'NW':315.0, 'WNW':337.5
    }.items(),
                key=(lambda _, v: abs(v - target)))[0]
        ]
    }
    storage_unit = 'deg'
    alias_units = {'°': 'deg'}
    nanvalues = ['VRB','NDV']
    nonnumeric = ['compass']

class Distance(Quantity):
    units = {
        'km': 1000,
        'm': 1,
        'cm': 0.01,
        'mm': 0.001,
        'ft': 0.3048,
        'in': 2.54e-2,
        'mi': 1609.344,
        'NM': 1852,
        'FL': 3048,
    }
    storage_unit = 'm'
    alias_units = {'SM': 'mi'}
    
class Height(Quantity):
    units = {
        'km': 1000/0.3048,
        'm': 1/0.3048,
        'cm': 0.01/0.3048,
        'mm': 0.001/0.3048,
        'ft': 1,
        'in': 1/12,
        'mi': 5280,
        'NM': 1852/0.3048,
        'FL': 100,
    }
    storage_unit = 'ft'
    alias_units = {'SM': 'mi'}
    
    default_unit = 'ft'
    @classmethod
    def of_cloud(cls,value):
        value, _ = cls.number_from_str(value)
        if not pd.isnull(value):
            value = (value-90)*1000 if value>100 else value*100
        return cls(value,'ft')

class Speed(Quantity):
    units = {
        'm/s': 1,
        'km/h': (1/3.6),
        'kt': (1852/3600),
        'mi/h': 0.44704,
        'ft/s': 0.3048,
    }
    storage_unit = 'm/s'
    alias_units = {'mps': 'm/s',
        'kph': 'km/h', 'kmph': 'km/h',
        'kts': 'kt', 'kn':'kt', 'knts': 'kt', 'kns': 'kt', 'knots': 'kt',
        'LT': 'kt', 'k': 'kt', 't': 'kt'}

class Temperature(Quantity):
    units = {
        'K': [lambda temp_k: temp_k - 273.15,
              lambda temp_c: temp_c + 273.15],
        '°C': 1,
        'd°C': .1,
        '°F': [lambda temp_f: (temp_f - 32)*(5/9),
              lambda temp_c: temp_c*(9/5) + 32],
        '°R': [lambda temp_r: (temp_r - 491.67)*(5/9),
              lambda temp_c: (temp_c + 273.15)*(9/5)],
        '°De':[lambda temp_de: 100-temp_de*(2/3),
              lambda temp_c: (100-temp_c)*(3/2)],
        '°N': 100/33,
        '°Ré': 1.25,
        '°Ro': [lambda temp_ro: (temp_ro-7.5)*(40/21),
               lambda temp_c: temp_c*(21/40)+7.5]}
    storage_unit = '°C'
    alias_units = {'°Rø': '°Ro'}

class Pressure(Quantity):
    units = {
        'Pa': 0.01,
        'daPa': 0.1,
        'hPa': 1,
        'kPa': 10,
        'inHg': 33.8639,
        '10thHg': 0.338639,
    }
    storage_unit = 'hPa'
    alias_units = {'A': '10thHg', 'ALSTG': '10thHg', 'Q': 'hPa', 'QNH': 'hPa'}
    @classmethod
    def missing_first_digit_daPa(cls,value):
        value, _ = cls.number_from_str(value)
        if not pd.isnull(value):
            value += 9000. if value>=500 else 10000.
        return cls(value,'daPa')

class Fraction(Quantity):
    units = {
        'frac': 1,
        '%': 1e-2,
        'pm': 1e-3,
        'pmy': 1e-4,
        'pcm': 1e-5,
        'ppm': 1e-6,
        'ppb': 1e-9,
        'ppt': 1e-12,
        'ppq': 1e-15
    }
    storage_unit = 'frac'
    alias_units = {
        'pc': '%',
        'percent': '%',
        'per cent': '%',
        '-': 'frac',
        '/1': 'frac'
    }
    unitmods = [
        lambda x: x,
        lambda x: x.lower(),
        lambda x: unicodedata.normalize('NFD', x).encode(
            'ascii', 'ignore').decode("utf-8"),
        lambda x: unicodedata.normalize('NFD', x).encode(
            'ascii', 'ignore').decode("utf-8").lower(),
    ]
