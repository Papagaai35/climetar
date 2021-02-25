import json
import os

import numpy as np
import matplotlib as mpl

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
    def to_color(cls,color,default_alpha=None,dict_keys=None,deep_dict_keys=None,default_color='k'):
        if default_alpha is None:
            default_alpha = [1]
        if hasattr(dict_keys,'__iter__') and not isinstance(dict_keys,list):
            dict_keys = list(dict_keys)
        
        if isinstance(color,np.ndarray):
            kwargs = dict(default_alpha=default_alpha,
                            dict_keys=dict_keys,
                            deep_dict_keys=deep_dict_keys,
                            default_color=default_color)
            if len(color.shape)==1:
                return cls.to_color(tuple(list(color)),**kwargs)
            else:
                return cls.to_color(list(color),**kwargs)
        elif isinstance(color,dict):
            if dict_keys is None:
                raise ValueError('Error 1: A list of colors is expected')
            key_seen = set()
            key_seen_add = key_seen.add
            keys = [k for k in (dict_keys+list(color.keys())) if not (k in key_seen or key_seen_add(k))]
            colordict = {}
            for k in keys:
                ddk = None
                if isinstance(deep_dict_keys,dict) and k in deep_dict_keys:
                    ddk = deep_dict_keys[k]
                if k in color:
                    colordict[k] = cls.to_color(color[k],default_alpha,dict_keys=ddk)
                elif 'default' in color:
                    colordict[k] = cls.to_color(color['default'],default_alpha,dict_keys=ddk)
                else:
                    colordict[k] = cls.to_color(default_color,default_alpha,dict_keys=ddk)
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
        default_color = kwargs.get('default_color','k')
        if fnname in self.theme:
            c = colors if colors is not None else self.theme[fnname].get('colors',default_color)
            e = edgecolor if edgecolor is not None else self.theme[fnname].get('edgecolor','none')
            l = linewidth if linewidth is not None else self.theme[fnname].get('linewidth',.5)
        else:
            c = colors if colors is not None else default_color
            e = edgecolor if edgecolor is not None else 'none'
            l = linewidth if linewidth is not None else .5
        c = self.to_color(c,**kwargs)
        e = self.to_color(e)[0] if e not in ['none',None] else 'none'
        l = 0 if e=='none' else l
        return c,e,l
    def cmapcel(self,fname,colors=None,edgecolor=None,linewidth=None,cmap=None,
                default_alpha=None,dict_keys=None,default_cmap=None,default_color='k'):
        if fname in self.theme:
            cm = cmap if cmap is not None else self.theme[fnname].get('cmap')
            c = colors if colors is not None else self.theme[fnname].get('colors')
            e = edgecolor if edgecolor is not None else self.theme[fnname].get('edgecolor','none')
            l = linewidth if linewidth is not None else self.theme[fnname].get('linewidth',.5)
        else:
            cm, c = cmap, colors
            e = edgecolor if edgecolor is not None else 'none'
            l = linewidth if linewidth is not None else .5
        if cm is None and default_cmap:
            cm = default_cmap
        if c is None:
            if cm is not None:
                cmap_obj = mpl.cm.get_cmap(cm)
                if dict_keys is not None:
                    cs = cmap_obj(np.linspace(0,1,max(1,len(dict_keys))))
                    c = dict(zip(dict_keys,cs))
                else:
                    c = cmap_obj(max(1,len(default_alpha or [1])))                
            else:
                c = default_color
        c = self.to_color(c,default_alpha=default_alpha,dict_keys=dict_keys)
        e = self.to_color(e)[0] if e not in ['none',None] else 'none'
        l = 0 if e=='none' else l
        return cm, c, e, l
