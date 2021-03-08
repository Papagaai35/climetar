import json
import os

import numpy as np
import matplotlib as mpl

class MetarTheme(object):
    default_theme = './resources/theme2_RHS.json'
    def __init__(self,json_or_file=None):
        self.theme = {}
        self.load(json_or_file if json_or_file is not None else self.default_theme)
    def load(self,json_or_file):
        if len(json_or_file)>255:
            self.load_json_str(json_or_file)
        else:
            if os.path.exists(json_or_file) and os.path.isfile(json_or_file):
                with open(json_or_file,'r') as fh:
                    self.theme = json.load(fh)
            else:
                self.load_json_str(json_or_file)
        if self.theme is None:
            raise ValueError('Invalid theme passed:\n%s'%json_or_file)
    def load_json_str(self,jsonstr):
        try:
            self.theme = json.loads(jsonstr)
        except:
            pass
    def validate_theme(self):
        assert("facecolor" in self.theme)
        assert("facealpha" in self.theme)
        assert("edgecolor" in self.theme)
        assert("edgealpha" in self.theme)
        assert("linewidth" in self.theme)
    
    @classmethod
    def argsplit(cls,args):
        return list((".".join(args)).split("."))
    @classmethod
    def is_digit(cls,digit):
        try:
            int(digit)
            return True
        except ValueError:
            return False
    def get_type(self,*args):
        args = ".".join(self.argsplit(args))
        types = {
            'line': ['line','median'],
            'bar': ['bar'],
        }
        for k,v in types.items():
            if any([args.startswith(s) for s in v]):
                return k
        return None
    def themeget(self,*args):
        args = self.argsplit(args)
        result = self.theme
        skiped_arg = 0
        for arg in list(args):
            if arg in result:
                skiped_arg = 0
                result = result[arg]
                if not isinstance(result,dict):
                    break
            else:
                if skiped_arg>1:
                    break
                skiped_arg += 1
        
        if isinstance(result,dict) and 'default' in result:
            return result['default']
        return result
    def to_style(self,fc,ec,lw,type=None):
        fc = fc if isinstance(fc,list) else [fc]
        ec = ec if isinstance(ec,list) else [ec]
        lw = lw if isinstance(lw,list) else [lw]
        maxlen = max(len(fc),len(ec),len(lw))
        result = []
        for i in range(maxlen):
            if type=='line':
                result.append({'color':ec[i%len(ec)],'linewidth':lw[i%len(ec)]})
            elif type=='bar':
                result.append({'color':fc[i%len(fc)],'edgecolor':ec[i%len(ec)],'linewidth':lw[i%len(ec)]})
            else:
                result.append({'facecolor':fc[i%len(fc)],'edgecolor':ec[i%len(ec)],'linewidth':lw[i%len(ec)]})
        return result
    def to_color(self,color,alpha='ff'):
        if isinstance(color,dict):
            return {k:self.to_color(c,alpha) for k,c in color.items() if '__' not in k}
        elif isinstance(color,list):
            return [self.to_color(c,alpha) for c in color]
        elif isinstance(alpha,dict):
            return {k:self.to_color(color,a) for k,a in alpha.items() if '__' not in k}
        elif isinstance(alpha,list):
            return [self.to_color(color,a) for a in alpha]
        else:
            return self.color_plus_alpha(color,alpha)
    def color_plus_alpha(self,color,alpha='ff'):
        if color=='none':
            color,alpha =  '#000000','00'
        alpha = alpha if isinstance(alpha,float) else int(alpha,16)/255
        return mpl.colors.to_hex(mpl.colors.to_rgba(color,alpha),keep_alpha=True)
    def get(self,*args):
        args = self.argsplit(args)
        fc = self.themeget('facecolor',*args),self.themeget('facealpha',*args)
        fc = self.to_color(*fc)
        ec = self.themeget('edgecolor',*args),self.themeget('edgealpha',*args)
        ec = self.to_color(*ec)
        lw = self.themeget('linewidth',*args)
        lw = 0. if (ec[0]=='#' and ec[-2:]=='00') or ec=='none' else lw
        return self.to_style(fc, ec, lw, self.get_type(*args))
    def get_ci(self,*args):
        return self.get("median",*args) + self.get("confidence",*args)
    def get_set(self,*args):
        args = self.argsplit(args)
        cs = self.themeget('colorsets',*args)
        fa = self.themeget('facealpha',*args)
        ec = self.themeget('edgecolor',*args),self.themeget('edgealpha',*args)
        ec = self.to_color(*ec)
        lw = self.themeget('linewidth',*args)
        lw = 0. if (ec[0]=='#' and ec[-2:]=='00') or ec=='none' else lw
        ty = self.get_type(*args)
        
        if all(self.is_digit(k) for k in cs.keys()):
            cs = dict(sorted({int(k):v for k,v in cs.items()}.items(),key=lambda i: i[0]))
        
        result = {}
        for k,c in cs.items():
            if isinstance(k,str) and '__' in k:
                continue
            result[k] = []
            fa = fa if isinstance(fa,list) else [fa]
            for a in fa:
                result[k].append(self.to_style(self.to_color(c,a),ec,lw,ty)[0])
        return result
    def get_setT(self,*args,indexes=None):
        stylesetT = []
        styleset = self.get_set(*args)
        sublevels = max(len(v) for v in styleset.values())
        indexes = indexes if indexes is not None else list(styleset.keys())
        for i in range(sublevels):
            stylesT = {}
            for name,style in styleset.items():
                if name not in indexes:
                    continue
                for k,v in style[i%len(style)].items():
                    if k not in stylesT:
                        stylesT[k] = []
                    stylesT[k].append(v)
            stylesetT.append(stylesT)
        if len(stylesetT)==1:
            return stylesetT[0]
        return stylesetT
    def get_from_set(self,*args):
        args = self.argsplit(args)
        fc = self.themeget('colorsets',*args),self.themeget('facealpha',*args)
        fc = self.to_color(*fc)
        ec = self.themeget('edgecolor',*args),self.themeget('edgealpha',*args)
        ec = self.to_color(*ec)
        lw = self.themeget('linewidth',*args)
        lw = 0. if (ec[0]=='#' and ec[-2:]=='00') or ec=='none' else lw
        return self.to_style(fc, ec, lw, self.get_type(*args))