import sys
import logging
import traceback

def getLogger(name):
    logging.setLogRecordFactory(getNewLogRecordFactory())
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(getLogFileHandler())
    logger.addHandler(getStdOutHandler())
    logger.addHandler(getStdErrHandler())
    try:
        ipython_shell = get_ipython()
        ipython_shell.set_custom_exc((Exception,),getIpythonExceptionHandler(logger))
    except NameError:
        sys.excepthook = getExceptionHandler(logger)
    return logger

class TracebackInfoFilter(logging.Filter):
    def __init__(self, clear=True):
        self.clear = clear
    def filter(self, record):
        if self.clear:
            record._exc_info_hidden, record.exc_info = record.exc_info, None
            record.exc_text = None
        elif hasattr(record, "_exc_info_hidden"):
            record.exc_info = record._exc_info_hidden
            del record._exc_info_hidden
        return True
class LevelFilter(logging.Filter):
    def __init__(self, low, high):
        self._low = low
        self._high = high
        logging.Filter.__init__(self)
    def filter(self, record):
        if self._low <= record.levelno <= self._high:
            return True
        return False

class LogFileFormatter(logging.Formatter):
    def __init__(self,indent=4):
        fmt = '%(levelname)-8s %(asctime)s %(name)-20s %(pathname)s:%(lineno)d %(funcName)s\n%(message)s'
        self.indent = indent
        logging.Formatter.__init__(self, fmt)
    def format(self,record):
        msg = logging.Formatter.format(self, record).split("\n")
        msgi = [msg[0]] + [self.indent*" "+m for m in msg[1:]]
        return "\n".join(msgi)

def getNewLogRecordFactory():
    oldFactory = logging.getLogRecordFactory()
    def newLogRecordFactory(name,level,fn,lno,msg,args,exc_info,func=None,sinfo=None,**kwargs):
        if exc_info is not None and len(exc_info)>2 and level>=logging.CRITICAL:
            tb = traceback.extract_tb(exc_info[2])
            fn = tb[-1].filename
            lno = tb[-1].lineno
            func = tb[-1].name
        return oldFactory(name,level,fn,lno,msg,args,exc_info,func,sinfo,**kwargs)
    return newLogRecordFactory

def getLogFileHandler(filename='climetar.log'):
    maxBytes = 15*(1024**2) #15 MiB
    fhand = logging.handlers.RotatingFileHandler(filename,'a',maxBytes=maxBytes,backupCount=2)
    fhand.setLevel(logging.INFO)
    fhand.addFilter(TracebackInfoFilter(clear=False))
    fhand.setFormatter(LogFileFormatter())
    return fhand
def getStdOutHandler(stream=sys.stdout):
    shand = logging.StreamHandler(stream=stream)
    shand.setLevel(logging.WARNING)
    shand.addFilter(LevelFilter(25,35))
    shand.addFilter(TracebackInfoFilter(clear=True))
    sfmtr = logging.Formatter('[Waarschuwing] %(message)s')
    shand.setFormatter(sfmtr)
    return shand
def getStdErrHandler(stream=sys.stderr):
    shand = logging.StreamHandler(stream=stream)
    shand.setLevel(logging.ERROR)
    shand.addFilter(TracebackInfoFilter(clear=True))
    sfmtr = logging.Formatter('[Fout] %(message)s\nZie climetar.log voor meer info...')
    shand.setFormatter(sfmtr)
    return shand

def getExceptionHandler(logger):
    def handle_exception(exc_type,exc_value,exc_traceback):
        if issubclass(exc_type,KeyboardInterrupt):
            logger.error('Interrupted by user',exc_info=(exc_type, exc_value, exc_traceback))
            #sys.__excepthook__(exc_type, exc_value, exc_traceback)
        else:
            logger.critical(('%s: %s'%(exc_type.__name__,exc_value)),exc_info=(exc_type, exc_value, exc_traceback))
    return handle_exception
def getIpythonExceptionHandler(logger):
    handle_exception = getExceptionHandler(logger)
    def handle_ipython_exception(shell,exc_type,exc_value,exc_traceback,tb_offset=None):
        handle_exception(exc_type,exc_value,exc_traceback)
        return traceback.format_tb(exc_traceback)
    return handle_ipython_exception
