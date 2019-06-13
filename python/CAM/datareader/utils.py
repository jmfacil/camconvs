from threading import Thread
import sys
import time
write = sys.stdout.write
flush = sys.stdout.flush
sleep = time.sleep
from IPython.display import display, HTML, clear_output



class MessageWaiting(Thread):
    def __init__(self,msg='Waiting',waiting_char='.', count = 3,end_msg = 'DONE',**kargs):
        """Constructor"""
        Thread.__init__(self,**kargs)
        self.msg = msg
        self.end_msg = end_msg
        self.c = waiting_char
        self.done = False
        self.count = count
    def finish_wait(self,new_end = ''):
        if not (new_end == ''):
            self.end_msg = new_end
        self.done = True
        self.join()
    def run(self,):
        #shell = sys.stdout.shell
        write(self.msg)
        write(' ')
        flush()
        dotes = int(self.count) * self.c
        while not self.done:
            for dot in dotes:
                write(dot)
                flush()
                sleep(0.1)
            for dot in dotes:
                write('\b \b')
                flush()
        for _ in self.msg:
            write('\b \b')
            flush()
        write(self.msg)
        write(' ')
        for dot in dotes:
            write(' ')
            flush()
        
        write(self.end_msg)
        write('\n')
        flush()
import numpy as np
class ProgressBar(Thread):
    def __init__(self,msg='Progressing',waiting_char='█', length = 20,end_msg = 'COMPLETED',total=1.0,**kargs):
        """Constructor"""
        Thread.__init__(self,**kargs)
        self.msg = msg
        self.end_msg = end_msg
        self.c = waiting_char
        self.done = False
        self.length = length
        self.total = float(total)
        self.current = float(0)
        
    def finish_wait(self,new_end = ''):
        if not (new_end == ''):
            self.end_msg = new_end
        self.done = True
        self.join()
    def update(self,current = -1):
        if not (current == -1):
            self.current = float(current)
    def run(self,):
        def charbar(prog,lon):
            _bar = ''
            for i in range(lon):
                if  float(i)/float(lon)<prog:
                    _bar+=self.c
                else:
                    _bar+=' '
            return _bar
        if self.total==0:
            return 
        initial_time = time.time()
        write(' ')
        w = 1
        while not self.done:
            try:
                get_ipython
                clear_output(True)
            except:
                write('\b'*w)
            
            #write("\033[F")
            flush()
            line = self.msg+' '
            bar = '['+charbar(self.current/self.total,self.length)+']  '
            percent = '{0:.2%} '.format(self.current/self.total)
            elapsed_time = time.time()-initial_time
            
            stime = '| elapsed time: '+time.strftime('%H:%M:%S',time.gmtime(elapsed_time))
            if (self.current/self.total)<=0:
                time_to_finish = 0
            else:
                time_to_finish = elapsed_time/(self.current/self.total)*(1.-self.current/self.total)
                stime=stime+' | time to finish: '+time.strftime('%H:%M:%S',time.gmtime(time_to_finish))
            toolbar = line+bar+percent+stime
            w = len(toolbar)
            write(line+bar+percent+stime)
            flush()
            sleep(0.1)
        write('   '+self.end_msg)
        write('\n')
        flush()
        
        
