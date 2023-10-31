'''
Created on Mar. 25, 2022

@author: cefect
'''
import os
proj_dir = r'C:\LS\09_REPOS\02_JOBS\2210_AggFSyn'
src_dir = proj_dir
src_name='aggF'

 

logcfg_file=r'C:\LS\09_REPOS\01_COMMON\coms\logger.conf'

root_dir=r'C:\LS\10_IO\2112_Agg'
wrk_dir=root_dir
 

#add latex engine
os.environ['PATH'] += r";C:\Users\cefect\AppData\Local\Programs\MiKTeX\miktex\bin\x64"