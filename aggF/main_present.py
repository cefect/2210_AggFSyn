'''
Created on Dec. 31, 2021

@author: cefect

presentation plots
see aggF.main.py for additional docstring
'''


#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy, logging, sys
import pandas as pd
import numpy as np
 

import scipy.stats 
 
print('loaded scipy: %s'%scipy.__version__)

start = datetime.datetime.now()
print('start at %s' % start)


 
idx = pd.IndexSlice

#===============================================================================
# setup logger----
#===============================================================================
logging.basicConfig(
                #filename='xCurve.log', #basicConfig can only do file or stream
                force=True, #overwrite root handlers
                stream=sys.stdout, #send to stdout (supports colors)
                level=logging.INFO, #lowest level to display
                )
 
    
    
#===============================================================================
# setup matplotlib----------
#===============================================================================
output_format='svg'
usetex=False #not working with font
add_stamp=True
 
os.environ['PATH'] += r";C:\Users\cefect\AppData\Local\Programs\MiKTeX\miktex\bin\x64"

cm = 1/2.54
import matplotlib
#matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt
 
#set teh styles
plt.style.use('default')
 
#font
font_size=18
matplotlib.rc('font', **{'family' : 'sans-serif','sans-serif':'Tahoma','weight' : 'normal','size':12})
 
 
for k,v in {
    'axes.titlesize':font_size,
    'axes.labelsize':font_size,
    'xtick.labelsize':12,
    'ytick.labelsize':12,
    'figure.titlesize':12,
    'figure.autolayout':False,
    'figure.figsize':(32 * cm, 14 * cm),
    'legend.title_fontsize':'large',
    'text.usetex':usetex,
    }.items():
        matplotlib.rcParams[k] = v
  
print('loaded matplotlib %s'%matplotlib.__version__)

#===============================================================================
# custom imports
#===============================================================================
 
from aggF.da import Session_AggF
        
        
 
def _pipe_funcs_synthX(ses, fp_d, vid_l, legend=True):
    """pipeline for plot_matrix_funcs_synthX
    
    todo: add this to the class"""
    plt.close('all')
    #===========================================================================
    # load
    #===========================================================================
    
    if not 'rl_xm_stats_dxcol' in fp_d:
        dx = ses.get_rl_xm_stats_dxcol(fp_d['rl_xmDisc_dxcol'], vid_l, write=True)
    else:
        dx = pd.read_pickle(fp_d['rl_xm_stats_dxcol']).loc[idx[vid_l, :, :, :], :]
 
    # #load vfuncs
 
    if not 'f_serx' in fp_d:
        f_serx = ses.load_vfunc_ddf(vid_l)
    else:
        f_serx = pd.read_pickle(fp_d['f_serx']).loc[idx[vid_l, :]]
    #=======================================================================
    # plot
    #=======================================================================
 
    
    return  ses.plot_matrix_funcs_synthX(dx, f_serx=f_serx, write=True, figsize=(32 * cm, 7 * cm),
                                         legend=legend)


def _pipe_rlDelta_xb(ses, fp_d, vid_l, 
                    aggLevel_l=[5, 100],
                    legend=True):
    """Pipeline for plot_matrix_rlDelta_xb
    
    
    """
    plt.close('all')
    log = ses.logger.getChild('rlDelta')
    #=======================================================================
    # load
    #=======================================================================
    # xmean RL values per AggLevel
    rl_dxcol = pd.read_pickle(fp_d['rl_dxcol'])
    mdex = rl_dxcol.columns  # vid_l = list(mdex.unique('df_id'))
    
    # function meta
    vid_df = ses.build_vid_df(vid_l=list(mdex.unique('df_id')), write=False, write_model_summary=True)  # vfunc data
    log.info('loaded %i models from %i libraries' % (len(vid_l), len(vid_df['model_id'].unique())))
    
    #=======================================================================
    # compute deltas
    #=======================================================================
    # get s1 base values
    rl_s1_dxcol = rl_dxcol.loc[:, idx[:,:, 0]].droplevel('aggLevel', axis=1)
    
    # get deltas
    serx = rl_dxcol.subtract(rl_s1_dxcol).drop(0, level='aggLevel', axis=1).unstack()
    
    
    
    # add model_id to idnex
    serx.index = serx.index.join(
        pd.MultiIndex.from_frame(vid_df['model_id'].reset_index()))
    # get model ids of interesplot_matrix_funcs_synthXt from function list
    df = serx.index.to_frame()
    mid_l = df[df['df_id'].isin(vid_l)]['model_id'].unique()
    
    #=======================================================================
    # plot deltas
    #=======================================================================
    if aggLevel_l is None:
        aggLevel_l = serx.index.unique('aggLevel').tolist()
 
    """color map can only support 8"""
    #filter
    serx_slice = serx.loc[idx[:,:,aggLevel_l,:,
            mid_l]]
            # [6, 16, 17, 27, 37, 44]
    log.info('for models\n    %s' % serx_slice.index.unique('model_id'))
    
    
    #===========================================================================
    # #plot
    #===========================================================================
     
    
    
    return ses.plot_matrix_rlDelta_xb(serx_slice,
 
        color_d={3:'#d95f02', 37:'#1b9e77'}, # divergent for colobrlind
        legend=legend, output_format='png',
        ) 


def _pipe_errorAreaBoxes(ses, fp_d, vid_l = [
                #798,
                 811, 
                 49,
                 ],
                 aggLevel_l=[5, 100],
                 ):
    """pipeline for plot_eA_box (get_eA_box_fig)
        (matrix of box plots showing error metrics by model id)
    
    TODO: add this to session
    remove plot_eA_box and just call get_eA_box_fig directly
    """
    log = ses.logger.getChild('errorAreaBoxes')
    plt.close('all')
    #=======================================================================
    # load
    #=======================================================================
    # xmean RL values per AggLevel
    rl_dxcol = pd.read_pickle(fp_d['rl_dxcol'])
    mdex = rl_dxcol.columns
    
    # function meta
    vid_df = ses.build_vid_df(vid_l=list(mdex.unique('df_id')), write=False, write_model_summary=True)  # vfunc data
    log.info('loaded %i models from %i libraries' % (len(vid_l), len(vid_df['model_id'].unique())))
    
    if 'errArea_dxcol' in fp_d:
        errArea_dxcol = pd.read_pickle(fp_d['errArea_dxcol'])
    else:
        errArea_dxcol = ses.build_model_metrics(dxcol=rl_dxcol)  # see AggSession1F.calc_areas()
    #=======================================================================
    # prep data
    #=======================================================================
    serx = errArea_dxcol.unstack()
    # add model_id to idnex
    serx.index = serx.index.join(pd.MultiIndex.from_frame(vid_df['model_id'].reset_index())).reorder_levels(['model_id', 'df_id', 'xvar', 'aggLevel', 'errArea_type'])
    serx = serx.sort_index(sort_remaining=True)
    
    if aggLevel_l is None:
        aggLevel_l = serx.index.unique('aggLevel').tolist()
        
    #slice to the error metric of interest
    errArea_dxcol1=errArea_dxcol.loc[['total'],idx[:, :, aggLevel_l]]
    #=======================================================================
    # plot
    #=======================================================================
 
    # # set of box plots for area errors, on different groups
 
    return ses.plot_eA_box(dxcol=errArea_dxcol1, vid_df=vid_df,
        grp_colns=['model_id'],
        #figsize=(17 * cm, 11 * cm),
        sharex='all', sharey='all', add_subfigLabel=False, set_ax_title=False,
        ylab='$e$',
        ylims=(-21, 21), logger=log)


def workflow_plotting(
        fp_d={},
        **kwargs):
    """generic workflow for all the plots"""
    
    with Session_AggF(
        logger = logging.getLogger('r'),output_format=output_format,add_stamp=add_stamp,
        **kwargs) as ses:
        
 
        #=======================================================================
        # defaults
        #=======================================================================
        log = ses.logger.getChild('r')
 
        #=======================================================================
        # dfunc vs. agg RL------
        #=======================================================================
        """plotting individually"""
        vid_l = [
                #798,
                 811, 
                 #49,
                 ]
        
        #_pipe_funcs_synthX(ses, fp_d, vid_l, legend=False)
  
        #=======================================================================
        # rl mean vs. xb--------
        #=======================================================================
        vid_l = [
                #798,
                 811, 
                 49,
                 ]
                
        #_pipe_rlDelta_xb(ses, fp_d, vid_l, legend=True)
 
        
        
 
        #=======================================================================
        # error area---------
        #=======================================================================
        _pipe_errorAreaBoxes(ses, fp_d, vid_l = [
                #798,
                 811, 
                 49,
                 ])        
        
 
        
        #=======================================================================
        # write stats
        #=======================================================================
        # calc some stats and write to xls
        
        #=======================================================================
        # ses.run_err_stats(dxcol = errArea_dxcol, vid_df=vid_df)
        out_dir = ses.out_dir
        #=======================================================================
        
    return out_dir
        

 
 
    
 
    
def all_r0_plot(**kwargs):
    return workflow_plotting(
        run_name='r0_present',
        fp_d = {        
                'rl_xmDisc_dxcol':  r'C:\LS\10_IO\2112_Agg\outs\agg1F\r0_all\20220205\working\aggErr1_r2_0104_rl_xmDisc_dxcol.pickle',
                #'rl_xmDisc_xvals':  r'C:\LS\10_IO\2112_Agg\outs\agg1F\r0_all\20220205\working\aggErr1_r2_0104_rl_xmDisc_xvals.pickle',
                'rl_dxcol':         r'C:\LS\10_IO\2112_Agg\outs\agg1F\r0_all\20220205\working\aggErr1_r3_0104_rl_dxcol.pickle',
                'errArea_dxcol':    r'C:\LS\10_IO\2112_Agg\outs\agg1F\r0\20221026\agg1F_r0_1026_model_metrics.pickle',
                'rl_xm_stats_dxcol':r'C:\LS\10_IO\2112_Agg\outs\agg1F\r0\20221028\agg1F_r0_1028_rl_xm_stats_dxcol.pkl',
                'f_serx':           r'C:\LS\10_IO\2112_Agg\outs\agg1F\r0\20221028\agg1F_r0_1028_vfunc_df.pkl',
            },
                
        **kwargs)
    


if __name__ == "__main__": 
    
    #run_aggErr1(vid_l = [798,811,49])
    output=all_r0_plot()
 
    
    #output=r1_3mods()
    #output=r1_3mods_plot()
    
 
    #output=dev()
    #output=dev_plot()
    
    
    
 
 
 
    
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))
