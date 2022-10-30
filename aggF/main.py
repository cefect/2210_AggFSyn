'''
Created on Dec. 31, 2021

@author: cefect

explore errors in impact estimates as a result of aggregation using pdist generated depths
    let's use hp.coms, but not Canflood
    using damage function csvs from figurero2018 (which were pulled from a db)
    
trying a new system for intermediate results data sets
    key each intermediate result with corresponding functions for retriving the data
        build: calculate this data from scratch (and other intermediate data sets)
        compiled: load straight from HD (faster)
        
    in this way, you only need to call the top-level 'build' function for the data set you want
        it should trigger loads on lower results (build or compiled depending on what filepaths have been passed)
        
        
TODO: migrate to new oop
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
output_format='pdf'
usetex=True
if usetex:
    os.environ['PATH'] += R";C:\Users\cefect\AppData\Local\Programs\MiKTeX\miktex\bin\x64"

cm = 1/2.54
import matplotlib
#matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt
 
#set teh styles
plt.style.use('default')
 
#font
matplotlib.rc('font', **{
        'family' : 'serif', 
        'weight' : 'normal',
        'size'   : 8})
 
 
for k,v in {
    'axes.titlesize':10,
    'axes.labelsize':10,
    'xtick.labelsize':8,
    'ytick.labelsize':8,
    'figure.titlesize':12,
    'figure.autolayout':False,
    'figure.figsize':(17 * cm, 10 * cm),
    'legend.title_fontsize':'large',
    'text.usetex':usetex,
    }.items():
        matplotlib.rcParams[k] = v
  
print('loaded matplotlib %s'%matplotlib.__version__)

#===============================================================================
# custom imports
#===============================================================================
from aggF.scripts import AggSession1F, view
from aggF.da import Session_AggF
        
        
def run_plotVfunc( 
        tag='r1',
        
 
        #data selection
        #vid_l=[796],
        vid_l=None,
        gcoln = 'model_id', #how to spilt figures
        style_gcoln = 'sector_attribute',
        max_mod_cnt=10,
        vid_sample=10,
        
         selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[
                              1, 2, 
                              3, #flemo 
                              4, 6, 7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47
                              ],
                          'function_formate_attribute':['discrete'], #discrete
                          'damage_formate_attribute':['relative'],
                          'coverage_attribute':['building'],
                         
                         },
        
        #style
        figsize=(6,6),
        xlims=(0,2),
        ylims=(0,100),
        #=======================================================================
        # #debugging controls
        #=======================================================================
        #=======================================================================
 
        # 
        #by record count
        #debug_len=20,

        # 
        # #use some preloaded data (saves lots of time during loading)
        # debug_fp=r'C:\LS\10_OUT\2107_obwb\outs\9vtag\r0\20211230\raw_9vtag_r0_1230.csv',
        # #debug_fp=None,
        #=======================================================================

        ):
 
    
 
    with Session(tag=tag,  overwrite=True,  name='plotAll',
                 bk_lib = {
                     'vid_df':dict(
                      vid_l=vid_l, max_mod_cnt=max_mod_cnt, selection_d=selection_d, vid_sample=vid_sample
                                    ),
                     }
                 ) as ses:
        
 
        
 
        vid_df = ses.retriee('vid_df')
 
 
        """
        view(vid_df)
        view(gdf)
        
        """
        
        for k, gdf in vid_df.groupby(gcoln):
            if not len(gdf)<=20:
                ses.logger.warning('%s got %i...clipping'%(k, len(gdf)))
                gdf = gdf.iloc[:20, :]
                                   
 
            phndl_d = ses.get_plot_hndls(gdf, coln=style_gcoln)
            
            fig = ses.plot_all_vfuncs(phndl_d=phndl_d, vid_df=gdf,
                         figsize=figsize, xlims=xlims,ylims=ylims,
                         title='%s (%s) w/ %i'%(gdf.iloc[0,:]['abbreviation'], k, len(phndl_d)))
            
            ses.output_fig(fig, fname='%s_vfunc_%s'%(ses.resname, k))
            
            #clear vcunfs
            del ses.data_d['vf_d']
            
        
        out_dir = ses.out_dir
        
    return out_dir


def run_aggErr1(#agg error per function

        
        #selection

        vid_l=None,
        
        vid_sample=None,
        max_mod_cnt=None,
        
         selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[
                              #1, 2, #continous 
                              3, #flemo 
                              4, 6, 
                              7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47
                              ],
                          'function_formate_attribute':['discrete'], #discrete
                          'damage_formate_attribute':['relative'],
                          'coverage_attribute':['building'],
                         
                         },
         
         #run control
         rl_xmDisc_dxcol_d = dict(
                xdomain=(0,2), #min/max of the xdomain
                xdomain_res = 10, #number of increments for the xdomain
                
                aggLevels_l= [2, 
                             #5, 
                             10,
                             ],
                
                #random depths pramaeters
                xvars_ar = np.linspace(.5,1,num=2), #varice values to iterate over
                statsFunc = scipy.stats.norm, #function to use for the depths distribution
                depths_resolution=500,  #number of depths to draw from the depths distritupion
                          ),
         plot_rlMeans = True,
         overwrite=True,
 
         #simple retrival
         rl_xmDisc_dxcol=None,
        
        **kwargs):
 
    with AggSession1F(overwrite=overwrite,
                 bk_lib={
                     'vid_df':dict(
                            selection_d=selection_d, vid_l=vid_l, vid_sample=vid_sample, max_mod_cnt=max_mod_cnt,
                                    ),
                     'rl_xmDisc_dxcol':rl_xmDisc_dxcol_d,
                     'rl_dxcol':dict(plot=plot_rlMeans),
                            },
                 # figsize=figsize,
                 logger = logging.getLogger('r'),
                 **kwargs) as ses:
        ses.plt = plt
 
        # plot discretization figures (very slow)
        # ses.plot_xmDisc()        
        
        #=======================================================================
        # retrieve for subsetting
        #=======================================================================
        vid_df = ses.build_vid_df(vid_l=vid_l, write=False) #vfunc data
        vf_d = ses.build_vf_d(vid_df=vid_df) #vfunc workers
        rl_xmDisc_dxcol = ses.build_rl_xmDisc_dxcol(vf_d=vf_d, **rl_xmDisc_dxcol_d) #aggregation erros mean-discretized for all vfuncs
        
        
        #=======================================================================
        # ploters
        #=======================================================================
        # nice plot per-func of means at different ag levels
        """workaround as this is a combined calc+plotting func"""        
        if plot_rlMeans:
            rl_dxcol = ses.build_rl_dxcol(dxcol=rl_xmDisc_dxcol, plot=plot_rlMeans, vf_d=vf_d, vid_df=vid_df)
 
        
        out_dir = ses.out_dir
        
    return out_dir

def plot_aggF_errs(
        fp_d={},
        **kwargs):
    
    with Session_AggF(
        logger = logging.getLogger('r'),output_format=output_format,
        **kwargs) as ses:
        
        ses.output_format
        #=======================================================================
        # defaults
        #=======================================================================
        log = ses.logger.getChild('r')
        
        #=======================================================================
        # data explore
        #=======================================================================
        #=======================================================================
        # dxcol_xvals = pd.read_pickle(fp_d['rl_xmDisc_xvals']) #raw xamples
        # dxcol = pd.read_pickle(fp_d['rl_xmDisc_dxcol']) #raw xamples
        #=======================================================================
        
        #=======================================================================
        # rl mean vs. xb--------
        #=======================================================================
        #=======================================================================
        # load        
        #=======================================================================
        #xmean RL values per AggLevel
        rl_dxcol = pd.read_pickle(fp_d['rl_dxcol'])        
        mdex = rl_dxcol.columns        
        vid_l = list(mdex.unique('df_id'))
        
        #function meta
        vid_df = ses.build_vid_df(vid_l=vid_l,write=False, write_model_summary=True) #vfunc data
        
        """
        vid_df.to_csv(os.path.join(ses.out_dir, 'vid_df.csv'))
        view(vid_df)
        
        vid_df.drop_duplicates('model_id').to_csv(os.path.join(ses.out_dir, 'vid_df_models.csv'))
        """
        
        log.info('loaded %i models from %i libraries'%(len(vid_l), len(vid_df['model_id'].unique())))
        
        #=======================================================================
        # compute deltas
        #=======================================================================
        #get s1 base values
        rl_s1_dxcol = rl_dxcol.loc[:, idx[:, :, 0]].droplevel('aggLevel', axis=1)
        
        #get deltas
        serx = rl_dxcol.subtract(rl_s1_dxcol).drop(0, level='aggLevel', axis=1).unstack()
        
        #add model_id to idnex        
        serx.index = serx.index.join(
            pd.MultiIndex.from_frame(vid_df['model_id'].reset_index())
            )
        
        #=======================================================================
        # plot deltas
        #=======================================================================
        
        #serx = serx.drop(3, level='model_id')
        """color map can only support 8"""
        serx = serx.loc[idx[:,:,:,:,[6, 16, 17, 27, 37, 44]]]
        log.info('for models\n    %s'%serx.index.unique('model_id'))
        
        
        ses.plot_matrix_rlDelta_xb(serx)
        
 
        return
        #=======================================================================
        # error area---------
        #=======================================================================
        if 'errArea_dxcol' in fp_d:
            errArea_dxcol = pd.read_pickle(fp_d['errArea_dxcol'])
        else:
            #see AggSession1F.calc_areas()
            errArea_dxcol = ses.build_model_metrics(dxcol=rl_dxcol)
        
        #=======================================================================
        # prep data
        #=======================================================================
        serx = errArea_dxcol.unstack() 
        #add model_id to idnex        
        serx.index = serx.index.join(
            pd.MultiIndex.from_frame(vid_df['model_id'].reset_index())
            ).reorder_levels(['model_id', 'df_id', 'xvar', 'aggLevel', 'errArea_type']
                             )
        serx = serx.sort_index(sort_remaining=True)
        #=======================================================================
        # plot
        #=======================================================================
        
        ses.plot_matrix_rlDelta_xb(serx)
        #=======================================================================
        # # set of box plots for area errors, on different groups
        #======================================================================= 
        #get_eA_box_fig()
        ses.plot_eA_box(dxcol=errArea_dxcol.loc[['total'], :], vid_df=vid_df,
                        grp_colns=['model_id'],
                        figsize=(17 * cm, 12 * cm),
                        sharex='all', sharey='all', add_subfigLabel=True, set_ax_title=False,
                        ylab = '$e_{total}$',
                        ylims=(-21, 21),
                        )
        
 
        
        
 
        # per-model bar plots of error area at different aggLevels and xvars
        #ses.plot_eA_bars()
        
        #=======================================================================
        # write stats
        #=======================================================================
        # calc some stats and write to xls
        
        ses.run_err_stats(dxcol = errArea_dxcol, vid_df=vid_df)
        out_dir = ses.out_dir
        
    return out_dir
        


def r1_3mods(#just those used in p2
             #reversed delta values
        ):
    
    return run_aggErr1(
        
            #model selection
            run_name='r1_3mods',
            vid_l=[798,811, 49] ,
            
                     
            #run control
            overwrite=True,
            rl_xmDisc_dxcol_d = dict(
                xdomain=(0,2), #min/max of the xdomain
                xdomain_res = 30, #number of increments for the xdomain
                
                aggLevels_l= [2, 
                             5, 
                             100,
                             ],
                
                #random depths pramaeters
                xvars_ar = np.linspace(.1,1,num=3), #varice values to iterate over
                statsFunc = scipy.stats.norm, #function to use for the depths distribution
                depths_resolution=2000,  #number of depths to draw from the depths distritupion
                          ),
            
            plot_rlMeans=True,
                 
                 
            compiled_fp_d = {
 
                        },
        
        )
    
def r1_3mods_plot():
    return plot_aggF_errs(
        run_name='r1_3mods',
        fp_d = {
                'rl_xmDisc_dxcol':r'C:\LS\10_IO\2112_Agg\agg1\outs\pdist\r1_3mods\20221025\pdist_r1_3mods_1025_rl_xmDisc_dxcol.pickle',
                'rl_xmDisc_xvals':r'C:\LS\10_IO\2112_Agg\agg1\outs\pdist\r1_3mods\20221025\pdist_r1_3mods_1025_rl_xmDisc_xvals.pickle',
                'rl_dxcol':r'C:\LS\10_IO\2112_Agg\agg1\outs\pdist\r1_3mods\20221025\pdist_r1_3mods_1025_rl_dxcol.pickle',
            })
    
def all_r0(#results presented at supervisor meeting on Jan 4th
           #focused on vid 027, 811, and 798
           #but included some stats for every curve in the library
           #the majority of these are FLEMO curves
           #takes ~1hr to run
        ):
    
    return run_aggErr1(
        
            #model selection
            run_name='r0',
            #vid_l=[811,798, 410] ,
            selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[
                              #1, 2, #continous 
                              3, #flemo 
                              4, 6, 
                              7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47
                              ],
                          'function_formate_attribute':['discrete'], #discrete
                          'damage_formate_attribute':['relative'],
                          'coverage_attribute':['building'],
                         
                         },
                     
            #run control
            overwrite=False,
            rl_xmDisc_dxcol_d = dict(
                xdomain=(0,2), #min/max of the xdomain
                xdomain_res = 30, #number of increments for the xdomain
                
                aggLevels_l= [2, 
                             5, 
                             100,
                             ],
                
                #random depths pramaeters
                xvars_ar = np.linspace(.1,1,num=3), #varice values to iterate over
                statsFunc = scipy.stats.norm, #function to use for the depths distribution
                depths_resolution=2000,  #number of depths to draw from the depths distritupion
                          ),
            
            plot_rlMeans=True,
                 
                 
            compiled_fp_d = {
                'rl_xmDisc_dxcol':  r'C:\LS\10_IO\2112_Agg\outs\agg1F\r0_all\20220205\working\aggErr1_r2_0104_rl_xmDisc_dxcol.pickle',
                'rl_xmDisc_xvals':  r'C:\LS\10_IO\2112_Agg\outs\agg1F\r0_all\20220205\working\aggErr1_r2_0104_rl_xmDisc_xvals.pickle',
                'rl_dxcol':         r'C:\LS\10_IO\2112_Agg\outs\agg1F\r0_all\20220205\working\aggErr1_r3_0104_rl_dxcol.pickle',
                #'model_metrics':    r'C:\LS\10_IO\2112_Agg\outs\agg1F\r0_all\20220205\working\aggErr1_r3_0104_model_metrics.pickle'
                        },
        
        )
    
def all_r0_plot(**kwargs):
    return plot_aggF_errs(
        run_name='r0',
        fp_d = {        
                'rl_xmDisc_dxcol':  r'C:\LS\10_IO\2112_Agg\outs\agg1F\r0_all\20220205\working\aggErr1_r2_0104_rl_xmDisc_dxcol.pickle',
                'rl_xmDisc_xvals':  r'C:\LS\10_IO\2112_Agg\outs\agg1F\r0_all\20220205\working\aggErr1_r2_0104_rl_xmDisc_xvals.pickle',
                'rl_dxcol':         r'C:\LS\10_IO\2112_Agg\outs\agg1F\r0_all\20220205\working\aggErr1_r3_0104_rl_dxcol.pickle',
                'errArea_dxcol':    r'C:\LS\10_IO\2112_Agg\outs\agg1F\r0\20221026\agg1F_r0_1026_model_metrics.pickle'
            },
                
        **kwargs)
    
def r0_noFlemo(
        
        ):
    return run_aggErr1(
            #model selection
            tag='r0_noFlemo',
            #vid_l=[811,798, 410] ,
            selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[
                              #1, 2, #continous 
                              #3, #flemo 
                              4, 6, 
                              7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47
                              ],
                          'function_formate_attribute':['discrete'], #discrete
                          'damage_formate_attribute':['relative'],
                          'coverage_attribute':['building'],
                         
                         },
                     
            #run control
            overwrite=False,
            rl_xmDisc_dxcol_d = dict(
                xdomain=(0,2), #min/max of the xdomain
                xdomain_res = 30, #number of increments for the xdomain
                
                aggLevels_l= [2, 
                             5, 
                             100,
                             ],
                
                #random depths pramaeters
                xvars_ar = np.linspace(.1,1,num=3), #varice values to iterate over
                statsFunc = scipy.stats.norm, #function to use for the depths distribution
                depths_resolution=2000,  #number of depths to draw from the depths distritupion
                          ),
            
            plot_rlMeans=True)

def dev(
        
        ):
    
    return run_aggErr1(
        
            #model selection
            run_name='dev',
            vid_l=[
                    796, #Budiyono (2015) 
                   #402, #MURL linear
                   #852, #Dutta (2003) nice and parabolic
                   33, #FLEMO resi...
                   332, #FLEMO commericial
                   ], #running on a single function
        
            #vid_l=[811,798, 410] ,
            #vid_sample = 3,
            selection_d = { #selection criteria for models. {tabn:{coln:values}}
                          'model_id':[
                              #1, 2, #continous 
                              3, #flemo 
                              4, 6, 
                              #7, 12, 16, 17, 20, 21, 23, 24, 27, 31, 37, 42, 44, 46, 47
                              ],
                          'function_formate_attribute':['discrete'], #discrete
                          'damage_formate_attribute':['relative'],
                          'coverage_attribute':['building'],
                         
                         },
                     
            #run control
            overwrite=True,
            rl_xmDisc_dxcol_d = dict(
                xdomain=(0,2), #min/max of the xdomain
                xdomain_res = 5, #number of increments for the xdomain
                
                aggLevels_l= [2, 
                             #5, 
                             100,
                             ],
                
                #random depths pramaeters
                xvars_ar = np.linspace(.1,1,num=3), #varice values to iterate over
                statsFunc = scipy.stats.norm, #function to use for the depths distribution
                depths_resolution=100,  #number of depths to draw from the depths distritupion
                          ),
            
            plot_rlMeans=True,
                 
                 
            compiled_fp_d = {
 
                        },
        
        )
    
def dev_plot(**kwargs):
    return plot_aggF_errs(
        run_name='dev',
        fp_d = {        
                'rl_xmDisc_dxcol':r'C:\LS\10_IO\2112_Agg\agg1\outs\pdist\dev\20221025\pdist_dev_1025_rl_xmDisc_dxcol.pickle',
            'rl_xmDisc_xvals':r'C:\LS\10_IO\2112_Agg\agg1\outs\pdist\dev\20221025\pdist_dev_1025_rl_xmDisc_xvals.pickle',
            'rl_dxcol':r'C:\LS\10_IO\2112_Agg\agg1\outs\pdist\dev\20221025\pdist_dev_1025_rl_dxcol.pickle',
            },
                
        **kwargs)

if __name__ == "__main__": 
    
    output=all_r0_plot()
    #output=all_r0()
    
    #output=r1_3mods()
    #output=r1_3mods_plot()
    
 
    #output=dev()
    #output=dev_plot()
    
    
    
 
 
 
    
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))
