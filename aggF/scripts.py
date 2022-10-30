'''
Created on Feb. 5, 2022

@author: cefect
'''
#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy
import pandas as pd
import numpy as np
 

import scipy.stats 
import scipy.integrate



from hp.oop import Basic
from hp.pd import view, get_bx_multiVal
from hp.plot import Plotr

from agg.coms.scripts import Vfunc
from agg.coms.scripts import AggSession1

idx = pd.IndexSlice

import matplotlib.pyplot as plt
import matplotlib

class AggSession1F(Plotr, AggSession1):
    
    
    data_d = dict() #datafiles loaded this session
    
    ofp_d = dict() #output filepaths generated this session
    

    colorMap = 'cool'
    
    bar_colors_d = {'total':'black', 'positives':'orange', 'negatives':'blue'}
    
    color_lib=dict() #container of loaded color librarires
    
    def __init__(self, 
                 name='agg1F',
                 **kwargs):
        

        
        
        
        data_retrieve_hndls = { #function mappings for loading data types
                        #compiled takes 1 kwarg (fp)
                        #build takes multi kwargs

                           
            'df_d':{ #no compiled version
                #'compiled':lambda x:self.build_df_d(fp=x),
                'build':lambda **kwargs:self.build_df_d(**kwargs),
                },
            

            
            'vid_df':{
                'build':lambda **kwargs:self.build_vid_df(**kwargs)
                
                },
            
            'vf_d':{
                'build':lambda **kwargs:self.build_vf_d(**kwargs)
                
                },
            
            'rl_xmDisc_dxcol':{
                'compiled':lambda **kwargs:self.load_aggRes(**kwargs),
                'build':lambda **kwargs:self.build_rl_xmDisc_dxcol(**kwargs),
                },
            
            'rl_xmDisc_xvals':{
                'compiled':lambda **kwargs:self.load_aggRes(**kwargs),
                #'build':lambda **kwargs:self.build_rl_xmDisc_dxcol(**kwargs),
                },
            
            'rl_dxcol':{
                'compiled':lambda **kwargs:self.load_aggRes(**kwargs),
                'build':lambda **kwargs:self.build_rl_dxcol(**kwargs),
                },
            
            'model_metrics':{
                'compiled':lambda **kwargs:self.load_aggRes(**kwargs),
                'build':lambda **kwargs:self.build_model_metrics(**kwargs),
                },
 
            }
        
        super().__init__(proj_name=name,
                         data_retrieve_hndls=data_retrieve_hndls,
                         **kwargs)
 
        
        

    #===========================================================================
    # COMPMILED-------------
    #===========================================================================


    
    def load_aggRes(self, #loading dxcol from pickels for rl_xmDisc_dxcol and rl_dxcol
                    fp,
                    dkey=None,
                    ):
        
        assert os.path.exists(fp), fp
        assert fp.endswith('.pickle')
        with open(fp, 'rb') as f: 
            dxcol_raw = pickle.load(f)
        
        #basic checks
        
        assert isinstance(dxcol_raw, pd.DataFrame), type(dxcol_raw)
        
        #check index
        mdex = dxcol_raw.columns
        """I guess only the level0 value is consistent"""
        for i, name in enumerate([self.vidnm]):
            assert name == mdex.names[i], 'bad mdex %i, %s !=%s'%(i, name, mdex.names[i])
        
        #=======================================================================
        # fix selection behavior
        #=======================================================================
        #swap default vid selection from random to explicit
        """
        while technically the user is making a bad request by random sampling while loading an intermediate result
            the expected behaior is to only apply random sampling during build
            
        leaving vid_l as passed so the user could request specific bids WITHIN hte loaded dxcol
        """
        if 'vid_df' in self.bk_lib:
            if 'vid_sample' in self.bk_lib['vid_df']:
                if not self.bk_lib['vid_df']['vid_sample'] is None:
                    self.logger.warning('replacing random with explicit smplaing on %s.%s'%('vid_df', 'vid_sample'))
                    assert self.bk_lib['vid_df']['vid_l'] is None, 'passing vid_df.vid_sample and vid_df.vid_l not allowed'
                    self.bk_lib['vid_df']['vid_l'] = dxcol_raw.columns.get_level_values(0).unique().tolist()
                    self.bk_lib['vid_df']['vid_sample'] = None
                
        
        
        #=======================================================================
        # check selection
        #=======================================================================
        #need to make sure the current selection matches whats loaded
        vid_df = self.retrieve('vid_df')
        
        
        #check lvl0:vid
        vid_l = vid_df.index
        res_vids_l = mdex.get_level_values(0).unique().tolist()
        l = set(vid_l).difference(res_vids_l)
        assert len(l) == 0, '%i requested vids not in the results... are you random sampling?: \n%s'%(len(l), l)
        
        
        #=======================================================================
        # #trim to selection
        #=======================================================================
        dxcol = dxcol_raw.sort_index(axis=1)

        """may need to fix this for rl_dxcol"""
        dxcol = dxcol.loc[:, idx[vid_l, :,:]]
        mdex = dxcol.columns
        
        
        #check
        l = set(mdex.get_level_values(0).unique().tolist()).symmetric_difference(vid_l)
        assert len(l)==0
        
        return dxcol
    
    #===========================================================================
    # BUILDERS-------------
    #===========================================================================

 
        
        


    
    def build_rl_xmDisc_dxcol(self,  #get 
                 vf_d=None,
                 
                 #vfunc selection

                 
                 
                 #vid_df keys
                 dkey='rl_xmDisc_dxcol',
                 
                 #chunk_size=1,
                 logger=None,
                 #get_rl_xmDisc control
                 **kwargs
                 ):
        """synthetic depth generation and function calculation 
        mean-discretized for all vfuncs
        
        iterate function
            iterate xmean. get_rl_xmDisc()
                calc_agg_imp()
        
        Returns
        ----------
        pd.DataFrame: rl_xmDisc_dxcol
            dxcol of synthetic depths and resulting RLs
                columns
                    df_id: function id
                    xmean: mean for synthetic depth distrubution
                    xvar: variance for synthetic depth distrubtion
                    aggLevel: aggregation size
                    vars: water depth (x) or relative loss (y)
        
        
 
        
        """

        if logger is None: logger=self.logger        
        log = logger.getChild('r')
        assert dkey== 'rl_xmDisc_dxcol'
 
        #=======================================================================
        # #load dfuncs
        #=======================================================================
        if vf_d is None:
            vf_d = self.retrieve('vf_d')
            
        assert len(vf_d)>0
        log.info(f'on {len(vf_d)} vfuncs')
        #=======================================================================
        # setup
        #=======================================================================
        out_dir = os.path.join(self.out_dir, self.tmp_dir, dkey)
        if not os.path.exists(out_dir): 
            os.makedirs(out_dir)
 
        #======================================================================
        # calc for each
        #======================================================================
        #master_vids, dump_fp_d= dict(), dict()
 
        
        res_lib,  err_d, xvals_lib = dict(), dict(), dict()
 
        for i, (vid, vfunc) in enumerate(vf_d.items()):
            log.info('%i/%i on %s'%(i+1, len(vf_d), vfunc.name))
            try:
                res_lib[vfunc.vid], xvals_lib[vfunc.vid] = self.get_rl_xmDisc(vfunc,logger=log.getChild(str(vid)), **kwargs)
                #master_vids[vid] = vfunc.vid
            except Exception as e:
                msg = 'failed on %i w/ %s'%(vid, e)
                log.error(msg)
                err_d[i] = msg
            

            
        assert len(res_lib)>0
        #=======================================================================
        # write
        #=======================================================================
        dxcol = pd.concat(res_lib, axis=1, 
                              names=[self.vidnm] + list(res_lib[vfunc.vid].columns.names))
            
        self.ofp_d[dkey] = self.write_dxcol(dxcol, dkey, logger=log)
        """
        view(dxcol.head(100))
        """
        
        #=======================================================================
        # random xvals
        #=======================================================================
        #=======================================================================
        # #not sure what I was thinking... these are cocntained in the 'wd' column of the above
        # """writing these so we can plot later"""
        # dkey = 'rl_xmDisc_xvals'
        # xvals_dxcol = pd.concat(xvals_lib, axis=1, 
        #                       names=[self.vidnm] + list(xvals_lib[vfunc.vid].columns.names))
        # 
        # self.ofp_d[dkey] = self.write_dxcol(xvals_dxcol, dkey, logger=log)
        # """
        # view(xvals_dxcol.head(100))
        # 
        # xvals_dxcol[796].loc[:, idx[0.0, 0.1]].sort_values()
        # """
        #=======================================================================
 
        #=======================================================================
        # error check
        #=======================================================================
        for k, msg in err_d.items():
            log.error(msg)
            
        if len(err_d)>0:
            raise AssertionError('got %i errors'%len(err_d))
        
 
 
        return dxcol
    
 
    def build_rl_dxcol(self, #
                        dxcol=None,
                        vf_d = None,
                        vid_df=None,
                        
                        #vid selection
 
                        
                        #run control
                        dkey='rl_dxcol',
                        **kwargs
                        ):
        """
        combine discretized xmean results onto single domain
        
        Parameters
        ---------
        dxcol: pd.DataFrame
            aggregation erros mean-discretized for all vfuncs
        
        Returns
        ----------
        pd.DataFrame
            columns: df_id, xvar, aggLevel
            index: means
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild(dkey)
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        #=======================================================================
        # get rloss per-mean results---------
        #=======================================================================
        #=======================================================================
        # retrieve
        #=======================================================================
        if dxcol is None:
            dxcol = self.retrieve('rl_xmDisc_dxcol') #aggF.scripts.AggSession1F.build_rl_xmDisc_dxcol()
        

        
        #get vid_df for slicing
        if vid_df is None:
            vid_df = self.retrieve('vid_df')
            
            
        
        vid_l = vid_df.index.tolist()
        
        #=======================================================================
        # check
        #=======================================================================
        #check index
        mdex = dxcol.columns
        names_d = {lvlName:i for i, lvlName in enumerate(mdex.names)}    
        
        assert np.array_equal(np.array(mdex.names), np.array([vidnm, 'xmean', 'xvar', 'aggLevel', 'vars']))
        
            
        #check lvl0:vid
        res_vids_l = mdex.get_level_values(0).unique().tolist()
        l = set(vid_df.index).symmetric_difference(res_vids_l)
        assert len(l) == 0, '%i requested vids not in the results... are you random sampling?: \n%s'%(len(l), l)
        
        #check lvl4:vars
        l = set(mdex.get_level_values(4).unique()).symmetric_difference([xcn, ycn])
        assert len(l)==0, l
        
        #check lvl3:aggLevel
        assert 'int' in mdex.get_level_values(3).unique().dtype.name
        

        """only relevant when loading from compiled... so do this there
        #=======================================================================
        # #trim to selection
        #=======================================================================
 
        """
        
        
        log.info('loaded  rl_xmDisc_dxcol w/ %s'%str(dxcol.shape))
        
        #=======================================================================
        # get vfuncs-----
        #=======================================================================
        if vf_d is None: 
            vf_d =  self.retrieve('vf_d', vid_df = vid_df)
            
        #check
        l = set(vf_d.keys()).symmetric_difference(vid_l)
        assert len(l)==0, l
 
        #=======================================================================
        # loop on xmean--------
        #=======================================================================
        
        log.info('on %i vfuncs: %s'%(len(vid_l), vid_l))
        res_d = dict()
        for i, (vid, gdf0) in enumerate(dxcol.groupby(level=0, axis=1)):
            log.info('%i/%i on %s'%(i+1, len(vf_d), vf_d[vid].name))
            
            #impact vs depth analysis on multiple agg levels for a single vfunc
            res_d[vid] = self.calc_rlMeans(gdf0.droplevel(0, axis=1), vf_d[vid], **kwargs)
            
            
        #=======================================================================
        # wrap
        #=======================================================================
        """this is a collapsed version of the rdxcol... wich only mean rloss values"""
        rdxcol = pd.concat(res_d, axis=1, names=[vidnm]+ list(res_d[vid].columns.names))
        self.ofp_d[dkey] = self.write_dxcol(rdxcol, dkey,  logger=log)
        
        log.info('finished w/ %s'%str(rdxcol.shape))
        return rdxcol


    def build_model_metrics(self, #
                        dxcol=None, #depth-damage at different AggLevel and xvars
                        
 
                        dkey='model_metrics',
                        **kwargs
                        ):
        """
        integrate delta areas. compute total areas
        
        Returns
        --------
        pd.DataFrame
            columns: df_id, xvar, aggLevel
            index: 3 types of area between s1 and s2 curves (total, positive, negative)
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild(dkey)
        assert dkey == 'model_metrics'
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        #=======================================================================
        # get rloss per-mean results---------
        #=======================================================================
        #=======================================================================
        # retrieve
        #=======================================================================
        if dxcol is None:
            dxcol = self.retrieve('rl_dxcol')
            
        mdex = dxcol.columns
        names_d = {lvlName:i for i, lvlName in enumerate(mdex.names)} 
        vid_l = mdex.get_level_values(0).unique().tolist()
        #=======================================================================
        # check
        #=======================================================================
        #check index
        
        assert np.array_equal(np.array(mdex.names), np.array([vidnm, 'xvar', 'aggLevel']))
        
        
        #log.info('loaded  rl_dxcol w/ %s'%str(dxcol.shape))
        """
        view(dxcol)
        """
        #=======================================================================
        # loop on vid--------
        #=======================================================================
        plt.close('all')
        log.info('on %i vfuncs: %s'%(len(vid_l), vid_l))
        res_d = dict()
        for vid, gdf0 in dxcol.groupby(level=0, axis=1):
            
            res_d[vid] = self.calc_areas(gdf0.droplevel(0, axis=1), logger=log.getChild('v%i'%vid), **kwargs)
            
            
        log.info('got %i'%len(res_d))
        rdxcol = pd.concat(res_d, axis=1, names=names_d.keys())

        rdxcol.index.name='errArea_type'
        
        #=======================================================================
        # write
        #=======================================================================
        self.ofp_d[dkey] = self.write_dxcol(rdxcol, dkey, write_csv=True)
        
        return rdxcol
 

 
    #===========================================================================
    # TOP RUNNERS---------
    #===========================================================================
    
    def plot_eA_box(self,
                        dxcol=None,  # depth-damage at different AggLevel and xvars
                        vid_df=None,
                        
                        # value grouping
                        g0_coln=None,  # coln for subfolder division
                        g1_coln='errArea_type',  # coln for dividing figures
                        grp_colns=['model_id', 'sector_attribute'],  # coln for xaxis division (matrix rows)
                        
                        # style control

                        ** kwargs
                        ):
        """plot multiple boxes of errors"""
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_eA_box')
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        #=======================================================================
        # get rloss per-mean results---------
        #=======================================================================
        #=======================================================================
        # retrieve
        #=======================================================================
        if dxcol is None:
            dxcol = self.retrieve('model_metrics')
            
        if vid_df is None:
            vid_df = self.retrieve('vid_df')
            
        #=======================================================================
        # get macro stats
        #=======================================================================
        
        # move the vid to the index
        dx1 = dxcol.T.unstack(level=vidnm).T.swaplevel(axis=0).sort_index(level=0)
        
        # calc each
        #=======================================================================
        # d = dict()
        # for stat in ['mean', 'min', 'max']:
        #     d[stat] = getattr(dx1, stat)()
        #     
        # # add back
        # # dx2 = dx1.append(pd.concat([pd.concat(d, axis=1).T], keys=['stats']))
        # stats_df = pd.DataFrame.from_dict(d)
        #=======================================================================
        
        #=======================================================================
        # get grouped stats-----
        #=======================================================================
        
        # move index to columns
        dx2 = dx1.unstack(level='errArea_type')
        names_d = {lvlName:i for i, lvlName in enumerate(dx2.columns.names)}
        
        # retrieve attribute info
        
        
        l = set(grp_colns).difference(vid_df.columns)
        assert len(l) == 0, l
        
        # add dummy indexer
        if g0_coln is None:
            g0_coln = 'group0'
            vid_df[g0_coln] = True
        
        #=======================================================================
        # #get figure for each attribute of interest
        #=======================================================================
        for i0, (g0val, vid_gdf) in enumerate(vid_df.groupby(g0_coln)):
            
            # setup suubfolder
            if len(vid_gdf) == len(vid_df):
                out_dir = os.path.join(self.out_dir, 'plot_eA_box')
            else:
                out_dir = os.path.join(self.out_dir, 'plot_eA_box', str(g0val))
        
            for i, coln in enumerate(grp_colns  
                                     # ['all']
                                     ):
                plt.close('all')
     
                # get grouping info
                if coln == 'all':
                    gser = pd.Series(True, name='all', index=vid_gdf.index)
                else:
                    gser = vid_gdf[coln]
                
                for gval, gdxcol in dx2.groupby(g1_coln, axis=1):
                    """need this extra divisor"""
     
                    # build the matrix fig
                    fig = self.get_eA_box_fig(gdxcol.droplevel(names_d[g1_coln], axis=1),
                                         gser, fig_id=i,
                                        logger=log.getChild(coln), **kwargs)
                    
                    # post
                    # fig.suptitle(gval)
                    
                    self.output_fig(fig, out_dir=out_dir, 
                                    fname=f'errBox_{gval}',
                                    overwrite=True,
                                    transparent=False)
                
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished')
        plt.close('all')
        
        return  
        """
        view(rdxcol)
        """
        
    def plot_eA_bars(self, #integrate delta areas
                        dxcol=None, #depth-damage at different AggLevel and xvars
                        
 
                        
                        grp_colns_fig = [ #pulling from vid_df
                            'model_id', 'sector_attribute'
                            ],
 
                        #dkey='rl_dxcol',
                        **kwargs
                        ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('pBars')
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        #=======================================================================
        # get rloss per-mean results---------
        #=======================================================================
        #=======================================================================
        # retrieve
        #=======================================================================
        if dxcol is None:
            dxcol = self.retrieve('model_metrics')
            
        vid_df = self.retrieve('vid_df')
        
        """
        view(vid_df)
        """
            
        #=======================================================================
        # data prep
        #=======================================================================

        
        #move the vid to the index
        dx1 = dxcol.T.unstack(level=vidnm).T.swaplevel(axis=0).sort_index(level=0)
        
        
        #move index to columns
        dx2 = dx1.unstack(level='errArea_type')
        
        
        #check
        l = set(grp_colns_fig).difference(vid_df.columns)
        assert len(l)==0, l
        
        #=======================================================================
        # join in grouping values
        #=======================================================================
        dx3 = dx2.copy()
        
        #build new index
        dfi = pd.Series(dx3.index).to_frame().join(vid_df.loc[:, grp_colns_fig], on=vidnm)
        dx3.index = pd.MultiIndex.from_frame(dfi)
        
        inames_d= {lvlName:i for i, lvlName in enumerate(dx3.index.names)}
        
        #=======================================================================
        # loop each fig group
        #=======================================================================
        for i, (gvals, gdx) in enumerate(dx3.groupby(level=[inames_d[k] for k in grp_colns_fig])):
            gk_d = dict(zip(grp_colns_fig, gvals))
            gv_str = '_'.join([str(e) for e in gvals])
        
            
            #drop extra levels
            dxcol_i = gdx.droplevel([inames_d[k] for k in grp_colns_fig])
            
            #plot
            plt.close('all')
            fig = self.get_eA_bars_fig(dxcol_i, fig_id=i, logger=log.getChild(gv_str), **kwargs)
            
            #post
            fig.suptitle(gv_str)
            
            
            self.output_fig(fig, out_dir=os.path.join(self.out_dir, 'plot_eA_bars'), 
                            fname='%s_%s'%(self.fancy_name,gv_str))
            
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished')
        
        
    def plot_xmDisc(self,
                    dxcol=None,
                    xvals_dxcol=None, #random xvals used to generate dxcol results
                    vf_d = None,
 
                        #run control
 
                        **kwargs
                        ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_xmDisc')
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        
        od0 = os.path.join(self.out_dir, 'models')
        start = datetime.datetime.now()
        #=======================================================================
        # get rloss per-mean results---------
        #=======================================================================
        #=======================================================================
        # retrieve
        #=======================================================================
        if dxcol is None:
            dxcol = self.retrieve('rl_xmDisc_dxcol')
            
        if xvals_dxcol is None:
            xvals_dxcol = self.retrieve('rl_xmDisc_xvals')
            
        if vf_d is None:
            vf_d = self.retrieve('vf_d')
        

        log.info('on %s'%str(dxcol.shape))
        
 
 
        #=======================================================================
        #  loop each dataset
        #=======================================================================
        names_d = {lvlName:i for i, lvlName in enumerate(dxcol.columns.names)}
        
        ofp_d = dict()
        for i, ((vid, xmean), gdx) in enumerate(
            dxcol.groupby(level=[names_d[e] for e in ['df_id', 'xmean']], axis=1)):
            
            #===================================================================
            # retrieve
            #===================================================================
            xvals_df = xvals_dxcol.loc[:, idx[vid, xmean, :]].droplevel([0,1], axis=1)
            gdf = gdx.droplevel([names_d[e] for e in ['df_id', 'xmean']], axis=1)
            vfunc = vf_d[vid]
 
            #===================================================================
            # #plot
            #===================================================================
            plt.close('all')
            fig = self.get_agg_imp_fig(gdf,
                                       xmean=xmean,
                                       xvals_df = xvals_df,
                                       vfunc=vfunc,
                                       fig_id=0, **kwargs)
            
            #write the figure
            ofp_d[i] = self.output_fig(fig,
                    out_dir=os.path.join(od0,str(vfunc.model_id), str(vfunc.vid)), #matching calc_rlMeans
                    fname='_'.join([self.fancy_name, 'v%i'%vid, 'm%.1f'%xmean, 'xmDisc']),
                    fmt='svg', dpi=400,transparent=False
                    )
        
        #=======================================================================
        # wrap
        #=======================================================================
        tdelta = datetime.datetime.now() - start
        log.info('finished on %i in %.1f secs to \n    %s'%(len(ofp_d),tdelta.total_seconds(), od0))
        
        return ofp_d
        
                            
 
    def plot_all_vfuncs(self,
                 #data selection

                 
                 vf_d=None,
                 vid_df=None,
                 
 
                 
                 #formatting
                 phndl_d=None, #optional plot handles
                 xlims = None,ylims=None,
                 figsize=(10,10),
                 lineKwargs = { #global kwargs
                     'linewidth':0.5
                     },
                 
                 
                 
                 title=None,
                 ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_all_vfuncs')
 
        vidnm=self.vidnm
        
        if vf_d is None: 
            vf_d=self.retrieve('vf_d', vid_df=vid_df)
        
        log.info('on %i'%len(vf_d))
        
        if phndl_d is None:
            phndl_d = {k:{} for k in vf_d.keys()}
            
        
        #=======================================================================
        # setup figure
        #=======================================================================
        plt.close('all')
        fig = plt.figure(0,
            figsize=figsize,
            tight_layout=False,
            constrained_layout=True)
        
        ax = fig.add_subplot(111)
        
        """
        plt.show()
        phndl_d.keys()
        """
        
        
        
        for vid, vfunc in vf_d.items():
            log.info('on %s'%vfunc.name)
            
            if not vid in phndl_d:
                log.warning('%s missing handles'%vfunc.name)
                phndl_d[vid]=dict()
 
            
            vfunc.plot(ax=ax, lineKwargs = {**lineKwargs, **phndl_d[vid]}, logger=log)

            
        log.info('added %i lines'%len(vf_d))
        
        #=======================================================================
        # style
        #=======================================================================
        xvar, yvar = vfunc.xcn, vfunc.ycn
        
        if not xlims is None:
            ax.set_xlim(xlims)
        if not ylims is None:
            ax.set_ylim(ylims)
        ax.set_xlabel('\'%s\' water depth (m)'%xvar)
        ax.set_ylabel('\'%s\' relative loss (pct)'%yvar)
        ax.grid()
 
        if title is None:
            title = '\'%s\' vs \'%s\' on %i'%(xvar, yvar, len(vf_d))
        ax.set_title(title)
        
        #legend
        #fig.legend()
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)
        
        #=======================================================================
        # write figure
        #=======================================================================
        return fig
    
    def run_err_stats(self, #calc some stats and write to xls
                      dxcol=None,
                      vid_df=None,
                      grp_colns =  ['model_id', 'sector_attribute'],
                      ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('rdd')
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
 
        #=======================================================================
        # retrieve
        #=======================================================================
        if dxcol is None:
            dxcol = self.retrieve('model_metrics')
            
        if vid_df is None:
            vid_df = self.retrieve('vid_df')
            
        #retrieve attribute info
        
        #=======================================================================
        # setup
        #=======================================================================
        res_lib = dict()

        
        #move the vid to the index
        dx1 = dxcol.T.unstack(level=vidnm).T.swaplevel(axis=0).sort_index(level=0)
 
 
        
        
        #move index to columns
        dx2 = dx1.unstack(level='errArea_type')
        names_d = {lvlName:i for i, lvlName in enumerate(dx2.columns.names)}
        

        #=======================================================================
        # combined stats
        #=======================================================================
        res_d = dict()
        res_d2 = dict()
        for eatype, gdf in dx2.groupby(level=names_d['errArea_type'], axis=1):
            df = gdf.droplevel(names_d['errArea_type'], axis=1)
            
            #thresholds
            d = {'df<0':(df<0).sum(),
                 'df>0':(df>0).sum(),
                 'total':df.notna().sum()}
            
            res_d[eatype] = pd.concat(d, axis=1).T
            
            #stats
            d=dict()
            for stat in ['mean', 'min', 'max']:
                d[stat] = getattr(df, stat)()
                
            res_d2[eatype] = pd.concat(d, axis=1).T
 
 
        res_lib['combined_thresh'] = pd.concat(res_d)
        
        res_lib['combined_stats'] =pd.concat(res_d2)
        
        
        #=======================================================================
        # all models
        #=======================================================================
        
        gcoln = 'model_id'
        dx3 = dx2.copy()
        #add model id for easy lookup
        meta_ser = vid_df[gcoln] 
 
        dx3.index=pd.MultiIndex.from_frame(pd.Series(dx2.index).to_frame().join(meta_ser, on=meta_ser.index.name))
        
        res_lib['all'] = dx3
 
 
        
        #=======================================================================
        # get grouped stats-----
        #=======================================================================
        
 
        
        
        for gcoln in  grp_colns:
            res_d = dict()
            for i, (gval, vid_gdf) in enumerate(vid_df.groupby(gcoln)):
                #get these errors
                res_d[gval] = dx2.loc[vid_gdf.index, :].sum().rename(gval).to_frame().T
            
            #reassemble
            rdxcol = pd.concat(res_d).droplevel(level=0, axis=0)
            rdxcol.index.name=gcoln
            
            #add some meta columns
            meta_ser = vid_df[gcoln].groupby(vid_df[gcoln]).count().rename('%s_cnt'%self.vidnm)
            mdex = pd.MultiIndex.from_frame(pd.Series(rdxcol.index).to_frame().join(meta_ser, on=gcoln))
            rdxcol.index=mdex
            
            
            res_lib[gcoln] = rdxcol
            
        #=======================================================================
        # write
        #=======================================================================
        out_fp=os.path.join(self.out_dir, '%s_err_stats.xls'%self.fancy_name)
        with pd.ExcelWriter(out_fp) as writer:
            for tabnm, df in res_lib.items():
                df.to_excel(writer, sheet_name=tabnm, index=True, header=True)
                
        log.info('wrote %i tabs: %s \n    %s'%(len(res_lib), list(res_lib.keys()), out_fp))
        
        return out_fp
        
                
 
        
 
    
    #===========================================================================
    # PLOTING--------
    #===========================================================================
    def get_eA_bars_fig(self,
                        dxcol,
                        col_coln='xvar', #matrix plot rows
                        row_coln='aggLevel',#matrix plot columns
 
                        #plot style
                        colorMap=None,
                        bar_colors_d=None,
                        
                        fig_id=None,
                        ylims = (-30,30),
                        
                        #run control
                        logger=None,
                        ):
            
        #=======================================================================
        # defaults
        #=======================================================================
        log = logger.getChild('get_eA_bars_fig')
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        if colorMap is None: colorMap=self.colorMap
        if bar_colors_d is None: bar_colors_d=self.bar_colors_d
        #=======================================================================
        # retrieve meta
        #=======================================================================
        mdex = dxcol.columns
        cnames_d= {lvlName:i for i, lvlName in enumerate(mdex.names)}
        
        xvar_l =  mdex.get_level_values(cnames_d[col_coln]).unique().tolist()
        rows_l = mdex.get_level_values(cnames_d[row_coln]).unique().tolist()
        
        
        #=======================================================================
        # #setup figure
        #=======================================================================
        
        fig, ax_d = self.get_matrix_fig(rows_l,xvar_l, 
                                        figsize=( len(xvar_l)*4, len(rows_l)*4),
                                        constrained_layout=True,
                                        fig_id=fig_id)
        
        #fig.suptitle(ser.name)
        """
        fig.show()
        """
        
        #get color map for aggLevels
        cmap = plt.cm.get_cmap(name=colorMap) 
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k,ni in dict(zip(rows_l, np.linspace(0, 1, len(rows_l)))).items()}
             
        
        #=======================================================================
        # populate each axis
        #=======================================================================       
        for gvals, gdf in dxcol.groupby(level=[cnames_d[row_coln],cnames_d[col_coln]],  axis=1):
            #===================================================================
            # setup this group
            #===================================================================
            gk_d = dict(zip([row_coln,col_coln], gvals))
            gv_str = '_'.join([str(e) for e in gvals])
            
            """transforming to match other funcs"""
            df = gdf.droplevel([cnames_d[row_coln],cnames_d[col_coln]], axis=1).T
            
 
            #===================================================================
            # setup axis
            #===================================================================
            ax = ax_d[gvals[0]][gvals[1]]
            ax.set_title(', '.join(['%s=%s'%(k,v) for k,v in gk_d.items()]))
            
            
 
            #===================================================================
            # add bars per xgroupn
            #===================================================================
            locs = np.linspace(.1,.9,len(df.columns))
            width = 0.1/len(locs)
            
            """
            ax.clear()
            """
            
            for i, (label, row) in enumerate(df.iterrows()):
                ax.bar(locs+i*width, row.values, label=label, width=width, alpha=0.5,
                       color=bar_colors_d[label],
                       tick_label=row.index)
                
        
        #=======================================================================
        # post axisforamtting
        #=======================================================================
        firstRow = True
        for rowVal, d in ax_d.items():
            
            firstCol =True
            for colVal, ax in d.items():
 
                
                
                
                if firstRow:
                    ax.set_xlabel(df.columns.name)
                    
                    
                    
                    if firstCol: ax.legend()
                        
 
                    
                if firstCol:
                    ax.set_ylabel('area under rl difference curve')
                    ax.set_ylim(ylims)
                    
                    firstCol = False
            
            #wrap row
            if firstRow: 
                firstRow=False
            
        log.debug('finished')
        return fig
    
    def get_eA_box_fig(self,  #
            dxcol, ser,
            row_coln='aggLevel',  # how to divide matrix plot rows
            logger=None,
            
            # plot control
            add_text=False,
            ylims=(-30, 30),
            fig_id=None,
            linewidth=0.75,
 
            colorMap=None,
            figsize=None,
            
            xlab = 'model id',
            ylab = '$e_{total}$',
            **kwargs):
        """get grouped (gser) bar plots per aggLevel + xvar 
        highlight
            aggLevel
            gropuVal
        
        matrix plot of boxplots
            columns: xvar
            rows:eatypes_l
                axis groups: gvals
            
        plt.close('all')
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = logger.getChild('get_eA_box_fig')
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        if colorMap is None: colorMap = self.colorMap

        #=======================================================================
        # retrieve meta
        #=======================================================================
        mdex = dxcol.columns
        cnames_d = {lvlName:i for i, lvlName in enumerate(mdex.names)}
        
        xvar_l = mdex.get_level_values(0).unique().tolist()
        rows_l = mdex.get_level_values(cnames_d[row_coln]).unique().tolist()
        
        if figsize is None:
            figsize=(len(xvar_l) * 4, len(rows_l) * 4)
        #=======================================================================
        # #setup figure
        #=======================================================================
        
        fig, ax_d = self.get_matrix_fig(rows_l, xvar_l,
                                        figsize=figsize,
                                        constrained_layout=True,
                                        fig_id=fig_id, **kwargs)
        
        # fig.suptitle(ser.name)
        
        # get color map for aggLevels
        cmap = plt.cm.get_cmap(name=colorMap) 
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k, ni in dict(zip(rows_l, np.linspace(0, 1, len(rows_l)))).items()}
            
        #=======================================================================
        # join in grouping values
        #=======================================================================
        dxcol1 = dxcol.copy()
        
        dxcol1.index = pd.MultiIndex.from_frame(pd.Series(dxcol.index).to_frame().join(ser.rename('group'), on=vidnm))
        inames_d = {lvlName:i for i, lvlName in enumerate(dxcol1.index.names)}
 
        for i, (rowVal, gdx0) in enumerate(dxcol1.groupby(level=cnames_d[row_coln], axis=1)):
            
            # xvar loop
            for xvar, gdx1  in  gdx0.groupby(level=0, axis=1):
                
                #===============================================================
                # data prep
                #===============================================================
                
                gd = {k:v.values.reshape(-1) for k, v in gdx1.groupby(level=inames_d['group'], axis=0)}
                #===============================================================
                # plotting prep
                #===============================================================
                ax = ax_d[rowVal][xvar]


                #===============================================================
                # add the boxplot
                #===============================================================
                skwargs = dict(linewidth=linewidth, color='black')
                boxres_d = ax.boxplot(gd.values(), labels=gd.keys(), meanline=False, 
                            boxprops=skwargs,whiskerprops=skwargs,medianprops=skwargs,capprops=skwargs,
                            flierprops={'markersize':3, 'marker':'o', 'markeredgewidth':0.5, 'alpha':0.5}, 
                            )
                
                #===============================================================
                # for boxLine in boxres_d['boxes']:
                #     boxLine.set_color(newColor_d[rowVal])
                #     boxLine.set_linewidth(0.5)
                #===============================================================
                
                
                if not ylims is None:
                    ax.set_ylim(ylims)
                
                # add the xaxis line
                
                ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], color='red', linewidth=0.5, linestyle='solid', zorder=0)
                
                #===============================================================
                # add extra text
                #===============================================================
                ax.text(0.98, 0.95, '$s=%i$, $\sigma=%.2f$'%(rowVal, xvar),transform=ax.transAxes, va='top', ha='right')
                """
                plt.show()
                """
                # counts on median bar
                #===============================================================
                # for gval, line in dict(zip(gd.keys(), boxres_d['medians'])).items():
                #     x_ar, y_ar = line.get_data()
                #     ax.text(x_ar.mean(), y_ar.mean(), 'n%i' % len(gd[gval]),
                #             # transform=ax.transAxes, 
                #             va='bottom', ha='center', fontsize=8)
                #     
                #     if add_text:
                #         ax.text(x_ar.mean(), ylims[0] + 1, 'mean=%.2f' % gd[gval].mean(),
                #             # transform=ax.transAxes, 
                #             va='bottom', ha='center', fontsize=8, rotation=90)
                #===============================================================
                

                
        #=======================================================================
        # post
        #=======================================================================
        row_keys = rows_l
        col_keys = xvar_l
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
                #ax.set_title('%s=%s, xvar=%.2f' % (row_coln, rowVal, xvar))
                
                #last row
                if row_key == row_keys[-1]:
                    ax.set_xlabel(xlab)
                    
                #first col
                if col_key==col_keys[0]:
                    ax.set_ylabel(ylab)
                
        #=======================================================================
        # wrap
        #=======================================================================
                
        return fig
    
    def plot_rv(self, #get a plot of a scipy distribution
                rv,
                ax=None,
                figNumber = 1,
                color='red',
                xlab=None,
                title=None,
                lineKwargs = {
                    #'color':'red'
                    }
                ):
        
        #=======================================================================
        # setup data
        #=======================================================================
        #get x domain
        xar = np.linspace(rv.ppf(0.01),rv.ppf(0.99), 100)
        
        
        #=======================================================================
        # setup figure
        #=======================================================================
        assert not ax is None
 

            
        #=======================================================================
        # #plot
        #=======================================================================
        #pdf
        ax.plot(xar, rv.pdf(xar), label='%s pdf'%rv.dist.name, color=color,
                 **lineKwargs)
        
        
        
        #=======================================================================
        # style
        #=======================================================================
        ax.grid()
        
        if not title is None: ax.set_title(title)
        if not xlab is None: ax.set_xlabel(xlab)
        ax.set_ylabel('frequency')
        
        d = {k: getattr(rv, k)() for k in ['mean', 'std', 'var']}
        
        text = '\n'.join(['%s:%.2f'%(k,v) for k,v in d.items()])
        anno_obj = ax.text(0.1, 0.9, text, transform=ax.transAxes, va='center',
                           fontsize=8, color=color)
        
 
        return ax
        """
        plt.show()
        """

 
 
    def get_plot_hndls(self,
                         df_raw,
                         coln = 'model_id', #style group
                         vidnm = None, #indexer for damage functions
                         colorMap = 'gist_rainbow',
                         ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if vidnm is None: vidnm=self.vidnm
        
       
        from matplotlib.lines import Line2D
        log = self.logger.getChild('plot_hndls')
        
        #=======================================================================
        # checks
        #=======================================================================
        assert len(df_raw)<=20, 'passed too many plots... this is over plotting'
        #=======================================================================
        # color by model_id
        #=======================================================================
        if df_raw[coln].isna().any():
            log.warning('got %i/%i nulls on \'%s\'... filling with mode'%(
                df_raw[coln].isna().sum(), len(df_raw), coln))
            
            df1 = df_raw.fillna(df_raw[coln].mode()[0])
        else:
            df1 = df_raw.copy()
            
            
        """
        view(df1)
        """
        res_df = pd.DataFrame(index=df1.index)
        #=======================================================================
        # #retrieve the color map
        #=======================================================================
        
        groups = df1.groupby(coln)
        
        cmap = plt.cm.get_cmap(name=colorMap)
        #get a dictionary of index:color values           
        d = {i:cmap(ni) for i, ni in enumerate(np.linspace(0, 1, len(groups)))}
        newColor_d = {i:matplotlib.colors.rgb2hex(tcolor) for i,tcolor in d.items()}
        
        #markers
        #np.repeat(Line2D.filled_markers, 3) 
        markers_d = Line2D.markers
        
        mlist = list(markers_d.keys())*3
        
        for i, (k, gdf) in enumerate(groups):
            
            #single color per model type
            res_df.loc[gdf.index.values, 'color'] = newColor_d[i]
            
 
            # marker within model_id
            if len(gdf)>len(markers_d):
                log.warning('\'%s\' more lines (%i) than markers!'%(k, len(gdf)))
 
 
            res_df.loc[gdf.index, 'marker'] = np.array(mlist[:len(gdf)])
             
            
 
        if not res_df.notna().all().all():
            log.warning('got %i nulls'%res_df.isna().sum().sum())
        
        
        #dummy for now
        return res_df.to_dict(orient='index')
    

    
    
    #===========================================================================
    # CALCs---------------
    #===========================================================================
    def calc_areas(self, #calc areas for a single vfunc
            dxcol,
            logger=None,
            ):
        """
        compute the area of the difference between the baseline
        
        not setup super well... we're also calcing this in calc_rlMeans to get the nice bar charts
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = logger.getChild('calc_area')
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        
        mdex = dxcol.columns
        names_d = {lvlName:i for i, lvlName in enumerate(mdex.names)}
        #=======================================================================
        # loop on xvar
        #=======================================================================
        res_lib = dict()
        for xvar, gdf0 in dxcol.groupby(level=names_d['xvar'], axis=1):
            res_lib[xvar] = dict()
            for aggLevel, gdf1 in gdf0.groupby(level=names_d['aggLevel'], axis=1):
                
                #===============================================================
                # get data
                #===============================================================
                
                #retrieve array
                yar = gdf1.droplevel(level=0, axis=1).iloc[:, 0].values
                

                #get base
                if aggLevel==0:
                    xar = gdf1.droplevel(level=0, axis=1).index.values
                    yar_base = yar
                    continue
                
                #get deltas
                diff_ar = yar_base - yar
            
                res_d = dict()
                #===============================================================
                # total area
                #===============================================================
                """negatives often balance positives"""

                
                #integrate
                res_d['total'] = scipy.integrate.trapezoid(diff_ar, xar)
                
                #===============================================================
                # positives areas
                #===============================================================
                #clear out negatives
                diff1_ar = diff_ar.copy()
                diff1_ar[diff_ar<=0] = 0
                
                res_d['positives'] = scipy.integrate.trapezoid(diff1_ar, xar)
                
                #===============================================================
                # negative  areas
                #===============================================================
                #clear out negatives
                diff2_ar = diff_ar.copy()
                diff2_ar[diff_ar>=0] = 0
                
                res_d['negatives'] = scipy.integrate.trapezoid(diff2_ar, xar)
                
                #===============================================================
                # wrap
                #===============================================================
                res_lib[xvar][aggLevel] = res_d
                
                """
                plt.plot(xar, diff_ar)
                plt.plot(xar, diff1_ar)
                plt.plot(xar, diff2_ar)
                """
            #===================================================================
            # wrap xvar
            #===================================================================
            res_lib[xvar] = pd.DataFrame.from_dict(res_lib[xvar])
                
        #=======================================================================
        # wrap
        #=======================================================================
        log.debug('finished on %i'%len(res_lib))
        
        rdxcol = pd.concat(res_lib, axis=1)
        
        return rdxcol
        
 
            
            
            
    def calc_rlMeans(self, #
                     dxcol, vfunc,
                     
                     quantiles = (0.05, 0.95), #rloss qantiles to plot
                     
                     ylims_d = {
                         'dd':(0,100), 'delta':(-30,15), 'bars':(-30,15)
                         },
                     
                     xlims = (0,2),
 
                     
                     #plot style
                     plot=True,
                     colorMap=None,
                     prec=4, #rounding table values
                     title=None,
                     
                     #run control
                     plot_xmean_scatter=None, #index for plotting the scatter on aggLevel=0
                     out_dir=None,
                     
                     ):
        """
        impact vs depth analysis on multiple agg levels for a single vfunc        
        
        called by build_rl_dxcol (dkey='rl_dxcol')
        
        very nasty function... lots of extra stuff just to get the plot nice
        this is a combined calc and plot function
        
        TODO: separate out plotting and calcing
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('calc_rl_v%s' % vfunc.vid)
        vidnm = self.vidnm
        xcn, ycn = self.xcn, self.ycn
        if colorMap is None: colorMap = self.colorMap
        plt = self.plt
        
        # output dir
        if out_dir is None: 
            out_dir = os.path.join(self.out_dir, 'models', str(vfunc.model_id))
        
        if not os.path.exists(out_dir): os.makedirs(out_dir)
 
        #=======================================================================
        # check mdex
        #=======================================================================
        mdex = dxcol.columns
        assert np.array_equal(np.array(mdex.names), np.array(['xmean', 'xvar', 'aggLevel', 'vars']))
        names_d = {lvlName:i for i, lvlName in enumerate(mdex.names)}
        
        #=======================================================================
        # #retrieve domain info
        #=======================================================================
        lVals_d = dict()
        for lvlName, lvl in names_d.items():
            lVals_d[lvlName] = np.array(dxcol.columns.get_level_values(lvl).unique())
 
        log.debug('for ' + ', '.join(['%i %sz' % (len(v), k) for k, v in lVals_d.items()]))
        #===================================================================
        # setup figure
        #===================================================================
        
        # get color map for aggLevels
        cmap = plt.cm.get_cmap(name=colorMap)
        l = lVals_d['aggLevel']        
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k, ni in dict(zip(l, np.linspace(0, 1, len(l)))).items()}

        plt.close('all')
        fig, ax_d = self.get_matrix_fig(list(ylims_d.keys()), lVals_d['xvar'].tolist(),
                                        figsize=(len(lVals_d['xvar']) * 4, 3 * 4),
                                        constrained_layout=True)
 
        if title is None:
            title = vfunc.name
        fig.suptitle(title)
 
        #===================================================================
        # loop by xvar--------
        #===================================================================
        res_lib = dict()
        """splitting this by axis... so needs to be top for loop"""
        for xvar, gdf0 in dxcol.groupby(level=names_d['xvar'], axis=1):
            
            """
            view(gdf0.droplevel(names_d['xvar'], axis=1))
            """
            
            #===================================================================
            # loop aggleve
            #===================================================================
            """probably best to loop this next as they'll share plotting pars"""
            res_d, area_lib = dict(), dict()
            for aggLevel, gdf1 in gdf0.groupby(level=names_d['aggLevel'], axis=1):
                
                #===============================================================
                # plotting defaults
                #===============================================================
                color = newColor_d[aggLevel]
                
                #===============================================================
                # setup data
                #===============================================================
                # get a clean frame for this aggLevel
                rdXmean_dxcol = gdf1.droplevel(level=[names_d['aggLevel'], names_d['xvar']], axis=1
                                           ).dropna(how='all', axis=0)
                
                # drop waterDepth info
                rXmean_df = rdXmean_dxcol.loc[:, idx[:, ycn]].droplevel(1, axis=1)
                
                """
                vfunc.ddf
                view(gdf1)
                view(rXmean_df)
                fig.show()
                ax.clear()
                """
                
                #===============================================================
                # mean impacts per depth 
                #===============================================================
                
                # calc
                rser1 = rXmean_df.mean(axis=0)
                res_d[aggLevel] = rser1.copy()
                res_lib[xvar] = pd.concat(res_d, axis=1)
                if not plot: continue
                
                # plot
                ax = ax_d['dd'][xvar]
                pkwargs = dict(color=color, linewidth=0.5, marker='x', markersize=3, alpha=0.8)  # using below also
                ax.plot(rser1, label='aggLevel=%i (rl_mean)' % aggLevel, **pkwargs)
                
                #===============================================================
                # #quartiles
                #===============================================================
                    
                if not quantiles is None:
                    ax.fill_between(
                        rser1.index,  # xvalues
                         rXmean_df.quantile(q=quantiles[1], axis=0).values,  # top of hatch
                         rXmean_df.quantile(q=quantiles[0], axis=0).values,  # bottom of hatch
                         color=color, alpha=0.1, hatch=None)
                    
                # wrap impacts-depth
                ax.legend()
                
                #===============================================================
                # special scatter
                #===============================================================
                if aggLevel == 0 and not plot_xmean_scatter is None:
                    # get this xmean
                    xmean = rXmean_df.columns[plot_xmean_scatter]
                    
                    dd_df = rdXmean_dxcol.loc[:, idx[xmean,:]].droplevel(0, axis=1).sample(50)
                    
                    ax.scatter(x=dd_df[xcn], y=dd_df[ycn], color=color, s=20, marker='$%.1f$' % xmean, alpha=0.1,
                               label='raws xmean=%.2f' % xmean)
                
                #===============================================================
                # deltas----------
                #===============================================================
                #===============================================================
                # get data
                #===============================================================
                # retrieve array
                yar = rser1.values
                
                # get base
                if aggLevel == 0:
                    xar = rser1.index.values
                    yar_base = yar
                    continue
                
                # get deltas
                diff_ar = yar - yar_base
                
                #===============================================================
                # plot
                #===============================================================
                ax = ax_d['delta'][xvar]
                ax.plot(xar, diff_ar, label='aggLevel=%i (rl_mean - true)' % aggLevel, **pkwargs)
                #===============================================================
                # areas----
                #===============================================================
                area_d = dict()
                """negatives often balance positives
                fig.show()
                """
                # integrate
                area_d['total'] = scipy.integrate.trapezoid(diff_ar, xar)
                
                #===============================================================
                # positives areas
                #===============================================================
                # clear out negatives
                diff1_ar = diff_ar.copy()
                diff1_ar[diff_ar <= 0] = 0
                
                area_d['positives'] = scipy.integrate.trapezoid(diff1_ar, xar)
                
                #===============================================================
                # negative  areas
                #===============================================================
                # clear out negatives
                diff2_ar = diff_ar.copy()
                diff2_ar[diff_ar >= 0] = 0
                
                area_d['negatives'] = scipy.integrate.trapezoid(diff2_ar, xar)
                
                # ax.legend()
                
                #===============================================================
                # wrap agg level
                #===============================================================
                area_lib[aggLevel] = area_d
                
                log.debug('finished aggLevel=%i' % aggLevel)
            
            if plot:
                ax.legend()  # turn on the diff legend
                #===================================================================
                # add area bar charts
                #===================================================================
     
                ax = ax_d['bars'][xvar]
                
                adf = pd.DataFrame.from_dict(area_lib).round(prec)
                adf.columns = ['aL=%i' % k for k in adf.columns]
                assert adf.notna().all().all()
                
                locs = np.linspace(.1, .9, len(adf.columns))
                for i, (label, row) in enumerate(adf.iterrows()):
                    ax.bar(locs + i * 0.1, row.values, label=label, width=0.1, alpha=0.5,
                           color=self.bar_colors_d[label],
                           tick_label=row.index)
                
                # first row
                if xvar == min(lVals_d['xvar']):
                    ax.set_ylabel('area under rl difference curve')
                    ax.legend()
 
            #===================================================================
            # #wrap xvar
            #===================================================================
            
            log.debug('finished xvar=%.2f' % xvar)
            
        #=======================================================================
        # post plot --------
        #=======================================================================
        if plot:
            #===================================================================
            # draw the vfunc  and setup the axis
            #===================================================================
                
            lineKwargs = {'color':'black', 'linewidth':1.0, 'linestyle':'dashed'}
     
            #=======================================================================
            # #depth-damage
            #=======================================================================
            first = True
            for xvar, ax in ax_d['dd'].items():
            
                vfunc.plot(ax=ax, label=vfunc.name, lineKwargs={**{'marker':'o', 'markersize':3, 'fillstyle':'none'}, **lineKwargs})
                ax.set_title('xvar = %.2f' % xvar)
                ax.set_xlabel(xcn)
                ax.grid()
                
                if first:
                    ax.set_ylabel(ycn)
                    first = False
    
            #=======================================================================
            # #delta
            #=======================================================================
            
            first = True
            for xvar, ax in ax_d['delta'].items():
                ax.set_xlabel(xcn)
                ax.grid()
                
                ax.hlines(0, -5, 5, label='true', **lineKwargs)
                
                if first:
                    ax.set_ylabel('%s (mean - true)' % ycn)
                    first = False
                            
                # ax.legend() #need to turn on at the end
                
            # set limits
            for rowName in ax_d.keys():
                for colName, ax in ax_d[rowName].items():
                    
                    # xlims
                    if not rowName == 'bars' and not xlims is None:
                        ax.set_xlim(xlims)
                        
                    # ylims
                    if not ylims_d[rowName] is None:
                        ax.set_ylim(ylims_d[rowName])
                        
            self.output_fig(fig, fname='rlMeans_%s_%s' % (vfunc.vid, self.fancy_name), out_dir=out_dir,
                            transparent=False, overwrite=True)
         
        #=======================================================================
        # write data
        #=======================================================================
        newNames_l = [{v:k for k, v in names_d.items()}[lvl] for lvl in [1, 2]]  # retrieve labes from original dxcol
        
        res_dxcol = pd.concat(res_lib, axis=1, names=newNames_l)
        
        return res_dxcol

    def calc_agg_imp(self, vfunc, 
                        
                        xmean, 
                        
                          #aggrevation levels to consider
                          aggLevels_l= [2, 
                                        5, 
                                        100,
                                        ],
                        
                          #random depths pramaeters
                          xvars_ar = np.linspace(.1,1,num=3), #varice values to iterate over
                        statsFunc = scipy.stats.norm, #function to use for the depths distribution
                         depths_resolution=2000,  #number of depths to draw from the depths distritupion
                        
 
                        logger=None,
 
                        ):
        """generate synthetic depths of different variance then compute aggregated rloss 
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('calc_agg_imp')
        log.debug('on %s'%vfunc.name)
        
 
        #===================================================================
        # #depth variances (for generating depth values)-------
        #===================================================================
        res_lib, xvals_lib = dict(), dict()
        for xvar in xvars_ar:
            res_d = dict()
            title = 'xmean=%.2f xvar=%.2f' % (xmean, xvar)
            log.debug(title)
            #===============================================================
            # get depths
            #===============================================================
            #build depths distribution
 
            rv = statsFunc(loc=xmean, scale=xvar)
            #self.plot_rv(rv, ax=ax, xlab='depth (m)', title=title)
            #get depths set
            depths_ar = rv.rvs(size=depths_resolution)
            xvals_lib[xvar] = pd.Series(depths_ar, dtype=np.float32, name=xvar)
            #===============================================================
            # #get average impact (w/o aggregation)
            #===============================================================
 
            res_d[0] = self.get_rloss(vfunc, depths_ar, annotate=True, ax=None)
            #===============================================================
            # #get average impacts for varios aggregation levels
            #===============================================================
            for aggLevel in aggLevels_l:
                log.debug('%s aggLevel=%s' % (title, aggLevel))
                #get aggregated depths
                depMean_ar = np.array([a.mean() for a in np.array_split(depths_ar, math.ceil(len(depths_ar) / aggLevel))])
                
                # get these losses
                res_d[aggLevel] = self.get_rloss(vfunc, depMean_ar, ax=None)
            
            #===================================================================
            # wrap
            #===================================================================
            res_lib[xvar] = pd.concat(res_d, axis=1)
            
        #=======================================================================
        # wrap
        #=======================================================================
        
        return pd.concat(res_lib, axis=1), pd.concat(xvals_lib, axis=1)
    
    def get_agg_imp_fig(self, 
                        dxcol,
                        
                        #plotting random vars
                        xmean=None,
                        xvals_df = None, #random xvalues
                        statsFunc = scipy.stats.norm, #function to use for the depths distribution+
                        
                        #plotting dep-dmt
                        vfunc=None,
                        colorMap=None,
                        
                        #plot conrols                        
                        fig_id=0,
                        
                         ylims_d = {
                                  'dep-dmg':(0,30), 'rl_box':(0,50)
                                 },
                         xlims = (-1, 3),
                        
                         logger=None,
                        ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_agg_imp_fig')
        log.info('on %s'%vfunc.name)
        
        if colorMap is None: colorMap=self.colorMap
        xcn, ycn = self.xcn, self.ycn
        
        names_d = {lvlName:i for i, lvlName in enumerate(dxcol.columns.names)}
        #=======================================================================
        # retrieve
        #=======================================================================
        aggLevels_l = dxcol.columns.get_level_values(names_d['aggLevel']).unique().tolist()
        xvars_ar = dxcol.columns.get_level_values(names_d['xvar']).unique()
        
        
        
        
        #get color map for aggLevels
        """aggLevlel=0 should be included to match the rl_dxcol plots"""
        cmap = plt.cm.get_cmap(name=colorMap)        
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k,ni in dict(zip(aggLevels_l, np.linspace(0, 1, len(aggLevels_l)))).items()}
        
        #=================================================================== 
        # setup working fig 
        #===================================================================
        fig, ax_d = self.get_matrix_fig(['depths_dist', 'dep-dmg', 'rl_box'], xvars_ar.tolist(), 
            figsize=(4 * len(xvars_ar), 8), fig_id=fig_id)
        
 
            
        """
        fig.show()
        """
        #===================================================================
        # #depth variances (for generating depth values)-------
        #===================================================================
 
        for xvar, gdx0 in dxcol.groupby(level=names_d['xvar'], axis=1):
 
            res_d = dict()
            #===================================================================
            # plot random depths-------
            #===================================================================
            if not xmean is  None:
                axName = 'depths_dist'
                ax = ax_d[axName][xvar]
                
                title = 'xmean=%.2f xvar=%.2f' % (xmean, xvar)
     
 
                #build depths distributio
                rv = statsFunc(loc=xmean, scale=xvar)
                self.plot_rv(rv, ax=ax, xlab='depth (m)', title=title)
                
                #get depths set
                if not xvals_df is None:
                    
                    depths_ar = xvals_df.loc[:, xvar].values
                    
                    ax.hist(depths_ar, color='blue', alpha=0.3, label='%i samps' %len(depths_ar), density=True, bins=20)
                    #ax.text(0.5, 0.9, '%i samples' % depths_resolution, transform=ax.transAxes, va='center', fontsize=8, color='blue')
                    
                #===============================================================
                # wrap
                #===============================================================
                if axName in ylims_d: ax.set_ylim(ylims_d[axName])
                ax.set_xlim(xlims)
                
                ax.legend()
 
            #===============================================================
            # dep-dmt plots--------
            #===============================================================
            #===================================================================
            # scatter
            #===================================================================
            axName ='dep-dmg'
            ax = ax_d[axName][xvar]
            
            res_d=dict() #collecting for the box plot
            for aggLevel, gdf1 in gdx0.droplevel(names_d['xvar'], axis=1).groupby(level=0, axis=1):
                
                dd_df = gdf1.droplevel(0, axis=1).dropna(axis=0).reset_index(drop=True)
 
                #add jitter to depths?
 
                self.get_rloss(vfunc, dd_df[xcn].values, rl_ar=dd_df[ycn].values,
                                             ax=ax, annotate=False, color=newColor_d[aggLevel], 
                                             label='aL=%i (%i)'%(aggLevel, len(dd_df)))
            
                res_d[aggLevel] = dd_df[ycn].values
            """
            view(gdx0)
            """
            #===================================================================
            # add lines
            #===================================================================
            
            
            #plot base f unction
            vfunc.plot(ax=ax, lineKwargs = dict(alpha=1.0, color='black', linewidth=0.75, linestyle='dashed'),
                       #label='base',
                       )
            
            #plot vertical line for xmean
            ax.axvline(x=xmean, color='red', alpha=0.8, linewidth=1, linestyle='dashed', label='xmean=%.2f'%xmean)
            
            #===================================================================
            # wrap
            #===================================================================
            
            ax.legend() #add legend to scatter
            
            ax.set_ylabel(ycn)
            ax.grid()
            
            if axName in ylims_d: ax.set_ylim(ylims_d[axName])
            ax.set_xlim(xlims)
            #===============================================================
            # box plot plotting---------
            #===============================================================
            axName='rl_box'
            ax = ax_d[axName][xvar]
 
            #add box plots from container
            boxres_d = ax.boxplot(res_d.values(), labels=['aL=%i' % k for k in res_d.keys()])
 
            #change box colors
            for aggLevel, boxLine in dict(zip(res_d.keys(), boxres_d['boxes'])).items():
                boxLine.set_color(newColor_d[aggLevel])
 
            
            #wrap
            ax.set_ylabel(ycn)
            
            if axName in ylims_d: ax.set_ylim(ylims_d[axName])
            #ax.set_xlim(xlims)
            
            
            
 
        
        return fig

    def get_rl_xmDisc(self,  # 
                          vfunc,
 
                          # xvalues to iterate (xmean)
                          xdomain=(0, 2),  # min/max of the xdomain
                          xdomain_res=30,  # number of increments for the xdomain
                          
                          # plotting parameters
 
                        logger=None,
                          **kwargs):
        """
        xmeans iteration wrapper for calc_agg_imp()
        """

        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('get_rl_xmDisc')
        log.info('on %s' % vfunc.name)
        
        fig_lib, xvals_lib, res_lib, ax_lib = dict(), dict(), dict(), dict()
        
        #=======================================================================
        # loop mean deths 
        #=======================================================================
 
        xmean_ar = np.linspace(xdomain[0], xdomain[1], num=xdomain_res)
        for i, xmean in enumerate(xmean_ar):
            log.info('%i/%i xmean=%.2f' % (i, len(xmean_ar), xmean))
            
            res_lib[xmean], xvals_lib[xmean] = self.calc_agg_imp(vfunc, xmean,
                                                    logger=log.getChild(str(i)), **kwargs)
                
            #===============================================================
            # wrap mean
            #===============================================================
 
            log.debug('finished xmean = %.2f' % xmean)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.debug('finsihed on %i' % len(res_lib))
        
        #=======================================================================
        # #=======================================================================
        # # #write the xmean plots---------
        # #=======================================================================
        # """
        # plt.show()
        # """
        # if plot:
        #     for xmean, fig in fig_lib.items():
        #         log.info('handling figure %i'%fig.number)
        #         #harmonize axis
        #         self.apply_axd('set_xlim', ax_lib[xmean], rowNames=['depths_dist', 'dep-dmg'], 
        #                        left=self.master_xlim[0], right=self.master_xlim[1])
        #         
        #         #print(xmean)
        #         self.output_fig(fig, fname='%s_xmean%.2f'%(self.fancy_name, xmean),
        #                         fmt='png',
        #                         out_dir=out_dir)
        #=======================================================================
 
        #=======================================================================
        # wrap
        #=======================================================================
        
        dxcol = pd.concat(res_lib, axis=1, names=['xmean', 'xvar', 'aggLevel', 'vars'])
 
        return dxcol, pd.concat(xvals_lib, axis=1, names=['xmean', 'xvar'])
    
    def apply_axd(self, #apply a method to an axis within an ax_d 
                  methodName, 
                  ax_d,
                  rowNames=None, #default: apply to all
                  colNames=None,
                  **kwargs
                  ):
        #=======================================================================
        # #defaults
        #=======================================================================
        if rowNames is None:
            rowNames = list(ax_d.keys())
        
        
        if colNames is None:
            colNames =  list(ax_d[list(ax_d.keys())[0]].keys())
            
        #=======================================================================
        # check selection
        #=======================================================================
        l = set(rowNames).difference(ax_d.keys())
        assert len(l)==0, l
        
        l = set(colNames).difference(ax_d[list(ax_d.keys())[0]].keys())
        assert len(l)==0, l
        
        #=======================================================================
        # apply
        #=======================================================================
        res_lib = {k:dict() for k in ax_d.keys()}
        for rowName, ax_d0 in ax_d.items():
            if not rowName in rowNames: continue
 
            for colName, ax in ax_d0.items():
                if not colName in colNames: continue
                
                res_lib[rowName][colName] = getattr(ax, methodName)(**kwargs)
                
        return res_lib
                
 

    
    def get_rloss(self, 
                    vfunc,
                    depths_ar, 
                    rl_ar=None,
                    ax=None,
                    label='base',
                    annotate=False,
                    color='black',
                    linekwargs = dict(s=7, marker='x', alpha=0.4)
                    ):
                    #vfunc, ax_d, res_d, xvar, ax, depths_ar):
        """seems better to keeep these on the vfunc"""
        #=======================================================================
        # get impacts
        #=======================================================================
        if rl_ar is None:
            rl_ar = vfunc.get_rloss(depths_ar)
 
        assert isinstance(rl_ar, np.ndarray)
        #=======================================================================
        # #plot these
        #=======================================================================
        if not ax is None:
            """
            plt.show()
            """
            ax.scatter(depths_ar, rl_ar,  color=color,label=label, **linekwargs)
            
            #label
            if annotate:
                #ax.legend()
                ax.set_title(vfunc.name)
                ax.set_ylabel(vfunc.ycn)
                ax.set_xlabel(vfunc.xcn)
        
 
        
        return pd.DataFrame([depths_ar, rl_ar], index=[vfunc.xcn, vfunc.ycn]).T.sort_values(vfunc.xcn).reset_index(drop=True) 
            



    def write_dxcol(self, dxcol, dkey,
                    write_csv=False,
                    out_fp = None, 
                    logger=None,
                    overwrite=None,
                    ):
        
        #=======================================================================
        # defautls
        #=======================================================================
        if logger is None: logger=self.logger
        if overwrite is None: overwrite=self.overwrite
        log=logger.getChild('write_dxcol')
        assert isinstance(dxcol, pd.DataFrame)
        
        #=======================================================================
        # add to data container
        #=======================================================================
        assert not dkey in self.data_d
        self.data_d[dkey] = dxcol.copy()
        #=======================================================================
        # #picklle
        #=======================================================================
        if out_fp is None:
            out_fp = os.path.join(self.out_dir, '%s_%s.pickle' % (self.fancy_name, dkey))
            
        else:
            write_csv=False
            
        if os.path.exists(out_fp):
            assert overwrite, 'file exists and overwrite=False\n    %s'%out_fp
            
        with open(out_fp, 'wb') as f:
            pickle.dump(dxcol, f, pickle.HIGHEST_PROTOCOL)
        log.info('wrote \'%s\' (%s) to \n    %s' % (dkey, str(dxcol.shape), out_fp))
        
        
        #=======================================================================
        # #csv
        #=======================================================================
        if write_csv:
            out_fp2 = os.path.join(self.out_dir, '%s_%s.csv' % (self.fancy_name, dkey))
            dxcol.to_csv(out_fp2, 
                index=True) #keep the names
            
            log.info('wrote %s to \n    %s' % (str(dxcol.shape), out_fp2))
        
        return out_fp
    
 
    def __exit__(self, #destructor
                 *args, **kwargs):
        
        print('Session.__exit__ on \'%s\''%self.__class__.__name__)
        
        #=======================================================================
        # log major containers
        #=======================================================================
        print('__exit__ w/ data_d.keys(): %s'%(list(self.data_d.keys())))
        
        if len(self.ofp_d)>0:
            print('__exit__ with %i ofp_d:'%len(self.ofp_d))
            for k,v in self.ofp_d.items():
                print('    \'%s\':r\'%s\','%(k,v))
              
              
        
        
        super().__exit__(*args, **kwargs)
        
    
