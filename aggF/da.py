'''
Created on Oct. 26, 2022

@author: cefect
'''
import os
import pandas as pd
idx = pd.IndexSlice

import matplotlib.pyplot as plt
from aggF.scripts import AggSession1F, view
cm = 1/2.54

class Plot_funcs_synthX(object):
    """plot an example function and the aggregated synthetic data"""
    def get_rl_xm_stats_dxcol(self,
                              rl_xmDisc_dxcol_fp,vid_l, write=True,
                              **kwargs):
        """rl stats (mean, quartiles) from synthetic data"""
        log, tmp_dir, out_dir, ofp, _, write = self._func_setup('rl_xm_stats_dxcol', ext='.pkl', **kwargs)
        #synthetic depths and rl at different aggregatins. see build_rl_xmDisc_dxcol()
        dxcol_raw = pd.read_pickle(rl_xmDisc_dxcol_fp)
        
        #selected functions
        dxcol = dxcol_raw.loc[:, idx[vid_l, :, :, :]]
        del dxcol_raw
        
        #clean up and drop depths
        dxcol_rl = dxcol.loc[:, idx[:, :, :, :, 'rl']].droplevel('vars', axis=1
                 ).swaplevel('xmean', 'aggLevel', axis=1).sort_index(sort_remaining=True, axis=1)
                 
        #get stats for plotting
        dx = pd.concat({'mean':dxcol_rl.mean(),
                 'qlo':dxcol_rl.quantile(q=0.05),
                 'qhi':dxcol_rl.quantile(q=0.95),
                 }, axis=1)
        
        if write: 
            dx.to_pickle(ofp)
            log.info(f'wrote {str(dx.shape)} to \n    {ofp}')
            
            
        return dx
            
    def plot_matrix_funcs_synthX(self, 
                                 dx,
                                 f_serx=None,
                                 output_fig_kwargs=dict(),
                                 **kwargs):
        """plot functions and aggregated RL values"""
        
        log, tmp_dir, out_dir, _, _, write = self._func_setup('p_fs_rlVx', **kwargs)
        
        #relative loss plots mean plots
        fig, ax_d = self.get_matrix_lines(dx['mean'])
        
        #add quantiles ranges
        self.add_q_fills(fig, ax_d, dx.drop('mean', axis=1))
        
        #add functions
        if not f_serx is None:
            self.add_funcs(fig, ax_d, f_serx)
            
        #=======================================================================
        # #add legend
        #=======================================================================
        
        handles, labels = ax_d[49][1.0].get_legend_handles_labels() #get legned info 
        fig.legend( handles, labels, ncols=2, loc='upper right', 
                     borderaxespad=0.,
                    #mode='expand',
                    bbox_to_anchor=(1.0,0.98))
        
        #=======================================================================
        # output
        #=======================================================================
        if write:
            return self.output_fig(fig, 
                                   ofp=os.path.join(out_dir, f'p_fs_rlVx.'+self.output_format), 
                                   logger=log, **output_fig_kwargs)
        else:
            return fig, ax_d
        
        """
        plt.show()
        """
        
        
        
    def get_matrix_lines(self,serx,
                 map_d={'row':'df_id', 'col':'xvar', 'color':'aggLevel', 'x':'xmean'},
               matrix_kwargs=dict(figsize=(17 * cm, 17 * cm), set_ax_title=False, add_subfigLabel=True, fig_id=0, constrained_layout=False),
               
               colorMap = 'cool', 
               output_fig_kwargs=dict(),
                               
                               **kwargs):
        """matrix plot"""
        log, tmp_dir, out_dir, _, _, write = self._func_setup('ax_mat', **kwargs)
        
        
        #=======================================================================
        # extract data
        #=======================================================================
        assert len(serx) > 0
        assert serx.notna().any().any()
        
        mdex = serx.index
        keys_all_d = {k:mdex.unique(v).tolist() for k, v in map_d.items()}  # order matters
        
        # check the keys
        for k, v in keys_all_d.items():
            assert len(v) > 0, k
            
        #=======================================================================
        # setup figure
        #=======================================================================
        plt.close('all') 
 
        fig, ax_d = self.get_matrix_fig(keys_all_d['row'], keys_all_d['col'], 
                                    sharey='all',sharex='all',  
                                    logger=log, **matrix_kwargs)
        
        #colors
        color_d = self._get_color_d(map_d['color'], keys_all_d['color'], colorMap=colorMap)
        self.color_d=color_d.copy()
        
        """
        plt.show()
        """
        #=======================================================================
        # loop and plot
        #=======================================================================
        levels = [map_d[k] for k in ['row', 'col']]
        for gk0, gsx0 in serx.groupby(level=levels):
            #===================================================================
            # setup
            #===================================================================
            ax = ax_d[gk0[0]][gk0[1]]
            keys_d = dict(zip(levels, gk0))            
 
            #===================================================================
            # loop each df_id series (faint)
            #===================================================================
            for gk1, gsx1 in gsx0.groupby(level=map_d['color']):
                #keys_d[map_d['color']] = gk1
                xar, yar = gsx1.index.get_level_values(map_d['x']).values, gsx1.values
 
                ax.plot(xar, yar, color=color_d[gk1],
                        label='$s=%i$'%gk1,
                        zorder=1,
                        **{'linestyle':'solid', 'marker':None, 'markersize':2, 'alpha':1.0, 'linewidth':1.0}
                        )
                
        #=======================================================================
        # post format
        #=======================================================================
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
                
 
                #first col
                if col_key == keys_all_d['col'][0]:
                    ax.set_ylabel('$RL$ (percent)')
                    
                    
                #last row
                if row_key==keys_all_d['row'][-1]:
                    ax.set_xlabel('$x$ (m)')
                    
                #labels
                ax.text(0.98,0.05, '$f=%i$, $\sigma=%.2f$'%(row_key, col_key),transform=ax.transAxes, va='bottom', ha='right')
                
        #=======================================================================
        # warp
        #=======================================================================
        return fig, ax_d
    
    def add_q_fills(self,fig, ax_d, dx,
                    map_d={'row':'df_id', 'col':'xvar', 'color':'aggLevel', 'x':'xmean'},
                    color_d=None,
                     **kwargs):
        """add the quantile fills"""
        log, tmp_dir, out_dir, _, _, write = self._func_setup('add_q_fills', **kwargs)
        
        #=======================================================================
        # extract data
        #=======================================================================
        assert len(dx) > 0
        assert dx.notna().any().any()
        
        mdex = dx.index
        keys_all_d = {k:mdex.unique(v).tolist() for k, v in map_d.items()}  # order matters
        
        # check the keys
        for k, v in keys_all_d.items():
            assert len(v) > 0, k
        
        if color_d is None:
            color_d=self.color_d
            #color_d = self._get_color_d(map_d['color'], keys_all_d['color'], colorMap=colorMap)
        #=======================================================================
        # loop and plot
        #=======================================================================
        levels = [map_d[k] for k in ['row', 'col']]
        for gk0, gx0 in dx.groupby(level=levels):
            #===================================================================
            # setup
            #===================================================================
            ax = ax_d[gk0[0]][gk0[1]]
            keys_d = dict(zip(levels, gk0))            
 
            #===================================================================
            # loop each df_id series (faint)
            #===================================================================
            for gk1, gx1 in gx0.groupby(level=map_d['color']):
                
                ax.fill_between(
                        gx1.index.get_level_values(map_d['x']).values,  # xvalues
                         gx1['qhi'],  # top of hatch
                         gx1['qlo'],  # bottom of hatch
                         color=color_d[gk1], alpha=0.1, hatch=None)
                
        #=======================================================================
        # wrap
        #=======================================================================
        return  
    
    def add_funcs(self, fig, ax_d, serx,
                  **kwargs):
        """add raw functions"""
        
        log, tmp_dir, out_dir, _, _, write = self._func_setup('add_funcs', **kwargs)
 
        #=======================================================================
        # loop on each function
        #=======================================================================
        for gk, ser in serx.groupby('df_id'):
            ser = ser.droplevel('df_id')
            #===================================================================
            # add to each axis
            #===================================================================
            for i, ax in ax_d[gk].items():
                xlims, ylims = ax.get_xlim(), ax.get_ylim()
                ax.plot(ser.index, ser.values, color='black', linestyle='dashed',
                        label='raw $f$', alpha=0.8, linewidth=0.8)
                
                ax.set_xlim(xlims)
                ax.set_ylim(ylims)
                
        #=======================================================================
        # wrap
        #=======================================================================
        return 
 
 
        
        
        """
        plt.show()
        """
        
        
class Plot_rlDelta_xb(object):
    def plot_matrix_rlDelta_xb(self,serx,
                               map_d={'row':'aggLevel', 'col':'xvar', 'color':'model_id', 'x':'xmean'},
                               matrix_kwargs=dict(figsize=(17 * cm, 17 * cm), set_ax_title=False, add_subfigLabel=True, fig_id=0, constrained_layout=False),
                               
                               colorMap = 'Dark2',
                               #plot_kwargs_lib=None,
                               #plot_kwargs={'linestyle':'solid', 'marker':None, 'markersize':7, 'alpha':0.8, 'linewidth':0.5},
                               output_fig_kwargs=dict(),
                               
                               **kwargs):
        """matrix plot of all functions rlDelta vs. xb"""
        
        log, tmp_dir, out_dir, _, _, write = self._func_setup('p_rlD_sb', **kwargs)
        
        #=======================================================================
        # extract data
        #=======================================================================
        assert len(serx) > 0
        assert serx.notna().any().any()
        
        mdex = serx.index
        keys_all_d = {k:mdex.unique(v).tolist() for k, v in map_d.items()}  # order matters
        
        # check the keys
        for k, v in keys_all_d.items():
            assert len(v) > 0, k
            
        #=======================================================================
        # setup figure
        #=======================================================================
        plt.close('all') 
 
        fig, ax_d = self.get_matrix_fig(keys_all_d['row'], keys_all_d['col'], 
                                    sharey='all',sharex='all',  
                                    logger=log, **matrix_kwargs)
        
        #colors
        color_d = self._get_color_d(map_d['color'], keys_all_d['color'], colorMap=colorMap)
        
        
        #plot kwargs
        #=======================================================================
        # """here we populate with blank kwargs to ensure every series has some kwargs"""
        # if plot_kwargs_lib is None: plot_kwargs_lib=dict()
        # for k in keys_all_d['color']:
        #     if not k in plot_kwargs_lib:
        #         plot_kwargs_lib[k] = plot_kwargs
        #     else:
        #         plot_kwargs_lib[k] = {**plot_kwargs, **plot_kwargs_lib[k]} #respects precedent
        #=======================================================================
        #=======================================================================
        # loop and plot
        #=======================================================================
        levels = [map_d[k] for k in ['row', 'col']]
        for gk0, gsx0 in serx.groupby(level=levels):
            #===================================================================
            # setup
            #===================================================================
            ax = ax_d[gk0[0]][gk0[1]]
            keys_d = dict(zip(levels, gk0))
            
 
            #===================================================================
            # loop each df_id series (faint)
            #===================================================================
            for gk1, gsx1 in gsx0.groupby(level=[map_d['color'], 'df_id']):
                #keys_d[map_d['color']] = gk1
                xar, yar = gsx1.index.get_level_values(map_d['x']).values, gsx1.values
 
                ax.plot(xar, yar, color=color_d[gk1[0]],label=None,zorder=1,
                        **{'linestyle':'solid', 'marker':None, 'markersize':7, 'alpha':0.4, 'linewidth':0.2}
                        )
                
            #===================================================================
            # loop and plot model average
            #===================================================================
            for gk1, gsx1 in gsx0.groupby(level=map_d['color']):
                ser = gsx1.groupby('xmean').mean()
                xar, yar = ser.index.values, ser.values
                ax.plot(xar, yar, color=color_d[gk1],label=gk1,zorder=2,
                        **{'linestyle':'solid', 'marker':None, 'markersize':7, 'alpha':0.9, 'linewidth':1.0}
                        )
                
        #=======================================================================
        # post format
        #=======================================================================
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
                
 
                #first col
                if col_key == keys_all_d['col'][0]:
                    ax.set_ylabel('$RL_{s2}-RL_{s1}$ (frac)')
                    
                    
                #last row
                if row_key==keys_all_d['row'][-1]:
                    ax.set_xlabel('$x_{b}$ (m)')
                    
                #labels
                ax.text(0.98, 0.95, '$s=%i$, $\sigma=%.2f$'%(row_key, col_key),transform=ax.transAxes, va='top', ha='right')
        
        """
        plt.show()
        """
        #=======================================================================
        # legend
        #=======================================================================
        handles, labels = ax.get_legend_handles_labels() #get legned info 
        fig.legend( handles, labels, ncols=len(labels), loc='upper right', 
                     borderaxespad=0.,
                    #mode='expand',
                    bbox_to_anchor=(1.0,0.75))
        
        """not working.. not sure how to adjust the subplots
        fig.subplots_adjust(top=0.2)"""
        #=======================================================================
        # output
        #=======================================================================
        if write:
            return self.output_fig(fig, 
                                   ofp=os.path.join(out_dir, f'rlErr_xb.'+self.output_format), 
                                   logger=log, **output_fig_kwargs)
        else:
            return fig, ax_d
                
 
           
        
class Session_AggF(AggSession1F, Plot_rlDelta_xb, Plot_funcs_synthX):
    def load_vfunc_ddf(self, vid_l, write=True, **kwargs):
        """loadd rl vs. x data for a collection of vfuncs"""
        log, tmp_dir, out_dir, ofp, _, write = self._func_setup('vfunc_df', ext='.pkl', **kwargs)
        
        
        vid_df = self.build_vid_df(vid_l=vid_l, write=False)
        vf_d = self.build_vf_d(vid_df=vid_df)
        
        d = dict()
        for k,o in vf_d.items():
            d[k] = o.ddf.copy()
        
        #wd v RL for each function
        f_serx = pd.concat(d, names=['df_id', 'index'], axis=0).set_index('wd', drop=True, append=True).droplevel('index').iloc[:, 0]
        
        if write:
            f_serx.to_pickle(ofp)
            log.info(f'wrote {str(f_serx.shape)} to \n    {ofp}')        
        
        return f_serx
