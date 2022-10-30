'''
Created on Oct. 26, 2022

@author: cefect
'''
import os
import matplotlib.pyplot as plt
from aggF.scripts import AggSession1F, view
cm = 1/2.54
 
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
        assert len(serx)>0
        assert serx.notna().any().any()
        
        
        mdex = serx.index
        keys_all_d = {k:mdex.unique(v).tolist() for k,v in map_d.items()} #order matters
        
        #check the keys
        for k,v in keys_all_d.items():
            assert len(v)>0, k
            
            
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
                
 
           
        
class Session_AggF(AggSession1F, Plot_rlDelta_xb):
    pass