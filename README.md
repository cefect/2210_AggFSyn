# Flood damage model bias caused by aggregation

Scripts for computing potential flood damage function error from aggregation against synthetically produced depths. 

ICFM9 work. PIAHS publication

## Use

### synthetic data
original results files are here: `l:\10_IO\2112_Agg\outs\agg1F\`
- generate synthetic depths of different variance then compute aggregated rloss: `aggF.scripts.AggSession1F.calc_agg_imp()`

### Figures
All plots are called from `aggF.main.all_r0_plot()`:
- Figure 1. Relative loss vs. anchor depth (xb) for three example functions and three synthetic depth generation variances (σ2) computed against the depth values and aggregation (s) described in the text. Shaded areas show the corresponding q95 and q5 for the RL computed for each level of aggregation: `aggF.da.plot_matrix_funcs_synthX()`
- Figure 2. Relative loss error for three levels of aggregation (s) and three synthetic depth families (σ2) for a selection of the models: `aggF.da.plot_matrix_rlDelta_xb()`


## Install
- clone repo
- create ./definitions.py (see below)
- build with conda using environment.yml
- add submodules


## Submodule
git submodule add -b 2210_AggFSyn https://github.com/cefect/coms.git


### definitions.py

```
import os
proj_dir = r'C:\LS\09_REPOS\02_JOBS\2210_AggFSyn'
src_dir = proj_dir
src_name='aggF'

 

logcfg_file=r'C:\LS\09_REPOS\01_COMMON\coms\logger.conf'

root_dir=r'C:\LS\10_IO\2112_Agg'
wrk_dir=root_dir
 

#add latex engine
os.environ['PATH'] += r";C:\Users\cefect\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
```