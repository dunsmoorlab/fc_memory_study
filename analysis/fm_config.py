import os
import sys
import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import matplotlib.ticker as mtick

from matplotlib import rcParams
from wesanderson import wes_palettes
from numpy.random import RandomState
from sklearn.linear_model import LogisticRegression
from matplotlib.ticker import MultipleLocator, IndexLocator, FixedLocator
from scipy.special import expit
from matplotlib.patches import Patch
from scipy.stats import chisquare, zscore

sns.set_context('notebook',font_scale=1.4)
sns.set_style('ticks', {'axes.spines.right':False, 'axes.spines.top':False})
# sns.set_style({'axes.facecolor':'.9','figure.facecolor':'.9'})
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['savefig.dpi'] = 300
# rcParams['savefig.format'] = 'png'

cons = ['CS+','CS-']
phases = ['baseline','acquisition','extinction']
groups = ['healthy','ptsd']

sub_args = [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,23,24,25,26]
p_sub_args = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,118, 120, 121, 122, 123, 124, 125]
all_sub_args = sub_args + p_sub_args

smt_sub_args = [2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,101,102,103,104,105,106,107,108,109,110,111,112,113,114,116,117,118,120]
xcl_sub_args = [2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,19,21,101,102,103,104,105,106,107,108,109,110,111,112,113,114,116,117,118]

subjects = {'healthy':sub_args,
            'ptsd':p_sub_args,
            'all':all_sub_args}

gpal = list((wes_palettes['Zissou'][0],wes_palettes['Royal1'][1]))
cpal = ['darkorange','grey']
spal = list((wes_palettes['Darjeeling1'][-1],wes_palettes['Darjeeling1'][0],wes_palettes['Darjeeling1'][1],))
spal = sns.color_palette(spal,n_colors=3,desat=.8)
tpal = list((wes_palettes['Chevalier'][0],wes_palettes['Chevalier'][1]))
cpoint = sns.color_palette(cpal,n_colors=2,desat=.75)
phase_convert = {1:'baseline',2:'acquisition',3:'extinction'}
phase2int = {'baseline':1,'acquisition':2,'extinction':3}

# WORK = '/work/05426/ach3377/lonestar/'
# HOME = '/home1/05426/ach3377/'
# SCRATCH = '/scratch/05426/ach3377/'
# gPPI_codebase = HOME + 'gPPI/'

def mkdir(path,local=False):
    if not local and not os.path.exists(path):
        os.makedirs(path)
def lgroup(x):
    if x > 100: return 'ptsd'
    else: return 'healthy'

#these are BIDS-app made
data_dir = os.path.join(os.path.expanduser('~'),'Documents','fearmem')
bids_dir = os.path.join(data_dir,'fm-bids')

# bids_dir = os.path.join(SCRATCH,'fc-bids')
# deriv    = os.path.join(bids_dir, 'derivatives')
# prep_dir = os.path.join(deriv,'fmriprep')
# fs_dir   = os.path.join(deriv,'freesurfer')

#these are user made
# model   = os.path.join(deriv,'model');#mkdir(model)
# preproc = os.path.join(deriv,'preproc');#mkdir(preproc)
# group_masks = os.path.join(deriv,'group_masks')



# std_1mm_brain = os.path.join(WORK,'standard','MNI152_T1_1mm_brain.nii.gz')
# std_3mm_brain = os.path.join(WORK,'standard','MNI152_T1_3mm_brain.nii.gz')
# std_3mm_brain_mask = os.path.join(WORK,'standard','MNI152_T1_3mm_brain_mask.nii.gz')
# std_2009_brain = os.path.join(SCRATCH,'standard','MNI152NLin2009cAsym_T1_1mm_brain.nii.gz')
# std_2009_brain_mask = os.path.join(SCRATCH,'standard','MNI152NLin2009cAsym_T1_1mm_brain_mask.nii.gz')
# std_2009_brain_3mm = os.path.join(SCRATCH,'standard','MNI152NLin2009cAsym_T1_3mm_brain.nii.gz')
# std_2009_brain_mask_3mm = os.path.join(SCRATCH,'standard','MNI152NLin2009cAsym_T1_3mm_brain_mask.nii.gz')

tasks = {'baseline':{'n_trials':48,'ses':1,'n_tr':259},
         'acquisition':{'n_trials':48,'ses':1,'n_tr':259},
         'extinction':{'n_trials':48,'ses':1,'n_tr':259},
         'renewal':{'n_trials':24,'ses':2,'n_tr':135},
         'memory_run-01':{'n_trials':80,'ses':2,'n_tr':310},
         'memory_run-02':{'n_trials':80,'ses':2,'n_tr':310},
         'memory_run-03':{'n_trials':80,'ses':2,'n_tr':310},
         'localizer_run-01':{'n_trials':24,'ses':2,'n_tr':240},
         'localizer_run-02':{'n_trials':24,'ses':2,'n_tr':240},
         'source_memory_typicality':{},
         }

slices={'CS+':{
                 'baseline':{'encoding':slice(0,24),
                             'retrieval':slice(144,168)},
              
              'acquisition':{'encoding':slice(24,48),
                             'retrieval':slice(168,192)},

         'early_extinction':{'encoding':slice(48,56),
                            'retrieval':slice(192,200)},
               
               'extinction':{'encoding':slice(56,72),
                             'retrieval':slice(200,216)}},
        'CS-':{
                 'baseline':{'encoding':slice(72,96),
                            'retrieval':slice(216,240)},
              
              'acquisition':{'encoding':slice(96,120),
                            'retrieval':slice(240,264)},

         'early_extinction':{'encoding':slice(120,128),
                            'retrieval':slice(264,272)},
               
               'extinction':{'encoding':slice(128,144),
                            'retrieval':slice(272,288)}}}

mem_slices = {'CS+':{
                 'baseline':slice(0,24),
              
              'acquisition':slice(24,48),

         'early_extinction':slice(48,56),
               
               'extinction':slice(56,72),

                     'foil':slice(72,120)},

        'CS-':{
                 'baseline':slice(120,144),
              
              'acquisition':slice(144,168),

         'early_extinction':slice(168,176),
               
               'extinction':slice(176,192),

                     'foil':slice(192,240)}}

class bids_meta(object):

    def __init__(self, sub):
     
        self.num = int(sub)
        
        self.fsub = 'sub-FC{0:0=3d}'.format(self.num)

        self.subj_dir = os.path.join(bids_dir, self.fsub)
        self.events   = os.path.join(self.subj_dir, 'events')

        self.behav = {}
        for task in tasks: self.behav[task] = self.load(task)
        self.cs_lookup()

        self.mem_df = pd.concat([self.behav['memory_run-01'],self.behav['memory_run-02'],self.behav['memory_run-03']]).reset_index(drop=True)

    def load(self,task):
        try:
            file = pd.read_csv(os.path.join(self.events,self.fsub+'_task-'+task+'_events.tsv'),sep='\t')
            file['subject'] = self.num
            return file
        except FileNotFoundError:
            pass
    
    def cs_lookup(self):    
        if self.behav['acquisition'].loc[0,'stimulus'][0] == 'a':
            self.csplus = 'animals'
            self.csminus = 'tools'
        elif self.behav['acquisition'].loc[0,'stimulus'][0] == 't':
            self.csplus = 'tool'
            self.csminus = 'animal'

            # self.prep_dir = os.path.join(prep_dir,self.fsub)
            # self.fs_dir   = os.path.join(fs_dir,self.fsub)

            # self.model_dir   = os.path.join(model,self.fsub);mkdir(self.model_dir,local)
            # self.feat_dir    = os.path.join(self.model_dir,'feats');mkdir(self.feat_dir,local)
            # self.preproc_dir = os.path.join(preproc,self.fsub);mkdir(self.preproc_dir,local)
            
            # self.reference    = os.path.join(self.preproc_dir,'reference');mkdir(self.reference,local)
            # self.t1           = os.path.join(self.reference,'T1w.nii.gz')
            # self.t1_mask      = os.path.join(self.reference,'T1w_mask.nii.gz')
            # self.t1_brain     = os.path.join(self.reference,'T1w_brain.nii.gz')
            # self.refvol       = os.path.join(self.reference,'boldref.nii.gz')
            # self.refvol_mask  = os.path.join(self.reference,'boldref_mask.nii.gz')
            # self.refvol_brain = os.path.join(self.reference,'boldref_brain.nii.gz')
            
            # self.ref2std      = os.path.join(self.reference,'ref2std.mat')
            # self.std2ref      = os.path.join(self.reference,'std2ref.mat')

            # self.ref2std3     = os.path.join(self.reference,'ref2std3.mat')
            # self.std32ref     = os.path.join(self.reference,'std32ref.mat')


            # self.ref2t1       = os.path.join(self.reference,'ref2t1.mat')
            # self.t12std_nii   = os.path.join(self.reference,'t12std')
            # self.t12std       = os.path.join(self.reference,'t12std.mat')
            # self.t12std_warp  = os.path.join(self.reference,'t12std_warp')

            # self.func = os.path.join(self.preproc_dir,'func');mkdir(self.func,local)
            # self.beta = os.path.join(self.preproc_dir,'lss_betas');mkdir(self.beta,local) 

            # self.fs_regmat = os.path.join(self.reference,'RegMat.dat')
            # self.faa       = os.path.join(self.reference,'aparc+aseg.nii.gz')
            # self.saa       = os.path.join(self.reference,'std_aparc+aseg.nii.gz')
            
            # self.masks  = os.path.join(self.preproc_dir,'masks');mkdir(self.masks,local)
            # self.weights = os.path.join(self.preproc_dir,'rsa_weights');mkdir(self.weights,local)

            # self.rsa = os.path.join(self.model_dir,'rsa_results');mkdir(self.rsa,local)

def pdm(x,y,tail='two',nperm=10000):
    '''ASSUMES PAIRED DATA (x,y)
    tail = 'two' (default) or "greater" ''' 
    
    if type(x) == pd.core.series.Series:
        x = x.values

    if type(y) == pd.core.series.Series:
        y = y.values

    assert x.shape == y.shape

    if True in np.isnan(x) or True in np.isnan(y):
        del_x = np.where(np.isnan(x) == True)[0]
        del_y = np.where(np.isnan(y) == True)[0]
        del_ = np.unique(np.concatenate((del_x,del_y)))

        x = np.delete(x,del_)
        y = np.delete(y,del_)
    
    _n = x.shape[0]

    diff = x - y
    fixed = diff.mean()


    R = RandomState(42)
    perm_res = np.zeros(nperm)
    for i in range(nperm):
        flip = R.choice([-1,1],_n)
        samp = diff * flip
        perm_res[i] = samp.mean()

    if tail == 'greater':
        p = np.mean(perm_res > fixed)
    elif tail == 'two':
        p = np.mean(np.abs(perm_res) > np.abs(fixed))

    print(pg.ttest(x,y,paired=True))
    return fixed, p, perm_res

def onesample_bdm(x,mu=0,tail='two-tailed',n_boot=10000):
    R = np.random.RandomState(42)

    boot_res = np.zeros(n_boot)

    for i in range(n_boot):
        boot_res[i] = R.choice(x,size=x.shape,replace=True).mean()
    avg = x.mean()

    if tail == 'two-tailed':
        if avg > mu:
            p = (1 - np.mean(boot_res > mu)) * 2
        else:
            p = (1 - np.mean(boot_res < mu)) * 2
        ci = (np.percentile(boot_res,2.5),np.percentile(boot_res,97.5))

    elif tail == 'greater':
        p = 1 - np.mean(boot_res > mu)
        ci = (np.percentile(boot_res,5),np.percentile(boot_res,100))

    elif tail == 'less':
        p = 1 - np.mean(boot_res < mu)
        ci = (np.percentile(boot_res,0),np.percentile(boot_res,95))

    if p == 0.0: p = 1/n_boot

    res = pd.DataFrame({'mu':mu,'avg':avg,'CI_l':ci[0],'CI_u':ci[1],'p':p,'tail':tail},index=[0])
    return res.round(4)