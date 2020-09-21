import os
import numpy as np
import pandas as pd

from fm_config import *
from fm_behavioral import *

# def sm_convert(x):
#     if x == 1: return 'baseline'
#     elif x == 2: return 'acquisition'
#     elif x == 3: return 'extinction'

def clean_sm():
    raw_dir = '../source_mem_raw'
    
    for sub in sm_sub_args:
        subj = bids_meta(sub)
        #read in files
        in_file = os.path.join(raw_dir,'Typicality_Source_mem-%s-1.csv'%(sub))
        df = pd.read_csv(in_file,header=1)
        
        #isolate columns we need and rename them
        df = df[['stims','SourceMem.RESP','Typicality.RESP']]
        df = df.rename(columns={'stims':'stimulus','SourceMem.RESP':'source_memory','Typicality.RESP':'typicality'})

        #clean up the stimulus columns
        df.stimulus = df.stimulus.apply(lambda x: x[7:])

        #add trial_type
        df['trial_type'] = df.stimulus.apply(lambda x: 'CS+' if subj.csplus in x else 'CS-')

        df['encode_phase'] = ''
        df['response'] = ''
        #add true encoding phase and grab the recognition memory
        for i in df.index:
            stim = df.loc[i,'stimulus']
            mem_loc = np.where(subj.mem_df.stimulus == stim)[0][0]
            df.loc[i,['encode_phase','response']] = subj.mem_df.loc[mem_loc,['encode_phase','response']]
        df = df.rename(columns={'response':'recognition_memory'})

        out_file = os.path.join(subj.events,subj.fsub+'_task-source_memory_typicality_events.tsv')
        df.to_csv(out_file,sep='\t',index=False)

#save full sm with hc acc for lmm
dat = {}
for sub in smt_sub_args:
    subj = bids_meta(sub)
    subj.behav['source_memory_typicality']['subject'] = subj.num
    dat[sub] = subj.behav['source_memory_typicality']
df = pd.concat(dat.values()
        ).set_index(['subject']
        ).drop([18,20,120]
        ).reset_index()
df['group'] = df.subject.apply(lgroup)
df['hc_acc'] = df.recognition_memory.apply(lambda x: 1 if x == 'DO' else 0)
df.to_csv('../cleaned_full_sm.csv',index=False)

#save avg. sm for graphing
df = smt().sm_df.set_index(['subject']
        ).drop([18,20,120]
        ).reset_index(
        ).to_csv('../cleaned_avg_sm.csv',index=False)

#clean and save typicality
df = smt().ty_df.set_index(['subject']
        ).drop([18,20,120]
        ).reset_index(
        ).to_csv('../cleaned_avg_ty.csv',index=False)


#getting together the difference scores for both memory tests
cr = pd.read_csv('../cleaned_corrected_recognition.csv'
    ).drop(columns='group'
    ).set_index(['condition','subject','encode_phase'])
cr = (cr.loc['CS+'] - cr.loc['CS-']).reset_index()
cr['group'] = cr.subject.apply(lgroup)

sm = pd.read_csv('../cleaned_avg_sm.csv'
    ).drop(columns='group'
    ).set_index(['condition','subject','encode_phase','response_phase'])
sm = (sm.loc['CS+'] - sm.loc['CS-']).reset_index()

cr = cr.set_index(['subject','encode_phase'])
sm = sm.set_index(['subject','encode_phase'])
sm['cr'] = 0.0
sm['hr'] = 0.0
for sub in xcl_sub_args:
    for phase in phases:
        sm.loc[(sub,phase),'cr'] = cr.loc[(sub,phase),'cr']
        sm.loc[(sub,phase),'hr'] = cr.loc[(sub,phase),'hr']
sm = sm.reset_index()
sm['group'] = sm.subject.apply(lgroup)
sm.to_csv('../memory_difference_scores.csv',index=False)



###collect stim list for memorability

#this parapgrah just proves that all subs had the same 
s1 = bids_meta(1).mem_df
s1 = s1.sort_values(by='stimulus')

for sub in all_sub_args:
    s2 = bids_meta(sub).mem_df
    s2 = s2.sort_values(by='stimulus')

    assert np.array_equal(s1.stimulus.values,s2.stimulus.values)

s1['stim_out'] = 'https://github.com/dunsmoorlab/fc_memory_study/stims/' + s1.stimulus

s1.stim_out.to_csv('../stimulus_list.txt',index=False)


#add trial number to subjects source memory data
for sub in smt_sub_args:
    subj = bids_meta(sub)
    sm = subj.behav['source_memory_typicality'].copy().set_index('stimulus')
    sm['trial_number'] = 0
    for encode_phase in ['baseline','acquisition','extinction']:
        enc = subj.behav[encode_phase].copy()
        enc['trial_number'] = 0
        for con in cons:
            enc.loc[enc.trial_type == con,'trial_number'] = np.arange(1,25)
        for s, stim in enumerate(enc.stimulus):
            sm.loc[stim,'trial_number'] = enc.loc[s,'trial_number']
    sm.to_csv(f'{subj.events}/{subj.fsub}_task-source_memory_typicality_events.tsv',sep='\t')
