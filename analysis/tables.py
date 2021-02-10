from paper_graphics_style import p_convert
'''code for grabbing all the data necessary for the tables of memory data'''
ctx_rname = {'baseline':'pre-conditioning','acquisition':'fear conditioning','extinction':'post-conditioning','':''}
ctx_rname_short = {'baseline':'pre','acquisition':'cond.','extinction':'post'}

'''recognition memory'''
df = pd.read_csv('../cleaned_corrected_recognition.csv')

table = df.groupby(['condition','encode_phase'])['cr'].apply(onesample_bdm)[['avg','CI_l','CI_u']]
table['sem'] = df.groupby(['condition','encode_phase'])['cr'].sem()
table['95% CI'] = table[['CI_l','CI_u']].values.tolist()
table = table.reset_index().drop(columns=['level_2','CI_l','CI_u']
            ).rename(columns={'condition':'CS Type',
                              'encode_phase':'Temporal Context',
                              'avg':'Mean',
                              'sem':'Std. Error'})
table = table[['Temporal Context','CS Type','Mean','95% CI','Std. Error']]

#and false alarms
fadf = df[df.encode_phase=='baseline'].drop(columns='encode_phase')
fatable = fadf.groupby(['condition'])['fa'].apply(onesample_bdm)[['avg','CI_l','CI_u']]
fatable = fatable.reset_index().set_index('condition').drop(columns='level_1')
fatable['sem'] = fadf.groupby('condition')['fa'].sem()
fatable['95% CI'] = fatable[['CI_l','CI_u']].values.tolist()
fatable = fatable.reset_index().drop(columns=['CI_l','CI_u']
            ).rename(columns={'condition':'CS Type',
                              'encode_phase':'Temporal Context',
                              'avg':'Mean',
                              'sem':'Std. Error'})
fatable['Temporal Context'] = ''
fatable = fatable[['Temporal Context','CS Type','Mean','95% CI','Std. Error']]

table = pd.concat((table,fatable))

table['Temporal Context'] = table['Temporal Context'].apply(lambda x: ctx_rname[x])
table['Temporal Context'] = pd.Categorical(table['Temporal Context'],['pre-conditioning','fear conditioning','post-conditioning',''],ordered=True)

table.sort_values(by='Temporal Context').round(4).to_csv('../paper/corrected_recognition_table.csv',index=False)

'''Replicating Dunsmoor (2015) Nature - mean proportion of responses'''
dfs = {}
for sub in all_sub_args: 
    if sub in [18,20,120]:
        pass
    else:
        subj = bids_meta(sub)
        old = subj.mem_df[subj.mem_df.memory_condition == 'Old'].reset_index(drop=True)[['trial_type','encode_phase','response','stimulus']]
        old.response[old.response.isna()] = 'DN'
        old = old.groupby(['trial_type','encode_phase','response']).count() / 24

        new = subj.mem_df[subj.mem_df.memory_condition == 'New'].reset_index(drop=True)[['trial_type','encode_phase','response','stimulus']]
        new.response[new.response.isna()] = 'DN'
        new = new.groupby(['trial_type','encode_phase','response']).count() / 48

        sub_df = pd.concat((old,new))

        for con in cons:
            for phase in ['baseline','acquisition','extinction','foil']:
                for resp in ['DN','DO','MN','MO']:
                    try:
                        sub_df.loc[(con,phase,resp)]
                    except:
                        sub_df.loc[(con,phase,resp),'stimulus'] = 0.0

        sub_df['subject'] = sub
        dfs[sub] = sub_df.sort_index()

table = pd.concat(dfs.values()
     ).rename(columns={'stimulus':'proportion'}
     ).drop(columns=['subject']
     ).reset_index(
     ).groupby(['trial_type','encode_phase','response']
     ).mean(
     ).unstack(-1)
table.columns = table.columns.droplevel()
table = table[['DO','MO','MN','DN']]
table = table.reset_index()

table.encode_phase = pd.Categorical(table.encode_phase,['baseline','acquisition','extinction','foil'],ordered=True)
table = table.set_index(['encode_phase','trial_type']).sort_index()
table.round(3).to_csv('../paper/tables/Dunsmoor_2015_rep_table.csv')

'''Source memory'''
df = pd.read_csv('../cleaned_avg_sm.csv')

df['encode_phase'] = df['encode_phase'].apply(lambda x: ctx_rname[x])
df['encode_phase'] = pd.Categorical(df['encode_phase'],['pre-conditioning','fear conditioning','post-conditioning'],ordered=True)
df['response_phase'] = df['response_phase'].apply(lambda x: ctx_rname_short[x])
df['response_phase'] = pd.Categorical(df['response_phase'],['pre','cond.','post'],ordered=True)

table = df.groupby(['encode_phase','condition','response_phase'])['prop'].apply(onesample_bdm,(1/3))[['avg','CI_l','CI_u','p','tail']]
table['95% CI'] = table[['CI_l','CI_u']].values.tolist()
table = table.droplevel(-1).drop(columns=['CI_l','CI_u','tail']
                ).reset_index().rename(columns={'avg':'Mean',
                                  'sem':'Std. Error',
                                  'p':'P',
                                  'encode_phase':'Temporal Context',
                                  'condition':'CS Type',
                                  'response_phase':'Response'})
table = table[['Temporal Context','CS Type','Response','Mean','95% CI','P']]
table['_sig'] = table.P.apply(p_convert) 
table.round(4).to_csv('../paper/source_memory_table.csv',index=False)


'''Source memory betas'''
df = pd.concat((pd.read_csv('../paper/acquisition_sm_betas.csv'),
                pd.read_csv('../paper/baseline_sm_betas.csv'),
                pd.read_csv('../paper/extinction_sm_betas.csv'))).reset_index(drop=True)
df = df[['encode_phase','condition','response_phase','beta','ci','p']]
df['encode_phase'] = df['encode_phase'].apply(lambda x: ctx_rname[x])
df['encode_phase'] = pd.Categorical(df['encode_phase'],['pre-conditioning','fear conditioning','post-conditioning'],ordered=True)
df['response_phase'] = df['response_phase'].apply(lambda x: ctx_rname_short[x])
df['response_phase'] = pd.Categorical(df['response_phase'],['pre','cond.','post'],ordered=True)
table = df.rename(columns={'encode_phase':'Temporal Context',
                           'condition':'CS Type',
                           'response_phase':'Response',
                           'beta':'Mean',
                           'ci':'95% CI',
                           'p':'P'})
table.P = table.P.apply(lambda x: 0.0001 if x == 0 else x)
table = table.sort_values(by=['Temporal Context','CS Type','Response'])
table['_sig'] = table.P.apply(p_convert) 
table.round(4).to_csv('../paper/souce_memory_betas_table.csv',index=False)
#do the stuff

'''Typicality betas'''
df = pd.read_csv('../paper/tables/typicality_betas.csv')
df = df[['phase','condition','beta','ci','p']]
df['phase'] = df['phase'].apply(lambda x: ctx_rname[x])
df['phase'] = pd.Categorical(df['phase'],['pre-conditioning','fear conditioning','post-conditioning'],ordered=True)
table = df.rename(columns={'phase':'Temporal Context',
                           'condition':'CS Type',
                           'beta':'Mean',
                           'ci':'95% CI',
                           'p':'P'})
table = table.sort_values(by=['Temporal Context','CS Type'])
table['_sig'] = table.P.apply(p_convert)
table.round(4).to_csv('../paper/tables/typicality_table.csv',index=False)


'''table with hit rate split by source memory responses'''
df = pd.read_csv('../cleaned_full_sm.csv')

df['encode_phase'] = df['encode_phase'].apply(lambda x: ctx_rname[x])
df['encode_phase'] = pd.Categorical(df['encode_phase'],['pre-conditioning','fear conditioning','post-conditioning'],ordered=True)
df['source_memory'] = df['source_memory'].apply(lambda x: ctx_rname_short[x])
df['source_memory'] = pd.Categorical(df['source_memory'],['pre','cond.','post'],ordered=True)

df = df.groupby(['encode_phase','condition','source_memory','subject']).mean()
table = df.groupby(['encode_phase','condition','source_memory'])['hc_acc'].mean()
table = table.reset_index().set_index(['encode_phase','condition','source_memory'])
table['sem'] = df.groupby(['encode_phase','condition','source_memory'])['hc_acc'].sem()
table = table.reset_index().rename(columns={'hc_acc':'Mean',
                                  'sem':'Std. Error',
                                  'encode_phase':'Temporal Context',
                                  'condition':'CS Type',
                                  'source_memory':'Response'})
table.round(4).to_csv('../paper/tables/hit_rate_by_source_memory.csv',index=False)