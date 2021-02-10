from fm_behavioral import *
from robust_corr import skipped_corr
'''RECOGNITION MEMORY'''
#a priori memory enhancement effect (CS+ > CS-)
df = pd.read_csv('../cleaned_corrected_recognition.csv')
df = df.set_index(['group','condition','encode_phase','subject']).sort_index()

'''Retroactive effect: Baseline CS+ >  CS-'''
pg.wilcoxon(df.loc[('healthy','CS+','baseline'),'cr'],df.loc[('healthy','CS-','baseline'),'cr'],tail='greater')

pg.wilcoxon(df.loc[('ptsd','CS+','baseline'),'cr'],df.loc[('ptsd','CS-','baseline'),'cr'],tail='greater')

#Emotional enhancement: Acquisition CS+ > CS-
pg.wilcoxon(df.loc[('healthy','CS+','acquisition'),'cr'],df.loc[('healthy','CS-','acquisition'),'cr'],tail='greater')

pg.wilcoxon(df.loc[('ptsd','CS+','acquisition'),'cr'],df.loc[('ptsd','CS-','acquisition'),'cr'],tail='greater')

#Proactive effect: Extinction CS+ > CS-
pg.wilcoxon(df.loc[('healthy','CS+','extinction'),'cr'],df.loc[('healthy','CS-','extinction'),'cr'],tail='greater')

pg.wilcoxon(df.loc[('ptsd','CS+','extinction'),'cr'],df.loc[('ptsd','CS-','extinction'),'cr'],tail='greater')

'''FALSE ALARM?'''
#phase doesnt really matter here since the false alarms are repeated across all phases
pg.wilcoxon(df.loc[('healthy','CS+','extinction'),'fa'],df.loc[('healthy','CS-','extinction'),'fa'])
pg.wilcoxon(df.loc[('ptsd','CS+','extinction'),'fa'],df.loc[('ptsd','CS-','extinction'),'fa'])


'''COLLAPSING ACROSS GROUPS FOR RECOGNITION MEMORY'''
df = pd.read_csv('../cleaned_corrected_recognition.csv'
    ).set_index(['condition','encode_phase','subject']).sort_index()
for phase in phases:
    print(phase,'\n')
    print(pg.wilcoxon(df.loc[('CS+',phase),'cr'], df.loc[('CS-',phase),'cr'], tail='two-sided'))



'''GROUP COMPARISONS - RECOGNITION MEMORY'''
df = pd.read_csv('../cleaned_corrected_recognition.csv'
    ).drop(columns='group'
    ).set_index(['condition','subject','encode_phase'])
df = (df.loc['CS+'] - df.loc['CS-']).reset_index()
df['group'] = df.subject.apply(lgroup)
df = df.set_index(['group','encode_phase'])

pg.mwu(df.loc[('healthy','baseline'),'cr'], df.loc[('ptsd','baseline'),'cr'])
pg.mwu(df.loc[('healthy','acquisition'),'cr'], df.loc[('ptsd','acquisition'),'cr'])
pg.mwu(df.loc[('healthy','extinction'),'cr'], df.loc[('ptsd','extinction'),'cr'])

#again phase doesn't matter here for false alarms
pg.mwu(df.loc[('healthy','baseline'),'fa'], df.loc[('ptsd','baseline'),'fa'])


'''source memory'''
#normality
df = pd.read_csv('../cleaned_avg_sm.csv')
# pg.normality(df,dv='prop',group='group')
df = df.set_index(['group','condition','encode_phase','response_phase','subject'])

#in baseline, comp of CS+ and CS- assigned to acquisition
pg.wilcoxon(df.loc[('healthy','CS+','baseline','acquisition'),'prop'],df.loc[('healthy','CS-','baseline','acquisition'),'prop'],tail='greater')
pg.wilcoxon(df.loc[('ptsd','CS+','baseline','acquisition'),'prop'],df.loc[('ptsd','CS-','baseline','acquisition'),'prop'],tail='greater')

#baseline, but with memory proportion
pg.wilcoxon(df.loc[('healthy','CS+','baseline','acquisition'),'mem_prop'],df.loc[('healthy','CS-','baseline','acquisition'),'mem_prop'],tail='greater')
pg.wilcoxon(df.loc[('ptsd','CS+','baseline','acquisition'),'mem_prop'],df.loc[('ptsd','CS-','baseline','acquisition'),'mem_prop'],tail='greater')

#extiction, CS+ vs. CS- assigned to acquisition
pg.wilcoxon(df.loc[('healthy','CS+','extinction','acquisition'),'prop'],df.loc[('healthy','CS-','extinction','acquisition'),'prop'],tail='greater')
pg.wilcoxon(df.loc[('ptsd','CS+','extinction','acquisition'),'prop'],df.loc[('ptsd','CS-','extinction','acquisition'),'prop'],tail='greater')

#extinction, but with memory proportion
pg.wilcoxon(df.loc[('healthy','CS+','extinction','acquisition'),'mem_prop'],df.loc[('healthy','CS-','extinction','acquisition'),'mem_prop'],tail='greater')
pg.wilcoxon(df.loc[('ptsd','CS+','extinction','acquisition'),'mem_prop'],df.loc[('ptsd','CS-','extinction','acquisition'),'mem_prop'],tail='greater')

'''SOURCE MEMORY COLLAPSED'''
df = pd.read_csv('../cleaned_avg_sm.csv'
    ).set_index(['condition','encode_phase','response_phase','subject']).sort_index()

'''one sample bootstrap means'''
for encode_phase in phases:
    for con in cons:
        for response_phase in phases:
            x = df.loc[(con,encode_phase,response_phase),'prop']
            print(encode_phase,con,response_phase,'\n')
            onesample_bdm(x=x,mu=1/3,tail='two-tailed')
            print('\n')

'''CS+*_A vs CS-*_A'''
for encode_phase in phases:
    print(encode_phase)
    print(pg.wilcoxon(df.loc[('CS+',encode_phase,'acquisition'),'prop'],df.loc[('CS-',encode_phase,'acquisition'),'prop'],tail='two-sided'))

'''chisqaure'''
freq = df.groupby(['condition','encode_phase','response_phase']).mean()
for encode_phase in phases:
    for con in cons:
        print(encode_phase, con)
        print(chisquare(freq.loc[(con,encode_phase),'_count'].values,[8,8,8]))



'''CORRELATIONS WITH RECOGNITION MEMORY'''
df = pd.read_csv('../memory_difference_scores.csv'
        # ).set_index(['group','encode_phase','response_phase','subject'])
        ).set_index(['encode_phase','response_phase','subject'])

# #baseline correlation with all prop
# pg.corr(df.loc[('healthy','baseline','acquisition'),'cr'],df.loc[('healthy','baseline','acquisition'),'prop'])
# pg.corr(df.loc[('ptsd','baseline','acquisition'),'cr'],df.loc[('ptsd','baseline','acquisition'),'prop'])

# #baseline correlation with memory prop
# pg.corr(df.loc[('healthy','baseline','acquisition'),'cr'],df.loc[('healthy','baseline','acquisition'),'mem_prop'])
# pg.corr(df.loc[('ptsd','baseline','acquisition'),'cr'],df.loc[('ptsd','baseline','acquisition'),'mem_prop'])

# #extinction correlation with all prop
# pg.corr(df.loc[('healthy','extinction','acquisition'),'cr'],df.loc[('healthy','extinction','acquisition'),'prop'])
# pg.corr(df.loc[('ptsd','extinction','acquisition'),'cr'],df.loc[('ptsd','extinction','acquisition'),'prop'])

# #extinction correlation with memory prop
# pg.corr(df.loc[('healthy','extinction','acquisition'),'cr'],df.loc[('healthy','extinction','acquisition'),'mem_prop'])
# pg.corr(df.loc[('ptsd','extinction','acquisition'),'cr'],df.loc[('ptsd','extinction','acquisition'),'mem_prop'])

#group collapse
pg.corr(df.loc[('baseline','acquisition'),'prop'],df.loc[('baseline','acquisition'),'cr'])[['r','p-val']]
pg.corr(df.loc[('acquisition','acquisition'),'prop'],df.loc[('acquisition','acquisition'),'cr'])[['r','p-val']]
pg.corr(df.loc[('extinction','acquisition'),'prop'],df.loc[('extinction','acquisition'),'cr'])[['r','p-val']]

q = sns.lmplot(data=df.reset_index().query('response_phase == "acquisition"'),
            x='mem_count',y='cr',col='encode_phase',palette=spal[0])
q.set_axis_labels('"Acquisition" response difference\n(only using remembered items)','Corrected recognition difference')

#testing without outliers
q = df.loc[('baseline','acquisition')]
q = q.drop(index=q.cr.idxmin())
pg.corr(q.prop,q.cr)

'''Typicality'''
df = pd.read_csv('../cleaned_avg_ty.csv')
pg.normality(df,dv='typicality',group='group')

df = df.set_index(['group','condition','encode_phase','subject'])

for group in groups:
    for phase in phases:
        print(group,phase,'\n')
        print(pg.ttest(df.loc[(group,'CS+',phase),'typicality'],df.loc[(group,'CS-',phase),'typicality'],paired=True))

'''TYPICALITY COLLAPSED'''
df = pd.read_csv('../cleaned_avg_ty.csv'
                ).drop(columns='group'
                ).groupby(['condition','subject']
                ).mean().sort_index()
pg.ttest(df.loc['CS+','typicality'],df.loc['CS-','typicality'],paired=True)

#getting data for ANOVA with category
df = pd.read_csv('../cleaned_full_sm.csv'
        ).rename(columns={'trial_type':'condition'})
df['category'] = df.stimulus.apply(lambda x: 'animals' if 'animals' in x else 'tools')

df = df.groupby(['condition','category','subject','encode_phase']).mean()
df = df.reset_index().set_index(['subject','condition','encode_phase']).sort_index()
for sub in xcl_sub_args:
    df.loc[(sub,'CS-',slice('acquisition','extinction')),'category'] = df.loc[(sub,'CS+','acquisition'),'category']
pg.mixed_anova(data=df.reset_index(),dv='typicality',subject='subject',within='condition',between='category')

'''typicality by phase CS+ vs. CS-'''
df = pd.read_csv('../cleaned_avg_ty.csv'
                ).drop(columns='group'
                ).set_index(['encode_phase','condition','subject']
                ).sort_index()
for phase in phases:
    print(phase)
    print(pg.ttest(df.loc[(phase,'CS+'),'typicality'],df.loc[(phase,'CS-'),'typicality'],paired=True))

'''typicality to CR regressions'''
df = pd.read_csv('../memory_difference_scores.csv')
df = df[df.response_phase == 'acquisition']
df = df.set_index(['encode_phase','subject']).sort_index()
for phase in phases:
    skipped_corr(df.loc[phase,'prop'],df.loc[phase,'cr'])
    skipped_corr(df.loc[phase,'typicality'],df.loc[phase,'cr'])
    
'''super-subject logistic regression'''
R = np.random.RandomState(42)

def sourcebeta(dat,subjects,n_boot=10000):
    #initialize logistic
    logreg = LogisticRegression(solver='lbfgs')
    
    #create output
    boot_res = np.zeros((n_boot,3))
    
    onehot = pd.get_dummies(dat.source_memory)
    onehot = onehot.rename(columns=phase_convert)

    for i in range(n_boot):
        y = np.zeros(816) #i just hardcode this for some reason, its 408 if doing by group, or 816 if both groups
        while len(np.unique(y)) == 1:
            _samp = R.choice(subjects,len(subjects))
            X = onehot.loc[_samp].values
            y = dat.loc[_samp,'hc_acc'].values
        
        logreg.fit(X,y)
        boot_res[i] = logreg.coef_

    pvals = 1 - np.mean(boot_res > 0, axis=0)
    print(pvals)

    return boot_res
    

df = pd.read_csv('../cleaned_full_sm.csv'
        ).rename(columns={'trial_type':'condition'}
        ).set_index(['condition','encode_phase','subject']
        ).sort_index()

betas = {}
# for group in groups:
#     betas[group] = {}
#     #get the right subject args, too lazy to put this in config
#     if group == 'healthy': subjects = [i for i in xcl_sub_args if i < 100]
#     elif group == 'ptsd':  subjects = [i for i in xcl_sub_args if i > 100]
    
for con in cons:
    betas[con] = {}    
    for phase in phases:
        print(con,phase)
        dat = df.loc[con,phase].copy()

        betas[con][phase] = sourcebeta(dat,xcl_sub_args)

with open('../sourcemem_logreg_betas.p','wb') as file:
    pickle.dump(betas,file,protocol=pickle.HIGHEST_PROTOCOL)

#see graphing script for p-values for the above, just easier to do it there


'''Typicality logreg'''
R = np.random.RandomState(42)

def typbeta(dat,subjects,n_boot=10000):
    #initialize logistic
    logreg = LogisticRegression(solver='lbfgs')
    
    #create output
    boot_res = np.zeros(n_boot)
    curve    = np.zeros((n_boot,100))

    for i in range(n_boot):
        y = np.zeros(2448)
        while len(np.unique(y)) == 1:
            _samp = R.choice(subjects,len(subjects))
            X = dat.loc[_samp,'typicality'].values.reshape(-1,1)
            y = dat.loc[_samp,'hc_acc'].values
        
        logreg.fit(X,y)
        boot_res[i] = logreg.coef_
        curve[i,:] = expit(np.linspace(1,7,100) * logreg.coef_ + logreg.intercept_)[0]

    pval = (1 - np.mean(boot_res > 0))*2
    print(pval)

    return boot_res, curve

df = pd.read_csv('../cleaned_full_sm.csv'
        ).rename(columns={'trial_type':'condition'}
        ).set_index(['condition','subject']
        ).sort_index()

betas = {}
curves = {}
for con in cons:
    print(con)
    dat = df.loc[con].copy()
    betas[con], curves[con] = typbeta(dat,xcl_sub_args)

diff = betas['CS+'] - betas['CS-']

with open('../typicality_logreg_betas.p','wb') as file:
    pickle.dump(betas,file,protocol=pickle.HIGHEST_PROTOCOL)
with open('../typicality_logreg_curves.p','wb') as file:
    pickle.dump(curves,file,protocol=pickle.HIGHEST_PROTOCOL)

'''typicality super-subject by phase'''
R = np.random.RandomState(42)

def typbeta_phase(dat,subjects,n_boot=10000):
    #initialize logistic
    logreg = LogisticRegression(solver='lbfgs')
    
    #create output
    boot_res = np.zeros(n_boot)
    curve    = np.zeros((n_boot,100))

    for i in range(n_boot):
        y = np.zeros(816)
        while len(np.unique(y)) == 1:
            _samp = R.choice(subjects,len(subjects))
            X = dat.loc[_samp,'typicality'].values.reshape(-1,1)
            y = dat.loc[_samp,'hc_acc'].values
        
        logreg.fit(X,y)
        boot_res[i] = logreg.coef_
        # curve[i,:] = expit(np.linspace(1,7,100) * logreg.coef_ + logreg.intercept_)[0]

    pval = (1 - np.mean(boot_res > 0))*2
    print(pval)

    return boot_res#, curve

df = pd.read_csv('../cleaned_full_sm.csv'
        ).rename(columns={'trial_type':'condition'}
        ).set_index(['encode_phase','condition','subject']
        ).sort_index()

betas = {}
curves = {}
for phase in phases:
    betas[phase] = {}
    curves[phase] = {}
    for con in cons:
        print(phase,con)
        dat = df.loc[(phase,con)].copy()
        # betas[phase][con], curves[phase][con] = typbeta_phase(dat,xcl_sub_args)
        betas[phase][con] = typbeta_phase(dat,xcl_sub_args)

with open('../typicality_logreg_betas.p','wb') as file:
    pickle.dump(betas,file,protocol=pickle.HIGHEST_PROTOCOL)

for phase in phases:
    diff = betas[phase]['CS+'] - betas[phase]['CS-']
    pval = (1 - np.mean(diff > 0))*2
    print(phase, pval)


'''Typicality logreg within subject'''
df = pd.read_csv('../cleaned_full_sm.csv'
        ).rename(columns={'trial_type':'condition'}
        ).set_index(['subject','condition']
        ).sort_index()
betas = pd.DataFrame(columns=['beta','intercept'],index=pd.MultiIndex.from_product([cons,xcl_sub_args],names=['condition','subject']))
logreg = LogisticRegression(solver='lbfgs')
for sub in xcl_sub_args:
    for con in cons:
        X = df.loc[(sub,con),'typicality'].values.reshape(-1,1)
        y = df.loc[(sub,con),'hc_acc'].values
        logreg.fit(X,y)
        betas.loc[(con,sub),('beta','intercept')] = logreg.coef_[0][0], logreg.intercept_[0]
betas = betas.astype(float)
betas.to_csv('../typicality_within_sub_logreg_betas.csv')
pg.ttest(betas.loc['CS+','beta'],0)
pg.ttest(betas.loc['CS-','beta'],0)
pg.ttest(betas.loc['CS+','beta'],betas.loc['CS-','beta'],paired=True)


'''multiple logistic'''
df = pd.read_csv('../cleaned_full_sm.csv'
        ).rename(columns={'trial_type':'condition'}
        ).set_index(['encode_phase','subject']
        ).sort_index()
df = df.drop(columns=['stimulus','trial_number','group','recognition_memory'])
df.condition = df.condition.apply(lambda x: 1 if x == 'CS+' else 0)
df['acq_resp'] = df.source_memory.apply(lambda x: 1 if x == 'acquisition' else 0)
df['ext_resp'] = df.source_memory.apply(lambda x: 1 if x == 'extinction' else 0)
# df['pre_resp'] = df.source_memory.apply(lambda x: 1 if x == 'baseline' else 0)
#df.source_memory = df.source_memory.apply(lambda x: 1 if x == 'acquisition' else 0) #this was in the preprint

#one phase
beta_df = {}
# for phase in phases:
for phase in ['baseline']:
    pdf = df.loc[phase].copy()

    betas = pd.DataFrame(columns=['typicality','condition','acq_resp','ext_resp','intercept'],index=pd.MultiIndex.from_product([xcl_sub_args],names=['subject']))
    logreg = LogisticRegression(solver='lbfgs')
    for sub in xcl_sub_args:
        pdf.loc[sub,'typicality'] = zscore(pdf.loc[sub,'typicality'])
        X = pdf.loc[sub,['typicality','condition','acq_resp','ext_resp']].values
        y = pdf.loc[sub,'hc_acc'].values
        logreg.fit(X,y)
        betas.loc[sub,('typicality','condition','acq_resp','ext_resp')] = logreg.coef_[0]
        betas.loc[sub,'intercept'] = logreg.intercept_
    betas = betas.astype(float)
    betas = betas.reset_index().melt(id_vars='subject',value_vars=['condition','typicality','acq_resp','ext_resp'],var_name='feature',value_name='beta')

    fig, ax = plt.subplots()
    sns.swarmplot(data=betas,x='feature',y='beta',zorder=1)
    sns.pointplot(data=betas,x='feature',y='beta',color='black',zorder=10000)
    ax.hlines(0,ax.get_xlim()[0],ax.get_xlim()[1],linestyle=':',color='black',zorder=0)
    betas['phase'] = phase
    beta_df[phase] = betas
    betas = betas.set_index(['feature','subject']).sort_index()
    print(pg.ttest(betas.loc['acq_resp','beta'],betas.loc['typicality','beta'],paired=True))
    print(pg.ttest(betas.loc['acq_resp','beta'],betas.loc['condition','beta'],paired=True))
    print(pg.ttest(betas.loc['condition','beta'],betas.loc['typicality','beta'],paired=True))

    print(pg.ttest(betas.loc['acq_resp','beta'],betas.loc['ext_resp','beta'],paired=True))
    print(pg.ttest(betas.loc['ext_resp','beta'],betas.loc['condition','beta'],paired=True))
    print(pg.ttest(betas.loc['ext_resp','beta'],betas.loc['typicality','beta'],paired=True))


    print(pg.ttest(betas.loc['condition','beta'],0))
    print(pg.ttest(betas.loc['acq_resp','beta'],0))
    print(pg.ttest(betas.loc['ext_resp','beta'],0))
    print(pg.ttest(betas.loc['typicality','beta'],0))

beta_df['baseline'].to_csv('../multiple_logreg_phase.csv',index=False)


'''SCR for a reviewer'''
df = pd.read_csv('../fc_scr_comp.csv').set_index(['subject']).loc[xcl_sub_args].reset_index()

acq = df.copy()
acq['phase'] = acq.quarter.apply(lambda x: 'acquisition' if x < 3 else 'not')
acq = acq.groupby(['phase','subject']).mean()

print(pg.ttest(acq.loc[('acquisition'),'scr'],0))

df = df.set_index(['quarter','subject']).sort_index()

print(pg.ttest(acq.loc[('acquisition'),'scr'], df.loc[(4),'scr'], paired=True))

'''expectancy for a reviewer'''
acq = pd.concat([bids_meta(sub).behav['acquisition'] for sub in xcl_sub_args])
ext = pd.concat([bids_meta(sub).behav['extinction'] for sub in xcl_sub_args])

acq.response = acq.response.apply(lambda x: 1 if x == 1 else 0)
acq = acq.groupby(['trial_type','subject']).mean()['response'].sort_index()
acq = acq.loc['CS+'] - acq.loc['CS-']

ext.response = ext.response.apply(lambda x: 1 if x == 1 else 0)
ext = ext.reset_index().rename(columns={'index':'trial'})
ext['half'] = ext.trial.apply(lambda x: 1 if x < 12 else 2)

ext = ext.groupby(['trial_type','half','subject']).mean()['response']
ext = ext.loc['CS+'] - ext.loc['CS-']

print(pg.ttest(acq,0))
print(pg.ttest(acq,ext.loc[2],paired=True))


'''TYPICALITY WITH THE SOURCE MEMORY RESPONSES'''
df = pd.read_csv('../cleaned_full_sm.csv'
        ).rename(columns={'trial_type':'condition'}
        ).groupby(['condition','source_memory','subject']
        ).mean(
        ).reset_index()
#jk its in R bc there are missing cells