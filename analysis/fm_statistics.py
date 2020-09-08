from fm_behavioral import *


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
    print(pg.wilcoxon(df.loc[('CS+',phase),'cr'], df.loc[('CS-',phase),'cr'], tail='greater'))



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

'''source memory collapsed'''
df = pd.read_csv('../cleaned_avg_sm.csv'
    ).set_index(['condition','encode_phase','response_phase','subject']).sort_index()

for encode_phase in phases:
    for con in cons:
        for response_phase in phases:
            x = df.loc[(con,encode_phase,response_phase),'prop']
            print(encode_phase,con,response_phase,'\n')
            onesample_bdm(x=x,mu=1/3,tail='two-tailed')
            print('\n')

for encode_phase in phases:
    # print(encode_phase)
    print(pg.wilcoxon(df.loc[('CS+',encode_phase,'acquisition'),'prop'],df.loc[('CS-',encode_phase,'acquisition'),'prop'],tail='greater'))

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
pg.corr(df.loc[('baseline','acquisition'),'mem_count'],df.loc[('baseline','acquisition'),'cr'])[['r','p-val']]
pg.corr(df.loc[('acquisition','acquisition'),'mem_count'],df.loc[('acquisition','acquisition'),'cr'])[['r','p-val']]
pg.corr(df.loc[('extinction','acquisition'),'mem_count'],df.loc[('extinction','acquisition'),'cr'])[['r','p-val']]

q = sns.lmplot(data=df.reset_index().query('response_phase == "acquisition"'),
            x='mem_count',y='cr',col='encode_phase',palette=spal[0])
q.set_axis_labels('"Acquisition" response difference\n(only using remembered items)','Corrected recognition difference')


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
        curve[i,:] = expit(np.linspace(0,1,100) * logreg.coef_ + logreg.intercept_)[0]

    pval = 1 - np.mean(boot_res > 0)
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

