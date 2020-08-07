from fm_config import *
from fm_behavioral import *

#recognition memory (CS+ > CS-)
df = pd.read_csv('../cleaned_corrected_recognition.csv')
df = df.drop(columns='group')
df = df.set_index(['condition','subject','encode_phase'])
df = (df.loc['CS+'] - df.loc['CS-']).reset_index()
df['group'] = df.subject.apply(lgroup)

fig, ax = plt.subplots()
sns.pointplot(data=df,x='encode_phase',y='cr',hue='group',palette=gpal,
                dodge=True,join=False,ax=ax)
ax.set_ylabel('CS+ - CS-\ncorrected recognition')
ax.set_xlabel('Encoding Phase')
ax.hlines(0,ax.get_xlim()[0],ax.get_xlim()[1],linestyle=':')
plt.tight_layout()


'''RECOGNITION MEMORY COLLAPSED ACROSS GROUPS'''
df = pd.read_csv('../cleaned_corrected_recognition.csv')

fig, ax = plt.subplots(1,3,figsize=(12,5),sharey=True)
for i, phase in enumerate(phases):
    dat = df[df.encode_phase == phase]

    # sns.stripplot(data=dat,x='encode_phase',y='cr',hue='condition',linewidth=1.5,hue_order=['CS-','CS+'],
    #             edgecolor='black',palette=[cpal[1],cpal[0]],dodge=True,ax=ax[i],jitter=1)


    ax[i].legend_.remove()    

    for sub in dat.subject.unique():
        csm = dat.cr[dat.condition == 'CS-'][dat.subject == sub].values[0]
        csp = dat.cr[dat.condition == 'CS+'][dat.subject == sub].values[0]
        ax[i].plot([-.1,.1],[csm,csp],c='grey',linewidth=.5,alpha=.8)
    
    sns.pointplot(data=dat,x='encode_phase',y='cr',hue='condition',hue_order=['CS-','CS+'],
            palette=[cpal[1],cpal[0]],ax=ax[i],dodge=True,saturation=.1)

    # sns.despine(ax=ax,trim=True)
    ax[i].set_ylim(-0.05,1)

    ax[i].yaxis.set_major_locator(MultipleLocator(.2))
    ax[i].yaxis.set_minor_locator(MultipleLocator(.1))

ax[0].set_ylabel('Corrected recognition')
ax[0].set_xlabel('')
ax[0].set_xticklabels(['Baseline'],ha='center')

ax[1].set_ylabel('')
ax[1].set_xlabel('Encoding phase')
ax[1].set_xticklabels(['Acquisition'],ha='center')

ax[2].set_ylabel('')
ax[2].set_xlabel('')
ax[2].set_xticklabels(['Extinction'],ha='center')

    # ax.hlines(0,ax.get_xlim()[0],ax.get_xlim()[1],color='black',linestyles='--')
    # ax.set_title(group)



#source memory
df = pd.read_csv('../cleaned_avg_sm.csv')

for phase in phases:
    dat = df[df.encode_phase == phase].copy()

    fig, ax = plt.subplots(1,2,sharey=True,figsize=(10,5))

    for i, group in enumerate(groups):
        d = dat[df.group == group].copy()
        sns.barplot(data=d,x='response_phase',y='prop',hue='condition',
                        palette=cpal,ax=ax[i])
        ax[i].set_ylabel('Proportion')
        ax[i].set_xlabel('Source Memory Response Phase')
        ax[i].set_title(group)
    plt.suptitle('Encoding Phase =\n%s'%(phase))

#source memory but just with remembered items
for phase in phases:
    dat = df[df.encode_phase == phase].copy()

    fig, ax = plt.subplots(1,2,sharey=True,figsize=(10,5))

    for i, group in enumerate(groups):
        d = dat[df.group == group].copy()
        sns.barplot(data=d,x='response_phase',y='mem_prop',hue='condition',
                        palette=cpal,ax=ax[i])
        ax[i].set_ylabel('Proportion of remembered items')
        ax[i].set_xlabel('Source Memory Response Phase')
        ax[i].set_title(group)
    plt.suptitle('Encoding Phase =\n%s'%(phase))


'''typicality'''
df = pd.read_csv('../cleaned_avg_ty.csv')
with sns.axes_style('whitegrid'):
    fig, ax = plt.subplots(1,2,figsize=(10,5),sharey=True,sharex=True)
    for i, group in enumerate(groups):
        dat = df[df.group == group].copy()
        sns.pointplot(data=dat,y='encode_phase',x='typicality',hue='condition',
                        ax=ax[i],palette=cpal,join=False,dodge=True)
        sns.stripplot(data=dat,y='encode_phase',x='typicality',hue='condition',
                        ax=ax[i],palette=cpal,dodge=True)
        ax[i].legend_.remove()
        ax[i].set_xlabel('Typicality')
        ax[i].set_title(group)

    ax[0].set_ylabel('Encoding Phase')
    ax[1].set_ylabel('')
    plt.suptitle('Typicality')

'''TYPICALITY COLLAPSED'''
df = pd.read_csv('../cleaned_avg_ty.csv'
                ).drop(columns='group'
                ).groupby(['subject','condition']
                ).mean(
                ).reset_index()
df['dummy'] = ''
with sns.axes_style('whitegrid'):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.pointplot(data=df,y='dummy',x='typicality',hue='condition',
                ax=ax,palette=cpal,join=False,dodge=True)
    sns.stripplot(data=df,y='dummy',x='typicality',hue='condition',
                    ax=ax,palette=cpal,dodge=True)
    ax.legend_.remove()
    ax.set_xlabel('Typicality')
    ax.set_ylabel('')

#recog and source correlations
df = pd.read_csv('../memory_difference_scores.csv')

for phase in ['baseline','extinction']:
    dat = df[df.encode_phase==phase][df.response_phase=='acquisition'].copy()
    sns.lmplot(data=dat,x='prop',y='cr',col='group',hue='group',palette=gpal)
    sns.lmplot(data=dat,x='mem_prop',y='cr',col='group',hue='group',palette=gpal)



'''LOGISTIC PREDICTIONS'''
with open('../logreg_betas.p','rb') as file:
    betas = pickle.load(file)

for phase in phases:
    fig, ax = plt.subplots(1,2,figsize=(10,5),sharey=True)
    for i, group in enumerate(groups):
        df = pd.DataFrame(np.concatenate((betas[group]['CS+'][phase],betas[group]['CS-'][phase])))
        df['condition'] = np.repeat(['CS+','CS-'],10000)
        df = df.rename(columns={0:'baseline',1:'acquisition',2:'extinction'})
        df = df.melt(id_vars=['condition'],var_name='response_phase',value_name='beta')
        
        sns.violinplot(data=df,x='response_phase',y='beta',hue='condition',
                        inner=None,palette=cpal,split=True,ax=ax[i],scale='count')
        ax[i].hlines(0,ax[i].get_xlim()[0],ax[i].get_xlim()[1],linestyle=':')

        ax[i].set_xlabel('Source Memory Response Phase')
        ax[i].set_title(group)

        pvals = df.groupby(['condition','response_phase']).apply(lambda x: 1 - np.mean(x > 0))
        pvals['p'] = pvals.beta.apply(lambda x: x if x < .5 else 1-x)
        pvals['tail'] = pvals.beta.apply(lambda x: 'greater' if x < .5 else 'less')
        pvals.p = pvals.p * 2
        pvals = pvals.reset_index()
        print(group,phase,'\n',pvals[pvals.p < .1],'\n\n')


    ax[0].set_ylabel('Logistic Regression\nBeta')
    # ax[1].set_ylabel('')
    plt.suptitle('Encoding Phase =\n%s'%(phase))



'''TYPICALITY LOGISTICS'''


'''TYPICALITY LOGISTIC CURVES'''
with open('../typicality_logreg_curves.p','rb') as file:
    curves = pickle.load(file)

fig, ax = plt.subplots(1,2,figsize=(10,5),sharey=True)
x = np.linspace(1,7,100)
for i, con in enumerate(cons):

    for c in range(10000):
        ax[i].plot(x,curves[con][c],alpha=.1,linewidth=.1,color=cpal[i])

    ax[i].plot(x,curves[con].mean(axis=0),linewidth=5,color='white')
    ax[i].set_title(con)
    ax[i].set_xlabel('Typicality')

ax[0].set_ylabel('High confidence "hit"')