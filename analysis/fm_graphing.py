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
    sns.pointplot(data=dat,x='encode_phase',y='cr',hue='condition',hue_order=['CS-','CS+'],
            palette=[cpal[1],cpal[0]],ax=ax[i],dodge=True)


    for sub in dat.subject.unique():
        csm = dat.cr[dat.condition == 'CS-'][dat.subject == sub].values[0]
        csp = dat.cr[dat.condition == 'CS+'][dat.subject == sub].values[0]
        ax[i].plot([-.05,.05],[csm,csp],c='grey',linewidth=.5,alpha=.8)
    


    ax[i].legend_.remove()    

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

#boring barplot version
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(data=df,x='encode_phase',y='cr',hue='condition',hue_order=['CS-','CS+'],
            palette=[cpal[1],cpal[0]])
ax.set_ylabel('Corrected recognition')
ax.set_xlabel('Encoding Phase')
ax.set_xticklabels(['Baseline','Acquisition','Extinction'],ha='center')



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

#source memory collapsed across groups
df = pd.read_csv('../cleaned_avg_sm.csv')
fig, ax = plt.subplots(1,3,figsize=(12,5),sharey=True)
for i, phase in enumerate(phases):
    dat = df[df.encode_phase == phase].copy()
    sns.barplot(data=dat,x='condition',y='prop',hue='response_phase',ax=ax[i],
        palette=spal)
    ax[i].hlines(1/3,ax[i].get_xlim()[0],ax[i].get_xlim()[1],linestyle=':')
    if i != 0:ax[i].set_ylabel('')
    else:ax[i].set_ylabel('Proportion of items')
    ax[i].legend_.remove()
legend_elements = [Patch(facecolor=spal[0],edgecolor=None,label='Baseline'),
                   Patch(facecolor=spal[1],edgecolor=None,label='Acquisition'),
                   Patch(facecolor=spal[2],edgecolor=None,label='Extinction')]
ax[0].legend(handles=legend_elements,loc='upper right')
ax[0].legend_.set_title('Source memory\n      response')
ax[0].set_xlabel('Baseline')
ax[1].set_xlabel('Acquisition')
ax[2].set_xlabel('Extinction')


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

'''TYPICALITY COLLAPSED BY STIM'''
df = pd.read_csv('../cleaned_full_sm.csv'
        ).rename(columns={'trial_type':'condition'})
df['category'] = df.stimulus.apply(lambda x: 'animals' if 'animals' in x else 'tools')

stim = df.groupby(['stimulus']).mean().reset_index().sort_values(by='hc_acc',ascending=False)
stim_order = stim.stimulus.values

#all stims no breakdown
sns.set_context('notebook',font_scale=1.4)
with sns.axes_style('whitegrid'):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.pointplot(data=df,x='stimulus',y='typicality',hue='category',
                    order=stim_order,ax=ax, palette=wes_palettes['Chevalier'], hue_order=['animals','tools'])
    ax.set_xticklabels('')
    ax.set_ylim(1,7)
    ax.set_ylabel('Typicality')#,fontsize=20)
    ax.set_xlabel('Stimulus')#, fontsize=20)
    # plt.setp(ax.collections, sizes=[1])

    #all stims hc hit rate
    fig, ax = plt.subplots(figsize=(8,5))
    sns.pointplot(data=df,x='stimulus',y='hc_acc',hue='category',
        order=stim_order,ax=ax, palette=wes_palettes['Chevalier'], hue_order=['animals','tools'])
    ax.set_xticklabels('')
    ax.set_ylim(0,1)
    ax.set_ylabel('High confidence\nhit rate')
    ax.set_xlabel('Stimulus')

    #now CS+/-
    fig, ax = plt.subplots(2,sharey=True,sharex=True,figsize=(8,10))
    for c, con in enumerate(cons):
        sns.pointplot(data=df.query("condition == @con"),x='stimulus',y='typicality',hue='category',
                        order=stim_order,ax=ax[c], palette=wes_palettes['Chevalier'], hue_order=['animals','tools'])
        ax[c].set_xticklabels('')
        ax[c].set_ylim(1,7)
        ax[c].set_ylabel('Typicality')
        ax[c].set_title(con)
    ax[0].set_xlabel('')
    ax[1].set_xlabel('Stimulus')
    ax[0].legend_.remove()

q = sns.catplot(data=df,x='stimulus',y='typicality',row='condition',col='category',kind='point')

'''RECOG AND SOURCE MEM CORRELATIONS'''
df = pd.read_csv('../memory_difference_scores.csv'
        # ).set_index(['group','encode_phase','response_phase','subject'])
        ).set_index(['encode_phase','response_phase','subject'])

q = sns.lmplot(data=df.reset_index().query('response_phase == "acquisition"'),
            x='mem_count',y='cr',col='encode_phase',palette=spal[0])
q.set_axis_labels('"Acquisition" response difference\n(only using remembered items)','Corrected recognition difference')



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

#same as above but without group
with open('../sourcemem_logreg_betas.p','rb') as file:
    betas = pickle.load(file)

fig, ax = plt.subplots(1,3,figsize=(12,5),sharey=True)
for i, phase in enumerate(phases):
    df = pd.DataFrame(np.concatenate((betas['CS+'][phase],betas['CS-'][phase])))
    df['condition'] = np.repeat(['CS+','CS-'],10000)
    df = df.rename(columns={0:'baseline',1:'acquisition',2:'extinction'})
    df = df.melt(id_vars=['condition'],var_name='response_phase',value_name='beta')
    
    sns.violinplot(data=df,x='condition',y='beta',hue='response_phase',
                    inner=None,palette=spal,ax=ax[i],scale='count',width=.6)

    ax[i].hlines(0,ax[i].get_xlim()[0],ax[i].get_xlim()[1],linestyle=':')

    if i != 0:ax[i].set_ylabel('')
    else:ax[i].set_ylabel('Logistic regression beta')
    ax[i].legend_.remove()

    pvals = df.groupby(['condition','response_phase']).apply(lambda x: 1 - np.mean(x > 0))
    pvals['p'] = pvals.beta.apply(lambda x: x if x < .5 else 1-x)
    pvals['tail'] = pvals.beta.apply(lambda x: '>' if x < .5 else '<')
    pvals.p = pvals.p * 2
    pvals.beta = df.groupby(['condition','response_phase']).mean()
    pvals['ci'] = df.groupby(['condition','response_phase']).apply(lambda x: np.round([np.percentile(x,2.5),np.percentile(x,97.5)],4))
    pvals = pvals.reset_index()
    pvals['encode_phase'] = phase
    print(phase)
    print(pvals[['condition','encode_phase','response_phase','beta','ci','p','tail']][pvals.p < .1],'\n\n')

legend_elements = [Patch(facecolor=spal[0],edgecolor=None,label='Baseline'),
                   Patch(facecolor=spal[1],edgecolor=None,label='Acquisition'),
                   Patch(facecolor=spal[2],edgecolor=None,label='Extinction')]
ax[2].legend(bbox_to_anchor=(1.3, 1),handles=legend_elements)
ax[2].legend_.set_title('Source memory\n      response')
ax[0].set_xlabel('Baseline')
ax[1].set_xlabel('Acquisition')
ax[2].set_xlabel('Extinction')



'''TYPICALITY LOGISTICS'''
with open('../typicality_logreg_betas.p','rb') as file:
    betas = pickle.load(file)
for con in cons:
    x = betas[con]
    print(x.mean())
    print(np.percentile(x,2.5),np.percentile(x,97.5))
    print((1 - np.mean(x > 0))*2)
x = betas['CS+'] - betas['CS-']
print(x.mean())
print(np.percentile(x,2.5),np.percentile(x,97.5))
print((1 - np.mean(x > 0))*2)




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

