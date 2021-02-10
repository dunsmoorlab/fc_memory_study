from fm_config import *
from robust_corr import skipped_corr
from paper_graphics_style import *

'''Fig 1. 24-hour corrected recognition'''
df = pd.read_csv('../cleaned_corrected_recognition.csv')

fig, ax = plt.subplots(figsize=mm2inch(100,100*.625))
sns.barplot(data=df,x='encode_phase',y='cr',hue='condition',
            palette=cpal,saturation=1)
ax.legend_.remove()
ax.set_ylabel('Corrected recognition')
ax.set_xlabel('')
ax.set_xticklabels(['Pre','Conditioning','Post'],ha='center')
ax.set_title('')
legend_elements = [Patch(facecolor=cpal[0],edgecolor=None,label='CS+'),
                   Patch(facecolor=cpal[1],edgecolor=None,label='CS-')]
ax.legend(handles=legend_elements,loc='upper right',bbox_to_anchor=(1,1),frameon=False)

upper = [line.get_ydata().max() for line in ax.lines]
paired_barplot_annotate_brackets('**',0,(upper[0],upper[3]),barh=.025)
paired_barplot_annotate_brackets('***',1,(upper[1],upper[4]),barh=.025)
paired_barplot_annotate_brackets('**',2,(upper[2],upper[5]),barh=.025)
ax.yaxis.set_major_locator(MultipleLocator(.2))

'''Fig 2. 24-hour source memory'''
df = pd.read_csv('../cleaned_avg_sm.csv')

fig, ax = plt.subplots(1,3,figsize=mm2inch(200,200*.4),sharey=False)
for i, phase in enumerate(phases):
    dat = df[df.encode_phase == phase].copy()
    sns.barplot(data=dat,x='condition',y='prop',hue='response_phase',ax=ax[i],
        palette=spal,saturation=1,zorder=2)
    ax[i].hlines(1/3,ax[i].get_xlim()[0],ax[i].get_xlim()[1],linestyle=':',color='black',zorder=1)
    if i != 0:
        ax[i].set_ylabel('')
        sns.despine(ax=ax[i],left=True,right=True,top=True)
    else:ax[i].set_ylabel('Proportion of items')
    ax[i].legend_.remove()
ax[0].set_xlabel('Pre')
ax[1].set_xlabel('Conditioning')
ax[2].set_xlabel('Post')
topy = np.max([ax[0].get_ylim()[1],ax[1].get_ylim()[1],ax[2].get_ylim()[1]])
boty = np.min([ax[0].get_ylim()[0],ax[1].get_ylim()[0],ax[2].get_ylim()[0]])
for i,Ax in enumerate(ax):
    Ax.set_ylim((boty,topy))
    Ax.set_yticks([0,.2,.4,.6,.8])
    if i != 0:
        Ax.set_yticks([])
#significant single bars above 0
label_bars(['','','***','','',''],ax[0],dh=.01)
label_bars(['','','***','','',''],ax[1],dh=.01)
label_bars(['','','***','','','*'],ax[2],dh=.01)

#acquisition CS+ vs CS-
simple_barplot_annotate_brackets('***',ax[0],dh=.1)
simple_barplot_annotate_brackets('***',ax[1],dh=.1)
simple_barplot_annotate_brackets('**',ax[2],dh=.1)

#set hatches for "correct" responses
ax[0].patches[0].set_hatch('/')
ax[0].patches[1].set_hatch('/')
ax[1].patches[2].set_hatch('/')
ax[1].patches[3].set_hatch('/')
ax[2].patches[4].set_hatch('/')
ax[2].patches[5].set_hatch('/')


legend_elements = [Patch(facecolor=spal[0],edgecolor=None,label='Pre'),
                   Patch(facecolor=spal[1],edgecolor=None,label='Conditioning'),
                   Patch(facecolor=spal[2],edgecolor=None,label='Post'),
                   Patch(facecolor='white',edgecolor='black',hatch='//',label='Correct responses')]
fig.legend(handles=legend_elements,ncol=4,frameon=False,title='Temporal context memory response',loc='upper center',bbox_to_anchor=(.5,1))
plt.tight_layout(rect=(0,0,1,.9))
ax[0].yaxis.set_major_locator(MultipleLocator(.2))

'''Fig 3. SM by CR correlations'''
df = pd.read_csv('../memory_difference_scores.csv'
        # ).set_index(['group','encode_phase','response_phase','subject'])
        ).set_index(['encode_phase','response_phase','subject'])

df = df.reset_index().query('response_phase == "acquisition"').set_index(['encode_phase','subject']).sort_index()

fig, ax = plt.subplots(1,3,sharey=True,sharex=True,figsize=mm2inch(200,200*.4))
for p, phase in enumerate(['baseline','acquisition','extinction']):
    skipped_corr(x=df.loc[phase,'prop'], y=df.loc[phase,'cr'], ax=ax[p], color=wes_palettes['Royal1'][0])
    # sns.regplot(data=df[df.encode_phase == phase], x='prop', y='cr',ax=ax[p],color=spal[1])
    ax[p].set_xlabel('')
for Ax in ax: Ax.set_xlabel('Temporal context memory\n"Conditioning" responses (CS+ - CS-)')
ax[0].set_title('Pre');ax[1].set_title('Conditioning');ax[2].set_title('Post')
ax[0].set_ylabel('Corrected recognition (CS+ - CS-)');ax[1].set_ylabel('');ax[2].set_ylabel('')
ax[0].set_xlim(-.35,.85)
plt.tight_layout()

rcParams['mathtext.default'] = 'regular'
ax[0].text(-0.25,.5,'$r_{skipped}$ = 0.34*',fontsize=rcParams['font.size'],va='bottom')
ax[1].text(-0.25,.5,'$r_{skipped}$ = 0.36*',fontsize=rcParams['font.size'],va='bottom')
ax[2].text(-0.25,.5,'$r_{skipped}$ = 0.37*',fontsize=rcParams['font.size'],va='bottom')

'''Fig. 4 source memory logistic predictions'''
with open('../sourcemem_logreg_betas.p','rb') as file:
    betas = pickle.load(file)

fig, ax = plt.subplots(1,3,figsize=mm2inch(200,200*.4),sharey=False)
for i, phase in enumerate(phases):
    df = pd.DataFrame(np.concatenate((betas['CS+'][phase],betas['CS-'][phase])))
    df['condition'] = np.repeat(['CS+','CS-'],10000)
    df = df.rename(columns={0:'baseline',1:'acquisition',2:'extinction'})
    df = df.melt(id_vars=['condition'],var_name='response_phase',value_name='beta')
    sns.violinplot(data=df,x='condition',y='beta',hue='response_phase',cut=0,saturation=1,
                    inner='mean_ci',palette=spal,ax=ax[i],scale='count',width=.6,zorder=1,edgecolor='white')

    sns.violinplot(data=df,x='condition',y='beta',hue='response_phase',cut=0,saturation=1,
                    inner='mean_ci',palette=spal,ax=ax[i],scale='count',width=.6,zorder=10,edgecolor='white')
    for v, viol in enumerate(ax[i].collections[6:]):
        viol.set_edgecolor((spal+spal)[v])
    ax[i].hlines(0,ax[i].get_xlim()[0],ax[i].get_xlim()[1],linestyle=':',color='black',zorder=0)


    if i != 0:
        ax[i].set_ylabel('')
        sns.despine(ax=ax[i],left=True,right=True,top=True)
        plt.setp(ax[i].get_yticklabels(), visible=False)
    else:ax[i].set_ylabel('Predictiveness of source memory\nresponse on recognition memory (β)')
    ax[i].legend_.remove()

    pvals = df.groupby(['condition','response_phase']).apply(lambda x: 1 - np.mean(x > 0))
    pvals['p'] = pvals.beta.apply(lambda x: x if x < .5 else 1-x)
    pvals['tail'] = pvals.beta.apply(lambda x: '>' if x < .5 else '<')
    pvals.p = pvals.p * 2
    pvals.beta = df.groupby(['condition','response_phase']).mean()
    pvals['ci'] = df.groupby(['condition','response_phase']).apply(lambda x: np.round([np.percentile(x,2.5),np.percentile(x,97.5)],4))
    pvals['topval'] = df.groupby(['condition','response_phase']).max()
    pvals['topval'] -= 0.045
    pvals['botval'] = df.groupby(['condition','response_phase']).min()
    pvals.botval -= .045
    # pvals = pvals.reset_index()
    pvals['encode_phase'] = phase
    pvals['star'] = pvals.p.apply(p_convert)
    pvals.to_csv(f'../paper/{phase}_sm_betas.csv')

    df = df.set_index(['condition','response_phase'])
    rpos = [-.2,0,.2]
    for xtick, con in enumerate(['CS+','CS-']):
        for r, response in enumerate(phases):
            if pvals.loc[(con,response),'beta'] > 0:
                topbot = 'topval'
                va = 'bottom'
            else:
                topbot = 'botval'
                va = 'top'
            ax[i].text(xtick+rpos[r],pvals.loc[(con,response),topbot],pvals.loc[(con,response),'star'],ha='center',va=va)
    #     #get the mean and 95%CI out here, and just copy paste into excel I think
    #     acq_bsl = p_convert(1 - (df.loc[(con,'acquisition'),'beta'].values > df.loc[(con,'baseline'),'beta'].values).mean())
    #     acq_ext = p_convert(1 - (df.loc[(con,'acquisition'),'beta'].values > df.loc[(con,'extinction'),'beta'].values).mean())
    #     print(con, phase, acq_bsl, acq_ext)
    #     y = pvals.loc[(con,'acquisition'),'topval']+.2
    #     if acq_bsl != '':
    #         barx = [xtick-.2,xtick-.2,xtick-.01,xtick-.01]
    #         bary = [y,y+.05,y+.05,y]
    #         ax[i].plot(barx, bary, c='black')
    #         mid = ((xtick-.2+xtick-.01)/2, y)
    #         ax[i].text(*mid,acq_bsl,ha='center',va='bottom')
    #     if acq_ext != '':
    #         barx = [xtick+.2,xtick+.2,xtick+.01,xtick+.01]
    #         bary = [y,y+.05,y+.05,y]
    #         ax[i].plot(barx, bary, c='black')
    #         mid = ((xtick+.2+xtick+.01)/2, y)
    #         ax[i].text(*mid,acq_ext,ha='center',va='bottom')
    # barx = [lx, lx, rx, rx]
    # bary = [y, y+barh, y+barh, y]
    # mid = ((lx+rx)/2, y+barh)
    '''post-hoc contrasts'''
    print(phase)
    # print(pvals.reset_index()[['condition','encode_phase','response_phase','beta','ci','p','tail']][pvals.p < .1],'\n\n')
ax[0].collections[0].set_hatch('//')
ax[0].collections[0].set_linewidth(.5)
ax[0].collections[6].set_facecolor('None')

ax[0].collections[3].set_hatch('//')
ax[0].collections[3].set_linewidth(.5)
ax[0].collections[9].set_facecolor('None')

ax[1].collections[1].set_hatch('//')
ax[1].collections[1].set_linewidth(.5)
ax[1].collections[7].set_facecolor('None')

ax[1].collections[4].set_hatch('//')
ax[1].collections[4].set_linewidth(.5)
ax[1].collections[10].set_facecolor('None')

ax[2].collections[2].set_hatch('//')
ax[2].collections[2].set_linewidth(.5)
ax[2].collections[8].set_facecolor('None')

ax[2].collections[5].set_hatch('//')
ax[2].collections[5].set_linewidth(.5)
ax[2].collections[11].set_facecolor('None')

topy = np.max([ax[0].get_ylim()[1],ax[1].get_ylim()[1],ax[2].get_ylim()[1]])
boty = np.min([ax[0].get_ylim()[0],ax[1].get_ylim()[0],ax[2].get_ylim()[0]])
for i,Ax in enumerate(ax):
    Ax.set_ylim((boty,topy))
    if i != 0:
        Ax.set_yticks([])
ax[0].set_xlabel('Pre')
ax[1].set_xlabel('Conditioning')
ax[2].set_xlabel('Post')
legend_elements = [Patch(facecolor=spal[0],edgecolor=None,label='Pre'),
                   Patch(facecolor=spal[1],edgecolor=None,label='Conditioning'),
                   Patch(facecolor=spal[2],edgecolor=None,label='Post'),
                   Patch(facecolor='white',edgecolor='black',hatch='//',label='Correct responses')]
fig.legend(handles=legend_elements,ncol=4,frameon=False,title='Temporal context memory response',loc='upper center',bbox_to_anchor=(.5,1))
plt.tight_layout(rect=(0,0,1,.9))


'''CS+ vs CS- Typicality'''
df = pd.read_csv('../cleaned_avg_ty.csv'
                ).drop(columns='group'
                ).groupby(['subject','condition']
                ).mean(
                ).reset_index()
df['dummy'] = ''
with sns.axes_style('whitegrid'):
    fig, ax1 = plt.subplots(figsize=mm2inch(100,100*1.4))
    # sns.stripplot(data=df,x='dummy',y='typicality',hue='condition',
    #                 ax=ax,palette=cpal,dodge=True,alpha=.8)
    sns.pointplot(data=df,x='dummy',y='typicality',hue='condition',
                ax=ax,palette=cpal,join=False,dodge=.3,saturation=1)
    ax.legend_.remove()
    ax.set_ylabel('Typicality')
    ax.set_xlabel('')
    for sub in df.subject.unique():
        csp = df.typicality[df.condition == 'CS+'][df.subject == sub].values[0]
        csm = df.typicality[df.condition == 'CS-'][df.subject == sub].values[0]
        
        ax.plot([-.1,.1],[csp,csm],c='grey',alpha=.5,zorder=1)
        ax.scatter(-.1,csp,color=cpal[0],alpha=1,edgecolor=None,zorder=2)
        ax.scatter(.1,csm,color=cpal[1],alpha=1,edgecolor=None,zorder=2)


    ax.plot([-.15,-.15,.15,.15],[6.25,6.35,6.35,6.25],c='black')
    ax.text(0,6.35,'*')
    ax.set_ylim(ax.get_ylim()[0],6.45)

    legend_elements = [Patch(facecolor=cpal[0],edgecolor=None,label='CS+'),
                       Patch(facecolor=cpal[1],edgecolor=None,label='CS-')]
    ax.legend(handles=legend_elements,ncol=2,frameon=False,title='CS type',loc='lower center',bbox_to_anchor=(.5,-.04))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(.5))
    # sns.despine(ax=ax,left=True,bottom=True)
    ax.grid(True,'both','y')
    plt.tight_layout()

'''Typicality CS+ vs CS- by phase'''
df = pd.read_csv('../cleaned_avg_ty.csv')
fig = plt.figure(figsize=mm2inch(100,200))
gs = fig.add_gridspec(3,1)
with sns.axes_style('whitegrid'):
    ax1 = fig.add_subplot(gs[0,0])
    sns.pointplot(data=df,x='encode_phase',y='typicality',hue='condition',
                palette=cpal,saturation=1,dodge=.4,join=False,ax=ax1)
    ax1.legend_.remove()
    ax1.set_ylabel('Typicality')
    ax1.set_xlabel('Encoding temporal context')
    ax1.set_xticklabels(['Pre','Conditioning','Post'],ha='center')
    ax1.set_title('')
    ax1.set_ylim(1,7)
    legend_elements = [Patch(facecolor=cpal[0],edgecolor=None,label='CS+'),
                       Patch(facecolor=cpal[1],edgecolor=None,label='CS-')]
    ax1.legend(handles=legend_elements,frameon=False,loc='lower right',bbox_to_anchor=(1,-0.025))

    upper = [line.get_ydata().max() for line in ax1.lines]
    # paired_barplot_annotate_brackets('*',0,(upper[0],upper[3]),ax.get_ylim(),barh=.025)
    # paired_barplot_annotate_brackets('*',1,(upper[1],upper[4]),ax.get_ylim(),barh=.025)
    # paired_barplot_annotate_brackets('*',2,(upper[2],upper[5]),ax.get_ylim(),barh=.025)
    ax1.yaxis.set_major_locator(FixedLocator([1,3,5,7]))
    ax1.yaxis.set_minor_locator(FixedLocator([2,4,6]))
    # sns.despine(ax=ax1,left=True,bottom=False,top=False)
    ax1.grid(True,'both','y')

for c in ax1.collections: c.set_linewidth(rcParams['lines.linewidth']*.3)
df = df.set_index(['encode_phase','condition','subject']).sort_index()
for p, phase in enumerate(phases):

    for sub in xcl_sub_args:
        csp = df.loc[(phase,'CS+',sub),'typicality']
        csm = df.loc[(phase,'CS-',sub),'typicality']
        
        x1, x2 = p-.1, p+.1
        ax1.plot([x1,x2],[csp,csm],c='black',alpha=.5,zorder=10,linewidth=rcParams['lines.linewidth']*.5)

paired_barplot_annotate_brackets('*',0,(6,6),ax1.get_ylim(),barh=.025,ax=ax1)
paired_barplot_annotate_brackets('*',1,(6.2,6.2),ax1.get_ylim(),barh=.025,ax=ax1)
paired_barplot_annotate_brackets('*',2,(5.8,5.8),ax1.get_ylim(),barh=.025,ax=ax1)

'''Typicaltiy logreg violins'''
with open('../typicality_logreg_betas.p','rb') as file:
    betas = pickle.load(file)

df = {}
for phase in phases:
    df[phase] = pd.DataFrame(np.concatenate((betas[phase]['CS+'],betas[phase]['CS-'])))
    df[phase]['condition'] = np.repeat(['CS+','CS-'],10000)
    df[phase] = df[phase].rename(columns={0:'beta'})
    df[phase]['phase'] = phase
df = pd.concat(df.values())

# fig, ax = plt.subplots(1,figsize=mm2inch(100,100*.6),sharey=False)
ax2 = fig.add_subplot(gs[1,0])
sns.violinplot(data=df,x='phase',y='beta',hue='condition',cut=0,saturation=1,
                inner='mean_ci',palette=cpal,ax=ax2,scale='count',width=.6,zorder=1,edgecolor='white')

sns.violinplot(data=df,x='phase',y='beta',hue='condition',cut=0,saturation=1,
                inner='mean_ci',palette=cpal,ax=ax2,scale='count',width=.6,zorder=10,edgecolor='white')

for v, viol in enumerate(ax2.collections[6:]):
    viol.set_edgecolor((cpal+cpal+cpal)[v])
ax2.hlines(0,ax2.get_xlim()[0],ax2.get_xlim()[1],linestyle=':',color='black',zorder=0)

ax2.set_ylabel('Predictiveness of typicality\non recognition memory (β)')
ax2.legend_.remove()
pvals = df.groupby(['phase','condition']).apply(lambda x: (1 - np.mean(x > 0))*2)
pvals['p'] = pvals.beta.apply(lambda x: x if x < .5 else 1-x)
pvals['tail'] = pvals.beta.apply(lambda x: '>' if x < .5 else '<')
pvals.beta = df.groupby(['phase','condition']).mean()
pvals['ci'] = df.groupby(['phase','condition']).apply(lambda x: np.round([np.percentile(x,2.5),np.percentile(x,97.5)],4))
pvals['topval'] = df.groupby(['phase','condition']).max()
# pvals = pvals.reset_index()
pvals['star'] = pvals.p.apply(p_convert)
pvals.to_csv(f'../paper/typicality_betas.csv')
pvals.loc[('baseline','CS+'),'star'] = '~'

df = df.set_index(['phase','condition'])
cpos = [-.15,.15]
for xtick, phase in enumerate(phases):
    for c, con in enumerate(['CS+','CS-']):
        ax2.text(xtick+cpos[c],pvals.loc[(phase,con),'topval'],pvals.loc[(phase,con),'star'],ha='center',va='bottom')
ax2.set_xticklabels(['Pre','Conditioning','Post'])
ax2.yaxis.set_major_locator(MultipleLocator(.2))
ax2.set_xlabel('Encoding temporal context')
#CS+ vs. CS-
for phase in phases:
    diff = betas[phase]['CS+'] - betas[phase]['CS-'] 
    diffp = [(1 - np.mean(diff > 0)) * 2, (1 - np.mean(diff < 0)) * 2] 
    diffp = np.min(diffp)
    print(phase,diffp)

#typicality by source memory
df = pd.read_csv('../cleaned_full_sm.csv'
        ).rename(columns={'trial_type':'condition'}
        ).groupby(['source_memory','subject']
        ).mean(
        ).reset_index()

with sns.axes_style('whitegrid'):
    ax3 = fig.add_subplot(gs[2,0])
    sns.pointplot(data=df,x='source_memory',y='typicality',ax=ax3,palette=spal,order=phases,saturation=1,zorder=2,join=False)
    ax3.set_xticklabels(['Pre','Conditioning','Post'])
    ax3.set_ylabel('Typicality')
    ax3.set_xlabel('Temporal context source memory response')
    ax3.yaxis.set_major_locator(MultipleLocator(.4))
    ax3.yaxis.set_minor_locator(MultipleLocator(.2))
    ax3.grid(True,'both','y')
    ax3.set_ylim((3.8,4.9))

fig.text(0.05,.99,'A',fontsize=16)
fig.text(0.05,.66,'B',fontsize=16)
fig.text(0.05,.33,'C',fontsize=16)
fig.tight_layout()



'''Typicality within subject beta curves'''
df = pd.read_csv('../typicality_within_sub_logreg_betas.csv').set_index(['condition','subject'])
x = np.linspace(1,7,100)
fig, ax = plt.subplots(1,2,figsize=mm2inch(200,200*.5),sharey=True)
for i, con in enumerate(cons):
    beta = onesample_bdm(df.loc[con,'beta'],0)
    intercept = onesample_bdm(df.loc[con,'intercept'],0)
    avg = expit(x * beta.loc[0,'avg'] + intercept.loc[0,'avg'])
    lower = expit(x * beta.loc[0,'CI_l'] + intercept.loc[0,'CI_l'])
    upper = expit(x * beta.loc[0,'CI_u'] + intercept.loc[0,'CI_u'])

    ax[i].plot(x,avg,color='black',linewidth=2,zorder=2)
    ax[i].plot(x,lower,color='black',linestyle='--',zorder=2)
    ax[i].plot(x,upper,color='black',linestyle='--',zorder=2)

    # ax[i].fill_between(x,lower,upper,alpha=.1,color=cpal[i])
    '''I dont think we want the sub curves'''
    for sub in xcl_sub_args:
        curve = expit(x * df.loc[(con,sub),'beta'] + df.loc[(con,sub),'intercept'])
        ax[i].plot(x,curve,alpha=.25,color=cpal[i],zorder=1)
    ax[i].set_title(con)
    ax[i].set_xlabel('Typicality')

ax[0].set_ylabel('Recognition memory\n(High confidence "hit")')
ax[0].text(6.6,.58,'***',ha='center',va='bottom')
ax[1].text(6.6,.46,'***',ha='center',va='bottom')
plt.tight_layout()


'''Final figure, multiple logistic regression'''
# df = pd.read_csv('../multiple_logreg.csv')
# fig, (ax1, ax2) = plt.subplots(1,2,figsize=mm2inch(200,200*.6),sharey=True)
# x_order = ['encode_phase','condition','source_memory','typicality']
# sns.swarmplot(data=df,x='feature',y='beta',color=wes_palettes['Royal1'][0],ax=ax1,zorder=1,alpha=.75,order=x_order)
# sns.pointplot(data=df,x='feature',y='beta',color='black',ax=ax1,zorder=10000,order=x_order)
# ax1.collections[4].set_linewidth(rcParams['lines.linewidth']*.3)
# ax1.hlines(0,ax1.get_xlim()[0],ax1.get_xlim()[1],linestyle=':',color='black',zorder=0)
# ax1.set_xticklabels(['Temporal\ncontext','CS\ntype','Source\nmemory','Typicality'],ha='center')
# ax1.set_ylabel('Logistic regression beta')
# ax1.set_xlabel('')
# ax1.set_title('All phases')

pdf = pd.read_csv('../multiple_logreg_phase.csv')
fig, ax2 = plt.subplots(1,figsize=mm2inch(100,100*.8),sharey=True)
sns.swarmplot(data=pdf.query('phase == "baseline"'),x='feature',y='beta',color=wes_palettes['Royal1'][0],ax=ax2,zorder=1,alpha=.75,order=['acq_resp','condition','typicality','ext_resp'])
sns.pointplot(data=pdf.query('phase == "baseline"'),x='feature',y='beta',color='black',ax=ax2,zorder=10000,order=['acq_resp','condition','typicality','ext_resp'])
ax2.collections[4].set_linewidth(rcParams['lines.linewidth']*.3)
ax2.hlines(0,*ax2.get_xlim(),linestyle=':',color='black',zorder=0)
ax2.set_xticklabels(['Conditioning\nmisattributions','CS type','Typicality','Post-\nconditioning\nmisattributions'],ha='center')
ax2.set_ylabel('Predictiveness of recognition memory')
ax2.set_xlabel('')
ax2.set_title('Pre-conditioning')

# y = pdf.beta[pdf.phase == 'baseline'].max() + .2
# barx = [0,0,1-.01,1-.01]
# bary = [y,y+.1,y+.1,y]
# ax2.plot(barx, bary, c='black')
# mid = (.5, y+.01)
# ax2.text(*mid,'*',ha='center',va='bottom')

# barx = [2,2,1+.01,1+.01]
# bary = [y,y+.1,y+.1,y]
# ax2.plot(barx, bary, c='black')
# mid = (1.5, y+.01)
# ax2.text(*mid,'**',ha='center',va='bottom')

plt.tight_layout()

'''Stimulus typicality difference for supplement'''
df = pd.read_csv('../cleaned_full_sm.csv'
        ).rename(columns={'trial_type':'condition'})

stim = df.groupby(['condition','stimulus']).mean()
stim = (stim.loc['CS+'] - stim.loc['CS-']).reset_index().sort_values(by='typicality',ascending=False)
stim['category'] = stim.stimulus.apply(lambda x: 'animals' if 'animals' in x else 'tools')

stim_order = stim.groupby(['stimulus']).mean().reset_index().sort_values(by='typicality',ascending=False).stimulus.values
stim_colors = [tpal[0] if stim[0] == 'a' else tpal[1] for stim in stim_order]

stim.typicality = (stim.typicality / 7) * 100
with sns.axes_style('whitegrid'):
    fig, ax = plt.subplots(figsize=mm2inch(200,200*.6))
    # sns.pointplot(data=stim,x='stimulus',y='typicality',hue='category',
    #                 order=stim_order,ax=ax, palette=tpal, hue_order=['animals','tools'],join=False)
    sns.barplot(data=stim,x='stimulus',y='typicality',order=stim_order,ax=ax, palette=stim_colors,saturation=1)
    ax.set_xticklabels('')
    # ax.set_ylim(1,7)
    ax.set_ylabel('Typicality (CS+ - CS-)')
    ax.set_xlabel('Stimulus')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_title('Mean difference in typicality for each stimulus')
legend_elements = [Patch(facecolor=tpal[0],edgecolor=None,label='Animal'),
                   Patch(facecolor=tpal[1],edgecolor=None,label='Tool')]
ax.legend(handles=legend_elements,loc='upper right',title='Category',bbox_to_anchor=(1,1),frameon=False)
ax.set_ylim([-30,30])
plt.tight_layout()