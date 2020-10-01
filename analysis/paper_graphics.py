from fm_config import *
def p_convert(x):
    if x < .001: return '***'
    elif x < .01: return '**'
    elif x < .05: return '*'
    else: return ''

def mm2inch(*tupl):
    inch = 25.4
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
def paired_barplot_annotate_brackets(txt, x_tick, height, y_lim=plt.gca().get_ylim(), dh=.05, barh=.05, fs=10, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param txt: string to write or number for generating asterixes
    :param x_tick: center of pair of bars
    :param height: heights of the errors in question
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(txt) is str:
        text = txt
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while txt < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = x_tick-.2, height[0]
    rx, ry = x_tick+.2, height[1]

    ax_y0, ax_y1 = y_lim
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)
def simple_barplot_annotate_brackets(txt, ax, dh=.05, barh=.05, fs=10, maxasterix=None):
    
    lx, rx = ax.get_xticks()
    upper = {line.get_xdata()[0]:line.get_ydata().max() for line in ax.lines}
    ly, ry = upper[lx], upper[rx]

    ax_y0, ax_y1 = ax.get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, txt, **kwargs)

def label_bars(txt,ax,dh=0.05,fs=10):
    #use the yerros as the reference since these contain the actual x-values where we want the text
    #for categorical variables, the list order is hue by x-values, so the first clump of a group of 3 is [0,2,4]

    ax_y0, ax_y1 = ax.get_ylim()
    dh *= (ax_y1 - ax_y0)

    x_vals = [line.get_xdata()[0] for line in ax.lines] 
    y_vals = [line.get_ydata().max() + dh for line in ax.lines]

    assert len(txt) == len(x_vals)
    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs
    for i, t in enumerate(txt):
        if t != '':
            ax.text(x_vals[i],y_vals[i],t,**kwargs)



sns.set_context('paper')
rcParams['savefig.dpi'] = 900


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

'''Fig 2. 24-hour corrected recognition'''
df = pd.read_csv('../cleaned_avg_sm.csv')

fig, ax = plt.subplots(1,3,figsize=mm2inch(200,200*.4),sharey=True)
for i, phase in enumerate(phases):
    dat = df[df.encode_phase == phase].copy()
    sns.barplot(data=dat,x='condition',y='prop',hue='response_phase',ax=ax[i],
        palette=spal,saturation=1)
    ax[i].hlines(1/3,ax[i].get_xlim()[0],ax[i].get_xlim()[1],linestyle=':',color='black')
    if i != 0:ax[i].set_ylabel('')
    else:ax[i].set_ylabel('Proportion of items')
    ax[i].legend_.remove()
ax[0].set_xlabel('Pre')
ax[1].set_xlabel('Conditioning')
ax[2].set_xlabel('Post')

#significant single bars
label_bars(['*','','***','','*',''],ax[0],dh=.01)
label_bars(['***','','***','','***',''],ax[1],dh=.01)
label_bars(['***','','***','','','*'],ax[2],dh=.01)

#acquisition CS+ vs CS-
simple_barplot_annotate_brackets('***',ax[0],dh=.1)
simple_barplot_annotate_brackets('***',ax[1],dh=.1)
simple_barplot_annotate_brackets('***',ax[2],dh=.1)

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

df = df.reset_index().query('response_phase == "acquisition"')

fig, ax = plt.subplots(1,3,sharey=True,sharex=True,figsize=mm2inch(200,200*.4))
for p, phase in enumerate(['baseline','acquisition','extinction']):
    sns.regplot(data=df[df.encode_phase == phase], x='prop', y='cr',ax=ax[p],color=spal[p])
    ax[p].set_xlabel('')
ax[1].set_xlabel('Temporal context memory\n"Conditioning" response (CS+ - CS-)')
ax[0].set_title('Pre');ax[1].set_title('Conditioning');ax[2].set_title('Post')
ax[0].set_ylabel('Corrected recognition (CS+ - CS-)');ax[1].set_ylabel('');ax[2].set_ylabel('')
ax[0].set_xlim(-.35,.85)
plt.tight_layout()

ax[0].text(.6,.3,'*',fontsize=12)
ax[1].text(.6,.5,'**',fontsize=12)
ax[2].text(.65,.45,'*',fontsize=12)

'''Fig. 4 source memory logistic predictions'''
with open('../sourcemem_logreg_betas.p','rb') as file:
    betas = pickle.load(file)

fig, ax = plt.subplots(1,3,figsize=mm2inch(200,200*.4),sharey=True)
for i, phase in enumerate(phases):
    df = pd.DataFrame(np.concatenate((betas['CS+'][phase],betas['CS-'][phase])))
    df['condition'] = np.repeat(['CS+','CS-'],10000)
    df = df.rename(columns={0:'baseline',1:'acquisition',2:'extinction'})
    df = df.melt(id_vars=['condition'],var_name='response_phase',value_name='beta')
    
    sns.violinplot(data=df,x='condition',y='beta',hue='response_phase',cut=0,saturation=1,
                    inner='mean_ci',palette=spal,ax=ax[i],scale='count',width=.6)

    ax[i].hlines(0,ax[i].get_xlim()[0],ax[i].get_xlim()[1],linestyle=':',color='black')
    # for viol in ax[i].collections: viol.set_edgecolor('black')

    if i != 0:ax[i].set_ylabel('')
    else:ax[i].set_ylabel('Logistic regression beta')
    ax[i].legend_.remove()

    pvals = df.groupby(['condition','response_phase']).apply(lambda x: 1 - np.mean(x > 0))
    pvals['p'] = pvals.beta.apply(lambda x: x if x < .5 else 1-x)
    pvals['tail'] = pvals.beta.apply(lambda x: '>' if x < .5 else '<')
    pvals.p = pvals.p * 2
    pvals.beta = df.groupby(['condition','response_phase']).mean()
    pvals['ci'] = df.groupby(['condition','response_phase']).apply(lambda x: np.round([np.percentile(x,2.5),np.percentile(x,97.5)],4))
    pvals['topval'] = df.groupby(['condition','response_phase']).max()
    # pvals = pvals.reset_index()
    pvals['encode_phase'] = phase
    pvals['star'] = pvals.p.apply(p_convert)

    df = df.set_index(['condition','response_phase'])
    rpos = [-.2,0,.2]
    for xtick, con in enumerate(['CS+','CS-']):
        for r, response in enumerate(phases):
            ax[i].text(xtick+rpos[r],pvals.loc[(con,response),'topval'],pvals.loc[(con,response),'star'],ha='center',va='bottom')

        acq_bsl = p_convert(1 - (df.loc[(con,'acquisition'),'beta'].values > df.loc[(con,'baseline'),'beta'].values).mean())
        acq_ext = p_convert(1 - (df.loc[(con,'acquisition'),'beta'].values > df.loc[(con,'extinction'),'beta'].values).mean())
        print(con, phase, acq_bsl, acq_ext)
        y = pvals.loc[(con,'acquisition'),'topval']+.2
        if acq_bsl != '':
            barx = [xtick-.2,xtick-.2,xtick-.01,xtick-.01]
            bary = [y,y+.05,y+.05,y]
            ax[i].plot(barx, bary, c='black')
            mid = ((xtick-.2+xtick-.01)/2, y)
            ax[i].text(*mid,acq_bsl,ha='center',va='bottom')
        if acq_ext != '':
            barx = [xtick+.2,xtick+.2,xtick+.01,xtick+.01]
            bary = [y,y+.05,y+.05,y]
            ax[i].plot(barx, bary, c='black')
            mid = ((xtick+.2+xtick+.01)/2, y)
            ax[i].text(*mid,acq_ext,ha='center',va='bottom')
    # barx = [lx, lx, rx, rx]
    # bary = [y, y+barh, y+barh, y]
    # mid = ((lx+rx)/2, y+barh)
    '''post-hoc contrasts'''
    # print(phase)
    # print(pvals.reset_index()[['condition','encode_phase','response_phase','beta','ci','p','tail']][pvals.p < .1],'\n\n')

ax[0].collections[0].set_hatch('//')
ax[0].collections[3].set_hatch('//')
ax[1].collections[1].set_hatch('//')
ax[1].collections[4].set_hatch('//')
ax[2].collections[2].set_hatch('//')
ax[2].collections[5].set_hatch('//')

ax[0].set_xlabel('Pre')
ax[1].set_xlabel('Conditioning')
ax[2].set_xlabel('Post')
legend_elements = [Patch(facecolor=spal[0],edgecolor=None,label='Pre'),
                   Patch(facecolor=spal[1],edgecolor=None,label='Conditioning'),
                   Patch(facecolor=spal[2],edgecolor=None,label='Post'),
                   Patch(facecolor='white',edgecolor='black',hatch='//',label='Correct responses')]
fig.legend(handles=legend_elements,ncol=4,frameon=False,title='Temporal context memory response',loc='upper center',bbox_to_anchor=(.5,1))
plt.tight_layout(rect=(0,0,1,.9))