from fm_config import *

class smt():

    def __init__(self):

        self.init_dfs()
        self.load_sub_data()

    def init_dfs(self):

        self.sm_df = pd.DataFrame({'prop':0.0,
                                   'mem_prop':0.0,
                                   '_count':0,
                                   'mem_count':0},
                                    index=pd.MultiIndex.from_product(
                                    [smt_sub_args,cons,phases,phases],
                                    names=['subject','condition','encode_phase','response_phase']))

        self.ty_df = pd.DataFrame({'typicality':0},index=pd.MultiIndex.from_product(
                                    [smt_sub_args,cons,phases],
                                    names=['subject','condition','encode_phase']))


    def load_sub_data(self):

        for sub in smt_sub_args:

            subj = bids_meta(sub)
            df = subj.behav['source_memory_typicality'].copy(
                ).set_index(['trial_type','encode_phase']
                ).sort_index()

            df['hc_acc'] = df.recognition_memory.apply(lambda x: 1 if x == 'DO' else 0)

            phase_convert = {1:'baseline',2:'acquisition',3:'extinction'}
            df.source_memory = df.source_memory.apply(lambda x: phase_convert[x])

            for con in cons:
                for encode_phase in phases:
                 
                    dat = df.loc[con,encode_phase].copy()
                    
                    for response_phase in phases:
                        _count = np.where(dat.source_memory == response_phase)[0].shape[0] 
                        self.sm_df.loc[(sub,con,encode_phase,response_phase),'prop'] = _count / 24
                        self.sm_df.loc[(sub,con,encode_phase,response_phase),'_count'] = _count

                        mem_dat = dat[dat.hc_acc == 1].copy()
                        mem_count = np.where(mem_dat.source_memory == response_phase)[0].shape[0] 
                        self.sm_df.loc[(sub,con,encode_phase,response_phase),'mem_prop'] = mem_count / mem_dat.shape[0] 
                        self.sm_df.loc[(sub,con,encode_phase,response_phase),'mem_count'] = mem_count


                    self.ty_df.loc[(sub,con,encode_phase),'typicality'] = dat.typicality.mean()

        self.sm_df = self.sm_df.reset_index()
        self.sm_df['group'] = self.sm_df.subject.apply(lgroup)
        
        self.ty_df = self.ty_df.reset_index()
        self.ty_df['group'] = self.ty_df.subject.apply(lgroup)

# s = smt()
# df = s.sm_df.copy()

# for group in ['healthy','ptsd']:
#     print(group)
#     sns.catplot(data=df[df.group == group],x='encode_phase',y='prop',
#                 col='response_phase',row='condition',
#                 kind='bar')

class recognition_memory():
    #exclude subs = [18,20,120]

    def __init__(self):
        
        self.init_dfs()
        self.load_sub_data()

    def init_dfs(self):

        self.df = pd.DataFrame({'hr':0.0,#hitrate
                                'fa':0.0},#false alarm rate
                                    index=pd.MultiIndex.from_product(
                                    [all_sub_args,cons,phases],
                                    names=['subject','condition','encode_phase']))
    def load_sub_data(self):

        for sub in all_sub_args:

            subj = bids_meta(sub)
            df = subj.mem_df.set_index(['trial_type','encode_phase']
                ).sort_index().copy()

            #grab hitrates first
            for con in cons:
                for phase in phases:
                    dat = df.loc[con,phase].copy()
                    dat.high_confidence_accuracy = dat.high_confidence_accuracy.apply(lambda x: 1 if x == 'H' else 0)
                    self.df.loc[(sub,con,phase),'hr'] = dat.high_confidence_accuracy.sum() / dat.shape[0]

                #next false alarms
                dat = df.loc[con,'foil'].copy()
                dat.high_confidence_accuracy = dat.high_confidence_accuracy.apply(lambda x: 1 if x == 'FA' else 0)
                self.df.loc[(sub,con),'fa'] = dat.high_confidence_accuracy.sum() / dat.shape[0]

        self.df['cr'] = self.df.hr - self.df.fa

        self.df = self.df.reset_index()
        self.df['group'] = self.df.subject.apply(lgroup)

    def explore_outliers(self):

        df = self.df.copy()

        nrs = {}
        for sub in all_sub_args:
            subj = bids_meta(sub)
            nrs[sub] = subj.mem_df.response.isna().sum() / 240

        self.nrs = pd.DataFrame.from_dict(nrs,orient='index').reset_index(
                    ).rename(columns={'index':'subject',0:'missing'})

        fig, ax = plt.subplots()
        # sns.boxplot(data=self.nrs,y='missing')
        sns.swarmplot(data=self.nrs,y='missing')
        ax.set_ylabel('% missing trials')

        print(self.nrs.loc[np.where(self.nrs.missing >= .05)[0]])


        df = df.set_index(['subject']
            ).drop([20,120]
            ).reset_index()

        for group in groups:

            dat = df[df.group == group].copy()
            fig, ax = plt.subplots()
            sns.boxplot(data=dat,x='encode_phase',y='cr',
                        hue='condition',palette=cpal,ax=ax)
            sns.swarmplot(data=dat,x='encode_phase',y='cr',
                        hue='condition',palette=cpal,ax=ax,dodge=True,edgecolor='black',linewidth=1)
            ax.set_title(group)
            ax.set_ylabel('high confidence\ncorrected recognition')
            ax.hlines(0,ax.get_xlim()[0],ax.get_xlim()[1],linestyle=':')


        print(df.loc[np.where(df.cr <= .05)[0],['group','subject','condition','encode_phase','cr']].sort_values(by=['encode_phase','condition','group']))


        df = df.set_index(['subject']
            ).drop([18]
            ).reset_index()

        print(pg.normality(df,dv='cr',group='group'))        

        df.to_csv('../cleaned_corrected_recognition.csv',index=False)
# df = cr.cr
# sns.catplot(data=df,x='encode_phase',y='cr',
#             hue='condition',row='group',
#             kind='bar')
