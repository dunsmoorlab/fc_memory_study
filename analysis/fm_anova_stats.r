require(ez)
require(permuco)
require(dplyr)
require(afex)
setwd("C:/Users/ACH/Documents/fearmem")
#setwd('/Users/ach3377/Documents/fearmem')

df <- read.csv('cleaned_corrected_recognition.csv')
str(df)
df$subject <- as.factor(df$subject)

full.form <- cr ~ group * condition * encode_phase + Error(subject/(condition*encode_phase))
full.aov <- aovperm(full.form,df,np=10000)
summary(full.aov)
ezANOVA(df,dv=cr,wid=subject,within=.(condition,encode_phase),between=.(group))


collapse.form <- cr ~ condition * encode_phase + Error(subject/(condition*encode_phase))
collapse.aov <- aovperm(collapse.form,df,np=10000)
select(collapse.aov$table, dfn, dfd, F, "permutation P(>F)")
ezANOVA(df,dv=cr,wid=subject,within=.(condition,encode_phase))

#source memory
df <- read.csv('cleaned_avg_sm.csv')
str(df)
df$subject <- as.factor(df$subject)

full.source.form <- prop ~ group * response_phase * condition * encode_phase + Error(subject/(response_phase * condition * encode_phase))
full.source.aov <- aovperm(full.source.form,df,np=10000)
summary(full.source.aov)
ezANOVA(df,dv=prop,wid=subject,within=.(condition,encode_phase,response_phase),between=.(group))


collapse.source.form <- prop ~ response_phase * condition * encode_phase + Error(subject/(response_phase * condition * encode_phase))
collapse.source.aov <- aovperm(collapse.source.form,df,np=10000)
select(collapse.source.aov$table, dfn, dfd, F, "permutation P(>F)")
ezANOVA(df,dv=prop,wid=subject,within=.(condition,encode_phase,response_phase))

#subset baseline
baseline = df[with(df,encode_phase == 'baseline'),]
extinction = df[with(df,encode_phase == 'extinction'),]


phase.form <- prop ~ group * condition * response_phase + Error(subject/(condition*response_phase))
mem.form <- mem_prop ~ group * condition * response_phase + Error(subject/(condition*response_phase))

prop.res <- aovperm(phase.form,extinction,np=10000) #change data here
mem.res <- aovperm(mem.form,extinction,np=10000) #change data here




#regular anovas here for typicality
df <- read.csv('cleaned_avg_ty.csv')
ty_aov <- ezANOVA(df,dv=.(typicality),wid=.(subject),within=.(condition,encode_phase))
#ty_aov <- ezANOVA(df,dv=.(typicality),wid=.(subject),within=.(condition),between=.(group))

#typicality & source memory lmms
set_sum_contrasts()

df <- read.csv('cleaned_full_sm.csv')

typ.mod <- mixed(typicality ~ condition*source_memory*encode_phase + (1|subject), data=df, REML=FALSE, method="LRT")#, test_intercept=TRUE)
summary(typ.mod)
anova(typ.mod)
typ.comp <- emmeans(typ.mod, pairwise ~ source_memory)
summary(typ.comp)
