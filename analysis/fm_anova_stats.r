require(ez)
require(wPerm)
require(permuco)
require(lme4)
require(afex)
require(lmerTest) #Package must be installed first
require(emmeans)
require(irr)

setwd("C:/Users/ACH/Documents/fearmem")

df <- read.csv('cleaned_corrected_recognition.csv')
str(df)
df$subject <- as.factor(df$subject)

full.form <- cr ~ group * condition * encode_phase + Error(subject/(condition*encode_phase))
full.cr <- aovperm(full.form,df,np=10000)



#source memory
df <- read.csv('cleaned_avg_sm.csv')
str(df)
df$subject <- as.factor(df$subject)

#subset baseline
baseline = df[with(df,encode_phase == 'baseline'),]
extinction = df[with(df,encode_phase == 'extinction'),]


phase.form <- prop ~ group * condition * response_phase + Error(subject/(condition*response_phase))
mem.form <- mem_prop ~ group * condition * response_phase + Error(subject/(condition*response_phase))

prop.res <- aovperm(phase.form,extinction,np=10000) #change data here
mem.res <- aovperm(mem.form,extinction,np=10000) #change data here




#regular anovas here for typicality
df <- read.csv('cleaned_avg_ty.csv')
ty_aov <- ezANOVA(df,dv=.(typicality),wid=.(subject),within=.(condition,encode_phase),between=.(group))





#source memory lmm
df <- read.csv('cleaned_full_smt.csv')
str(df)
#df$subject <- as.factor(df$subject)
df$subject <- as.numeric(unlist(df$subject))
df$hc_acc <- as.numeric(unlist(df$hc_acc))
#df$hc_acc <- as.factor(df$hc_acc)
df$source_memory <- as.factor(df$source_memory)

baseline = df[with(df,encode_phase == 'baseline'),]

wo_mem <- lmer(hc_acc ~ trial_type * group + (1|subject), REML = FALSE, data = baseline)
w_mem <- lmer(hc_acc ~ trial_type + group + source_memory + (1|subject), REML = FALSE, data = baseline)
anova(wo_mem,w_mem)


glm(hc_acc~1+(1|subject), family = "binomial",baseline)






