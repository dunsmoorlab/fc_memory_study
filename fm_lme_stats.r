require(lme4)
require(lmerTest)
require(emmeans)
require(magrittr)
require(ggplot2)
phases = c('baseline','acquisition','extinction')
cons = c('CS+','CS-')

#behavior example
setwd('/Users/ach3377/Documents/fearmem')
df <- read.csv('cleaned_avg_sm.csv')
df$subject <- factor(df$subject)
beh <- lmer(prop~encode_phase*condition*response_phase + (1|subject),data=df,REML=FALSE)
anova(beh)
beh.emm <- emmeans(beh,list(pairwise ~ encode_phase:condition:response_phase),adjust="None")
beh.emmc <- broom::tidy(beh.emm$`pairwise differences of encode_phase, condition, response_phase`)

df <- read.csv('cleaned_full_sm.csv')
df$subject <- factor(df$subject)
df$typicality <- ave(df$typicality,df$subject,FUN=scale)

simple_mod <- glmer(hc_acc ~ factor(encode_phase) + factor(condition) + (1|subject),data=df,family="binomial")
source_mod <- glmer(hc_acc ~ factor(encode_phase) + factor(condition) + factor(source_memory) + (1|subject),data=df,family="binomial")
ty_mod <- glmer(hc_acc ~ factor(encode_phase) + factor(condition) + typicality + (1|subject),data=df,family="binomial")
full_mod <- glmer(hc_acc ~ factor(encode_phase) + factor(condition) + factor(source_memory) + typicality + (1|subject),data=df,family="binomial")
#summary(full_mod)
anova(full_mod,ty_mod,method="LRT")

lrtest(ty_mod,source_mod)
sjPlot::plot_model(mod)
lh <- linearHypothesis(mod,c(".5*factor(source_memory)baseline + .5*factor(source_memory)extinction = typicality"))
lh2 <- linearHypothesis(mod,c("factor(encode_phase)baseline + factor(encode_phase)extinction = typicality"))


require(afex)
full_mod <- mixed(hc_acc ~ encode_phase + condition + source_memory + typicality + (1|subject),data=df,family="binomial", method="LRT")










#roi data
setwd('/Users/ach3377/Documents/gPPI/sm_events')
df <- read.csv('Response_extracted_pe.csv')
df$subject <- factor(df$subject)

roi.df <- df[which(df$roi == 'dACC'),]
roi.lme <- lmer(pe~encode_phase*condition*source_memory + (1+run|subject), data=roi.df, REML=FALSE)
anova(roi.lme)
roi.emm <- emmeans(roi.lme, pairwise~condition*source_memory, adjust='None')
roi.emmc <- data.frame(roi.emm$contrasts)
roi.emmc[c(68,87),]

roi.emm.df <- data.frame(roi.emm$emmeans)# %>% broom::tidy()
roi.emm.df %>%
  ggplot(aes(source_memory, emmean, ymin=lower.CL, ymax=upper.CL)) +
  geom_pointrange() +
  ylab("RT")

