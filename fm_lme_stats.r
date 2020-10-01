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
beh.emmc <- broom::tidy(m.emm$`pairwise differences of encode_phase, condition, response_phase`)

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

