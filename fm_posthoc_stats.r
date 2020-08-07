require(ez)
require(permuco)
require(dplyr)

setwd("C:/Users/ACH/Documents/fearmem")
df <- read.csv('cleaned_avg_sm.csv')
str(df)
df$subject <- as.factor(df$subject)

x = df[with(df,encode_phase == 'baseline' & response_phase == 'acquisition' & condition == 'CS+'),'prop']
y = df[with(df,encode_phase == 'baseline' & response_phase == 'acquisition' & condition == 'CS-'),'prop']

acq = c[with(c,group == group & phase %in% c('fear1','fear2')),]
