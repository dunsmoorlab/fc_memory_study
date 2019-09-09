#Overview of Univariate Analysis Model Setups

(Written by Emily, edited for markdown by Gus)

## Encoding:
Functional data from each encoding run was modeled with separate regressors for each combination of CS Type x Subsequent Memory.  For now, we are trying this three different ways, due to variation in trial numbers per condition among participants:  1) Model 001 – Remembered and Forgotten, 2) Model 002 -- Remembered High Confidence and Guessing/Forgotten, and 3) Model 003 – parametric modulation of CS type by subsequent memory (2 for Remembered High confidence, 1 for Remembered low confidence, -1 for forgotten low confidence, -2 for forgotten high confidence).  TBD which is best.
For each of the three models, then did a repeated measures ANOVA with factors of Encoding Phase, CS Type, Group, and (for Models 001-002) Subsequent Memory (either Remembered and Forgotten, or Remembered High Conf and Guessing/Forgotten). For the parametric modulation, subsequent memory is the modulator so it doesn’t make sense to include a factor for this.

### Model 001: remembered vs. forgotten

1st level:  BOLD data from each encoding run modeled with regressors for: 

1. CS+ Remembered
2. CS+ Forgotten and no responses 
3. CS- Remembered 
4. CS- Forgotten  and no responses

Higher level: repeated measures ANOVA with factors for:

1.  Phase (Baseline, FC, Extinction)
2.  CS Type (CS+, CS-)
3.  Subsequent Memory (Remembered, Forgotten)
4.  Group (Healthy, PTSD)

### Model 002: remembered high confidence vs. guessing/forgotten (like the Hermans paper)

1st level:  BOLD data from each encoding run modeled with regressors for:

1.  CS+ Remembered with High Confidence
2.  CS+ Remembered with Low Confidence (considered guessing), forgotten, and no responses
3.  CS- Remembered with high confidence
4.  CS- Remembered with low confidence (considered guessing), forgotten, and no responses

Higher level: repeated measures ANOVA with factors for:

1.  Phase (Baseline, FC, Extinction)
2.  CS Type (CS+, CS-)
3.  Subsequent Memory (High Confidence Remembered, Guessing/Forgotten)
4.  Group (Healthy, PTSD)

### Model 003: parametric modulation for subsequent memory
1st level: BOLD data from each phase modeled with regressors for:

1.  CS+, modulated by subsequent memory (2 for Remembered high confidence, 1 for Remembered low confidence, -1 for Forgotten low confidence (and no responses), -2 for Forgotten high confidence)
2.  CS-, modulated by subsequent memory (2 for Remembered high confidence, 1 for Remembered low confidence, -1 for Forgotten low confidence (and no responses), -2 for Forgotten high confidence)

Higher level: repeated measures ANOVA on the modulated BOLD data with factors for:
1.  Phase (Baseline, FC, Extinction)
2.  CS Type (CS+, CS-)
3.  Group (Healthy, PTSD)


## Recognition:
Functional data from each recognition block was modeled with separate regressors for each combination of CS Type x Prior Encoding Phase (for old items) x Recognition Judgment.  As with the encoding data, this was done three different ways (Models 001-003). Followed by 2nd level analysis to model within subject effects across recognition blocks.
For each of the three models, then did a repeated measures ANOVA with factors of CS Type x Group x Prior Encoding Phase (for old items) x (for Models 001-002 only) Old/New and Recognition Decision.  Will have to think about how we would include New items in Model 003 if we wanted to.  Otherwise, would just include Old items for the ANOVA in Model 003.

### Model 001: remembered vs. forgotten
1st level: BOLD data from each recognition run modeled with regressors for:

1.  CS+ Remembered from Baseline
2.  CS+ Remembered from FC
3.  CS+ Remembered from Extinction
4.  CS+ Forgotten and no responses (to old items) from Baseline
5.  CS+ Forgotten and no responses (to old items) from FC
6.  CS+ Forgotten and no responses (to old items) from Extinction
7.  CS+ False Alarms and no responses (to new items)
8.  CS+ Correct Rejections
9.  CS-Remembered from Baseline
10. CS-Remembered from FC
11. CS-Remembered from Extinction
12. CS- Forgotten and no responses (to old items) from Baseline
13. CS- Forgotten and no responses (to old items) from FC
14. CS- Forgotten and no responses (to old items) from Extinction
15. CS- False Alarms and no responses (to new items)
16. CS- Correct Rejections

Higher level: repeated measures ANOVA with factors for:

1.  Prior Encoding Phase (Baseline, FC, Extinction) 
2.  CS Type (CS+, CS-)
3.  Old New Condition (Old, New)
4.  Recognition Accuracy (Correct, Incorrect)
5.  Group (Healthy, PTSD)

optionally, could do a higher level analysis on only the old items (so it would just be memory accuracy—remembered/forgotten)

### Model 002: remembered high confidence vs. guessing/forgetting
1st level: BOLD data from each recognition run modeled with regressors for:

1.  CS+ Remembered with High confidence from Baseline
2.  CS+ Remembered with High confidence from FC
3.  CS+ Remembered with High confidence from Extinction
4.  CS+ Remembered with low confidence (guessing), forgotten, and no responses (to old items) from Baseline
5.  CS+ Remembered with low confidence (guessing), forgotten, and no responses (to old items) from FC
6.  CS+ Remembered with low confidence (guessing), forgotten, and no responses (to old items) from Extinction
7.  CS+ False Alarms and no responses (to new items)
8.  CS+ Correct Rejections
9.  CS- Remembered from Baseline
10. CS- Remembered from FC
CS- Remembered from Extinction
11. CS- Remembered with low confidence (guessing), forgotten, and no responses (to old items) from Baseline
12. CS- Remembered with low confidence (guessing), forgotten, and no responses (to old items) from FC
13. CS- Remembered with low confidence (guessing), forgotten, and no responses (to old items) from Extinction
14. CS- False Alarms and no responses (to new items)
15. CS- Correct Rejections

Higher level: repeated measures ANOVA with factors for:

1.  Prior Encoding Phase (Baseline, FC, Extinction) 
2.  CS Type (CS+, CS-)
3.  Old/New Condition (Old, New)
4.  Recognition Accuracy (Correct, Incorrect)
5.  Group (Healthy, PTSD)

Optionally, could do a higher level analysis on only the old items (so it would just be memory accuracy—remembered/forgotten)

### Model 003: parametric modulation for memory for old items
1st level: BOLD data from each phase modeled with regressors for:

1.  CS+ old items from Baseline, modulated by memory accuracy for old items (2 for Remembered high confidence, 1 for Remembered low confidence, -1 for Forgotten low confidence (and no responses), -2 for Forgotten high confidence)
2.  CS+ old items from FC, modulated by memory accuracy for old items (2 for Remembered high confidence, 1 for Remembered low confidence, -1 for Forgotten low confidence (and no responses), -2 for Forgotten high confidence)
3.  CS+ old items from Extinction, modulated by memory accuracy for old items (2 for Remembered high confidence, 1 for Remembered low confidence, -1 for Forgotten low confidence (and no responses), -2 for Forgotten high confidence)
4.  CS- old items from Baseline, modulated by memory accuracy for old items (2 for Remembered high confidence, 1 for Remembered low confidence, -1 for Forgotten low confidence (and no responses), -2 for Forgotten high confidence)
5.  CS- old items from FC, modulated by memory accuracy for old items (2 for Remembered high confidence, 1 for Remembered low confidence, -1 for Forgotten low confidence (and no responses), -2 for Forgotten high confidence)
6.  CS- old items from Extinction, modulated by memory accuracy for old items (2 for Remembered high confidence, 1 for Remembered low confidence, -1 for Forgotten low confidence (and no responses), -2 for Forgotten high confidence)
7.  CS+ False Alarms to new items
8.  CS+ Correct Rejections to new items
9.  CS- False alarms to new items
10. CS- correct rejections to new items

Alternatively, could do parametric modulation on new items as well, based on accuracy—but not clear if you would want False alarms high confidence as 2 or -2, etc.

Higher level: repeated measures ANOVA on the modulated BOLD data with factors for:

1.  Phase (Baseline, FC, Extinction)
2.  CS Type (CS+, CS-)

Optionally, could do a higher level analysis with new items, but not as straightforward given the parametric modulation for old item recognition accuracy


### Planned Conjunction Analyses of Encoding and Retrieval within certain conditions
Could take encoding data based only on CS type and phase (regardless of subsequent memory), or could do it based on CS type and phase x subsequent memory.  Want to examine to what extent CS+ effects (or CS-) from Fear Conditioning or Extinction return to mind during retrieval.

1.  CS+ from FC  ∩ Correctly remembered CS+ encoded during FC
2.  CS- from FC ∩ Correctly remembered CS- encoded during FC
3.  CS+ from Ext ∩ Correctly remembered CS+ encoded during Ext
4.  CS- from Ext ∩ Correctly remembered CS- encoded during Ext
5.  CS+ from FC  ∩ Correctly remembered CS+ encoded during Extinction
6.  CS+ from Ext ∩ Correctly remembered CS+ encoded during FC
