The remaining tasks needs to use module_LP:py



# FLEXIBLE LOADS MODELING 
the previous quantity lower bound = 80% of the bid quantity was making the flexible loads actually inflexible,since they were never curtailed.ù
the right approach is to have 0, in order to allow the curtailment.

CHANGES branch step 5:

Saved classes in seperate module'module LP' to be imported for each step 
'load_casestudy' to use for import of case study data, change date and time here
import results directly from step1 for step 5, so not to run optimization again
step 1 commented plotting out just to not see it each time 
