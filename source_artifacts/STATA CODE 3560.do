***Import data

***Panel data
xtset SN Post

***Descriptive Stats
***Controls
tab Gender, gen(M)
tab Ethnicity, gen(R)
tab HighestEducationAttained, gen(EDU)
tab Fieldofstudy, gen(B)
tab IndustryExperience, gen(EXPF)
tab Round, gen(ROUND)
tab Sector, gen(INDUSTRY)

***STEM Education
gen STEM=0
replace STEM=1 if B3==1
replace STEM=1 if B5==1 
replace STEM=1 if B9==1 

***EDUCATION
***Education is coded as a categorical variable capturing the participant’s highest level of education. Education was equal to 1 in case of no tertiary education, 2 for Bachelor, 3 for Master, and 4 for PhD.
gen Education=1 if EDU2==1
replace Education=2 if EDU1==1
replace Education=3 if EDU3==1
replace Education=4 if EDU4==1

***STEM Experience
gen STEMEXP=0
replace STEMEXP=1 if EXPF6==1
replace STEMEXP=1 if EXPF16==1

***RANDOMIZATION CHECK (TABLE 1)
***Demand Pull Group is indicated as Treatment Group
***Resource Push Group is indicated as Control Group

***T-TEST
ttest Age if Post==0, by(Treatment)
ttest M1 if Post==0, by(Treatment)
ttest R1 if Post==0, by(Treatment)
ttest Education if Post==0, by(Treatment)
ttest STEM if Post==0, by(Treatment)
ttest B2 if Post==0, by(Treatment)
ttest Working if Post==0, by(Treatment)
ttest Studying if Post==0, by(Treatment)
ttest EntrepExperience if Post==0, by(Treatment)

***The categorical variable Work experience reflected the participant’s past work experience. Work experience was set to 0 in case of no prior work experience, 1 for work experience between 1 to 5 years, 2 for 10 years, 3 for 11 to 20 years, and 4 for above 20 years.

ttest WorkExp if Post==0, by(Treatment)
ttest STEMEXP if Post==0, by(Treatment)
ttest TeamSize if Post==0, by(Treatment)
ttest Registered if Post==0, by(Treatment)

***Participants’ business ideas were classified into four broad industries. INUSTRY1 refers to Commerce & e-Commerce, INDUSTRY2 refers to Online Platform & App, INUSTRY4 refers to Services and INUSTRY3 refers to Others. The category Others involved business ideas that could not be classified in the previous categories as well as ideas that were too broad, vague, or unspecified by the participant in the enrolment phase.

ttest INDUSTRY1 if Post==0, by(Treatment)
ttest INDUSTRY2 if Post==0, by(Treatment)
ttest INDUSTRY4 if Post==0, by(Treatment)
ttest INDUSTRY3 if Post==0, by(Treatment)

***Revenue and Customers are reported in categorical values to to preserve the anonymity & confidentiality of our participants
***Revenue(likert) has value 1 if sales were 0, 2 for sales between 1 and 1,000 SGD, 3 for sales between 1,001 and 5,000 SGD, 4 for sales between 5,001 and 10,000 SGD, and 5 for sales exceeding 10,000 SGD. 
***Customers(likert) has value of 1 if there were no new customers, 2 for 1 to 10 new customers, 3 for 11 to 50 new customers, 4 for 51 to 100 new customers, and 5 for over 100 new customers. 
 

ttest RevenueLikert if Post==0, by(Treatment)
ttest CustomersLikert if Post==0, by(Treatment)

ttest ROUND2 if Post==0, by(Treatment)

***Summary Table
eststo a: estpost ttest Age M1 R1 Education STEM B2 Working Studying EntrepExperience WorkExp STEMEXP TeamSize Registered INDUSTRY1 INDUSTRY2 INDUSTRY4 INDUSTRY3  RevenueLikert CustomersLikert ROUND2 if Post==0, by(Treatment)

esttab a, cells("mu_2 mu_1 b p")

***OUTCOME VARIABLES (TABLE 2)
***Customer Interaction measured how much the entrepreneur engaged with customers and includes the activities of customer interviews, customer observation (ethnography), market research, and surveys. Expert Interaction measured the degree of engagement with experts in the participant’s specific business sector. Networking captured how much the entrepreneur actively searched for potential partners, including team members, suppliers, or investors. The category Others comprised the remaining activities. The score 0 referred to no engagement with the activity while 5 referred to a strong focus on the activity.
***END OF INTERVENTION
sum CustomerInteraction if Post==1 & Treatment==1
sum CustomerInteraction if Post==1 & Treatment==0
sum ExpertInteraction if Post==1 & Treatment==1
sum ExpertInteraction if Post==1 & Treatment==0
sum Networking if Post==1 & Treatment==1
sum Networking if Post==1 & Treatment==0

***Summary Table
eststo clear
local descript = "CustomerInteraction ExpertInteraction Networking"
eststo a: quietly estpost summarize `descript' if Treatment==1 & Post==1
eststo b: quietly estpost summarize `descript' if Treatment==0 & Post==1 
esttab a b using "2.descriptives.rtf", cells("mean sd min max") label nogaps replace 

***END OF PROGRAM
sum RevenueLikert if Post==1 & Treatment==1
sum RevenueLikert if Post==1 & Treatment==0
sum CustomersLikert if Post==1 & Treatment==1
sum CustomersLikert if Post==1 & Treatment==0
sum Post if Treatment==1
sum Post if Treatment==0

***Summary Table
eststo clear
local descript = "RevenueLikert CustomersLikert"
eststo a: quietly estpost summarize `descript' if Treatment==1 & Post==1
eststo b: quietly estpost summarize `descript' if Treatment==0 & Post==1 
esttab a b using "2.descriptives.rtf", cells("mean sd min max") label nogaps replace 


***NON PARAMETRIC ANALYSIS (FIGURE 2)
***Kernel Density
grstyle init
grstyle set plain , horizontal
grstyle set legend 1 , inside
grstyle set lpattern

***REVENUE (LIKERT)
kdensity RevenueLikert if Treat==0 & Post==1, addplot(kdensity RevenueLikert if Treat==1 & Post==1) xtitle(RevenueLikert) ytitle(Kernel Density)

***NEW CUSTOMERS (LIKERT) (FIGURE A7, In Appendix)
kdensity CustomersLikert  if Treat==0 & Post==1, addplot(kdensity CustomersLikert if Treat==1 & Post==1) xtitle(CustomersLikert) ytitle(Kernel Density)

***TABLE 3 REGRESSIONS

***REVENUE Model (2)
xtreg RevenueLikert Post##Treat , fe robust
outreg2 using table3.doc, stats(coef se pval) bracket noaster dec(3) replace

***CUSTOMERS Model (4)
xtreg CustomersLikert Post##Treat , fe robust
outreg2 using table3.doc, stats(coef se pval) bracket noaster dec(3) 


***Manipulation Check (Table A1)
reg CustomerInteraction Treat if Post==1, robust
outreg2 using tableA1.doc, stats(coef se pval) bracket noaster dec(3) replace

reg ExpertInteraction Treat if Post==1, robust
outreg2 using tableA1.doc, stats(coef se pval) bracket noaster dec(3)

reg Networking Treat if Post==1, robust
outreg2 using tableA1.doc, stats(coef se pval) bracket noaster dec(3)




*******************************************************
****************ONLINE APPENDIX RESULTS****************
*******************************************************

***EXIT ANALYSIS (Table B1)
reg exit Treat if Post==0 , robust
outreg2 using tableB1.doc, stats(coef se pval) bracket noaster dec(3) replace

reg exit Treat i.Round if Post==0, robust
outreg2 using tableB1.doc, stats(coef se pval) bracket noaster dec(3)  

reg exit Treat i.Round Age M1 R1 EDU3 Registered Working Studying EntrepExperience RevenueLikert CustomersLikert if Post==0, robust
outreg2 using tableB1.doc, stats(coef se pval) bracket noaster dec(3)
 
***TIME EFFORT (TABLE B2)
reg TimeEffort Treat if Post==0 , robust
outreg2 using tableB2.doc, stats(coef se pval) bracket noaster dec(3) replace

reg TimeEffort Treat i.Round if Post==0, robust
outreg2 using tableB2.doc, stats(coef se pval) bracket noaster dec(3)  

reg TimeEffort Treat i.Round Age M1 R1 EDU3 Registered Working Studying EntrepExperience RevenueLikert CustomersLikert if Post==0, robust
outreg2 using tableB2.doc, stats(coef se pval) bracket noaster dec(3) 


***Including Exited Ventures (Table B3)

***Replace missing Final Revenue with Initial Revenue 
***REV***
gen HTotalREV=RevenueLikert
gen IREV=RevenueLikert if Post==0 
egen InitialREV=max(IREV), by(SN)
gen FREV=RevenueLikert if Post==1
egen FinalREV=max(FREV), by(SN)
replace HTotalREV=InitialREV if RevenueLikert==. & Post==1 & InitialREV!=.
replace HTotalREV=1 if InitialREV==. & FinalREV==.


***Replace missing Final Customers with Initial Customers 
***CUS
gen HTotalCUS=CustomersLikert
gen ICUS=CustomersLikert if Post==0 
egen InitialCUS=max(ICUS), by(SN)
gen FCUS=CustomersLikert if Post==1
egen FinalCUS=max(FCUS), by(SN)
replace HTotalCUS=InitialCUS if CustomersLikert==. & Post==1
replace HTotalCUS=1 if InitialCUS==. & FinalCUS==.
*******

***REGRESSIONS
xtreg HTotalREV Post##Treat , fe robust
outreg2 using tableB3.doc, stats(coef se pval) bracket noaster dec(3) replace

xtreg HTotalCUS Post##Treat , fe robust
outreg2 using tableB3.doc, stats(coef se pval) bracket noaster dec(3)


***Program Attendance (Table B4)
xtreg RevenueLikert Post##Treat if AttendanceCertificate==1 , fe robust
outreg2 using tableB4.doc, stats(coef se pval) bracket noaster dec(3) replace  

xtreg CustomersLikert Post##Treat if AttendanceCertificate==1, fe robust
outreg2 using tableB4.doc, stats(coef se pval) bracket noaster dec(3) 


***MEDIATION (Table B5)
reg CustomerInteraction Treat if Post==1
outreg2 using tableB5.doc, stats(coef se pval) bracket noaster dec(3) replace  
reg RevenueLikert Treat CustomerInteraction if Post==1
outreg2 using tableB5.doc, stats(coef se pval) bracket noaster dec(3) 
reg CustomersLikert Treat CustomerInteraction if Post==1
outreg2 using tableB5.doc, stats(coef se pval) bracket noaster dec(3) 

medeff (regress CustomerInteraction Treat ) (regress RevenueLikert CustomerInteraction Treat) if Post==1, mediate( CustomerInteraction ) treat(Treat)
medeff (regress CustomerInteraction Treat ) (regress CustomersLikert CustomerInteraction Treat) if Post==1, mediate( CustomerInteraction ) treat(Treat)

***SEPARATE ROUNDS 
***Round 1 (Table B6)
xtreg RevenueLikert Post##Treat if Round==1, fe robust
outreg2 using tableB6.doc, stats(coef se pval) bracket noaster dec(3) replace

xtreg CustomersLikert Post##Treat if Round==1, fe robust
outreg2 using tableB6.doc, stats(coef se pval) bracket noaster dec(3) 

***Round 2 (Table B7)
xtreg RevenueLikert Post##Treat if Round==2, fe robust
outreg2 using tableB7.doc, stats(coef se pval) bracket noaster dec(3)  

xtreg CustomersLikert Post##Treat if Round==2, fe robust
outreg2 using tableB7.doc, stats(coef se pval) bracket noaster dec(3) 



***Power Calculations 
***Power Analysis based on the increment in revenue
***Table B8
power twomeans 0 (1000 1500 2000 2500 3000), n(100 140 180 220 260) sd(6000) alpha(0.10)
***Table B9
power twomeans 0 (1000 1500 2000 2500 3000), n(100 140 180 220 260) sd(6000) alpha(0.05)

***Power Analysis based on the increment in revenue(cat)
***Table B10
power twomeans 0 (0.2 0.4 0.6 0.8 1), n(100 140 180 220 260) sd(1) alpha(0.10)
***Table B11
power twomeans 0 (0.2 0.4 0.6 0.8 1), n(100 140 180 220 260) sd(1) alpha(0.05)

***Power Analysis based on the increment in Customers
***Table B12
power twomeans 0 (10 15 20 25 30), n(100 140 180 220 260) sd(60) alpha(0.10)
***Table B13
power twomeans 0 (10 15 20 25 30), n(100 140 180 220 260) sd(60) alpha(0.05)




