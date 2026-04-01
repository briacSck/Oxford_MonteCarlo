rm(list=ls())
library(data.table)
library(ggplot2)
library(stargazer)
library(systemfit)
library(texreg)
library(stringr)
library(kableExtra)
library(knitr)
library(scales)
library(tikzDevice)
library(gridExtra)
library(moments)
library(lfe)
library(xtable)
library(cowplot)
library(extrafont)
library(ggpubr)

# DATA --------------------------------------------------------------------
# movie data
ratings <- fread("movie_data.csv")

# experiment data
pairs <- fread("experiment_pairs_data.csv")


##################################
# FIGURES 1 and 2
# https://www.imdb.com/title/tt1179034/
# https://www.imdb.com/title/tt1821641/

##################################
# FIGURE 3

p1 <- ggplot(ratings,aes(x=mean_pooled,y=sd_pooled,color=FLead, linetype = FLead)) +
  geom_smooth() + xlim(4.5,8.5) + 
  labs( y="Rating St. Dev.", x="Pooled Rating Mean") + 
  scale_color_discrete(name ="Lead Actor",labels=c("FALSE"="Male", "TRUE"="Female"))  +
  scale_linetype_discrete(name ="Lead Actor",labels=c("FALSE"="Male", "TRUE"="Female")) +
  theme_bw() + theme(text= element_text(family="Times New Roman"), panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank()) +
  theme(strip.background = element_rect(fill=NA), strip.text = element_text(size = 12))
p2 <- ggplot(ratings, aes(x=mean_pooled)) +
  geom_histogram(aes(fill = FLead), position = "identity", alpha = .5) + xlim(4.5,8.5) +
  labs( y="Count of films", x="Pooled Rating Mean") +
  theme_bw() + theme(text= element_text(family="Times New Roman"), panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank()) +
  geom_vline(data = ratings[, mean(mean_pooled), by = FLead], aes(xintercept=V1, color=FLead, linetype = FLead)) +
  scale_color_discrete(name ="Lead Actor",labels=c("FALSE"="Male", "TRUE"="Female"))  +
  scale_fill_discrete(name ="Lead Actor",labels=c("FALSE"="Male", "TRUE"="Female")) +
  scale_linetype_discrete(name ="Lead Actor",labels=c("FALSE"="Male", "TRUE"="Female"))
p <- ggpubr::ggarrange(p1, p2, ncol = 2, common.legend = T, legend = "bottom")
p

##################################
# FIGURE 4
ggplot() +
  stat_summary_bin(data = pairs[actually_watch.x >= 1 & actually_watch.y >= 1 & la_type != "mixed"], aes("1", absdiff, color = la_type, shape = la_type)) +
  stat_summary_bin(data = pairs[actually_watch.x >= 2 & actually_watch.y >= 2 & la_type != "mixed"], aes("2", absdiff, color = la_type, shape = la_type)) +
  stat_summary_bin(data = pairs[actually_watch.x >= 3 & actually_watch.y >= 3 & la_type != "mixed"], aes("3", absdiff, color = la_type, shape = la_type)) +
  stat_summary_bin(data = pairs[actually_watch.x >= 4 & actually_watch.y >= 4 & la_type != "mixed"], aes("4", absdiff, color = la_type, shape = la_type)) +
  stat_summary_bin(data = pairs[actually_watch.x >= 5 & actually_watch.y >= 5 & la_type != "mixed"], aes("5", absdiff, color = la_type, shape = la_type)) +
  stat_summary_bin(data = pairs[actually_watch.x >= 6 & actually_watch.y >= 6 & la_type != "mixed"], aes("6", absdiff, color = la_type, shape = la_type)) +
  stat_summary_bin(data = pairs[actually_watch.x >= 7 & actually_watch.y >= 7 & la_type != "mixed"], aes("7", absdiff, color = la_type, shape = la_type)) +
  stat_summary_bin(data = pairs[actually_watch.x >= 8 & actually_watch.y >= 8 & la_type != "mixed"], aes("8", absdiff, color = la_type, shape = la_type)) +
  stat_summary_bin(data = pairs[actually_watch.x >= 9 & actually_watch.y >= 9 & la_type != "mixed"], aes("9", absdiff, color = la_type, shape = la_type)) +
  stat_summary_bin(data = pairs[actually_watch.x >= 10 & actually_watch.y >= 10 & la_type != "mixed"], aes("10", absdiff, color = la_type, shape = la_type)) +
  labs(x = "Watch score threshold", y = "Rating diff.") +
  theme_bw() + theme(text= element_text(family="Times New Roman"), panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank()) +
  scale_color_discrete(name = "Lead actor") + scale_shape_discrete("Lead actor") +
  scale_x_discrete(limits = paste0(1:10))


##################################
# FIGURE 5
pairs[,watch := (actually_watch.x > 5 & actually_watch.y > 5)]
label <- paste0("Pairs with watch score > 5 \n (N=",
                prettyNum(pairs[watch==1 & la_type != "mixed" & team_type != "mixed" & rater_type != "mixed", .N], big.mark = ','),")")
c1 <- pairs[watch==1 & la_type=="male" & team_type=="exp" & rater_type=="female", mean(absdiff)] - 
  pairs[watch==1 & la_type=="female" & team_type=="exp" & rater_type=="female", mean(absdiff)]
c2 <- pairs[watch==1 & la_type=="male" & team_type=="inexp" & rater_type=="female", mean(absdiff)] - 
  pairs[watch==1 & la_type=="female" & team_type=="inexp" & rater_type=="female", mean(absdiff)]
c3 <- pairs[watch==1 & la_type=="male" & team_type=="exp" & rater_type=="male", mean(absdiff)] - 
  pairs[watch==1 & la_type=="female" & team_type=="exp" & rater_type=="male", mean(absdiff)]
c4 <- pairs[watch==1 & la_type=="male" & team_type=="inexp" & rater_type=="male", mean(absdiff)] - 
  pairs[watch==1 & la_type=="female" & team_type=="inexp" & rater_type=="male", mean(absdiff)]
f_labels <- data.frame(
  team_type = c("exp", "inexp", "exp", "inexp"),
  rater_type = c("female", "female", "male", "male"),
  label = c(paste("MF gap =",sprintf(c1,fmt = '%#.3f')),
            paste("MF gap =",sprintf(c2,fmt = '%#.3f')),
            paste("MF gap =",sprintf(c3,fmt = '%#.3f')),
            paste("MF gap =",sprintf(c4,fmt = '%#.3f'))
  )
)
team_type.labs <- c("experienced team", "inexperienced team", "mixed")
names(team_type.labs) <- c("exp", "inexp", "mixed")
rater_type.labs <- c("female audience", "male audience", "mixed")
names(rater_type.labs) <- c("female", "male", "mixed")
ggplot(pairs[watch==1 & la_type != "mixed" & team_type != "mixed" & rater_type != "mixed"])+
  stat_summary_bin(aes(x=la_type, y=absdiff))+
  facet_grid(rater_type~team_type, labeller = labeller(team_type = team_type.labs, rater_type = rater_type.labs))+
  labs(caption = label, x = "Lead actor", 
       y = "Rating diff.", fill = FALSE) +
  theme_bw() + theme(text= element_text(family="Times New Roman"), panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), strip.text.x = element_text(size = 11), strip.text.y = element_text(size = 11)) +
  geom_text(data = f_labels, mapping = aes(x = 1.5, y = 1.55, label = label, family="Times New Roman"))


##################################
# FIGURE 6
ggplot(ratings,
       aes(x=bom_year,y=as.integer(FLead), shape=major, linetype = major))+ 
  stat_smooth() + stat_summary_bin(size=.2) +
  scale_x_continuous(breaks = c(min(ratings$bom_year):max(ratings$bom_year))) +
  scale_y_continuous(breaks = c(0, 0.1, 0.2, 0.3, 0.4, 0.5), labels = scales::percent) +  
  xlab("Year of film release") +
  ylab("Lead actor is woman") +
  scale_shape_discrete("Studio type", labels = c("FALSE" = "independent", "TRUE" = "major")) +
  scale_linetype_discrete("Studio type", labels = c("FALSE" = "independent", "TRUE" = "major")) +
  theme_bw() + theme(text= element_text(family="Times New Roman"), axis.text.x=element_text(angle = -90, vjust = 0.5), panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank())


##################################
# FIGURE 7
stack <- rbind(
  ratings[,.(kim_violence_gore,value=mean_males,FLead,case="Male Audience")],
  ratings[,.(kim_violence_gore,value=mean_females,FLead,case="Female Audience")]
)
ggplot(stack, aes(x=kim_violence_gore, y= value, shape=FLead, color=FLead)) +
  stat_summary_bin() + 
  scale_x_continuous("Violence level", breaks=c(0:10)) +
  scale_y_continuous("Rating Mean") +
  scale_color_discrete(name ="Lead Actor",labels=c("FALSE"="Male", "TRUE"="Female")) +
  scale_shape_discrete(name ="Lead Actor",labels=c("FALSE"="Male", "TRUE"="Female")) +  theme_bw() + 
  theme(text= element_text(family="Times New Roman"), panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank()) +
  facet_wrap(~ case)

##################################
# FIGURE 8
stack <- rbind(
  ratings[,.(kim_violence_gore,value=sd_males,FLead,case="Male Audience")],
  ratings[,.(kim_violence_gore,value=sd_females,FLead,case="Female Audience")]
)
ggplot(stack, aes(x=kim_violence_gore, y= value, color=FLead,shape=FLead)) +
  stat_summary_bin() + 
  scale_x_continuous("Violence level", breaks=c(0:10)) +
  scale_y_continuous("Rating St. Dev.") +
  scale_color_discrete(name ="Lead Actor",labels=c("FALSE"="Male", "TRUE"="Female")) +
  scale_shape_discrete(name ="Lead Actor",labels=c("FALSE"="Male", "TRUE"="Female")) +
  theme_bw() + 
  theme(text= element_text(family="Times New Roman"), panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank()) +
  facet_wrap(~ case)

##################################
# TABLE 1
t <- ratings[,list(FLead,
                   kim_violence_gore,kim_sex_nudity,kim_language,
                   major,log(bom_opening_theaters),genres.count,
                   mean_pooled,sd_pooled,skew_pooled,
                   mean_males,sd_males,skew_males,log(men_all_count),
                   mean_females,sd_females,skew_females,log(women_all_count),
                   log(bom_total_gross)
                   )]
stargazer(t,type="text")
rm(t)              

##################################
# TABLE 2
m.pooled.mean <- lm(mean_pooled ~ FLead + 
                      kim_violence_gore + kim_sex_nudity + kim_language +
                      major +
                      log(bom_opening_theaters) +
                      genres.count + 
                      as.factor(bom_year) + as.factor(bom_open_month) +
                      genre.Action + genre.Adventure + genre.Animation + genre.Biography + genre.Comedy +
                      genre.Crime + genre.Drama + genre.Family + genre.Fantasy +
                      genre.History + genre.Horror + genre.Music + genre.Musical + genre.Mystery + 
                      genre.Romance + genre.Sci.Fi + genre.Sport +
                      genre.Thriller + genre.War + genre.Western,
                    data = ratings)
m.pooled.sd <- lm(sd_pooled ~ FLead + mean_pooled +
                    kim_violence_gore + kim_sex_nudity + kim_language +
                    major +
                    log(bom_opening_theaters) +
                    genres.count + 
                    as.factor(bom_year) + as.factor(bom_open_month) +
                    genre.Action + genre.Adventure + genre.Animation + genre.Biography + genre.Comedy +
                    genre.Crime + genre.Drama + genre.Family + genre.Fantasy +
                    genre.History + genre.Horror + genre.Music + genre.Musical + genre.Mystery + 
                    genre.News + genre.Romance + genre.Sci.Fi + genre.Short + genre.Sport +
                    genre.Thriller + genre.War + genre.Western ,
                  data = ratings) 
m.pooled.skew <- lm(skew_pooled ~ FLead + mean_pooled + sd_pooled  + 
                      kim_violence_gore + kim_sex_nudity + kim_language +
                      major +
                      log(bom_opening_theaters) +
                      genres.count + 
                      as.factor(bom_year) + as.factor(bom_open_month) +
                      genre.Action + genre.Adventure + genre.Animation + genre.Biography + genre.Comedy +
                      genre.Crime + genre.Drama + genre.Family + genre.Fantasy +
                      genre.History + genre.Horror + genre.Music + genre.Musical + genre.Mystery + 
                      genre.News + genre.Romance + genre.Sci.Fi + genre.Short + genre.Sport +
                      genre.Thriller + genre.War + genre.Western,
                    data = ratings)
stargazer(m.pooled.mean, m.pooled.sd, m.pooled.skew, omit.stat=c("f", "ser"), omit = c("bom_year", "bom_open_month", "genre\\."), type="text")
# LaTex output for paper
# TABLE 2
stargazer(m.pooled.mean, m.pooled.sd, m.pooled.skew,
          title = "Effect of lead gender on pooled ratings. The unit of analysis is a movie.",
          label = "tab:reg_pooled",
          order = c("^FLeadTRUE$",
                    "^mean_pooled$",
                    "^sd_pooled$",
                    "^kim_violence_gore$", "^kim_sex_nudity$", "^kim_language$",
                    "^majorTRUE$",
                    "^log\\(bom_opening_theaters\\)$",
                    "^genres.count$"),
          covariate.labels = c("Female lead actor",
                               "Ratings mean",
                               "Ratings stdev",
                               "Amount of violence", "Amount of sex/nudity", "Amount of profane language",
                               "Major studio (0/1)",
                               "log(Opening theaters)",
                               "Total genres"),
          header = FALSE, no.space = TRUE,
          omit.stat = c("f", "ser"),
          omit = c("bom_year", "bom_open_month", "genre\\."),
          omit.labels = c("Release year indicators", "Release month indicators", "Genre indicators"),
          dep.var.labels = c("Mean", "Stdev", "Skewness"),
          star.cutoffs = NA, notes = "Standard errors are in parentheses.", notes.append = F,
          type = "latex")

##################################
# TABLE 3
# SUR models for paper; must drop genre.News, genre.Short, genre.Western; SUR won't run without variance
# Mean SUR
m1.men.mean <- mean_males ~ FLead + 
  kim_violence_gore + kim_sex_nudity + kim_language +
  major +
  log(bom_opening_theaters) +
  genres.count + 
  as.factor(bom_year) + as.factor(bom_open_month) +
  genre.Action + genre.Adventure + genre.Animation + genre.Biography + genre.Comedy +
  genre.Crime + genre.Drama + genre.Family + genre.Fantasy +
  genre.History + genre.Horror + genre.Music + genre.Musical + genre.Mystery + 
  genre.Romance + genre.Sci.Fi + genre.Sport +
  genre.Thriller + genre.War
m2.women.mean <- mean_females ~ FLead + 
  kim_violence_gore + kim_sex_nudity + kim_language +
  major +
  log(bom_opening_theaters) +
  genres.count + 
  as.factor(bom_year) + as.factor(bom_open_month) +
  genre.Action + genre.Adventure + genre.Animation + genre.Biography + genre.Comedy +
  genre.Crime + genre.Drama + genre.Family + genre.Fantasy +
  genre.History + genre.Horror + genre.Music + genre.Musical + genre.Mystery + 
  genre.Romance + genre.Sci.Fi + genre.Sport +
  genre.Thriller + genre.War

fitsur.means <- systemfit(list(maleAudience = m1.men.mean, femaleAudience = m2.women.mean),
                          method = "SUR",
                          data = ratings)

# Comparison of means coefficients
restriction1 <- "maleAudience_FLeadTRUE =
        femaleAudience_FLeadTRUE"
fitsur.mean.test.FLeadTRUE <- linearHypothesis(fitsur.means, restriction1, test = "Chisq")

# Stdev SUR
m1.men.sd <- sd_males ~ FLead + mean_males + 
  kim_violence_gore + kim_sex_nudity + kim_language +
  major +
  log(bom_opening_theaters) +
  genres.count + 
  as.factor(bom_year) + as.factor(bom_open_month) +
  genre.Action + genre.Adventure + genre.Animation + genre.Biography + genre.Comedy +
  genre.Crime + genre.Drama + genre.Family + genre.Fantasy +
  genre.History + genre.Horror + genre.Music + genre.Musical + genre.Mystery + 
  genre.Romance + genre.Sci.Fi + genre.Sport + 
  genre.Thriller + genre.War
m2.women.sd <- sd_females ~ FLead + mean_females +
  kim_violence_gore + kim_sex_nudity + kim_language +
  major +
  log(bom_opening_theaters) +
  genres.count + 
  as.factor(bom_year) + as.factor(bom_open_month) +
  genre.Action + genre.Adventure + genre.Animation + genre.Biography + genre.Comedy +
  genre.Crime + genre.Drama + genre.Family + genre.Fantasy +
  genre.History + genre.Horror + genre.Music + genre.Musical + genre.Mystery + 
  genre.Romance + genre.Sci.Fi + genre.Sport +
  genre.Thriller + genre.War

fitsur.sd <- systemfit(list(maleAudience = m1.men.sd, femaleAudience = m2.women.sd),
                       method = "SUR",
                       data = ratings)

# Comparison of stdev coefficients
restriction1 <- "maleAudience_FLeadTRUE =
        femaleAudience_FLeadTRUE"
fitsur.sd.test.FLeadTRUE <- linearHypothesis(fitsur.sd, restriction1, test = "Chisq")

# Skew SUR
m1.men.skew <- skew_males ~ FLead + mean_males + sd_males + 
  kim_violence_gore + kim_sex_nudity + kim_language +
  major +
  log(bom_opening_theaters) +
  genres.count + 
  as.factor(bom_year) + as.factor(bom_open_month) +
  genre.Action + genre.Adventure + genre.Animation + genre.Biography + genre.Comedy +
  genre.Crime + genre.Drama + genre.Family + genre.Fantasy +
  genre.History + genre.Horror + genre.Music + genre.Musical + genre.Mystery + 
  genre.Romance + genre.Sci.Fi + genre.Sport +
  genre.Thriller + genre.War
m2.women.skew <- skew_females ~ FLead + mean_females + sd_females +
  kim_violence_gore + kim_sex_nudity + kim_language +
  major +
  log(bom_opening_theaters) +
  genres.count + 
  as.factor(bom_year) + as.factor(bom_open_month) +
  genre.Action + genre.Adventure + genre.Animation + genre.Biography + genre.Comedy +
  genre.Crime + genre.Drama + genre.Family + genre.Fantasy +
  genre.History + genre.Horror + genre.Music + genre.Musical + genre.Mystery + 
  genre.Romance + genre.Sci.Fi + genre.Sport +
  genre.Thriller + genre.War

fitsur.skew <- systemfit(list(maleAudience = m1.men.skew, femaleAudience = m2.women.skew),
                         method = "SUR",
                         data = ratings)

# Comparison of skew coefficients
restriction1 <- "maleAudience_FLeadTRUE =
        femaleAudience_FLeadTRUE"
fitsur.skew.test.FLeadTRUE <- linearHypothesis(fitsur.skew, restriction1, test = "Chisq")

# SUR output table for paper # Table 3
texreg(list(fitsur.means, fitsur.sd, fitsur.skew),
       label = "tab:split_audience",
       caption = "Seemingly unrelated regressions predicting rating outcomes from female and male audiences.",
       custom.header = list("Means" = 1:2, "Stdev" = 3:4, "Skew" = 5:6),
       omit.coef = "(genre\\.)|(bom_year)|(bom_open_month)",
       custom.gof.rows = list("Genre FEs" = rep("Yes", 6),
                              "Year FEs" = rep("Yes", 6),
                              "Month FEs" = rep("Yes", 6)),
       custom.coef.map = list(
         FLeadTRUE = "Female lead",
         mean_males = "Rating mean (M)",
         mean_females = "Rating mean (F)",
         sd_males = "Rating stdev (M)",
         sd_females = "Rating stdev (F)",
         kim_violence_gore = "Violence",
         kim_sex_nudity = "Sex/nudity",
         kim_language = "Language",
         majorTRUE = "Major studio",
         "log(bom_opening_theaters)" = "log(Theaters)",
         "genres.count" = "Total genres",
         "(Intercept)" = "Constant"),
       stars = numeric(0),
       custom.note = "\\emph{Note}: Standard errors are in parentheses.",
       beside = TRUE, doctype = F, caption.above = T,
       digits = 3,
       booktabs = T, use.packages = F,
       longtable = F)


##################################
# TABLE 4
# restrict to mediocre titles
nrow(ratings[mean_pooled > 5 & mean_pooled < 7.5])

fit.ltail <- lm(Ltail*100 ~ FLead + major + mean_pooled +
                  kim_violence_gore + kim_sex_nudity + kim_language +
                  log(bom_opening_theaters) +
                  genres.count + 
                  as.factor(bom_year) + as.factor(bom_open_month) +
                  genre.Action + genre.Adventure + genre.Animation + genre.Biography + genre.Comedy +
                  genre.Crime + genre.Drama + genre.Family + genre.Fantasy +
                  genre.History + genre.Horror + genre.Music + genre.Musical + genre.Mystery + 
                  genre.News + genre.Romance + genre.Sci.Fi + genre.Short + genre.Sport +
                  genre.Thriller + genre.War + genre.Western,
                data = ratings[mean_pooled>5 & mean_pooled<7.5])
fit.rtail <- lm(Rtail*100 ~ FLead + major + mean_pooled +
                  kim_violence_gore + kim_sex_nudity + kim_language +
                  log(bom_opening_theaters) +
                  genres.count + 
                  as.factor(bom_year) + as.factor(bom_open_month) +
                  genre.Action + genre.Adventure + genre.Animation + genre.Biography + genre.Comedy +
                  genre.Crime + genre.Drama + genre.Family + genre.Fantasy +
                  genre.History + genre.Horror + genre.Music + genre.Musical + genre.Mystery + 
                  genre.Romance + genre.Sci.Fi + genre.Sport +
                  genre.Thriller + genre.War,
                data = ratings[mean_pooled>5 & mean_pooled<7.5])
fit.female_count <- lm(log(women_all_count) ~ FLead*major + mean_pooled +
                         kim_violence_gore + kim_sex_nudity + kim_language +
                         log(bom_opening_theaters) +
                         genres.count + 
                         as.factor(bom_year) + as.factor(bom_open_month) +
                         genre.Action + genre.Adventure + genre.Animation + genre.Biography + genre.Comedy +
                         genre.Crime + genre.Drama + genre.Family + genre.Fantasy +
                         genre.History + genre.Horror + genre.Music + genre.Musical + genre.Mystery + 
                         genre.Romance + genre.Sci.Fi + genre.Sport +
                         genre.Thriller + genre.War,
                       data = ratings[mean_pooled>5 & mean_pooled<7.5])
fit.male_count <- lm(log(men_all_count) ~ FLead*major + mean_pooled +
                       kim_violence_gore + kim_sex_nudity + kim_language +
                       log(bom_opening_theaters) +
                       genres.count + 
                       as.factor(bom_year) + as.factor(bom_open_month) +
                       genre.Action + genre.Adventure + genre.Animation + genre.Biography + genre.Comedy +
                       genre.Crime + genre.Drama + genre.Family + genre.Fantasy +
                       genre.History + genre.Horror + genre.Music + genre.Musical + genre.Mystery + 
                       genre.Romance + genre.Sci.Fi + genre.Sport +
                       genre.Thriller + genre.War,
                     data = ratings[mean_pooled>5 & mean_pooled<7.5])
fit.BO_tix <- lm(log(tickets_sold_est) ~  FLead*major + mean_pooled + 
                   kim_violence_gore + kim_sex_nudity + kim_language +
                   log(bom_opening_theaters) + 
                   genres.count + 
                   as.factor(bom_year) + as.factor(bom_open_month) +
                   genre.Action + genre.Adventure + genre.Animation + genre.Biography + genre.Comedy +
                   genre.Crime + genre.Drama + genre.Family + genre.Fantasy +
                   genre.History + genre.Horror + genre.Music + genre.Musical + genre.Mystery + 
                   genre.Romance + genre.Sci.Fi + genre.Sport +
                   genre.Thriller + genre.War,
                 data = ratings[mean_pooled>5 & mean_pooled<7.5])

# Test of the addition of the two coefficients; stat for the text of the paper
linearHypothesis(fit.BO_tix, 'FLeadTRUE + FLeadTRUE:majorTRUE = 0')
linearHypothesis(fit.female_count, 'FLeadTRUE + FLeadTRUE:majorTRUE = 0')
linearHypothesis(fit.male_count, 'FLeadTRUE + FLeadTRUE:majorTRUE = 0')

stargazer(fit.ltail, fit.rtail, fit.BO_tix, fit.female_count, fit.male_count,
          omit.stat=c("f", "ser"), omit = c("bom_year", "bom_open_month", "genre\\."),type="text")


stargazer(fit.ltail, fit.rtail, fit.BO_tix, fit.female_count, fit.male_count,
          title = paste0("Effect of lead gender and studio type for ``mediocre'' movies (rated between 5 and 7.5 stars). These movies represent ", round(nrow(ratings[mean_pooled>5 & mean_pooled<7.5])/nrow(ratings), digits = 3) * 100, "\\% of the sample."),
          label = "tab:reg_tails_bo",
          order = c("^FLeadTRUE$",
                    "^majorTRUE$",
                    "^FLeadTRUE:majorTRUE$",
                    "^mean_pooled$",
                    "^kim_violence_gore$", "^kim_sex_nudity$", "^kim_language$",
                    "^log\\(bom_opening_theaters\\)$",
                    "^genres.count$"),
          covariate.labels = c("Female lead actor",
                               "Major studio (0/1)",
                               "Female × Major",
                               "Ratings mean",
                               "Violence", "Sex/nudity", "Profane language",
                               "log(Opening theaters)",
                               "Total genres"),
          header = FALSE, no.space = TRUE,
          omit.stat = c("f", "ser"),
          omit = c("bom_year", "bom_open_month", "genre\\."),
          omit.labels = c("Release year indicators", "Release month indicators", "Genre indicators"),
          dep.var.labels = c("Left tail", "Right tail", "Tickets sold (log)", "Female rating count", "Male rating count"),
          star.cutoffs = NA, notes = "Standard errors are in parentheses.", notes.append = F,
          column.sep.width = "-5pt",
          type = "latex")


##################################