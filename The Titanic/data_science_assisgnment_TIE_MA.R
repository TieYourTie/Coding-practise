#Tie Ma
#101310917


#cleaning the environment.
rm( list = ls())


#loading the package
library(tidyverse)
library(dplyr)
library(fpp2)
library(glmnet)
library(tidyr)
library(lmtest)
library(boot)
library(forecast)
library(readr)
library(ggfortify)
library(tseries)
library(urca)
library(readxl)
library( lubridate)
library(tsbox)
library(RColorBrewer)
library(wesanderson)
library(writexl)
library(gridExtra)
library(vars)
library(leaps)
library(broom)
library(fastDummies)
library(car)
library(randomForest)
library(caret)
####################
#The R package her just copy and past from my undergraduate assignment 
#As where all my command I know come from

##################

#setting the saving addreess
setwd("/Users/tie/Documents/GitHub/The-data-analysis-job--")

######################
#Step one: Clean the data
######################
  Titanic_train_raw <- read_csv("Titanic data/train.csv", 
                                col_types = cols( Name = col_skip(), Ticket = col_skip(), 
                                                 Cabin = col_skip()))
  
#At here I skiped their name which does not impact the model training, 
#their ticket number with is not important and their cabin which has limit number


print(colSums(is.na(Titanic_train_raw)))

################ The age. 
#I notice there are some age part are empty so i decided to use mean of age to fill the gap. 
#but before that I just need to check the distribution of age between the survived and died


#check the age of average by the survive or not 
mean_age_by_survival <- aggregate(Age ~ Survived, data = Titanic_train_raw, mean)

#print(mean_age_by_survival)
  #the average for people who died: 30.45
  #the average for people who does survived: 28.54


#calcuate the average of all the age 
The_average_age <- mean(Titanic_train_raw$Age, na.rm = TRUE)
#print(The_average_age)
  #the average age for the trainning data set is 29.69
  #they are close enough, now fill the age.

#fill the NA in the age variable

Titanic_train_raw$Age <- replace(Titanic_train_raw$Age, is.na(Titanic_train_raw$Age), mean(Titanic_train_raw$Age, na.rm = TRUE))
Titanic_train_without_age_NA <- Titanic_train_raw

#head()
#head(Titanic_train_without_age_NA)

#check again
print(colSums(is.na(Titanic_train_without_age_NA)))


#delete the 2 line in the embarked (since they are NA)
Titanic_train_cleaned <- na.omit(Titanic_train_without_age_NA)


#transfer the gender into the dummy variable. 
#For this model I will using sex_female
  # = 1: means female
  # = 0.: means male

TTD<- dummy_cols(Titanic_train_cleaned, select_columns = "Sex", remove_selected_columns = TRUE)



######################
#step two: check the data

par(mfrow = c(3, 2))
#the grpahy per class
barplot(table(TTD$Survived, TTD$Pclass), beside = TRUE, 
        main = "Survived by Pclass", xlab = "Pclass", ylab = "Count")
legend("top", legend = c("0: Died", "1: Survived"), fill = c("black", "white"))


# Age Boxplot
boxplot(Age ~ Survived, data = TTD, 
        col = c("orange", "gray"), 
        main = "Survived by Age", 
        xlab = "Survived", ylab = "Age")
legend("top", legend = c("0: Died", "1: Survived"), fill = c("orange", "gray"))


# SibSp
survival_table_sibsp <- table(TTD$Survived, TTD$SibSp)
barplot(survival_table_sibsp, beside = TRUE, col = c("black", "gray"), 
        main = "Survived by SibSp", xlab = "SibSp", ylab = "Count")
legend("topright", legend = c("0: Died", "1: Survived"), fill = c("black", "gray"))


# The parch
stripchart(Parch ~ Survived, data = TTD, vertical = TRUE, method = "jitter", 
           pch = 20, col = c("black", "gray"),
           main = " Survived by parch",
           xlab = "Survived", ylab = "Parch")
legend("top", legend = c("0: Died", "1: Survived"), fill = c("black", "gray"))

# The fare(The price of ticket)
boxplot(Fare ~ Survived, data = TTD, 
        col = c("orange", "gray"), 
        main = "Survived by Fare",
        xlab = "Survived", ylab = "Fare")
legend("top", legend = c("0: Died", "1: Survived"), fill = c("black", "gray"))

#The sex

female_table <- table(TTD$Sex_female, TTD$Survived)
barplot(female_table, beside = TRUE, col = c("black", "gray"),
        main = "Survived by Sex", xlab = "Sex", ylab = "Count",
        legend = F)
legend("center", legend = c("MALE", "Female"), fill = c("black", "gray"))
dev.off()


######## Test of the heteroscedasticity.
#crate a new matrix 
x <- TTD%>%
  dplyr::select(
    Pclass, Age, SibSp, Parch, Fare, Embarked, Sex_female
  ) %>%
  data.matrix()

#take out the survive
live <- TTD$Survived

#put two things together as data frame
data2 <- data.frame(x, live)

#The BP test
data_test_1 <- lm(live ~ . , data = data2)
bptest(data_test_1)

#Since the p value is 0.19 which is greater than 0.05, 
#we fail to reject the null hypothesis. we state that there is not enough statisticallty significant 
#evidence of heteroskedasticity exist in the data set.

####### The Multicollinearity
vif(data_test_1)
#All the VIF values for the variable are between 1 and 1.6. This indicates that 
#the training data show moderate correlation, which is common for real-life examples, and will not raise a red flag.


######################
#The linear regression and logistic regression model
######################


######################
#Step two: Sharking method
  #best subset selection 
  #Lasso regression
#####################
#using the best sub selection to find the best variable choose
#create a new matrix

x <- TTD %>%
  dplyr::select(
    Pclass, Age, SibSp, Parch, Fare, Embarked, Sex_female
  ) %>%
  data.matrix()

#take out the survived data.
live <- TTD$Survived

#The data frame time!
data2 <- data.frame(x, live)

#using regression to search all the possible output
regfit_all <- regsubsets(live ~ ., data = data2, nvmax = 10)

#take out the summary
reg_summary <- summary(regfit_all)

# ploting the result. 
plot(regfit_all, scale = "bic")

########## using the sharking method to determined the optimal output for the number of variable

par(mfrow=c(1,3))
# Plot RSS
plot(reg_summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")

# Plot BIC
plot(reg_summary$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
m.bic <- which.min(reg_summary$bic) # Find the index of minimum BIC
points(m.bic, reg_summary$bic[m.bic], col = "red", cex = 2, pch = 20) # Highlight the min point

# Plot Cp
plot(reg_summary$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
m.cp <- which.min(reg_summary$cp) # Find the index of minimum Cp
points(m.cp, reg_summary$cp[m.cp], col = "red", cex = 2, pch = 20) # Highlight the min point

#####################

#According to the best sub selection, I should choose two groups as the following variable
# the first group
  #Pclass, Age, sibsp, Sex_female
# the second group
  #Pclass, Age, sibsp, Embarked, Sex_female 



########## The Ridge Regression Time ##########

# using the 5 fold cross validation to find the optimal lambda
ridge.cv <- cv.glmnet(x, live, alpha = 0, nfolds = 5) 
# alpha = 0 Ridge regression
# alpha = 1 the lasso regressiom.

# input the cross validation
coef_ridge <- as.vector(coef(ridge.cv, s = "lambda.min")[-1]) 

#doing a linear regressuion and take out the OLS
ols_mode <- lm(live ~ ., data = data2)

#take everything except the intercept
coef_ols <- coef(ols_mode)[-1]  

# Create a data frame for comparison
variable_names <- names(coef_ols)


#put all the result together.
performence_table <- data.frame(
  Variable = variable_names,
  OLS = coef_ols,
  Ridge = coef_ridge)


# print the output table.
print(performence_table)

 

############ OLS model time
# the first group
#Pclass, Age, sibsp, Sex_female
# the second group
#Pclass, Age, sibsp, Embarked, Sex_female 


#####################
#model compare (both linear regression and logic model)
#####################

##############################
#3.The model comparsing
#TTD

#cut the date in the 2 part
#select 90% for the training
#select 10% for the testing
#note: I was planning to do 5 fold cross validation and R told me that I cannot do it.
      #so I just do a simple cut the data and test. 



#calculate how many row in the TTD data set 
cut_data<- sample(1:nrow(TTD), size = 9/10* nrow(TTD), replace = FALSE)

#select the 90% of the data set for the training
training_set <- TTD[cut_data, ]
#rest of 10% for the test 
testing_set <- TTD[-cut_data, ]

# Fit the logistic regression model
model1 <- lm(Survived ~ Pclass + Sex_female + Age + SibSp, data = training_set)
model2 <- glm(Survived ~ Pclass + Sex_female + Age + SibSp, family = binomial, data = training_set)
model3 <- lm(Survived ~ Pclass + Sex_female + Age + SibSp + Embarked, data = training_set)
model4 <- glm(Survived ~ Pclass + Sex_female + Age + SibSp + Embarked, family = binomial, data = training_set)


# Making predictions on the test set for each model
model1_pred <- predict(model1, newdata = testing_set)
model2_pred <- predict(model2, newdata = testing_set, type = "response")
model3_pred <- predict(model3, newdata = testing_set)
model4_pred <- predict(model4, newdata = testing_set, type = "response")


# Actual values
test_time <- testing_set$Survived


# Calculate MSE, RMSE, and MAE for each group
#for the model_1
model1_MSE <- mean((test_time -model1_pred)^2)
model1_RMSE <- sqrt(mean((test_time -model1_pred)^2))
model1_MAE <- mean(abs(test_time -model1_pred))

#for the model_2
model2_MSE <- mean((test_time -model2_pred)^2)
model2_RMSE <- sqrt(mean((test_time -model2_pred)^2))
model2_MAE <- mean(abs(test_time -model2_pred))

#for the model_3
model3_MSE <- mean((test_time -model3_pred)^2)
model3_RMSE <- sqrt(mean((test_time -model3_pred)^2))
model3_MAE <- mean(abs(test_time -model3_pred))

#for the model_4
model4_MSE <- mean((test_time -model4_pred)^2)
model4_RMSE <- sqrt(mean((test_time -model4_pred)^2))
model4_MAE <- mean(abs(test_time -model4_pred))

#kaggle scale
kaggle_scale <- c(0.76794, 0.75385, 0.77033,0.77272)

# The final result 
The_final_table <- data.frame(
  Test = c("MSE", "RMSE", "MAE", "Kaggle Scale"),
  Model_1_lm = c(model1_MSE, model1_RMSE, model1_MAE, kaggle_scale[1]),
  Model_1_glm = c(model2_MSE, model2_RMSE, model2_MAE, kaggle_scale[2]),
  Model_2_lm = c(model3_MSE, model3_RMSE, model3_MAE, kaggle_scale[3]),
  Model_2_glm = c(model4_MSE, model4_RMSE, model4_MAE, kaggle_scale[4])
)

# Print the final result
print(The_final_table)


#####################
#prediction time!
#####################

#Note : At here I just writing code for generated one predictions only.
      #but it also can be used to generate other prediction.
      
########### 

#constratined the model.
Winning_model_1_lm<- lm(Survived ~ Pclass + Sex_female + Age + SibSp + Embarked, data =TTD)
winning_Model_1_glm<- lm(Survived ~ Pclass + Sex_female + Age + SibSp + Embarked, data =TTD)
winning_Model_2_lm <- lm(Survived ~ Pclass + Sex_female + Age + SibSp data =TTD)
winning_Model_2_glm <- lm(Survived ~ Pclass + Sex_female + Age + SibSp, data =TTD)


#import hte data test data set. 
test_data_set <- read_csv("Titanic data/test.csv", 
                 col_types = cols(Name = col_skip(), Parch = col_skip(), 
                                  Ticket = col_skip(), Fare = col_skip(), 
                                  Cabin = col_skip()))

#check the data 
#head(test_data_set)

#check if there any NA here
print(colSums(is.na(test_data_set)))
#the age got 86 NA. 

#because both regression is training base on the average so I will using average
#age to fill the

#calcuate the average of all the age 
The_age_information <- summary(test_data_set$Age, na.rm = TRUE)
print(The_age_information)
#the average age is 30.27


# Fill in missing Age values with the mean Age, ensuring NA values are handled
test_data_set$Age <- replace(test_data_set$Age, is.na(test_data_set$Age), mean(test_data_set$Age, na.rm = TRUE))


#Transfer the sex to dummy variable 
test_data_ultra<- dummy_cols(test_data_set, select_columns = "Sex", remove_selected_columns = TRUE)



# Prediction!
prediction <- predict(Winning_model, newdata = test_data_ultra)

# transfer the probability to the 0 and 1 into the data set
predicted_01<- ifelse(prediction > 0.5, 1, 0)
#if the probability is greater than 0.5, output 1
#if the probbability is smaller than 0.5, output 0 

# write the prediction in to the test data swer
test_data_ultra$Predicted_Survived <- predicted_01


The_final_test <- data.frame(PassengerId = test_data_ultra$PassengerId, 
                             Survived = test_data_ultra$Predicted_Survived)

# Save to CSV
write.csv(The_final_test, "The_final_test_lm_2 .csv", row.names = FALSE)




#model 2: the randomForest
############
############data import and cleaning
#clean the enviroment and import the date
rm( list = ls())

#setting the saving addreess
setwd("/Users/tie/Documents/GitHub/The-data-analysis-job--")

#####################
#Step one: Clean the data
#####################
Titanic_train_raw <- read_csv("Titanic data/train.csv", 
                              col_types = cols( Name = col_skip(), Ticket = col_skip(), 
                                                Cabin = col_skip()))

#At here I skiped their name which does not impact the model training, 
#their ticket number with is not important and their cabin which has limit number

################ The age.

######### fill the age

#calcuate the average age
The_average_age <- mean(Titanic_train_raw$Age, na.rm = TRUE)

#fill the NA age 
Titanic_train_raw$Age <- replace(Titanic_train_raw$Age, is.na(Titanic_train_raw$Age), mean(Titanic_train_raw$Age, na.rm = TRUE))

#outupt it into the aerage
Titanic_train_without_age_NA <- Titanic_train_raw

#remove towo empty line of embraketd 
Titanic_train_cleaned <- na.omit(Titanic_train_without_age_NA)

#we got the final data
TTD <- dummy_cols(Titanic_train_cleaned, select_columns = "Sex", remove_selected_columns = TRUE)


#transfer the survived into factor 
TTD$Survived <- as.factor(TTD$Survived) 

#using the 10 ford cross validation to determine the number mtry win the 
control <- trainControl(method="cv", number=10, search="grid", allowParallel = TRUE) 

#check the all mtry from 1 to 10
ntreeGrid <- expand.grid(mtry=1:10)


#doing the model test
model_test <- train(Survived ~ ., data=TTD, method="rf", trControl=control, tuneGrid=ntreeGrid)

print(model_test)

# the optimual vaule for the mtry is 3
rf_model <- randomForest(Survived ~ ., data = TTD, 
                         ntree = 10000,  #how many 
                         mtry = 3, #the optimal 
                         nodesize = 1, #1 for classification (die or live)
                         importance = TRUE)
print(rf_model)

###
# prodce the forcast
####

setwd("/Users/tie/Documents/GitHub/The-data-analysis-job--")

# Import test datast
test_data_set <- read_csv("Titanic data/test.csv",col_types = cols( Name = col_skip(), Ticket = col_skip(), 
                                                                    Cabin = col_skip()))

# Fill in missing Age with the average ag
test_data_set$Age <- replace(test_data_set$Age, is.na(test_data_set$Age), mean(test_data_set$Age, na.rm = TRUE))


#Transfer the test set into the dummy variable
test_data_ultra<- dummy_cols(test_data_set, select_columns = "Sex", remove_selected_columns = TRUE)

#generate the prediction 
prediction <- predict(rf_model, newdata = test_data_ultra, type = "response")

# transfer it into 0 and 1, because the random forest are give 1 and 2 as output.
bi_prediction <- as.integer(prediction) - 1

#put the Binary prediction into the test data set.
test_data_ultra$Survived <- bi_prediction

#the final output!
final_output_evil <- data.frame(PassengerId = test_data_ultra$PassengerId, Survived = test_data_ultra$Survived)

#write its down
write.csv(final_output_evil, "final_predictions_rf.csv", row.names = FALSE)


