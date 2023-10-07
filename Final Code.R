# Prediction of Netflix Movies

# Data Preprocessing
# Install and load tidyverse package
install.packages("tidyverse")
library(tidyverse)
library(class)

# Install and load corrplot package
install.packages("corrplot")
library(corrplot)

# Install and load olsrr package
install.packages("olsrr")
library(olsrr)

# Install and load smotefamily package
install.packages("smotefamily")
library(smotefamily)

# Install dummies and load package
install.packages("dummies", repos = NULL, type="source")
library(dummies)

# Install and load e1071 package
install.packages("e1071")
library(e1071)

# Install and load rpart
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)

# Install and load neural net
install.packages("neuralnet")
library(neuralnet)
install.packages("lattice")
library(lattice)

# Set working directory to folder
setwd("C:/Users/ual-laptop/OneDrive - University of Arizona/Desktop/545 project")

# Read CSV file into a tibble and define column types
NetflixUSA <- read_csv(file = "netflixMovies.csv",
                       col_types = "cccfilcinnlf",
                       col_names = TRUE)

# Data Preprocessing ------------------------------------------------------
# Converting NetfilxUSA into a data frame
NetflixUSADataFrame <- data.frame(NetflixUSA)

# Converting age certification into dummies
NetflixUSA1 <- as_tibble(dummy.data.frame(data = NetflixUSADataFrame,
                                          names = "age_certification"))

# Converting NetflixUSA1 into a data frame
NetflixUSADataFrame2 <- data.frame(NetflixUSA1)

# Converting release decade into dummies
NetflixUSA2 <- as_tibble(dummy.data.frame(data = NetflixUSADataFrame2,
                                          names = "release_decade"))

# Remove unneccesary variables
NetflixUSAMain<- subset(NetflixUSA2, select = -c(id,
                                                 title,
                                                 description,
                                                 imdb_id,
                                                 age_certificationPG,
                                                 release_decade1960s))

# Removing outliers
NetflixUSAMain <- NetflixUSAMain %>%
  mutate(max1 = quantile(imdb_votes, .75) + (1.5 * IQR(imdb_votes))) %>%
  mutate(max2 = quantile(tmdb_score, .75) + (1.5 * IQR(tmdb_score))) %>%
  mutate(max3 = quantile(runtime, .75) + (1.5 * IQR(runtime))) %>%
  mutate(max4 = quantile(tmdb_popularity, .75) +
           (1.5 *IQR(tmdb_popularity))) %>%
  filter(imdb_votes <= max1) %>%
  filter(tmdb_score <= max2) %>%
  filter(runtime <= max3) %>%
  filter(tmdb_popularity <= max4) %>%
  select(-max1,-max2,-max3,-max4)

# Creating a function to display histograms
displayAllHistograms <- function(tibbleDataset) {
  tibbleDataset %>%
    keep(is.numeric) %>%
    gather() %>%
    ggplot() + geom_histogram(mapping = aes(x=value,fill=key),
                              color = "black")+
    facet_wrap( ~ key,scales= "free")+
    theme_minimal()
}

# Display the histogram of the tibble
displayAllHistograms(NetflixUSAMain)

# Display a correlation matrix rounded to two decimals
round(cor(NetflixUSAMain),digits = 2)

# Display of a cleaner correlation plot
corrplot(cor(NetflixUSAMain),
         method = "number",
         type = "lower",
         number.cex = 0.6,
         tl.cex = 0.6)

# Logistic Regression
# Splitting data into training and test data sets
set.seed(369)
sampleSetLog <- sample(nrow(NetflixUSAMain),
                    round(nrow(NetflixUSAMain)*.75),
                    replace = FALSE)


NetflixTrainingLog <- NetflixUSAMain[sampleSetLog, ]
NetflixTestLog <- NetflixUSAMain[-sampleSetLog, ]

# Display the test data set and convert variables into logical
summary(NetflixTestLog)

NetflixTestLog<-NetflixTestLog%>%
  mutate(hit = as.logical(hit),
         age_certificationPG.13 = as.logical(age_certificationPG.13),
         age_certificationG = as.logical(age_certificationG),
         age_certificationR = as.logical(age_certificationR),
         top_three = as.logical(top_three),
         age_certificationNC.17 = as.logical(age_certificationNC.17),
         release_decade2020s = as.logical(release_decade2020s),
         release_decade2010s = as.logical(release_decade2010s),
         release_decade1990s = as.logical(release_decade1990s),
         release_decade2000s = as.logical(release_decade2000s),
         release_decade1980s = as.logical(release_decade1980s),
         release_decade1970s = as.logical(release_decade1970s))

# Checking for class imbalance
summary(NetflixTrainingLog$hit)

# Dealing with class imbalance in training set using SMOTE function 
NetflixTrainingLogSmoted <- tibble(SMOTE(
  X=data.frame(NetflixTrainingLog),
  target = NetflixTrainingLog$hit,
  dup_size = 2)$data)

# Display the tibble after dealing with class imbalance 
summary(NetflixTrainingLogSmoted)

# Convert variable in training dataset into logical type
NetflixTrainingLogSmoted <- NetflixTrainingLogSmoted %>%
  mutate(hit = as.logical(hit),
         age_certificationPG.13 = as.logical(age_certificationPG.13),
         age_certificationG = as.logical(age_certificationG),
         age_certificationR = as.logical(age_certificationR),
         top_three = as.logical(top_three),
         age_certificationNC.17 = as.logical(age_certificationNC.17),
         release_decade2020s = as.logical(release_decade2020s),
         release_decade2010s = as.logical(release_decade2010s),
         release_decade1990s = as.logical(release_decade1990s),
         release_decade2000s = as.logical(release_decade2000s),
         release_decade1980s = as.logical(release_decade1980s),
         release_decade1970s = as.logical(release_decade1970s))

# Display Display the tibble after converting into logical type 
summary(NetflixTrainingLogSmoted)

# Get rid of "class" column in tibble (added by SMOTE())
NetflixTrainingLogSmoted <- NetflixTrainingLogSmoted %>%
  select(-class)

# Check for class imbalance in the training set
summary(NetflixTrainingLogSmoted)

# Generate logistic regression Model 
NetflixUSALogModel<- glm(data=NetflixTrainingLogSmoted, family=binomial, 
                          formula=hit ~ .)

# Display the logistic model summary
summary(NetflixUSALogModel)

# Use the model to predict outcomes in the testing dataset
NetflixUSALogPrediction <- predict(NetflixUSALogModel,
                                   NetflixTestLog,
                                    type='response')

# Display the test model
print(NetflixUSALogPrediction)

# Converting less than 0.5 as 0 and greater than 0.5 as 1
NetflixUSALogPrediction <- 
  ifelse(NetflixUSALogPrediction >= 0.5,1,0)

# Creating a mobile phone confusion matrix
NetflixUSALogConfusionMatrix <- table(NetflixTestLog$hit,
                                      NetflixUSALogPrediction)

# Display confusion matrix
print(NetflixUSALogConfusionMatrix)

# Calculating false positive
NetflixUSALogConfusionMatrix[1,2]/
  (NetflixUSALogConfusionMatrix[1,2]+NetflixUSALogConfusionMatrix[1,1])

# Calculating false negative
NetflixUSALogConfusionMatrix[2,1]/
  (NetflixUSALogConfusionMatrix[2,1]+NetflixUSALogConfusionMatrix[2,2])

# Calculating Model Prediction Accuracy
sum(diag(NetflixUSALogConfusionMatrix))/ nrow(NetflixTestLog)




# K-Nearest
# Splitting the data into two groups
NetflixUSAK_Labels <- NetflixUSAMain %>% select(hit)
NetflixUSAK <- NetflixUSAMain %>% select(-hit)

# Splitting mobilephone data into training and test data sets
set.seed(69)
sampleSetK <- sample(nrow(NetflixUSAMain),
                    round(nrow(NetflixUSAMain)*.75),
                    replace = FALSE)

# Put the records from 75% training into Training tibbles
# Put the records from 25% into Testing tibbles
NetflixUSAK_Training <- NetflixUSAMain[sampleSetK, ]
NetflixUSAK_Test <- NetflixUSAMain[-sampleSetK, ]

NetflixUSAK_LabelsTraining <- NetflixUSAK_Labels[sampleSetK, ]
NetflixUSAK_LabelsTest <- NetflixUSAK_Labels[-sampleSetK, ]

# Generate the K-nearest Model
NetflixUSAK_Prediction <- knn(train = NetflixUSAK_Training,
                                test = NetflixUSAK_Test,
                                cl = NetflixUSAK_LabelsTraining$hit,
                                k = 17)

# Display the predictions from the testing data on the console
print(NetflixUSAK_Prediction)

# Display the summary of prediction from the testing dataset
print(summary(NetflixUSAK_Prediction))

# Evaluate the model by forming confusion matrix
NetflixUSAK_ConfusionMatrix <- table(NetflixUSAK_LabelsTest$hit,
                                     NetflixUSAK_Prediction)

# Display the confusion matrix
print(NetflixUSAK_ConfusionMatrix)

# Calculating false positive
NetflixUSAK_ConfusionMatrix[1,2]/
  (NetflixUSAK_ConfusionMatrix[1,2]+NetflixUSAK_ConfusionMatrix[1,1])

# Calculating false negative
NetflixUSAK_ConfusionMatrix[2,1]/
  (NetflixUSAK_ConfusionMatrix[2,1]+NetflixUSAK_ConfusionMatrix[2,2])

# Calculate the predictive accuracy model
predictiveAccuracyK <- sum(diag(NetflixUSAK_ConfusionMatrix))/
  nrow(NetflixUSAK_Test)

# Display the predictive accuracy
print(predictiveAccuracyK)

# Create a Kvalue matrix along with their predictive accuracy
KValueMatrix <- matrix(data = NA,
                       nrow = 0,
                       ncol= 2)

# Adding column headings
colnames(KValueMatrix) <- c("k value","Predictive Accuracy")

# Looping through different values of k with the training dataset
for (kValue in 1:nrow(NetflixUSAK_Training)){
  # Calculate prdictive accuracy only if k value is odd
  if (kValue %% 2 !=0) {
    # Generate the Model
    NetflixUSAK_Prediction <- knn(train = NetflixUSAK_Training,
                                  test = NetflixUSAK_Test,
                                  cl = NetflixUSAK_LabelsTraining$hit,
                                  k = kValue)
    # Generate the confusion matrix
    NetflixUSAK_ConfusionMatrix <- table(NetflixUSAK_LabelsTest$hit,
                                         NetflixUSAK_Prediction)
    
    # Calculate the predictive accuracy
    predictiveAccuracyK <- sum(diag(NetflixUSAK_ConfusionMatrix))/
      nrow(NetflixUSAK_Test)
    
    # Adding a new row
    KValueMatrix <- rbind(KValueMatrix, c(kValue,predictiveAccuracyK))
  }
}

# Display the kValue Matrix
print(KValueMatrix)



# Naive Bayes

# Binning
# Binning Runtime
breaks_runtime <- c(0, 12, 24, 36, 48, 60,
                    72, 84, 96, 108, 120, 132, 180)
tags_runtime <- c("[0-12)", "[12-24)", "[24-36)", "[36-48)",
                  "[48-60)", "[60-72)", "[72-84)",
                  "[84-96)", "[96-108)",
                  "[96-120)", "[120-132)", "[132+)")
NetflixUSAMainNaive <- NetflixUSAMain %>% mutate(runtime_binned =
                                              cut(NetflixUSAMain$runtime,
                                                  breaks = breaks_runtime,
                                                  include.lowest = TRUE,
                                                  right = FALSE,
                                                  labels= tags_runtime))

# Binning IMDB Votes
breaks_imdb_votes <- c(0, 20000, 40000, 60000, 80000, 100000,
                       120000, 140000, 160000, 180000, 300000)
tags_imdb_votes <- c("[0-20k)", "[20-40k)", "[40-60k)",
                     "[60-80k)", "[80-100k)", "[100-120k)",
                     "[120-140k)", "[140-160k)", "[160-180k)", "[180k+)")
NetflixUSAMainNaive <- NetflixUSAMainNaive %>% mutate(imdb_votes_binned =
                                              cut(NetflixUSAMainNaive$imdb_votes
                                                  ,breaks = breaks_imdb_votes,
                                                  include.lowest = TRUE,
                                                  right = FALSE,
                                                  labels= tags_imdb_votes))

# Binning TMDB Popularity
breaks_tmdb_popularity <- c(0, 5, 10, 15, 20, 25, 30, 35,
                            40, 45, 50, 55, 60, 65, 100)
tags_tmdb_popularity <- c("[0-5)", "[5-10)", "[10-15)", "[15-20)", "[20-25)",
                          "[25-30)", "[30-35)", "[35-40)", "[40-45)",
                          "[45-50)", "[50-55)", "[55-60)", "[60-65)",
                          "[65+")
NetflixUSAMainNaive <- NetflixUSAMainNaive %>% 
  mutate(tmdb_popularity_binned =cut(NetflixUSAMainNaive$tmdb_popularity,
                                     breaks = breaks_tmdb_popularity,
                                     include.lowest = TRUE,
                                     right = FALSE,
                                     labels= tags_tmdb_popularity))

# Binning TMDB Score
breaks_tmdb_score <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
tags_tmdb_score <- c("[0-1)", "[1-2)", "[2-3)",
                     "[3-4)", "[4-5)", "[5-6)",
                     "[6-7)", "[7-8)", "[8-9)", "[9-10)")
NetflixUSAMainNaive <- NetflixUSAMainNaive %>%
  mutate(tmdb_score_binned =
           cut(NetflixUSAMainNaive$tmdb_score,
               breaks = breaks_tmdb_score,
               include.lowest = TRUE,
               right = FALSE,
               labels= tags_tmdb_score))

# Remove the unbinned continuous variables
NetflixUSAMainNaive <- NetflixUSAMainNaive %>% select(-c(imdb_votes,
                                               runtime,
                                               tmdb_popularity,
                                               tmdb_score
))

# Splitting data into training and test data sets
set.seed(369)
sampleSetNaive <- sample(nrow(NetflixUSAMainNaive),
                    round(nrow(NetflixUSAMainNaive)*.75),
                    replace = FALSE)

NetflixUSANaiveTraining <- NetflixUSAMainNaive[sampleSetNaive, ]
NetflixUSANaiveTest <- NetflixUSAMainNaive[-sampleSetNaive, ]

# Generates the naive-Bayes model
netflixNaiveModel <- naiveBayes(formula = hit ~ .,
                           data = NetflixUSANaiveTraining,
                           laplace = 1)

# Builds the probabilities
netflixNaiveProbability <- predict(netflixNaiveModel,
                                   NetflixUSANaiveTest,
                              type = "raw")

# Displays the probabilities
netflixNaiveProbability

# Predicts classes for each record in the test dataset
netflixNaivePrediction <- predict(netflixNaiveModel,
                                  NetflixUSANaiveTest,
                             type = "class")

# Displays the predictions
netflixNaivePrediction

# Evaluates the model with a confusion matrix and shows it
netflixNaiveConfusionMatrix <- table(NetflixUSANaiveTest$hit,
                                     netflixNaivePrediction)
# Display Confusion Matrix
print(netflixNaiveConfusionMatrix)

# calculates false positive rate
netflixNaiveConfusionMatrix[1, 2] /
  (netflixNaiveConfusionMatrix[1, 2] +
     netflixNaiveConfusionMatrix[1, 1])

# Calculates false negative rate
netflixNaiveConfusionMatrix[2, 1] /
  (netflixNaiveConfusionMatrix[2, 1] +
     netflixNaiveConfusionMatrix[2, 2])

# Calculates and shows the model's predictive accuracy
predictiveAccuracyNaive <- sum(diag(netflixNaiveConfusionMatrix)) / 
  nrow(NetflixUSANaiveTest)

# Display Predictive Accuracy
print(predictiveAccuracyNaive)



# Decision Trees

# Splitting dataset into training and testing with 369 as random seed
set.seed(369)
sampleSetDT <- sample(nrow(NetflixUSAMain),
                    round(nrow(NetflixUSAMain) * 0.75),
                    replace = FALSE)

NetflixUSADTTraining <- NetflixUSAMain[sampleSetDT, ]
NetflixUSADTTesting <- NetflixUSAMain[-sampleSetDT, ]

# Display summary of testing dataset
summary(NetflixUSADTTesting)

# Part 1 ------------------------------------------------------------------
# Generate decision tree model to predict hit based on other variables in the
# dataset, use 0.01 as complexity parameter
netflixDecisionTreeModel <- rpart(formula = hit ~ .,
                                  method = "class",
                                  cp = 0.01,
                                  data = NetflixUSADTTraining)

# Display decision tree visualization
rpart.plot(netflixDecisionTreeModel)

# Predict classes for each record in testing dataset,
# store them in netflixPrediction
netflixDTPrediction <- predict(netflixDecisionTreeModel,
                               NetflixUSADTTesting,
                               type = "class")

# Display netflixPrediction in console
print(netflixDTPrediction)

# Evaluate the model by forming a confusion matrix
netflixDTConfusionMatrix <- table(NetflixUSADTTesting$hit,
                                  netflixDTPrediction)

# Display confusion matrix in console
print(netflixDTConfusionMatrix)

# Calculate the confusion matrix's false positive rate
netflixDTConfusionMatrix[1, 2] / (netflixDTConfusionMatrix[1, 2] +
                                    netflixDTConfusionMatrix[1, 1])

# Calculate the confusion matrix's false negative rate
netflixDTConfusionMatrix[2, 1] / (netflixDTConfusionMatrix[2, 1] +
                                    netflixDTConfusionMatrix[2, 2])

# Calculate model's predictive accuracy, store to variable: predictiveAccuracy
predictiveAccuracyDT <- sum(diag(netflixDTConfusionMatrix)) /
  nrow(NetflixUSADTTesting)

# Display predictive accuracy in console
print(predictiveAccuracyDT)

# Part 2 ------------------------------------------------------------------
# Create new decision tree model using 0.007 as complexity parameter

netflixSecondDecisionTreeModel <- rpart(formula = hit ~ .,
                                        method = "class",
                                        cp = 0.007,
                                        data = NetflixUSADTTraining)

# Display second decision tree visualization
rpart.plot(netflixSecondDecisionTreeModel)

# Predict classes for each record in testing dataset
netflixSecondPrediction <- predict(netflixSecondDecisionTreeModel,
                                   NetflixUSADTTesting,
                                   type = "class")

# Display second prediction in console
print(netflixSecondPrediction)

# Evaluate the second model by forming a confusion matrix
netflixSecondConfusionMatrix <- table(NetflixUSADTTesting$hit,
                                      netflixSecondPrediction)

# Display second confusion matrix in console
print(netflixSecondConfusionMatrix)

# Calculate the confusion matrix's false positive rate
netflixSecondConfusionMatrix[1, 2] / (netflixSecondConfusionMatrix[1, 2] + 
                                        netflixSecondConfusionMatrix[1, 1])

# Calculate the confusion matrix's false negative rate
netflixSecondConfusionMatrix[2, 1] / (netflixSecondConfusionMatrix[2, 1] +
                                        netflixSecondConfusionMatrix[2, 2])

# Calculate second model's predictive accuracy
predictiveAccuracySecond <- sum(diag(netflixSecondConfusionMatrix)) /
  nrow(NetflixUSADTTesting)

# Display second predictive accuracy in console
print(predictiveAccuracySecond)


# Neural Networks
# Splitting data into training and test data sets
set.seed(369)
sampleSetNN <- sample(nrow(NetflixUSAMain),
                    round(nrow(NetflixUSAMain)*.75),
                    replace = FALSE)

NetflixUSANNTraining <- NetflixUSAMain[sampleSetNN, ]
NetflixUSANNTest <- NetflixUSAMain[-sampleSetNN, ]

# Generate the neural network
netflixUSANeuralNet <- neuralnet(
  formula = hit ~ .,
  data = NetflixUSANNTraining,
  hidden = 3,
  act.fct = "logistic",
  linear.output = FALSE)

# Display the neural network numeric results
print(netflixUSANeuralNet$result.matrix)

# Visualize the neural network
plot(netflixUSANeuralNet)

# Use netflixMoviesNeuralNet to generate probabilities on the 
# netflixMoviesTesting data set and store this in netflixMoviesProbability
netflixUSANNProbability <- neuralnet::compute(netflixUSANeuralNet,
                                   NetflixUSANNTest)

# Display the predictions from the testing dataset on the console
print(netflixUSANNProbability)

# Convert probability predictions into 0/1 predictions and store this into 
# netflixMoviesPrediction
netflixUSNNPrediction <-
  ifelse(netflixUSANNProbability$net.result>0.5,1,0)

# Disply the predictions on the console
print(netflixUSNNPrediction)

# Evaluate the model by forming a confusion matrix
netflixUSANNConfusionMatrix <- table(NetflixUSANNTest$hit,
                                     netflixUSNNPrediction)

# Display the confusion matrix on the console
print(netflixUSANNConfusionMatrix)

# Calculate the confusion matrix's false positivity rate
netflixUSANNConfusionMatrix[1, 2] / (netflixUSANNConfusionMatrix[1, 2] +
                                       netflixUSANNConfusionMatrix[1, 1])

# Calculate the confusion matrix's false negativity rate
netflixUSANNConfusionMatrix[2, 1] / (netflixUSANNConfusionMatrix[2, 1] +
                                       netflixUSANNConfusionMatrix[2, 2])

# Calculate the model predictive accuracy
predictiveAccuracyNN <- sum(diag(netflixUSANNConfusionMatrix)) /
  nrow(NetflixUSANNTest)

# Print the predictive accuracy on the console
print(predictiveAccuracyNN)
