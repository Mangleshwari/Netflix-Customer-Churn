##import_library

library(ggplot2)
library(patchwork)
library(dplyr)
library(caTools)
library(car)
library(corrplot)
library(scales)
library(caret)
library(glmnet)
library(pROC)
library(randomForest)

##import_data_set

data = read.csv(file.choose(), header = T)
attach(data)

##Exploratory_Data_Analysis(EDA)##

#general_overview

head(data)
summary(data)
any(is.na(data))
str(data)
churn_rate=table(churned)*100/length(churned)

#encoding

data$gender_encoded <- as.numeric(factor(gender, levels = c("Male", "Female", "Other")))-1
subscription_type_factor=as.factor(subscription_type)
data$subscription_type_encoded <- as.numeric(factor(subscription_type_factor, levels = c("Basic", "Premium", "Standard")))-1
data$region_encoded <- as.numeric(factor(region, levels = c("Africa", "Asia", "Europe", "North America", "Oceania", "South America")))-1
data$device_encoded <- as.numeric(factor(device, levels = c("Desktop", "Laptop", "Mobile", "Tablet", "TV")))-1


#uni_variate_analysis

churned_factor=as.factor(churned)
monthly_fee_factor=as.factor(monthly_fee)
number_of_profiles_factor = as.factor(number_of_profiles)
par(mfrow = c(2,2))
boxplot(watch_hours~churned_factor, main = "Watch Hours", xlab = "churned factor", ylab = "watch hours") #users with low watch time are churning more
boxplot(last_login_days~churned_factor, main = "Last Login Days", xlab = "churned factor", ylab = "last login days") #churned users have longer gaps since their last login
boxplot(data$age~churned_factor, main = "Age", xlab = "churned factor", ylab = "age")
boxplot(avg_watch_time_per_day~churned_factor, main = "Avg Watch Hours Per Day", xlab = "churned factor", ylab = "avg watch hour per day")
par(mfrow = c(1,1))
ggplot(data, aes(subscription_type))+ geom_bar(fill = "steelblue")+geom_text(stat = "count", aes(label = after_stat(count), vjust = -.5))+labs(title = "Bar Plot of Subscription Type", x = "Subscription Type", y = "Number of Users")+theme_minimal(base_size = 14)
ggplot(data, aes(gender))+ geom_bar(fill = "steelblue")+geom_text(stat = "count", aes(label = after_stat(count), vjust = -.5))+labs(title = "Bar Plot of Gender", x = "Gender", y = "Number of Users")+theme_minimal(base_size = 14) #balanced representation across genders
ggplot(data, aes(region))+ geom_bar(fill = "steelblue")+geom_text(stat = "count", aes(label = after_stat(count), vjust = -.5))+labs(title = "Bar Plot of Region", x = "Region", y = "Number of Users")+theme_minimal(base_size = 14)
ggplot(data, aes(device))+ geom_bar(fill = "steelblue")+geom_text(stat = "count", aes(label = after_stat(count), vjust = -.5))+labs(title = "Bar Plot of Device", x = "Device", y = "Number of Users")+theme_minimal(base_size = 14)
ggplot(data, aes(payment_method))+ geom_bar(fill = "steelblue")+geom_text(stat = "count", aes(label = after_stat(count), vjust = -.5))+labs(title = "Bar Plot of Payment Method", x = "Payment Method", y = "Number of Users")+theme_minimal(base_size = 14)
ggplot(data, aes(monthly_fee))+ geom_bar(fill = "steelblue")+geom_text(stat = "count", aes(label = after_stat(count), vjust = -.5))+labs(title = "Bar Plot of Monthly Fee", x = "Monthly Fee", y = "Number of Users")+theme_minimal(base_size = 14)
ggplot(data, aes(number_of_profiles))+ geom_bar(fill = "steelblue")+geom_text(stat = "count", aes(label = after_stat(count), vjust = -.5))+labs(title = "Bar Plot of Number of Profiles", x = "Number of Profiles", y = "Number of Users")+theme_minimal(base_size = 14)
ggplot(data, aes(favorite_genre))+ geom_bar(fill = "steelblue")+geom_text(stat = "count", aes(label = after_stat(count), vjust = -.5))+labs(title = "Bar Plot of Favorite Genre", x = "Favorite Genre", y = "Number of Users")+theme_minimal(base_size = 14)
ggplot(data, aes(churned))+ geom_bar(fill = "steelblue")+geom_text(stat = "count", aes(label = after_stat(count), vjust = -.5))+labs(title = "Bar Plot of Number of Customer Churned", x = "No. of Customer Churned", y = "Number of Users")+theme_minimal(base_size = 14)
ggplot(data, aes(age)) + geom_histogram(binwidth = 5, fill = "steelblue", color = "white") + labs(title = "Histogram of Age", x = "Age", y = "Count")+theme_minimal()
ggplot(data, aes(x = watch_hours)) + geom_histogram(binwidth = 5, fill = "steelblue", color = "white") + labs(title = "Histogram of Watch Hours", x = "Watch Hours", y = "Count")+theme_minimal()
ggplot(data, aes(x = avg_watch_time_per_day)) + geom_histogram(binwidth = .5, fill = "steelblue", color = "white") + labs(title = "Histogram of Average Watch Time Per Day", x = "Avg Watch Time Per Day", y = "Count")+theme_minimal()


#churned_vs_non_churned_customers
p1 = ggplot(data, aes(x = churned_factor, fill = gender)) +geom_bar(position = "stack", width = 0.5)+labs(title = "Gender vs Churned", x = "Churned", y = "Number of Users", fill = "Gender")
p2 = ggplot(data, aes(x = churned_factor, fill = subscription_type)) +geom_bar(position = "stack", width = 0.5)+labs(title = "Subscription type vs Churned", x = "Churned", y = "Number of Users", fill = "Subscription Type")
p3 = ggplot(data, aes(x = churned_factor, fill = region)) +geom_bar(position = "stack", width = 0.5)+labs(title = "Region vs Churned", x = "Churned", y = "Number of Users", fill = "Region")
p4 = ggplot(data, aes(x = churned_factor, fill = device)) +geom_bar(position = "stack", width = 0.2)+labs(title = "Device vs Churned", x = "Churned", y = "Number of Users", fill = "Device")
p5 = ggplot(data, aes(x = churned_factor, fill = monthly_fee_factor)) +geom_bar(position = "stack", width = 0.2)+labs(title = "Monthly fee vs Churned", x = "Churned", y = "Number of Users", fill = "Monthly fee")
p6 = ggplot(data, aes(x = churned_factor, fill = favorite_genre)) +geom_bar(position = "stack", width = 0.2)+labs(title = "Favorite Genre vs Churned", x = "Churned", y = "Number of Users", fill = "Favorite Genre")
(p1|p2|p3)/(p4|p5|p6)

##Hypothesis Testing

#to check whether the mean watch hours are the same for churned and non-churned customers.
t.test(watch_hours~churned) #Ho rejected

#to check whether mean monthly fee is the same for churned and non-churned customers.
t.test(monthly_fee~churned)  #Ho rejected

#to check whether there is no association between churn and subscription type
chisq.test(table(churned_factor, subscription_type_factor))  #Ho rejected

#to test whether avg watch hours significantly differs by subscription type
anova1=aov(watch_hours~subscription_type_factor) 
summary(anova1) #Ho accepted

#to test whether monthly fee significantly differs by churn
anova2=aov(churned~monthly_fee)
summary(anova2) #Ho rejected

# Splitting the dataset
split <- sample.split(data$churned, SplitRatio = 0.8)
train_reg <- subset(data, split == TRUE)
test_reg <- subset(data, split == FALSE)

# Logistic Regression Model
p7 = ggplot(data, aes(x = watch_hours)) + geom_histogram(binwidth = 5, fill = "steelblue", colour = "white")+labs(title = "Histogram of Watch Hours", x = "watch hours", y = "Count") +
  theme_minimal()
data$log_watch_hours = log1p(data$watch_hours)
p8 = ggplot(data, aes(x = log_watch_hours)) +
  geom_histogram(binwidth = 0.2, fill = "steelblue", color = "white") +
  labs(title = "Histogram of Log-Transformed Watch Hours", x = "log(Watch Hours + 1)", y = "Count") +
  theme_minimal()
p7|p8
###model_log <- glm(churned ~ log_watch_hours + other_features, data = data, family = "binomial")
train_reg$churned <- as.factor(train_reg$churned)
train_reg$log_watch_hours <- log1p(train_reg$watch_hours)
test_reg$log_watch_hours <- log1p(test_reg$watch_hours)
logistic_model <- glm(churned ~ monthly_fee + log_watch_hours, family = binomial, data = train_reg)
summary(logistic_model)
# Ridge regression
library(glmnet)
x = model.matrix(churned ~ watch_hours + monthly_fee, data = train_reg)[, -1]
y = train_reg$churned
ridge_model <- cv.glmnet(x, y, family = "binomial", alpha = 0)
coef(ridge_model, s = "lambda.min")
# Predicting on test set

predicted_prob <- predict(logistic_model, newdata = test_reg, type = "response")
predict_reg <- as.data.frame(predicted_prob)
head(predict_reg)

# 1. Converting predicted probabilities to binary class labels
predicted_class <- ifelse(predict_reg$predicted_prob > 0.5, 1, 0)

# 2. Actual labels from test set
actual_class <- test_reg$churned

# 3. Confusion Matrix and Metrics
library(caret)
conf_matrix <- confusionMatrix(as.factor(predicted_class), as.factor(actual_class))
print(conf_matrix)

# 4. Extract Accuracy, Precision, Recall, F1 Score
accuracy <- conf_matrix$overall['Accuracy']
precision <- conf_matrix$byClass['Precision']
recall <- conf_matrix$byClass['Recall']
f1 <- conf_matrix$byClass['F1']

# 5. ROC Curve and AUC
library(pROC)
roc_obj <- roc(actual_class, predict_reg$predicted_prob)
plot(roc_obj, col = "blue", main = "ROC Curve")
auc_value <- auc(roc_obj)
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lwd = 2)


rf_model <- randomForest(churned ~ ., data = train_reg)
importance(rf_model)

# 1. Predict class labels on test set
rf_pred_class <- predict(rf_model, newdata = test_reg)

# 2. Predict probabilities for ROC and AUC
rf_pred_prob <- predict(rf_model, newdata = test_reg, type = "prob")[, 2]

# 3. Actual labels from test set
actual_class_rf <- test_reg$churned

# 4. Confusion Matrix and Metrics
conf_matrix_rf <- confusionMatrix(as.factor(rf_pred_class), as.factor(actual_class_rf))
corrplot(cor_matrix, method = "color", addCoef.col = "black", tl.cex = 0.8, tl.srt = 45, mar = c(0, 0, 1, 0), number.cex = 0.7, cl.cex = 0.8,)
print(conf_matrix_rf)

# 5. Extract Accuracy, Precision, Recall, F1 Score
accuracy_rf <- conf_matrix_rf$overall['Accuracy']
precision_rf <- conf_matrix_rf$byClass['Precision']
recall_rf <- conf_matrix_rf$byClass['Recall']
f1_rf <- conf_matrix_rf$byClass['F1']

# 6. ROC Curve and AUC
library(pROC)
roc_obj_rf <- roc(actual_class_rf, rf_pred_prob)
plot(roc_obj_rf, col = "darkgreen", main = "ROC Curve - Random Forest")
auc_rf <- auc(roc_obj_rf)
legend("bottomright", legend = paste("AUC =", round(auc_rf, 3)), col = "darkgreen", lwd = 2)





cat("Logistic Regression:\n")
cat("Accuracy:", round(accuracy, 3), "\n")
cat("Precision:", round(precision, 3), "\n")
cat("Recall:", round(recall, 3), "\n")
cat("F1 Score:", round(f1, 3), "\n")
cat("AUC:", round(auc_value, 3), "\n\n")

cat("Random Forest:\n")
cat("Accuracy:", round(accuracy_rf, 3), "\n")
cat("Precision:", round(precision_rf, 3), "\n")
cat("Recall:", round(recall_rf, 3), "\n")
cat("F1 Score:", round(f1_rf, 3), "\n")
cat("AUC:", round(auc_rf, 3), "\n")

