##import_library

library(ggplot2)
library(dplyr)
library(caTools)
library(car)
library(corrplot)

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
boxplot(watch_hours~churned_factor, main = "Watch Hours") #users with low watch time are churning more
boxplot(last_login_days~churned_factor, main = "Last Login Days") #churned users have longer gaps since their last login
boxplot(age~churned_factor, main = "Age")
boxplot(avg_watch_time_per_day~churned_factor, main = "Avg Watch Hours Per Day")
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
ggplot(data, aes(x = watch_hours)) + geom_histogram(binwidth = .5, fill = "steelblue", color = "white") + labs(title = "Histogram of Watch Hours", x = "Watch Hours", y = "Count")+theme_minimal()
ggplot(data, aes(x = avg_watch_time_per_day)) + geom_histogram(binwidth = .5, fill = "steelblue", color = "white") + labs(title = "Histogram of Average Watch Time Per Day", x = "Avg Watch Time Per Day", y = "Count")+theme_minimal()


#churned_vs_non_churned_customers

ggplot(data, aes(x = churned_factor, fill = gender)) +geom_bar(position = "stack", width = 0.5)+labs(title = "Gender vs Churned", x = "Churned", y = "Number of Users", fill = "Gender")
ggplot(data, aes(x = churned_factor, fill = subscription_type)) +geom_bar(position = "stack", width = 0.5)+labs(title = "Subscription type vs Churned", x = "Churned", y = "Number of Users", fill = "Subscription Type")
ggplot(data, aes(x = churned_factor, fill = region)) +geom_bar(position = "stack", width = 0.5)+labs(title = "Region vs Churned", x = "Churned", y = "Number of Users", fill = "Region")
ggplot(data, aes(x = churned_factor, fill = device)) +geom_bar(position = "stack", width = 0.2)+labs(title = "Device vs Churned", x = "Churned", y = "Number of Users", fill = "Device")
ggplot(data, aes(x = churned_factor, fill = monthly_fee_factor)) +geom_bar(position = "stack", width = 0.2)+labs(title = "Monthly fee vs Churned", x = "Churned", y = "Number of Users", fill = "Monthly fee")
ggplot(data, aes(x = churned_factor, fill = favorite_genre)) +geom_bar(position = "stack", width = 0.2)+labs(title = "Favorite Genre vs Churned", x = "Churned", y = "Number of Users", fill = "Favorite Genre")

#correlation_heatmap

num_data=data[sapply(data, is.numeric)]
cor_matrix <- cor(num_data, use = "complete.obs")
corrplot(cor_matrix, method = "color", addCoef.col = "black", main = "correlation heatmap")

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

#Splitting_the_data_set

split = sample.split(data, SplitRatio = 0.8)
train_reg = subset(data, split == "TRUE")
test_reg = subset (data, split == "FALSE")

#multi_collinearity_test
#Building_the_model

logistic_model = glm(churned_factor ~ monthly_fee_factor + watch_hours, family = binomial, data = train_reg)
summary(logistic_model)
predicted_prob=predict(test_reg, type = "response")
predict_reg <- as.data.frame(predict_reg)
predict_reg
