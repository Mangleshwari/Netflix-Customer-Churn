##import_library
library(ggplot2)
library(dplyr)
library(caTools)
library(car)

##import_data_set

data = read.csv(file.choose(), header = T)
attach(data)

##Exploratory_Data_Analysis(EDA)##
#general overview

head(data)
summary(data)
any(is.na(data))
str(data)
table(churned)

#uni_variate_analysis

ggplot(data, aes(subscription_type))+ geom_bar(fill = "steelblue")+geom_text(stat = "count", aes(label = after_stat(count), vjust = -.5))+labs(title = "Bar Plot of Subscription Type", x = "Subscription Type", y = "Number of Users")+theme_minimal(base_size = 14)
ggplot(data, aes(gender))+ geom_bar(fill = "steelblue")+geom_text(stat = "count", aes(label = after_stat(count), vjust = -.5))+labs(title = "Bar Plot of Gender", x = "Gender", y = "Number of Users")+theme_minimal(base_size = 14)
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

#churned vs. non churned customers
churned_factor=as.factor(churned)
ggplot(data, aes(x = churned_factor, fill = gender)) +geom_bar(position = "stack", width = 0.5)+labs(title = "Gender vs Churned", x = "Churned", y = "Number of Users", fill = "Gender")
ggplot(data, aes(x = churned_factor, fill = subscription_type)) +geom_bar(position = "stack", width = 0.5)+labs(title = "Subscription type vs Churned", x = "Churned", y = "Number of Users", fill = "Subscription Type")
ggplot(data, aes(x = churned_factor, fill = region)) +geom_bar(position = "stack", width = 0.5)+labs(title = "Region vs Churned", x = "Churned", y = "Number of Users", fill = "Region")
ggplot(data, aes(x = churned_factor, fill = device)) +geom_bar(position = "stack", width = 0.2)+labs(title = "Device vs Churned", x = "Churned", y = "Number of Users", fill = "Device")
monthly_fee_factor=as.factor(monthly_fee)
ggplot(data, aes(x = churned_factor, fill = monthly_fee_factor)) +geom_bar(position = "stack", width = 0.2)+labs(title = "Monthly fee vs Churned", x = "Churned", y = "Number of Users", fill = "Monthly fee")
number_of_profiles_factor = as.factor(number_of_profiles)
ggplot(data, aes(x = churned_factor, fill = favorite_genre)) +geom_bar(position = "stack", width = 0.2)+labs(title = "Favorite Genre vs Churned", x = "Churned", y = "Number of Users", fill = "Favorite Genre")


##ML_Algorithms

#Splitting_the_data_set
split = sample.split(data, SplitRatio = 0.8)
train_reg = subset(data, split == "TRUE")
test_reg = subset (data, split == "FALSE")
#multi_collinearity_test
#Building_the_model
logistic_model = glm(churned_factor ~ monthly_fee_factor + subscription_type_factor + watch_hours, family = binomial)
logistic_model


#use anova for comparision of model
monthly_fee_factor

