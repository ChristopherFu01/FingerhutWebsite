# Data Analysis of Fingerhut Website Customer Traffic Behavior from Q4 2020 to Q3 2023

## Table of Contents

TO-DO

## Executive Summary

  In this report, we aim to determine flaws with the current Fingerhut website design and provide feedback for possible revisions to address these issues.
  Using Fingerhut’s ideal journey for labeling customers’ journeys, we performed binary classification using Logistic Regression, Random Forest, and K-Means Clustering with Classification. From the results of all models, we observe that a revised ideal journey would include the following stages: “First Purchase,” “Prospecting,” “Downpayment,” and “Orders Shipped.” The “Apply for Credit” stage lacks the same impact as these stages, thus requiring a revision to its individual events.
  We designed four combinations of predictor variables to ultimately understand which variable(s) heavily impact the response variable “Order Shipped.” We isolated predictor variables such as “Downpayment” and “Credit Account” and used “binary” and “values” datasets, subsequently performing Logistic Regression to confirm that “Downpayment” and “Credit Account” are highly correlated with “Order Shipped.” Overall, we analyzed the feature importance of our predictors in the Logistic Regression models we implemented. We ultimately determined that a customer deviating from the ideal was more likely to have an order shipped had they prospected more frequently than other customers. Furthermore, we were able to use the timestamp variables to find hotspots in the customer interaction cycle such that Fingerhut can maximize the percentage of customers that complete their credit rehabilitation journey and graduate to the next program “Fetti” which includes a revolving line of credit.
  To enhance customer conversion rate, specifically by encouraging customers to proceed from one milestone to the next one, we focus on predicting the time for the milestones, enabling us to customize incentives such as email and promotions accordingly. Our approach suggests that the LSTM (Long Short-Term Memory) model with the time elapsed and journey_steps_until end variable for prior milestones as predictors is a viable method for forecasting the time for the next milestone. While acknowledging the limitations in model accuracy, we believe that our approach serves as a solid foundation for developing more personalized engagement strategies.

## Acknowledgments

### Team Acknowledgment

I would like to extend my gratitude to the following members who contributed to the project:
- Charles Barnes
- Han Qiang
- Zach Godick

### Data Acknowledgment

The data used in this project was provided by Fingerhut, an American catalog/online retailer.

## Abstract

  This report examines the Fingerhut dataset. It is split into three questions, with each section having its own preprocessing steps, models, and results. We compare and contrast multiple models to draw conclusions about these questions as well as discover additional insights. Through this process, we aim to evaluate the model performance based on several metrics to see how valid and applicable the model results are. We also discuss different techniques including vectorization, supervised machine learning models, unsupervised machine learning models, and statistical modeling.

## Introduction

  Using machine learning models and statistical modeling that we have learned from our data theory courses at UCLA, we hoped to guide Fingerhut in optimal journeys taken by customers and provide suggestions. In this project, we aim to explore flaws inherent in Fingerhut’s website as well as customer journeys and provide recommendations to resolve such issues.
  This larger thematic focus was split into three smaller questions to provide the most context: (1) How optimal is Fingerhut’s ideal path for customer journeys? (2) How can we maximize a customer achieving the “Order shipped” milestone using their interaction with intermediate steps? (3) How can we maximize the customer conversion rate by predicting customer behaviors?
  We began with two .csv files: export.csv (the main Fingerhut dataset) and Event Definitions.csv. In export.csv, there are 64,911,906 entries for customers described by 7 columns: customer_id (the customer ID), account_id (the account ID), ed_id (the event ID), event_name (name of the event), event_timestamp (when the event occurred), and journey_steps_until_end (an integer count of which journey step the customer is on).

**Figure 0: Fingerhut Dataset**

## <u>**Question 1**</u>

### 1. Preprocessing Steps

#### 1.1 Exploratory Data Analysis

  Prior to data cleaning, we performed an initial analysis of the data to investigate our question in greater detail. There are a total of 7 named stages in addition to an unknown stage(s) for two events regarding promotions (event ID 1) and email clicks (event ID 24). The stages are as follows in no particular order: “Apply for Credit,” “Credit Account,” “Discover,” “Downpayment,” “First Purchase,” “Order Shipped,” “Prospecting,” and an 8th stage we will refer to as “Miscellaneous.”
  In Figure 1(a), we observe how many customers go through each of the stages over time, with the highest activity being “First Purchase” and the lowest activity being “Downpayment.” Observing how much activity occurs stage by stage can provide us insight into which stages are being ignored and require the most assistance.
  In Figure 1(b), we see a small sample of customers’ journeys and how they compare to each other. It seems that “First Purchase” yields the longest journey length whereas customers would change from stage to stage very frequently.
  In Figure 1(c), a heat matrix shows how closely similar these paths are to each other. We see that there are a few identical paths between different customers and a decent amount of paths with at least around 75% similarity. This suggests that a certain journey is preferable by multiple customers, which could help us confirm the validity of Fingerhut’s ideal path or provide a revision so that the Fingerhut website can yield more customers.

**Figure 1(a-c): Exploratory Data Analysis**

#### 1.2 Data Cleaning

  To clean the data, we first performed a left merge of the main dataset and the event definitions dataset on the event IDs to have each entry represented with a stage. We noticed that rows containing event IDs 1 and 24 (those being “promotion_created” and “campaignemail_clicked” respectively) had no associated stage; as a fix, we assigned them an arbitrary stage called “misc.” Since this question primarily focuses on stages and events, we only kept the following columns: the customer ID, the event ID (since event name and ID were semantically identical while the latter is visually easier to identify), and the journey_steps_until_end variable. One crucial assumption we made in the next step was to assume that a new journey for a customer restarted when the journey_steps_until_end reset to 1. We first cataloged the number of journeys taken for each customer, as indicated by a reset in the journey_steps_until_end variable, then combined all the event IDs and stages into their own lists for each journey per customer.

**Table 1: Customers with Labeled Multiple Journeys, and Stage and Path as Lists**

Performing EDA on our modified dataset provides a few insights:
  (1) The majority of customers stayed within the “First Purchase” stage, followed by “Apply for Credit” stage, while other stages saw diminished participation in other stages. Even if we take into account that some stages have more associated events than others (e.g. 8 events within “Apply for Credit” versus 5 events within “Discovery”), there appears to be a significant discrepancy in the ratio between the earlier stages (“Apply for Credit” and “First Purchase”) and the terminating stages (especially “Order Shipped”).
  (2) We see that the first journey had outliers with the longest journey lengths; surprisingly, the second journey also had outliers with long journey lengths. Otherwise, the boxplots indicate that the journey lengths were relatively on par according to the IQR of each plot. Note in the top-right figure that some journeys were labeled as the “0th” journey, indicating that our metric is flawed; however, the majority of journeys lie within the 1st journey.
  (3) The majority of journeys have lengths of less than approximately 250 steps. See the summary statistics for a more discrete observation of journey lengths.
  (4) Multiple journeys are rather uncommon; the majority of customers only did one journey, a few had a second journey, and a very tiny minority ever did more than three journeys total.

**Figure 2: EDA on Multiple Journeys**

**Table 2: Summary Statistics of Multiple Journeys**

#### 1.3 Feature Engineering

After cleaning the data, we begin with labeling stages in accordance with how “ideal” they were. In the case of testing how customers progress through stages, we used a metric of comparing a customer’s entire journey in stages to Fingerhut’s ideal stage route: (1) Apply for Credit (2) First Purchase (3) Down Payment (4) Order Shipped. A score is assigned to how many of these stages are reached in that specific order, then labeled with a 1 for “ideal” only if all four stages happen to be present within the customer’s stages.

**Figure 3: Imbalanced Dataset for Ideal and Non-Ideal Paths**

  Our initial labeling shows that the dataset is imbalanced with 317,592 ideal paths and 1,501,408 non-ideal paths (for a total of 1,819,000 labels), requiring us to balance the dataset. Based on our binary labeling of the paths, we see that roughly 21.15% of total journeys were considered ideal.
  Due to hardware constraints (i.e. the kernel continued to crash with memory-intensive actions), we were unable to balance the dataset in its rawest form with the labels without pre-processing. However, implementing pre-processing enables us to use techniques such as oversampling to balance out the dataset without encountering issues. We first converted the list of stages into a string version of the entire list, then applied a vectorizer (CountVectorizer) to turn these lists into numerical vectors that can be inputted into our models. The result is twelve different features representing the different stages. The table below shows each stage and its corresponding features:

**Table 3: Stages and Features**

After performing an 80/20 train-test split of the dataset, we proceeded to apply oversampling on the training data for the minority class (in this case, “Ideal” or 1’s).
Given how high-dimensional the data is, we employed Principal Component Analysis (PCA) as our dimension reduction technique for specific models to reduce the high-dimensional features down to lower-dimensional features, which helped accelerate and bolster model performance.

#### 1.4 Modeling

For best predicting ideal paths, we used three different models and evaluated their performance.

##### 1.4.1 Logistic Regression

Logistic Regression is a common supervised classification technique that can be applied to the scoring system we used for labeling ideal journeys. However, fitting the labels without further modification leads to this technique performing too well [1]. To counter the issue of overfitting, we applied L2 regularization as well as hyperparameter tuning, specifically for the inverse of regularization strength C; decreasing C will increase the regularization strength of the model. Hyperparameter tuning for the optimal C value produced C = 10. The results are shown below in Figure 4 and Table 4:

**Figure 4: ROC Curve for Logistic Regression (C = 10)**

**Table 4: Performance Metrics for Logistic Regression (L: Baseline, R: Best)**

##### 1.4.2 Random Forest

  Random Forest is made up of an ensemble of decision trees which all contribute to the final prediction. Examining the different decision paths for the features can provide insight into how the model predicts the ideal path, whether it be the ordering, length, or both regarding the list of stages.
  The Random Forest model is also able to assess the importance of different features from our pre-processed data in relation to the stages. We see that Features 1, 5, 9, and 12 have the highest values, which suggests that “Apply for Credit”, “First Purchase”, “Down Payment”, and “Order Shipped” are correlated when it comes to the ideal path; it seems that “First Purchase” and “Down Payment” yield the greatest importance.
  Pruning controls complexity and prevents overfitting by restricting the model’s ability to memorize training data. However, it appears that pruning does not have the largest effect on model performance.

**Figure 5: Random Forest Decision Tree Model**

**Table 5: Feature Importances**

**Table 6: Model Performance for Random Forest**

##### 1.4.3 K-Means Clustering — K-means Classifier

  In this approach, we will combine an unsupervised model with a supervised model. K-means clustering will contribute to greater dimensionality reduction and help improve model performance in the classification task. We will also use a metric called the silhouette score to evaluate the quality of the clusters produced by K-means, which measures how similar objects are to their own clusters (cohesion) versus other clusters (separation). Scores with magnitude closer to 1 indicate proper matching to clusters. In addition, we will use PCA to further reduce dimensionality [2].
  We will first apply K-means clustering without using dimension reduction on the techniques for our baseline model. Using an arbitrary choice of two clusters and fitting the K-means model onto our training data to produce cluster labels, we then train our Logistic Regression model onto our labels. We collect an accuracy score from our combined model, then obtain the silhouette score of the model based on the training data.
  For our best model, we find the cumulative explained variance ratio to determine the number of PCA components to use; 2 components explain at least 95% variance, which will be the optimal number of PCA components we use. We then apply the elbow method to determine the optimal number of K Clusters to use; past 3 clusters is when the graph begins to plateau, meaning the optimal number of K-clusters we use is 3. We proceed to scale the training and testing data for our inputs, apply PCA onto our inputs, and repeat the same steps as the baseline model; a grid search for hyperparameter-tuning for Logistic Regression shows that C = 10 is the most optimal. The following results are summarized by the figures and tables below:

**Figure 6: Determining the Optimal Number of PCA components and K-Clusters for Model**

**Figure 7: PCA Components for All Features**

**Figure 8: Classification Reports and Confusion Matrices for K-Means Cluster Model (L: Baseline, R: Best)**

**Figure 9: ROC Curve for K-Means Clustering with Logistic Regression Classifier (Before and After Optimization)**

**Table 7: Model Performance for K-Means Clustering with Logistic Regression Classifier**

**Figure 10: Visualization of 3 K-Means Clusters with 2 Principal Components**

##### 1.5 Results
  We observe that the supervised learning models performed too well (with approximately 100% accuracy), indicating that the data is too imbalanced, either due to noise or other inherent issues with the data, despite using optimization techniques to account for this issue, leading to biased model results; most of the improvements did not have a discernible effect on the model performance compared to the baseline.   However, the Random Forest model’s insights into feature importance, in particular how “First Purchase” and “Down Payment” are significant in a customer journey being classified as “Ideal.”
However, unsupervised learning models seem to perform more appropriately at the cost of lower accuracy. Combining an unsupervised model with a supervised model provided the best result; in the case of K-means clustering with the Logistic Regression classifier, a significant increase in the testing accuracy at the slight expense of the silhouette score suggests that objects were mostly successful in matching to the correct clusters. The clustering shape appears triangular, which may suggest uneven density and an underlying complexity that is not fully captured; however, model performance seems to be sufficient enough to have results be interpretable.
  Of most importance from our combined model are the resulting PCA components. For the 1st component, the following stages showed the greatest variations: “Apply for Credit,” “First Purchase,” “Downpayment,” and “Order Shipped;” for the 2nd component, the following stages showed the greatest variance: “Apply for Credit,” and “Prospecting”. In addition to the stages listed as being “Ideal” by Fingerhut, surprisingly the “Prospecting” stage has a significant effect when it comes to customer journeys. This may suggest that a more ideal stage would include “Prospecting” as one of the stages.

## References
[1] Subramanian, Jyothi, and Richard Simon. "Overfitting in prediction models–is it a problem only in high dimensions?." Contemporary clinical trials 36.2 (2013): 636-641
[2] Czarnowski, Ireneusz. "Cluster-based instance selection for machine classification." Knowledge and Information Systems 30 (2012): 113-133.
[3] Fingerhut, www.fingerhut.com/content/OurStory. Accessed 18 Mar. 2024.
