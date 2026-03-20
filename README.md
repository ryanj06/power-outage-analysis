# What Makes the Lights Stay Off? Analyzing Power Outage Duration in the U.S.
**by Ryan Jose**

## Introduction

This project analyzes major power outages across the United States from 2000 to 2016. The central question guiding this analysis is: **what factors are associated with longer power outage durations?**

Power outages are more than just an inconvenience. When the lights go out for days at a time, hospitals struggle to keep patients safe, small businesses lose revenue they cannot recover, and families are left without heat, refrigeration, or communication. Understanding what drives outage duration, whether it is the cause, the season, the region, or something else entirely, could help utilities respond faster and help policymakers invest in the right places before the next major event.

The dataset contains **1534 rows**, each representing one major outage event. The columns most relevant to this analysis are:

| Column | Description |
|--------|-------------|
| `outage_duration` | Duration of the outage in minutes |
| `cause_category` | General category of the outage cause (e.g. severe weather, intentional attack) |
| `customers_affected` | Number of customers who lost power |
| `climate_region` | U.S. climate region where the outage occurred |
| `month_name` | Month the outage began |
| `anomaly_level` | Oceanic El Niño/La Niña index at the time of the outage |
| `population` | Population of the state where the outage occurred |

## Data Cleaning and Exploratory Data Analysis

Several cleaning steps were necessary before analysis could begin. First, column names were standardized to lowercase with underscores for consistency. The `outage_start_date` and `outage_start_time` columns were combined into a single `outage_start` datetime column, and the same was done for the restoration columns. This makes it easier to work with time-based features and better reflects how the data was generated, as each outage has a single start and end time. Numeric columns stored as strings were converted to proper numeric types. An ordered categorical `month_name` column was created from the numeric month to preserve natural ordering. Finally, uninformative columns (`variables`, `obs`) were dropped.

After cleaning, the DataFrame has 1534 rows and 54 columns. The first few rows are shown below:

| year | month_name | u_s_state | cause_category | outage_duration | customers_affected | climate_region |
|------|------------|-----------|----------------|----------------|--------------------|----------------|
| 2011 | July | Minnesota | severe weather | 3060.0 | 70000.0 | East North Central |
| 2014 | May | Minnesota | intentional attack | 1.0 | NaN | East North Central |
| 2010 | October | Minnesota | severe weather | 3000.0 | 70000.0 | East North Central |
| 2012 | June | Minnesota | severe weather | 2550.0 | 68200.0 | East North Central |
| 2015 | July | Minnesota | severe weather | 1740.0 | 250000.0 | East North Central |

### Univariate Analysis

The distribution of outage duration is heavily right-skewed. The majority of outages are resolved within a few thousand minutes, but a small number of events last tens of thousands of minutes. This skew has important implications for modeling, as a handful of extreme events dominate the variance.

<iframe src="assets/uni1.html" width="800" height="450" frameborder="0"></iframe>

Severe weather is by far the most common cause of outages, accounting for nearly half of all events in the dataset. Intentional attacks are the second most common cause, though they tend to produce much shorter outages on average.

<iframe src="assets/uni2.html" width="800" height="450" frameborder="0"></iframe>

### Bivariate Analysis

There is a weak positive relationship between the number of customers affected and outage duration, with substantial variance throughout. This suggests that while larger outages tend to last longer, customer count alone is not a strong predictor of duration.

<iframe src="assets/bi1.html" width="800" height="450" frameborder="0"></iframe>

Severe weather and fuel supply emergencies produce the longest outages by a wide margin. Intentional attacks, despite being frequent, are typically resolved much more quickly, likely because they are easier to diagnose and repair than weather-related infrastructure damage.

<iframe src="assets/bi2.html" width="800" height="450" frameborder="0"></iframe>

### Interesting Aggregates

The table below shows the median outage duration in minutes broken down by cause category and climate region. Fuel supply emergencies in the East North Central region stand out with extremely long median durations, likely reflecting the region's dependence on fuel-based generation and the logistical difficulty of restoring supply during shortages.

<iframe src="assets/pivot.html" width="800" height="350" frameborder="0"></iframe>

## Assessment of Missingness

**NMAR Analysis:** The `demand_loss_mw` column is likely NMAR. Utilities may be less likely to report demand loss figures when the loss was very large or when the outage was caused by an intentional attack, for security or reputational reasons. Since the decision to report or withhold this value is plausibly tied to the value itself, this is consistent with NMAR. To potentially make it MAR, additional data such as mandatory NERC regulatory filings would be needed to explain why certain utilities chose not to report.

**Missingness Dependency:** The missingness of `customers_affected` was analyzed. This column is missing for 443 out of 1534 rows.

*Test 1: Does missingness of `customers_affected` depend on `cause_category`?*

TVD was used as the test statistic since `cause_category` is categorical. The observed TVD was 0.557 with a p-value of 0.0. The null hypothesis is rejected. The missingness of `customers_affected` does depend on cause category. Fuel supply emergencies, for example, have a much higher rate of missing customer data than other cause types.

<iframe src="assets/miss1.html" width="800" height="450" frameborder="0"></iframe>

*Test 2: Does missingness of `customers_affected` depend on `res_price`?*

The absolute difference in group means was used as the test statistic. The observed difference was 0.102 with a p-value of 0.577. The null hypothesis is not rejected. The missingness of `customers_affected` does not depend on residential electricity price.

<iframe src="assets/miss2.html" width="800" height="450" frameborder="0"></iframe>

## Hypothesis Testing

**Null Hypothesis:** The distribution of outage duration is the same for severe weather and non-severe weather outages. Any observed difference in means is due to random chance.

**Alternative Hypothesis:** Severe weather outages have a longer average duration than non-severe weather outages.

**Test Statistic:** Difference in group means (severe weather minus non-severe weather). This is an appropriate choice because the alternative hypothesis is directional and outage duration is a quantitative variable.

**Significance Level:** 0.05

The observed difference in means was 2537.81 minutes. After 1000 permutations, a p-value of 0.0 was obtained. The null hypothesis is rejected. The data are consistent with severe weather outages lasting significantly longer than non-severe weather outages, and this result is not likely due to random chance.

<iframe src="assets/hyp.html" width="800" height="450" frameborder="0"></iframe>

## Framing a Prediction Problem

The goal is to predict **outage duration in minutes**, which is a regression problem.

Outage duration was chosen as the response variable because it is the most practically meaningful quantity to predict. Knowing how long an outage is likely to last allows utilities and emergency responders to make better resource allocation decisions in real time.

**Evaluation metric:** RMSE is used over MAE because it penalizes large prediction errors more heavily, which is appropriate here since very long outages carry disproportionately high costs. RMSE is also preferred over R² because it is interpretable in the original units (minutes).

**Features known at time of prediction:** `cause_category`, `customers_affected`, `climate_region`, and `month_name` are all known at or very shortly after the start of an outage. Any features derived from the outage's end, such as restoration time, are excluded since those would not be available when a prediction needs to be made.

## Baseline Model

The baseline model uses a `LinearRegression` estimator inside a single `sklearn` Pipeline with two features:

- `customers_affected` — **quantitative**, passed through as-is
- `cause_category` — **nominal**, encoded using `OneHotEncoder` with `drop='first'`

| Split | RMSE |
|-------|------|
| Train | 4294.98 |
| Test | 4233.10 |

The train and test RMSE are nearly identical, which indicates the model is not overfitting. That said, an RMSE of roughly 4233 minutes is quite large relative to a typical outage. This is expected for a linear model with only two features. Cause category captures some variation in duration, but a linear relationship with customers affected is too simple to capture the nonlinear dynamics of how outages unfold. The baseline gives a reasonable starting point but leaves clear room for improvement.

## Final Model

Two new features were added on top of the baseline:

- `month_name` — **ordinal**, one-hot encoded. Outage duration varies seasonally. Summer storms and winter weather events tend to cause more widespread and longer-lasting damage than outages in other months, so including month allows the model to capture these patterns.
- `climate_region` — **nominal**, one-hot encoded. Different climate regions have different infrastructure characteristics, weather patterns, and utility response capabilities. An outage in the East North Central region during a polar vortex is a fundamentally different event from one in the Southwest during a heatwave, and climate region helps the model account for this.

A **log transformation was also applied to the target variable** before training. Because outage duration is heavily right-skewed, a small number of extreme events were dominating the loss function and pulling predictions off for the majority of outages. Log-transforming the target reduces the influence of these outliers and allows the model to fit the bulk of the data more accurately. Predictions were exponentiated back to the original scale for final evaluation.

A `RandomForestRegressor` was used because it captures nonlinear interactions between features that a linear model cannot. `GridSearchCV` with 5-fold cross-validation was used to tune `max_depth` and `n_estimators`. The `max_depth` parameter controls how deep each tree can grow, and `n_estimators` controls the number of trees in the forest. More trees generally reduce variance, while deeper trees can capture more complex patterns but risk overfitting.

**Best hyperparameters:** `max_depth=5`, `n_estimators=200`

| Split | RMSE |
|-------|------|
| Baseline Test | 4233.10 |
| Final Model Test | 3516.58 |
| Improvement | 716.52 minutes |

The final model reduced test RMSE by over 700 minutes compared to the baseline, showing that the added features and more flexible model architecture improved generalization to unseen data.

## Fairness Analysis

**Groups:** High population states (at or above the median state population) vs. low population states (below the median).

This grouping is worth examining because densely populated states tend to have more complex grid infrastructure, which could make outage duration harder to predict accurately.

**Evaluation metric:** RMSE

**Null Hypothesis:** The model's RMSE is the same for high and low population states. Any observed difference is due to random chance.

**Alternative Hypothesis:** The model performs worse (higher RMSE) for high population states than for low population states.

**Test Statistic:** Difference in RMSE (high population minus low population). **Significance level:** 0.05.

The observed difference in RMSE was 592.40 minutes. After 1000 permutations, a p-value of 0.217 was obtained. The null hypothesis is not rejected. There is no statistically significant evidence that the model performs worse for high population states, and the observed difference is consistent with what would be expected by random chance alone.

<iframe src="assets/fair.html" width="800" height="450" frameborder="0"></iframe>
