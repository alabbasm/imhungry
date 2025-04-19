<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>

# *I'm starving, how long is dinner gonna take?*  
*A multiple linear regression approach to predicting the number of minutes needed to make your dinner.*

## Introduction  
When you’re hungry and scrolling through recipes, one of the first questions on your mind is: *how long is this going to take me?* Some recipe sites give prep time estimates, but they’re often vague, inconsistent, or overly optimistic. Wouldn’t it be nice to have a more systematic way to predict cooking time from the actual recipe data?

This project uses a dataset from Food.com, which includes two main CSVs: one with detailed recipe information (ingredients, steps, nutrition, tags) and another with user-submitted ratings. After merging and cleaning this data, we aim to build a model that predicts the total number of **minutes** a recipe will take to make.

The main guiding question:
- **Is there a definable relationship between a recipe’s characteristics (like number of steps, ingredients, and calories) and how long it takes to cook?**

To explore this, we’ll start by loading and cleaning the data, then we’ll dive into some exploratory analysis, followed by building a predictive model with linear regression and then a more complex, feature-rich pipeline.

To start let's report some information about our datasets. We have two: `recipes.csv` and `ratings.csv`. The `recipes.csv` has 83782 rows or recipes. The `ratings.csv` contaings every interaction that a user has with a given recipe. That means there are significantly more rows in `ratings.csv` than `recipes.csv`. The columns in each row and descriptions of them are given below:

### Recipes Table

| Column           | Description                                                                                  |
|------------------|----------------------------------------------------------------------------------------------|
| `name`           | Recipe name                                                                                  |
| `id`             | Recipe ID                                                                                    |
| `minutes`        | Minutes to prepare recipe                                                                    |
| `contributor_id` | User ID who submitted this recipe                                                            |
| `submitted`      | Date recipe was submitted                                                                    |
| `tags`           | Food.com tags for recipe                                                                     |
| `nutrition`      | Nutrition information in the form `[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]`; PDV stands for “percentage of daily value” |
| `n_steps`        | Number of steps in recipe                                                                    |
| `steps`          | Text for recipe steps, in order                                                              |
| `description`    | User-provided description                                                                    |

### Ratings Table

| Column      | Description            |
|-------------|------------------------|
| `user_id`   | User ID                |
| `recipe_id` | Recipe ID              |
| `date`      | Date of interaction    |
| `rating`    | Rating given           |
| `review`    | Review text            |

## Data Cleaning and Exploratory Data Analysis  


### Cleaning the rating data  
We began by merging the two datasets (`recipes.csv` and `ratings.csv`) on their shared `id` column so that each recipe would include the user feedback. From the ratings, we computed the average rating per recipe and added this as a new column.

Interestingly, we noticed that some ratings were recorded as **0**, which isn’t part of the expected 1–5 scale. We interpreted these as **missing or invalid entries**, and treated them accordingly. For recipes with no valid ratings, we **imputed the missing average rating** using the overall mean rating from the dataset. This ensured that every recipe had a value for `avg_rating` and prevented missing data from derailing our model later.

The distribtuion of the `avg_rating` column pre and post imputation is shown below. 

<div class="centered-plot">
  <iframe src="assets/rating_kde_imputation.html" width="1000" height="600"></iframe>
  <div class="caption">Figure 1: Rating before and after mean imputation</div>
</div>

We can see that mean imputation does not destroy the over trend of the data, so it seems like a fine technique to use. 

The `nutrition` column of our resulting dataset was stored as string representations of Python lists — for example, the `ingredients`, `steps`, `tags`, and `nutrition` fields. This is a bit unqeildy, so we used the `ast.literal_eval()` function to convert them into actual lists. We then unpacked those lists into columns of their own giving us the following columns:
- `calories`
- `total fat`
- `sugar`
- `sodium`
- `protein`
- `saturated fat`
- `carbohydrates`

### Cleaning Outliers
During our EDA it was found that many of the numerical columns like `minutes`, `calories`, etc had some very obnoxious outliers that were not realistic values at all. Realistically, we plan for this model to be used by hungry and eachausted college students who come home and just want a quick estimate of how long dinner or any other meal is going to tak them to make. For this reason we decided to remove all recipes that aren't inline with the recommended number of macronutrients for a given meal:
- For the macronutrient columns (`calories`,`protein`,`total fat`,`sugar`,`saturated fat`), an arbitrary calorie cap of 1000 cal was chosen, and the bounds for the other macros were chosen based off the (AMDR's)[https://supplysix.com/blogs/all/acceptable-macronutrient-ranges-and-healthy-diets-for-adults?srsltid=AfmBOoqEKFfdXNCOsxzDD2wbRW8EXQl5jwfPpaY5jjuPer7TUMYY89yJ] guidelines for an average healthy adult's nutritional intake. Since the data was already in a PDV format, it was easy to filter based off the given AMDR guidelines.

- For the `minutes` column, a cutoff of 65 minutes was chosen in part because it is the 75th quartile of the minutes values, and because a person making dinner isn't likely to spend more than an hour making dinner.

To make it easier on a potential user of this model, the macronutrient columns (`calories`,`protein`,`total fat`,`sugar`,`saturated fat`), were converted from PDV to grams, as its more likely a user would know the macaronutrients in their recipe as grams, not as a PDV.  

 <div class="centered-plot">
  <iframe src="assets/minutes_before_filt.html" width="1000" height="600"></iframe>
  <div class="caption">Figure 2: Minutes Before Filtering</div>
</div>

<div class="centered-plot">
  <iframe src="assets/minutes_after_filt.html" width="1000" height="600"></iframe>
  <div class="caption">Figure 3: Minutes After Filtering</div>
</div>

### Determining recipe types  
Many recipes have a list of tags (like “vegetarian” or “dessert”), but we created a simpler **recipe type** feature using text analysis. We applied TF-IDF on the following columns post cleaning to find characteristic keywords for each recipe; `name`,`tags`,`steps`,`description`,`ingredients`,`review`.
Each recipe was then labeled with the top TF-IDF keyword as its “type.”  We then only kept the top 40 most frequent terms from our pseudo `recipe_types` column, and classified the other recipe types as `other`. This was becuase many of the low frequency terms returned some nonsensical word that did not accurately represent the recipe at all. A list of some of those top words are presented below:

'potatoes', 'rice', 'beans', 'asparagus', ...

It's important to note here that a lot of the values in `recipe_type` may not actually have names representative of the actual recipe, but may be associated with the recipe, ie. a major ingredient used in the recipe or something similar. Thus if a user doesn't find the name of their recipe in the column, they may use a common ingredient found in their recipe that is found in the column. Otherwise they must select the `other` option when using our model. 

Here's the head of `X_recipes`, the final cleaned dataframe that is used in the rest of our project.

|    | name                                |   minutes |   n_steps |   n_ingredients |   rating |   calories |   total_fat_g |   saturated_fat_g |   sugar_g |   protein_g |   sodium_mg |   carbohydrates_g | recipe_type   |
|---:|:------------------------------------|----------:|----------:|----------------:|---------:|-----------:|--------------:|------------------:|----------:|------------:|------------:|------------------:|:--------------|
| 13 | pinards en branche  sauted spinach  |        50 |        13 |               4 |     5    |       61.8 |          3.9  |               1.8 |       1   |         4.5 |         138 |              2.75 | spinach       |
| 16 | bbq spray recipe    it really works |         5 |         5 |               3 |     4.75 |       47.2 |          0    |               0   |       1   |         0   |           0 |              0    | other         |
| 17 | berry french toast  oatmeal         |        12 |         5 |               6 |     4.75 |      190.9 |          6.24 |               0.6 |       1   |         6.5 |           0 |             24.75 | other         |
| 32 | near east  rice pilaf  low fat      |        25 |         8 |               8 |     4.5  |      236.9 |          2.34 |               0.2 |       0.5 |         5   |           0 |             41.25 | chicken       |
| 33 | outback   steak rub                 |         5 |         1 |               8 |     4    |        6.6 |          0    |               0   |       0   |         0   |         874 |              0    | other         |


### Exploring the `minutes` column  
Now, let’s dive into the data! First up: **cooking time (`minutes`)**. How are recipe durations distributed? Is there a typical cook time most recipes fall under? In a histogram of `minutes`, we might see a peak around shorter times (e.g. many recipes take 20-40 minutes) and a long tail of recipes that take hours. 

<div class="centered-plot">
  <iframe src="assets/minutes_after_filt.html" width="1000" height="600"></iframe>
  <div class="caption">Figure 4: Distribution of Minutes</div>
</div>

The distribution of recipe cooking times is right-skewed, with most recipes taking under 60 minutes. This tells us that quick meals dominate the dataset, with some rough peaks aroung the 30-40 min range.

We can also see if certain types of recipes tend to take longer. For example, a boxplot of `minutes` grouped by `recipe_type` could show that *desserts* versus *main dishes* have different prep time distributions. We can see that there's definite variance between the different `recipe type`'s. A table grouped by recipe types showing the mean of every recipe type in ascending order shows this a bit better.

<div class="centered-plot">
  <iframe src="assets/min_by_r_type.html" width="1000" height="600"></iframe>
  <div class="caption">Figure 6: Minutes by Recipe Type</div>
</div>

 | recipe_type   |   minutes | recipe_type   |   minutes | recipe_type   |   minutes |
|:--------------|----------:|:--------------|----------:|:--------------|----------:|
| dough         |   38.9268 | spinach       |   25.8191 | sauce         |  21.4756  |
| potatoes      |   38.1304 | eggs          |   25.5667 | other         |  20.4615  |
| rice          |   32.7366 | shrimp        |   25.5432 | lemon         |  20.3387  |
| soup          |   31.2133 | pancakes      |   25.2162 | mint          |  16.2982  |
| chicken       |   30.8681 | beans         |   25.1371 | hummus        |  15.6949  |
| pizza         |   30.7119 | fish          |   24.8913 | salsa         |  15.6364  |
| beef          |   30.2439 | tomatoes      |   24.5263 | chocolate     |  15.5625  |
| mushrooms     |   30.2    | zucchini      |   24.4043 | salad         |  14.9028  |
| turkey        |   29.5405 | broccoli      |   23.8046 | dressing      |  11.4795  |
| bacon         |   29.1964 | tofu          |   23.75   | lime          |  10       |
| pasta         |   28.459  | corn          |   23.0204 | coffee        |   8.15842 |
| pumpkin       |   28.2292 | cheese        |   22.65   | drink         |   3.9     |
| flour         |   27.6122 | asparagus     |   22.38   | cocktail      |   3.62162 |
| bread         |   25.9518 | sesame        |   22.122  | nan           | nan       |

<p align="center"><em>Table 1: Average cooking time by recipe type</em></p>

What about relationships between `minutes` and other numeric features? Intuitively, recipes with more steps or ingredients might take more time. We explore scatter plots of `minutes` vs. `n_steps` (number of steps in the instructions) and vs. `n_ingredients`. As expected, there is a *slight* upward trend: recipes with more steps and ingredients do tend to require more minutes. It’s not a perfect correlation, but the positive association is there. 

<div class="centered-plot">
  <iframe src="assets/min_vs_step.html" width="1000" height="600"></iframe>
  <div class="caption">Figure 7: Scatter Plot of minutes vs n_step</div>
</div>

<div class="centered-plot">
  <iframe src="assets/min_vs_ing.html" width="1000" height="600"></iframe>
  <div class="caption">Figure 8: Scatter Plot of minutes vs n_ingredients</div>
</div>

## Framing a Prediction Problem  
After exploring, we decided our goal is to predict **cooking time (`minutes`)**. Cooking time is more directly useful to someone planning a meal – knowing if a dish takes 15 minutes versus 2 hours is valuable! Ratings are interesting but subjective, and calorie counts depend a lot on portion sizes, which vary by recipe.  

So, **formal prediction question:** *Given a recipe’s attributes (ingredients, steps, nutritional info, etc.), can we accurately predict how many minutes it will take to make?* 

We’ll treat this as a regression problem, where the target variable is `minutes`.  

We used standard multiple linear regression techniques built in scikit learn to achieve this task. A standard metric was chosen of MSE (Mean Squared Error) to evaluate the validity of our model.

To clarify, the features we plan to use include things like: the number of steps, number of ingredients, average user rating, nutritional stats (calories, fat, sugar, etc.), and the recipe type/category. We suspect these factors collectively influence cook time. For example, more steps or ingredients might mean more prep work, affecting the minutes. The recipe type might capture aspects like cooking techniques (a “bake” might take longer than a “salad”). Our hope is that by feeding all this information into a model, it can *learn* the typical patterns and give a reasonable time estimate for a new recipe.  

## Baseline Model  
Our first attempt is a straightforward **baseline model**. We chose a simple multiple linear regression using a few key features that we believed would be most predictive:  
- `n_steps`, a discrete numerical value describing the number of steps needed to complete a recipe 
- `calories`, a continuous numerical value 
- `recipe_type`, a nominal categorical feature that we'll one hot encode

Using these features, we fit a linear regression on 80% of the data (with 20% held out for testing). This is our baseline for comparison. 

**Baseline performance:** The baseline model’s predictions turned out okay but not amazing. The Mean Squared Error (MSE) on the test set was about **175.5**. In more intuitive terms, that means the root mean squared error is around $\sqrt{170} \approx 13 \ \text{minutes}$. So on average, our baseline predictions are about 13 minutes off from the actual time. That’s a sizable gap — if you’re expecting a 30-minute meal, it might actually take 43 minutes, which is the difference between a quick dinner and a long one! Clearly, there’s room for improvement.  

We also looked at the learned coefficients to interpret the baseline model. The coefficients suggested that recipes with more steps **do** take longer (each additional step adds roughly 1.3 minutes on average, according to the model). Surprisingly, the calorie count had a smaller effect (the coefficient for `calories` was very low, implying that an extra 100 calories only adds about 1 minute of cook time). This makes sense: calories are more about ingredients than process. The recipe type dummy variables showed slight shifts; for example, the model might have given a positive bump to categories like “roast” (meaning if a recipe is a roast, it predicts a longer time, all else equal) and a negative bump to quick categories like “salad.” However, many of those category effects weren’t very large in the linear model.  

| feature   |   weight | feature   |   weight | feature   |   weight |
|:----------|---------:|:----------|---------:|:----------|---------:|
| bacon     |     4.74 | eggs      |     2.65 | rice      |     9.5  |
| beans     |     4.29 | fish      |     4.2  | salad     |    -2.91 |
| beef      |     8.84 | flour     |     5.91 | salsa     |    -2.9  |
| bread     |     3.99 | hummus    |    -4.96 | sauce     |     1.95 |
| broccoli  |     2.01 | lemon     |    -0.56 | sesame    |     0.91 |
| cheese    |     1.22 | lime      |    -7.55 | shrimp    |     1.53 |
| chicken   |     6.62 | mint      |    -0.66 | soup      |     9.38 |
| chocolate |    -3.71 | mushrooms |     7.95 | spinach   |     2.71 |
| cocktail  |   -12.2  | other     |     0.11 | tofu      |    -0.23 |
| coffee    |    -8.58 | pancakes  |     2.97 | tomatoes  |     3.18 |
| corn      |     1.35 | pasta     |     3.77 | turkey    |     6.37 |
| dough     |     4.01 | pizza     |     4.53 | zucchini  |     3.03 |
| dressing  |    -5.83 | potatoes  |    14.37 | calories  |     0.01 |
| drink     |   -11.58 | pumpkin   |     6.9  | n_steps   |     1.27 |

<p align="center"><em>Table 2: Features and their respective weights computed by our model</em></p>


<div class="centered-plot">
  <iframe src="assets/min_vs_step_cal.html" width="1000" height="600"></iframe>
  <div class="caption">Figure 8: Overlaid predicition of our baseline model</div>
</div>

In summary, the baseline linear model captures some obvious signals (steps matter!), but it’s not very accurate yet. We’ll use this as a reference point for building a better model.  

## Final Model  
To improve on the baseline, we pulled out all the stops in feature engineering and modeling. Our final approach is still a regression model with **all** the features in our dataframe: , but with several enhancements:  

| Native Features  | Description                                                                                  |
|------------------|----------------------------------------------------------------------------------------------|
|  `minutes`       | Number of minutes taken to complete a recipe, numerical continous data                       |
| `rating`         | Rating of a recipe from 1-5, numerical discrete                                              |
| `calories`       | Number of calories in a recipe, numerical continuous                                         |
| `total_fat_g`    | Total grams of fat in a recipe, numerical continuous                                         |
| `saturated_fat_g`| Grams of saturated fat in a recipe, numerical continuous                                     |
| `sugar_g`        | Grams of (added) sugar in a recipe, numerical continuous                                     |
| `sodium_mg`      | Milligrams of sodium in a recipe, numerical continuous                                       |
| `protein_g`      | Grams of protein in a recipe, numerical continuous                                           |
| `carbohydrates_g`| Grams of carbohydrates in a recipe, numerical continuous                                     |
|   `n_steps`      | Number of steps to complete a recipe, numerical discrete                                     |
| `n_ingredients`  | Number of ingredients in a recipe, numerical discrete                                        |

- **Derived Features:** We created new features such as *normalized nutrition* stats. For each recipe, we took nutritional values like fat, sugar, etc., converted to calories and divided them by total calories to get the proportion of . The idea is to capture the composition of the recipe (e.g., how sugary it is) rather than absolute calories, since absolute calories already correlate with other things. This gives features like “sugar per calorie” which might relate to recipe type (desserts vs. savory).  

- **Binned Features:** We used a `KBinsDiscretizer` to turn some numerical features into categorical buckets, and one hot encoded them. For example, we binned `n_steps`, `n_ingredients`, and `calories` into ranges (10 bins each). This can allow the model to learn non-linear relationships (maybe recipes with 1-5 steps aren’t that different, but beyond 10 steps, the time jumps a lot – a linear model might miss that if not binned).  

- **Quantile Transformation**: A `QuantileTransformer` was used to transform our data to fit more of a normal distribution. This type of transformation is known to work quite well with skewed data, which many of our features are. It's also a monotonic transformation-it keeps the relative ordering of the data consistent. 

- **Polynomial/Interaction Features:** We considered that certain combinations of features or non-linear patterns could affect cook time. To allow for this, we added polynomial features (up to degree 3 for the numeric features). This means the model can account for, say, the squared effect of `n_steps` or an interaction between `n_steps` and `n_ingredients` if it helps.  

- **Log Transforms:** We introduced the possibility of log-transformed features. For instance, perhaps beyond a certain point, adding more steps has diminishing returns (a log relationship). We set up custom transformers that could apply a `np.log1p(x)` to features, with tunable parameters to adjust the curve. These were experimental features to see if any non-linear relationship would significantly improve the fit. The function itself looks something like this:
$$
\text{Log Transform}(X) = a\log{(1+X)}
$$  
Where $a$ is the decay/growth rate, and 1 is added to avoid division by 0. 

- **Model Choice:** We didn’t want to just assume linear regression is best. We also considered **Ridge regression** (a regularized linear model) to potentially handle multicollinearity among our many features and to prevent overfitting given the polynomial expansion. We treated the choice of using a plain Linear Regression vs. a Ridge regression (and the ridge penalty `alpha` value) as something to tune.  

Whew! That’s a lot of moving parts. To manage this systematically, we set up a machine learning pipeline and used **Grid Search with cross-validation** to tune the hyperparameters. The hyperparameters we tuned included: the polynomial degree (how complex the interactions could get), the parameter `a` in our custom log transformer (to adjust their shape), the number of bins (`n_bins`) used by our `KBinsDiscretizer`, and the Ridge `alpha` (if Ridge was used). We used 5-fold cross-validation on the training set to try out different combinations and find what worked best.  

| Hyperparameter         | Range Chosen              |  Optimal Parameter     |
|------------------------|---------------------------|------------------------|
|  `a` (log decay/growth)|     [-4,-3,...,3]         |         -3             |
| `n_bins`               |     [1,2,3,...,10]        |         9              |
| `degree`               |       [1,2,3]             |         1              |
| `alpha`                |     [0.01,0.1,1,5,10]     |          10            |

<p align="center"><em>Table 3: Chosen Hyper parameters and their optimal values</em></p>

After quite a bit of number crunching, we arrived at a final model. Interestingly, the best combination of transformations was to use a **logarithmic transform** (with a relatively high decay rate parameter) on certain features, along with polynomial features of degree 1. (It’s hard to interpret exactly what this means physically, but it suggests some diminishing returns in one area and maybe an overall linear growth in others). The best model ended up being a plain ridge regression with an alpha value of 10 (linear regression didn’t outperform it, though it was very close, implying our features were not causing huge overfitting issues).  

**Final model performance:** Drumroll, please... The improved model brought the test MSE down to about **145**. That’s roughly a 15% reduction in MSE compared to the baseline (170 → 145). In terms of root mean squared error, we went from ~14.8 minutes off to ~12 minutes off. So we gained about a one-minute improvement in average error by all that fancy feature engineering and tuning. Not a huge drop, but an improvement nonetheless!  


|                Model Type                |   Train MSE |   Test MSE |
|:-------------------------------|------------:|-----------:|
| Baseline Model                 |     172.191 |    166.858 |
| Linear Regression w Poly feat. |     144.52  |    147.089 |
| Linear Regression w Log feat.  |     144.519 |    147.089 |
| Ridge Regression w Poly feat.  |     145.313 |    145.24  |
| Ridge Regression w Log feat.   |     145.313 |    145.239 |
| Ridge Regression w both feat. (Final Model) |     145.314 |    145.246 |

<p align="center"><em>Table 4: All models trained and their respective training and testing MSE's</em></p>

We should ask: is this level of error acceptable? **12-13 minutes uncertainty** for a recipe’s cook time might be okay for some scenarios (predicting ~30 min vs actual 45 min is not too bad), but it’s still quite high if someone needs a very accurate estimate. It seems that predicting `minutes` is inherently tricky with the given data. Recipes can always have unobserved factors (technique difficulty, user skill, etc.) that affect prep time. Our model captures the obvious factors, but the variability in cooking is large.  

In the end, we chose to stick with the ridge regression model for our final solution (instead of the Ridge), since it performed essentially the same. The added regularization didn’t yield a noticeable benefit, and the simpler model is easier to interpret and explain.  

**Reflection:** So, *how long is dinner gonna take?* Our final model can give an estimate, but expect an error margin of about ±12 minutes. Not perfect, but better than a blind guess. Importantly, we confirmed some common-sense insights: the number of steps in a recipe is a strong indicator of prep time, and our model uses that heavily. We also learned that beyond a point, extra ingredients or calories don’t linearly increase cook time (hence the model leveraging log/exponential features). Perhaps truly nailing the prediction would require more detailed features (like parsing the text of the recipe for specific difficult techniques, or accounting for whether multiple steps can be done in parallel, etc.). Those are complexities beyond our current scope, but they hint at why this problem is challenging.  

On the bright side, if you’re ever unsure how long a recipe will take, our project shows you can plug in the recipe’s details and get a ballpark time estimate. It might just save you from starting a 2-hour recipe when you only have 30 minutes before dinner! *Bon appétit!*  
