# Classifying Game Success
### By: Adam Cisneros

## Purpose
This project aims to address challenges faced by multiple groups within the video game industry. The first group includes developers preparing to ship a game but uncertain about its potential reception among users. This work also examines a common question for developers, Do pre-release critic scores reliably indicate real-world user feedback upon release?

The second group includes users and consumers seeking to understand whether critic scores accurately reflect their personal enjoyment or whether a game meets their expectations for quality. This model is intended to support both groups by providing an analytically grounded framework for predicting game performance across user and critic spaces.

## Data
The initial phase of this project involved developing a comprehensive understanding of the dataset and performing cleaning procedures to ensure that the data was both meaningful and suitable for model training.

To build this understanding, I graphed the primary columns of the dataset. Key findings from this exploratory analysis are summarized below:
    * There were very few null values across most columns, with the exception of the “Type” column, which contained about 25% null entries. See graph below:
    ![Nulls](images/Nulls.png "Nulls")

    * I observed no null values in the “User Rating” column, which appeared unusual. After plotting the column as a continuous line chart, I found a significant number of zero values. Further investigation revealed that the API uses zeros in this column to represent null values. These zero entries introduced unnecessary noise and non-informative data. Additionally, this plot displayed a spike around the 1.0–1.5 range, indicating a skewed, multimodal distribution—an important observation that influenced later modeling decisions. See graph below:
    ![User Rating Distribution](images/URating.png "User Rating Distribution")

    * A plot of “Metacritic Score” vs. “User Rating” revealed no clear visual correlation between the two metrics. This lack of visible relationship suggested that a predictive model—such as the one developed in this project—could provide meaningful insight.See graph below:
    ![Rating Correlation](images/RatingCorr.png "Rating Correlation")

Using these insights, I then carried out the following data-cleaning steps:
    * Excluding the “Type” column from the feature set, due to the high proportion of null entries

    * Removing all remaining rows with null values

    * Removing all zero entries in the “User Rating” column (treated as nulls)

## Process
After cleaning, the dataset was temporally split to prevent information leakage later in the process. The data was sorted by release date, with the earliest 80% designated as training data and the most recent 20% as testing data. This temporal split ensured that calculated features would not inadvertently utilize future information.

Next, I engineered several calculated features to improve the model’s predictive capability. These included:
    * Mean user reviews per developer

    * Mean user reviews per publisher

    * Mean critic scores per developer

    * Mean critic scores per publisher

The final feature set therefore included:
Genres (TF-IDF encoded), Developers (TF-IDF encoded), Publishers (TF-IDF encoded), Mean User Reviews per Developer, Mean User Reviews per Publisher, Mean Critic Scores per Developer, and Mean Critic Scores per Publisher.

These features were used to train a Multi-Output Regressor, enabling simultaneous prediction of two regression outputs. An XGBRegressor was used within this multi-output framework, providing an efficient and unified training process for both target metrics.

After regression was done, I used a Gaussian Mixture Model (GMM) to compute quadrant boundaries. Instead of manually defining these boundaries or using basic statistical measures such as the mean or median, the GMM allowed the data to cluster naturally into mathematically meaningful groups. This approach was particularly well-suited to the skewed, multimodal distribution identified earlier in the “User Rating” column. Predicted values from the regressors were then compared against these boundaries to determine each game’s predicted classification.

## Evaluation Results
Model performance was evaluated using a combination of visual assessments and quantitative metrics.

First, a visual confusion matrix was generated to compare the model’s predicted classifications with the ground-truth classifications. Results indicated that the model performed strongly in identifying High User / High Critic and Low User / Low Critic outcomes. Performance was comparatively weaker in the High User / Low Critic and Low User / High Critic quadrants, indicating that the model could reliably identify overall good or bad performance but had more difficulty distinguishing cases where critic and user responses diverged. See graph below:

![Confusion Matrix](images/ConfMat.png "Confusion Matrix")

Next, standard regression and classification metrics were computed:
    * Mean Absolute Error (MAE): Approximately 2%–3% relative to the data range, indicating consistently accurate predictions

    * Root Mean Squared Error (RMSE): Roughly double the MAE, suggesting occasional but infrequent larger errors

    * R² Scores: 0.84 and 0.79 for the regression outputs, demonstrating strong modeling of variance and good overall fit

    * Accuracy: 87%, particularly impressive given dataset limitations

    * Macro F1 Score: 0.82, suggesting balanced classification performance across all four quadrants

Finally, I plotted the actual data (left) and predicted data (right) using the computed quadrant boundaries. Both plots demonstrated similar overall structures, with the primary difference being a slight translation in coordinate space. Specifically, the model tended to predict somewhat higher user ratings and a slightly more compact spread for both metrics. See graph below:

![Final Classification Comparison](images/FinalClass.png "Final Classification Comparison")

## Conclusion
With an accuracy of 87% and strong supporting metrics, this model provides a credible method for predicting game performance across both user and critic dimensions. While it demonstrates some difficulty in cases where user and critic responses diverge, it makes relatively few errors in the context of the overall dataset.

Accordingly, this model can serve as a decently reliable starting point for users seeking to assess whether a game will meet their expectations, as well as for developers interested in understanding whether user reception is likely to mirror critic assessments.