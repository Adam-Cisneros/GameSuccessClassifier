import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# initalize a classifier class with needed methods
class Classifier:
    def __init__(self, df):
        # Drop missing values in target columns
        self.df = df.dropna(subset=['user_rating', 'metacritic_score'])
        X = self.df[['genres', 'developer', 'publisher']]
        y = self.df[['user_rating', 'metacritic_score']]
        self.X_train, self.X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.y_user_train = y_train['user_rating']
        self.y_critic_train = y_train['metacritic_score']
        self.y_user_test = y_test['user_rating']
        self.y_critic_test = y_test['metacritic_score']

        self.categorical_features = ['genres', 'developer', 'publisher']
        self.preprocess = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ]
        )

    def plot_feature_importance(self, model="user"):
        """
        Plot feature importance for either the user or critic model.
        model="user"  -> user_reg_model
        model="critic" -> critic_reg_model
        """

        if model == "user":
            rf = self.user_reg_model.named_steps["regressor"]
            encoder = self.user_reg_model.named_steps["preprocess"].named_transformers_["cat"]
        elif model == "critic":
            rf = self.critic_reg_model.named_steps["regressor"]
            encoder = self.critic_reg_model.named_steps["preprocess"].named_transformers_["cat"]
        else:
            raise ValueError("model must be 'user' or 'critic'")

        # Get feature names after one-hot encoding
        encoded_feature_names = encoder.get_feature_names_out(self.categorical_features)

        # Match them with importances
        importances = rf.feature_importances_
        feature_data = pd.DataFrame({
            "feature": encoded_feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        # Plot top 20 for readability
        top_n = 20
        plt.figure(figsize=(10, 7))
        sns.barplot(y=feature_data.head(top_n)["feature"], 
                    x=feature_data.head(top_n)["importance"], 
                    orient="h")
        plt.title(f"Top {top_n} Feature Importances ({model.capitalize()} Rating Model)")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    def train(self):
        self.user_reg_model = Pipeline(steps=[
            ('preprocess', self.preprocess),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

        self.critic_reg_model = Pipeline(steps=[
            ('preprocess', self.preprocess),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

        self.user_reg_model.fit(self.X_train, self.y_user_train)
        self.critic_reg_model.fit(self.X_train, self.y_critic_train)

    def predict(self):
        self.df['pred_user'] = self.user_reg_model.predict(self.X_test)
        self.df['pred_critic'] = self.critic_reg_model.predict(self.X_test)
        self.user_median = self.df['pred_user'].median()
        self.critic_median = self.df['pred_critic'].median()
        self.df['quadrant'] = self.df.apply(self.get_quadrants, axis=1)
        return self.df[['title', 'genres', 'developer', 'publisher', 'pred_user', 'pred_critic', 'quadrant']]

    def get_quadrants(self, row):
        if row['pred_user'] >= self.user_median and row['pred_critic'] >= self.critic_median:
            return "High User / High Critic"
        elif row['pred_user'] >= self.user_median and row['pred_critic'] < self.critic_median:
            return "High User / Low Critic"
        elif row['pred_user'] < self.user_median and row['pred_critic'] >= self.critic_median:
            return "Low User / High Critic"
        else:
            return "Low User / Low Critic"

    def evaluate(self):
        # Compute predicted quadrants for X_test rows only
        preds_user = self.user_reg_model.predict(self.X_test)
        preds_critic = self.critic_reg_model.predict(self.X_test)

        self.user_median = np.median(preds_user)
        self.critic_median = np.median(preds_critic)

        # Compute metrics for user regression
        user_mae = mean_absolute_error(self.y_user_test, preds_user)
        user_rmse = np.sqrt(mean_squared_error(self.y_user_test, preds_user))
        user_r2 = r2_score(self.y_user_test, preds_user)

        # Compute metrics for critic regression
        critic_mae = mean_absolute_error(self.y_critic_test, preds_critic)
        critic_rmse = np.sqrt(mean_squared_error(self.y_critic_test, preds_critic))
        critic_r2 = r2_score(self.y_critic_test, preds_critic)

        print("\n=== User Rating Regression Performance ===")
        print(f"MAE:  {user_mae:.3f}")
        print(f"RMSE: {user_rmse:.3f}")
        print(f"R²:   {user_r2:.3f}")

        print("\n=== Critic Score Regression Performance ===")
        print(f"MAE:  {critic_mae:.3f}")
        print(f"RMSE: {critic_rmse:.3f}")
        print(f"R²:   {critic_r2:.3f}")

        # Compute medians from TRAIN ONLY so it's not cheating
        user_med = self.y_user_train.median()
        critic_med = self.y_critic_train.median()

        # Function to map values to quadrant label
        def map_quad(u, c):
            if u >= user_med and c >= critic_med:
                return "High User / High Critic"
            elif u >= user_med and c < critic_med:
                return "High User / Low Critic"
            elif u < user_med and c >= critic_med:
                return "Low User / High Critic"
            else:
                return "Low User / Low Critic"

        # True quadrants (based on actual y_test values)
        true_quads = [
            map_quad(u, c) for u, c in zip(self.y_user_test, self.y_critic_test)
        ]

        # Predicted quadrants
        pred_quads = [
            map_quad(u, c) for u, c in zip(preds_user, preds_critic)
        ]

        # Compute metrics
        acc = accuracy_score(true_quads, pred_quads)
        f1 = f1_score(true_quads, pred_quads, average="macro")

        print("\n=== Quadrant Classification Evaluation ===")
        print(f"Accuracy: {acc:.3f}")
        print(f"Macro F1 Score: {f1:.3f}")

        # Confusion matrix
        labels = [
            "High User / High Critic",
            "High User / Low Critic",
            "Low User / High Critic",
            "Low User / Low Critic"
        ]

        cm = confusion_matrix(true_quads, pred_quads, labels=labels)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title("Quadrant Prediction Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    
    def plot_magic_quadrant(self):
        plt.figure(figsize=(12, 8))

        # Scatter plot of actual ratings
        plt.scatter(self.df['user_rating'], self.df['metacritic_score'], alpha=0.6)

        # Threshold lines
        user_med = self.df['user_rating'].median()
        critic_med = self.df['metacritic_score'].median()

        plt.axvline(user_med, linestyle="--")
        plt.axhline(critic_med, linestyle="--")

        # Annotate quadrants
        plt.text(user_med + 0.05, critic_med + 0.5, "High User / High Critic", fontsize=12)
        plt.text(user_med + 0.05, critic_med - 0.5, "High User / Low Critic", fontsize=12)
        plt.text(user_med - 0.8, critic_med + 0.5, "Low User / High Critic", fontsize=12)
        plt.text(user_med - 0.8, critic_med - 0.5, "Low User / Low Critic", fontsize=12)

        plt.title("Magic Quadrant of Games (Actual Ratings)")
        plt.xlabel("User Rating")
        plt.ylabel("Critic Rating (Metacritic)")
        plt.grid(alpha=0.3)
        plt.show()

    def plot_predicted_quadrants(self):
        # Predict
        preds_user = self.user_reg_model.predict(self.X_test)
        preds_critic = self.critic_reg_model.predict(self.X_test)

        df_plot = self.X_test.copy()
        df_plot["pred_user"] = preds_user
        df_plot["pred_critic"] = preds_critic

        user_med = preds_user.mean()
        critic_med = preds_critic.mean()

        # Assign quadrant labels
        df_plot["quadrant"] = df_plot.apply(
            lambda r: self.get_quadrants({
                "pred_user": r["pred_user"],
                "pred_critic": r["pred_critic"]
            }), axis=1
        )

        # Final scatter
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df_plot,
                        x="pred_user", y="pred_critic",
                        hue="quadrant", palette="Set2")

        plt.axvline(user_med, linestyle="--")
        plt.axhline(critic_med, linestyle="--")

        plt.title("Magic Quadrant (Predicted Ratings)")
        plt.xlabel("Predicted User Rating")
        plt.ylabel("Predicted Critic Rating")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Load dataset rawg_games.csv
    df = pd.read_csv("rawg_games.csv")
    df = df[['title', 'type', 'genres', 'developer', 'publisher', 'user_rating', 'metacritic_score']]
    classifier = Classifier(df)
    classifier.train()
    classifier.evaluate()
    classifier.plot_feature_importance(model="user")
    classifier.plot_feature_importance(model="critic")
    classifier.plot_magic_quadrant()
    classifier.plot_predicted_quadrants()