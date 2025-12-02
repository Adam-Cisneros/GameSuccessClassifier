import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


class Classifier:
    def __init__(self, df):
        self.df = df.dropna(subset=['user_rating', 'metacritic_score'])
        self.df = self.df[(self.df['user_rating'].astype(float) > 0) & (self.df['metacritic_score'].astype(float) > 0)]
        self.df['genres'] = self.df['genres'].fillna("")
        self.df['developer'] = self.df['developer'].fillna("")
        self.df['publisher'] = self.df['publisher'].fillna("")

        X = self.df[['genres', 'developer', 'publisher', 'dev_game_count']]
        y = self.df[['user_rating', 'metacritic_score']]

        self.X_train, self.X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.y_user_train = y_train['user_rating']
        self.y_critic_train = y_train['metacritic_score']
        self.y_user_test = y_test['user_rating']
        self.y_critic_test = y_test['metacritic_score']
        self.num_features = ['dev_game_count']
        self.text_features = ['genres', 'developer', 'publisher']
        self.preprocess = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', self.num_features),
                ('genres_tfidf', TfidfVectorizer(max_features=100), 'genres'),
                ('dev_tfidf', TfidfVectorizer(max_features=100), 'developer'),
                ('pub_tfidf', TfidfVectorizer(max_features=100), 'publisher')
            ],
            remainder='drop'
        )


    # -----------------------------------------------------------
    # IMPROVED BOUNDARY CALCULATION
    # -----------------------------------------------------------
    def compute_boundaries(self, strategy="kmeans"):
        user = self.y_user_train
        critic = self.y_critic_train

        if strategy == "median":
            return user.median(), critic.median()

        elif strategy == "mean":
            return user.mean(), critic.mean()

        elif strategy == "percentile":
            return np.percentile(user, 50), np.percentile(critic, 50)

        elif strategy == "zscore":
            return user.mean(), critic.mean()

        elif strategy == "kmeans":
            km_user = KMeans(n_clusters=2, random_state=42).fit(user.to_frame())
            km_critic = KMeans(n_clusters=2, random_state=42).fit(critic.to_frame())

            u_centers = sorted(km_user.cluster_centers_.flatten())
            c_centers = sorted(km_critic.cluster_centers_.flatten())

            user_boundary = (u_centers[0] + u_centers[1]) / 2
            critic_boundary = (c_centers[0] + c_centers[1]) / 2

            return user_boundary, critic_boundary

        else:
            raise ValueError("Invalid strategy for compute_boundaries")

    # -----------------------------------------------------------
    # TRAIN MODELS
    # -----------------------------------------------------------
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

        # Use improved boundaries
        self.user_boundary, self.critic_boundary = self.compute_boundaries(strategy="kmeans")

    # -----------------------------------------------------------
    def get_quadrants(self, row):
        u = row["pred_user"]
        c = row["pred_critic"]

        if u >= self.user_boundary and c >= self.critic_boundary:
            return "High User / High Critic"
        elif u >= self.user_boundary and c < self.critic_boundary:
            return "High User / Low Critic"
        elif u < self.user_boundary and c >= self.critic_boundary:
            return "Low User / High Critic"
        else:
            return "Low User / Low Critic"

    # -----------------------------------------------------------
    def evaluate(self):
        preds_user = self.user_reg_model.predict(self.X_test)
        preds_critic = self.critic_reg_model.predict(self.X_test)

        user_boundary = self.user_boundary
        critic_boundary = self.critic_boundary

        # Regression metrics
        print("\n=== User Rating Regression ===")
        print("MAE:", mean_absolute_error(self.y_user_test, preds_user))
        print("RMSE:", np.sqrt(mean_squared_error(self.y_user_test, preds_user)))
        print("R²:", r2_score(self.y_user_test, preds_user))

        print("\n=== Critic Score Regression ===")
        print("MAE:", mean_absolute_error(self.y_critic_test, preds_critic))
        print("RMSE:", np.sqrt(mean_squared_error(self.y_critic_test, preds_critic)))
        print("R²:", r2_score(self.y_critic_test, preds_critic))

        # Quadrant true and predicted
        def quad(u, c):
            if u >= user_boundary and c >= critic_boundary:
                return "High User / High Critic"
            elif u >= user_boundary and c < critic_boundary:
                return "High User / Low Critic"
            elif u < user_boundary and c >= critic_boundary:
                return "Low User / High Critic"
            else:
                return "Low User / Low Critic"

        true_q = [quad(u, c) for u, c in zip(self.y_user_test, self.y_critic_test)]
        pred_q = [quad(u, c) for u, c in zip(preds_user, preds_critic)]

        print("\n=== Quadrant Classification ===")
        print("Accuracy:", accuracy_score(true_q, pred_q))
        print("Macro F1:", f1_score(true_q, pred_q, average="macro"))

        labels = ["High User / High Critic", "High User / Low Critic",
                  "Low User / High Critic", "Low User / Low Critic"]

        cm = confusion_matrix(true_q, pred_q, labels=labels)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title("Quadrant Prediction Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------
    # ACTUAL MAGIC QUADRANT
    # -----------------------------------------------------------
    def plot_magic_quadrant(self):
        plt.figure(figsize=(12, 8))

        plt.scatter(self.df['user_rating'], self.df['metacritic_score'], alpha=0.6)

        user_med = self.user_boundary
        critic_med = self.critic_boundary

        plt.axvline(user_med, linestyle="--")
        plt.axhline(critic_med, linestyle="--")

        plt.title("Magic Quadrant (Actual Ratings)")
        plt.xlabel("User Rating")
        plt.ylabel("Critic Rating")
        plt.grid(alpha=0.3)
        plt.show()

    # -----------------------------------------------------------
    # PREDICTED MAGIC QUADRANT
    # -----------------------------------------------------------
    def plot_predicted_quadrants(self):
        preds_user = self.user_reg_model.predict(self.X_test)
        preds_critic = self.critic_reg_model.predict(self.X_test)

        df_plot = self.X_test.copy()
        df_plot["pred_user"] = preds_user
        df_plot["pred_critic"] = preds_critic

        df_plot["quadrant"] = df_plot.apply(
            lambda r: self.get_quadrants(r), axis=1
        )

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df_plot, x="pred_user", y="pred_critic",
            hue="quadrant", palette="Set2"
        )

        plt.axvline(self.user_boundary, linestyle="--")
        plt.axhline(self.critic_boundary, linestyle="--")

        plt.title("Magic Quadrant (Predicted Ratings)")
        plt.xlabel("Predicted User Rating")
        plt.ylabel("Predicted Critic Rating")
        plt.grid(alpha=0.3)
        plt.show()

    # -----------------------------------------------------------
    # SIDE-BY-SIDE ACTUAL vs PREDICTED
    # -----------------------------------------------------------
    def plot_side_by_side_quadrants(self):
        preds_user = self.user_reg_model.predict(self.X_test)
        preds_critic = self.critic_reg_model.predict(self.X_test)

        df_pred = self.X_test.copy()
        df_pred["pred_user"] = preds_user
        df_pred["pred_critic"] = preds_critic
        df_pred["quadrant"] = df_pred.apply(lambda r: self.get_quadrants(r), axis=1)

        user_med = self.user_boundary
        critic_med = self.critic_boundary

        fig, axes = plt.subplots(1, 2, figsize=(22, 8))

        # ---- LEFT: Actual ----
        axes[0].scatter(
            self.df['user_rating'], self.df['metacritic_score'], alpha=0.6
        )
        axes[0].axvline(user_med, linestyle="--")
        axes[0].axhline(critic_med, linestyle="--")
        axes[0].set_title("Actual Magic Quadrant")
        axes[0].set_xlabel("User Rating")
        axes[0].set_ylabel("Critic Rating")
        axes[0].grid(alpha=0.3)

        # ---- RIGHT: Predicted ----
        sns.scatterplot(
            data=df_pred, x="pred_user", y="pred_critic",
            hue="quadrant", palette="Set2", ax=axes[1]
        )
        axes[1].axvline(user_med, linestyle="--")
        axes[1].axhline(critic_med, linestyle="--")
        axes[1].set_title("Predicted Magic Quadrant")
        axes[1].set_xlabel("Predicted User Rating")
        axes[1].set_ylabel("Predicted Critic Rating")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("games.csv")
    df = df[['genres', 'developer', 'publisher', 'dev_game_count', 'user_rating', 'metacritic_score']]

    classifier = Classifier(df)
    classifier.train()
    classifier.evaluate()
    classifier.plot_magic_quadrant()
    classifier.plot_predicted_quadrants()
    classifier.plot_side_by_side_quadrants()