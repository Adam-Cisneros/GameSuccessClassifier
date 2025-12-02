import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt


class Classifier:
    def __init__(self, df):
        # Clean base data
        self.df = df.dropna(subset=['user_rating', 'metacritic_score'])
        self.df = self.df[(self.df['user_rating'] > 0) & (self.df['metacritic_score'] > 0)]
        self.df['genres'] = self.df['genres'].fillna("")
        self.df['developer'] = self.df['developer'].fillna("")
        self.df['publisher'] = self.df['publisher'].fillna("")

        # Do a manual temporal train-test split to simulate real-world scenario and avoid temporal data leakage
        cutoff_index = int(0.8 * len(self.df))
        self.X_train = self.df.iloc[:cutoff_index]
        # Add historical averages to feature sets after split to avoid data leakage
        self.X_train['dev_avg_user'] = self.X_train.groupby('developer')['user_rating'].transform('mean')
        self.X_train['dev_avg_critic'] = self.X_train.groupby('developer')['metacritic_score'].transform('mean')
        self.X_train['pub_avg_user'] = self.X_train.groupby('publisher')['user_rating'].transform('mean')
        self.X_train['pub_avg_critic'] = self.X_train.groupby('publisher')['metacritic_score'].transform('mean')
        for col in ['dev_avg_user','dev_avg_critic','pub_avg_user','pub_avg_critic']:
            self.X_train[col] = self.X_train[col].fillna(self.X_train[col].mean())
        self.X_train.drop(columns=['user_rating', 'metacritic_score'], inplace=True)

        self.X_test = self.df.iloc[cutoff_index:].drop(columns=['user_rating', 'metacritic_score'])
        # Add historical averages of global data relative to developers/publishers in test set
        self.X_test['dev_avg_user'] = self.df.groupby('developer')['user_rating'].transform('mean').loc[self.X_test.index]
        self.X_test['dev_avg_critic'] = self.df.groupby('developer')['metacritic_score'].transform('mean').loc[self.X_test.index]
        self.X_test['pub_avg_user'] = self.df.groupby('publisher')['user_rating'].transform('mean').loc[self.X_test.index]
        self.X_test['pub_avg_critic'] = self.df.groupby('publisher')['metacritic_score'].transform('mean').loc[self.X_test.index]
        for col in ['dev_avg_user','dev_avg_critic','pub_avg_user','pub_avg_critic']:
            self.X_test[col] = self.X_test[col].fillna(self.X_train[col].mean())

        # Target sets
        self.y_train = self.df.iloc[:cutoff_index][['user_rating', 'metacritic_score']]
        self.y_test = self.df.iloc[cutoff_index:][['user_rating', 'metacritic_score']]

        # Features
        self.feature_cols = [
            'genres', 'developer', 'publisher',
            'dev_avg_user', 'dev_avg_critic',
            'pub_avg_user', 'pub_avg_critic'
        ]

        # Preprocess
        self.num_features = [
            'dev_avg_user','dev_avg_critic','pub_avg_user','pub_avg_critic'
        ]
        self.text_features = ['genres','developer','publisher']

        self.preprocess = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', self.num_features),
                ('genres_tfidf', TfidfVectorizer(max_features=150), 'genres'),
                ('dev_tfidf', TfidfVectorizer(max_features=150), 'developer'),
                ('pub_tfidf', TfidfVectorizer(max_features=150), 'publisher'),
            ]
        )

    def compute_boundaries(self):
        user = self.y_train['user_rating']
        critic = self.y_train['metacritic_score']

        gmm_user = GaussianMixture(n_components=2, random_state=42).fit(user.to_frame())
        gmm_critic = GaussianMixture(n_components=2, random_state=42).fit(critic.to_frame())

        u_means = sorted(gmm_user.means_.flatten())
        c_means = sorted(gmm_critic.means_.flatten())

        return (u_means[0] + u_means[1]) / 2, (c_means[0] + c_means[1]) / 2

    def train(self):
        self.model = Pipeline(steps=[
            ('preprocess', self.preprocess),
            ('regressor', MultiOutputRegressor(
                XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='reg:squarederror',
                    random_state=42
                )
            ))
        ])

        self.model.fit(self.X_train, self.y_train)

        # New GMM boundaries
        self.user_boundary, self.critic_boundary = self.compute_boundaries()

    def get_quadrant(self, u, c):
        if u >= self.user_boundary and c >= self.critic_boundary:
            return "High User / High Critic"
        if u >= self.user_boundary and c < self.critic_boundary:
            return "High User / Low Critic"
        if u < self.user_boundary and c >= self.critic_boundary:
            return "Low User / High Critic"
        return "Low User / Low Critic"

    def evaluate(self):
        preds = self.model.predict(self.X_test)
        preds_user = preds[:, 0]
        preds_critic = preds[:, 1]

        print("\n=== USER RATING REGRESSION ===")
        print("MAE:", mean_absolute_error(self.y_test['user_rating'], preds_user))
        print("RMSE:", np.sqrt(mean_squared_error(self.y_test['user_rating'], preds_user)))
        print("R²:", r2_score(self.y_test['user_rating'], preds_user))

        print("\n=== CRITIC SCORE REGRESSION ===")
        print("MAE:", mean_absolute_error(self.y_test['metacritic_score'], preds_critic))
        print("RMSE:", np.sqrt(mean_squared_error(self.y_test['metacritic_score'], preds_critic)))
        print("R²:", r2_score(self.y_test['metacritic_score'], preds_critic))

        # Quadrant metrics
        true_q = [
            self.get_quadrant(u, c)
            for u, c in zip(self.y_test['user_rating'], self.y_test['metacritic_score'])
        ]
        pred_q = [
            self.get_quadrant(u, c)
            for u, c in zip(preds_user, preds_critic)
        ]

        print("\n=== QUADRANT CLASSIFICATION ===")
        print("Accuracy:", accuracy_score(true_q, pred_q))
        print("Macro F1:", f1_score(true_q, pred_q, average="macro"))

        labels = [
            "High User / High Critic",
            "High User / Low Critic",
            "Low User / High Critic",
            "Low User / Low Critic"
        ]
        cm = confusion_matrix(true_q, pred_q, labels=labels)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title("Quadrant Prediction Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def plot_actual_quadrant(self):
        plt.figure(figsize=(10, 7))
        plt.scatter(self.df['user_rating'], self.df['metacritic_score'], alpha=0.6)
        plt.axvline(self.user_boundary, linestyle="--")
        plt.axhline(self.critic_boundary, linestyle="--")
        plt.title("Actual Magic Quadrant")
        plt.xlabel("User Rating")
        plt.ylabel("Critic Rating")
        plt.grid(alpha=0.3)
        plt.show()

    def plot_predicted_quadrant(self):
        preds = self.model.predict(self.X_test)
        preds_user = preds[:, 0]
        preds_critic = preds[:, 1]

        df_plot = self.X_test.copy()
        df_plot["pred_user"] = preds_user
        df_plot["pred_critic"] = preds_critic
        df_plot["quadrant"] = df_plot.apply(
            lambda r: self.get_quadrant(r["pred_user"], r["pred_critic"]), axis=1
        )

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df_plot, x="pred_user", y="pred_critic",
            hue="quadrant", palette="Set2"
        )
        plt.axvline(self.user_boundary, linestyle="--")
        plt.axhline(self.critic_boundary, linestyle="--")
        plt.title("Predicted Magic Quadrant")
        plt.xlabel("Predicted User Rating")
        plt.ylabel("Predicted Critic Rating")
        plt.grid(alpha=0.3)
        plt.show()
    
    def side_by_side_quadrant_plots(self):
        # Actual Quadrant
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(self.df['user_rating'], self.df['metacritic_score'], alpha=0.6)
        plt.axvline(self.user_boundary, linestyle="--")
        plt.axhline(self.critic_boundary, linestyle="--")
        plt.title("Actual Magic Quadrant")
        plt.xlabel("User Rating")
        plt.ylabel("Critic Rating")
        plt.grid(alpha=0.3)

        # Predicted Quadrant
        preds = self.model.predict(self.X_test)
        preds_user = preds[:, 0]
        preds_critic = preds[:, 1]

        df_plot = self.X_test.copy()
        df_plot["pred_user"] = preds_user
        df_plot["pred_critic"] = preds_critic
        df_plot["quadrant"] = df_plot.apply(
            lambda r: self.get_quadrant(r["pred_user"], r["pred_critic"]), axis=1
        )

        plt.subplot(1, 2, 2)
        sns.scatterplot(
            data=df_plot, x="pred_user", y="pred_critic",
            hue="quadrant", palette="Set2"
        )
        plt.axvline(self.user_boundary, linestyle="--")
        plt.axhline(self.critic_boundary, linestyle="--")
        plt.title("Predicted Magic Quadrant")
        plt.xlabel("Predicted User Rating")
        plt.ylabel("Predicted Critic Rating")
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("rawg_games.csv")
    df = df[['genres', 'developer', 'publisher', 'user_rating', 'metacritic_score']]

    classifier = Classifier(df)
    classifier.train()
    classifier.evaluate()
    classifier.plot_actual_quadrant()
    classifier.plot_predicted_quadrant()
    classifier.side_by_side_quadrant_plots()
