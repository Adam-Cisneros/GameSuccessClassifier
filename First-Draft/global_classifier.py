import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def build_heuristic_predictions(df_train, df_pred, k=5, weights=None):
    """
    df_train: DataFrame used to compute historical averages (training set)
    df_pred: DataFrame to produce heuristic predictions for
    k: shrinkage constant (higher -> more shrink toward global mean)
    weights: dict of weights for ['developer','publisher','genres']
    Returns df_pred with columns 'heur_user' and 'heur_critic'
    """
    if weights is None:
        weights = {'developer': 0.35, 'publisher': 0.15, 'genres': 0.5}

    # global means
    global_user = df_train['user_rating'].mean()
    global_critic = df_train['metacritic_score'].mean()

    # helper to compute blended mean for a grouping column and target
    def blended_group_mean(df, by_col, target_col):
        grp = df.groupby(by_col)[target_col].agg(['mean','count']).reset_index()
        grp['blended'] = (grp['count'] * grp['mean'] + k * df[target_col].mean()) / (grp['count'] + k)
        return grp.set_index(by_col)['blended'].to_dict(), grp.set_index(by_col)['count'].to_dict()

    dev_user_mean, dev_user_count = blended_group_mean(df_train, 'developer', 'user_rating')
    pub_user_mean, pub_user_count = blended_group_mean(df_train, 'publisher', 'user_rating')
    gen_user_mean, gen_user_count = blended_group_mean(df_train, 'genres', 'user_rating')

    dev_critic_mean, _ = blended_group_mean(df_train, 'developer', 'metacritic_score')
    pub_critic_mean, _ = blended_group_mean(df_train, 'publisher', 'metacritic_score')
    gen_critic_mean, _ = blended_group_mean(df_train, 'genres', 'metacritic_score')

    def safe_lookup(mapping, key, fallback):
        return mapping.get(key, fallback)

    heur_users = []
    heur_critics = []
    for _, row in df_pred.iterrows():
        dev = row['developer']
        pub = row['publisher']
        gen = row['genres']

        # lookups with fallback to global mean
        dev_user = safe_lookup(dev_user_mean, dev, global_user)
        pub_user = safe_lookup(pub_user_mean, pub, global_user)
        gen_user = safe_lookup(gen_user_mean, gen, global_user)

        dev_crit = safe_lookup(dev_critic_mean, dev, global_critic)
        pub_crit = safe_lookup(pub_critic_mean, pub, global_critic)
        gen_crit = safe_lookup(gen_critic_mean, gen, global_critic)

        # weighted combination
        heur_user = (weights['developer'] * dev_user +
                     weights['publisher'] * pub_user +
                     weights['genres'] * gen_user)
        heur_crit = (weights['developer'] * dev_crit +
                     weights['publisher'] * pub_crit +
                     weights['genres'] * gen_crit)

        heur_users.append(heur_user)
        heur_critics.append(heur_crit)

    df_pred = df_pred.copy()
    df_pred['heur_user'] = heur_users
    df_pred['heur_critic'] = heur_critics
    return df_pred

def make_quadrant_labels(df, user_col='heur_user', critic_col='heur_critic', user_med=None, critic_med=None):
    """
    Create quadrant labels based on medians.
    If user_med/critic_med None, use medians from df. Typically pass training medians to avoid leakage.
    """
    if user_med is None:
        user_med = df[user_col].median()
    if critic_med is None:
        critic_med = df[critic_col].median()

    def label(u, c):
        if u >= user_med and c >= critic_med:
            return "High User / High Critic"
        elif u >= user_med and c < critic_med:
            return "High User / Low Critic"
        elif u < user_med and c >= critic_med:
            return "Low User / High Critic"
        else:
            return "Low User / Low Critic"

    return df.apply(lambda r: label(r[user_col], r[critic_col]), axis=1), user_med, critic_med

def evaluate_quadrant_preds(true_labels, pred_labels, plot_cm=True):
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro')
    print(f"Accuracy: {acc:.3f}, Macro F1: {f1:.3f}")

    labels = ["High User / High Critic", "High User / Low Critic",
              "Low User / High Critic", "Low User / Low Critic"]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    if plot_cm:
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix (Heuristic)")
        plt.xticks(rotation=45, ha='right')
        plt.show()
    return acc, f1, cm

def plot_magic_quadrant_actual_vs_heuristic(df_actual, df_heur, user_med, critic_med, sample_labels=None):
    """
    df_actual: dataframe with actual user_rating/metacritic_score and title
    df_heur: dataframe with heur_user / heur_critic and title
    user_med, critic_med: medians used for quadrant split (usually training medians)
    sample_labels: list of game titles to annotate (optional)
    """
    plt.figure(figsize=(12,6))

    # Left: actual
    plt.subplot(1,2,1)
    plt.scatter(df_actual['user_rating'], df_actual['metacritic_score'], alpha=0.6)
    plt.axvline(user_med, linestyle='--'); plt.axhline(critic_med, linestyle='--')
    plt.title("Actual Ratings")
    plt.xlabel("User Rating"); plt.ylabel("Critic (Metacritic)")

    # Right: heuristic predictions
    plt.subplot(1,2,2)
    plt.scatter(df_heur['heur_user'], df_heur['heur_critic'], alpha=0.6)
    plt.axvline(user_med, linestyle='--'); plt.axhline(critic_med, linestyle='--')
    plt.title("Heuristic Predictions")
    plt.xlabel("Heuristic User"); plt.ylabel("Heuristic Critic")

    plt.tight_layout()
    plt.show()

    # optional annotation for a few games to show differences
    if sample_labels:
        plt.figure(figsize=(8,6))
        for t in sample_labels:
            arow = df_actual[df_actual['title']==t]
            hrow = df_heur[df_heur['title']==t]
            if not arow.empty and not hrow.empty:
                plt.scatter(arow['user_rating'], arow['metacritic_score'], c='blue', label='actual' if t==sample_labels[0] else "")
                plt.scatter(hrow['heur_user'], hrow['heur_critic'], c='red', marker='x', label='heuristic' if t==sample_labels[0] else "")
                plt.text(arow['user_rating'].values[0], arow['metacritic_score'].values[0], t, fontsize=8)
        plt.axvline(user_med, linestyle='--'); plt.axhline(critic_med, linestyle='--')
        plt.legend()
        plt.title("Selected Games: actual vs heuristic")
        plt.show()

if __name__ == "__main__":
    df = pd.read_csv("rawg_games.csv")
    df = df[['title','genres','developer','publisher','user_rating','metacritic_score']].dropna()

    # train/test split (use train medians for quadrant thresholds)
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=None)

    # build heuristic predictions on test set (using train stats)
    df_test_heur = build_heuristic_predictions(df_train, df_test, k=5)

    # compute true quadrants (based on actual ratings and TRAIN medians)
    _, user_med_train, critic_med_train = make_quadrant_labels(df_train, user_col='user_rating', critic_col='metacritic_score')

    true_quads = make_quadrant_labels(df_test, user_col='user_rating', critic_col='metacritic_score',
                                      user_med=user_med_train, critic_med=critic_med_train)[0]

    pred_quads = make_quadrant_labels(df_test_heur, user_col='heur_user', critic_col='heur_critic',
                                      user_med=user_med_train, critic_med=critic_med_train)[0]

    # evaluate
    evaluate_quadrant_preds(true_quads, pred_quads, plot_cm=True)

    # visualize actual vs heuristic scatter
    plot_magic_quadrant_actual_vs_heuristic(df_test, df_test_heur, user_med_train, critic_med_train,
                                           sample_labels=df_test['title'].sample(6, random_state=1).tolist())