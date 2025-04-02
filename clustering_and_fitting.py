"""
This is the template file for the clustering and fitting assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
Fitting should be done with only 1 target variable and 1 feature variable,
likewise, clustering should be done with only 2 variables.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score


def preprocessing(df):
    """
    Encodes 'species' as numeric and prints an overview of the dataset.
    Also shows correlation among features.
    Returns the preprocessed dataframe.
    """
    print("Data Overview (head):")
    print(df.head())
    print("\nData Overview (tail):")
    print(df.tail())
    print("\nData Statistics:")
    print(df.describe())

    df["species"] = LabelEncoder().fit_transform(df["species"])

    print("\nData Correlations:")
    print(df.corr())

    return df


def plot_relational_plot(df):
    """
    Plots a relational plot between sepal_width and sepal_length.
    Saves the figure as 'relational_plot.png'.
    """
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df["sepal_width"],
        y=df["sepal_length"],
        ax=ax
    )
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Sepal Length (cm)")
    plt.title("Sepal Width vs Sepal Length")
    plt.savefig("relational_plot.png")
    plt.close()


def plot_statistical_plot(df):
    """
    Plots a heatmap of the correlation matrix.
    Saves the figure as 'statistical_plot.png'.
    """
    fig, ax = plt.subplots()
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    plt.title("Feature Correlation Heatmap")
    plt.savefig("statistical_plot.png")
    plt.close()


def plot_categorical_plot(df):
    """
    Plots a boxplot of petal_length by species.
    Saves the figure as 'categorical_plot.png'.
    """
    fig, ax = plt.subplots()
    sns.boxplot(
        x=df["species"],
        y=df["petal_length"],
        ax=ax
    )
    plt.xlabel("Species")
    plt.ylabel("Petal Length (cm)")
    plt.title("Petal Length by Species")
    plt.savefig("categorical_plot.png")
    plt.close()


def statistical_analysis(df, col):
    """
    Computes mean, standard deviation, skewness, and kurtosis for 'col'.
    Returns a tuple: (mean, std, skew, kurt)
    """
    mean_val = df[col].mean()
    std_val = df[col].std()
    skew_val = ss.skew(df[col])
    kurt_val = ss.kurtosis(df[col])
    return mean_val, std_val, skew_val, kurt_val


def writing(moments, col):
    """
    Prints the statistical analysis results for 'col'.
    'moments' is the tuple from 'statistical_analysis'.
    """
    print(
        f"Mean={moments[0]:.2f}, SD={moments[1]:.2f}, "
        f"Skew={moments[2]:.2f}, Kurtosis={moments[3]:.2f}."
    )
    skew_type = "right" if moments[2] > 2 else (
        "left" if moments[2] < -2 else "not"
    )
    kurt_type = (
        "leptokurtic" if moments[3] > 2 else (
            "platykurtic" if moments[3] < -2 else "mesokurtic"
        )
    )
    print(f"The data was {skew_type} skewed and {kurt_type}.")
    print()


def perform_clustering(df, col1, col2):
    """
    Performs K-Means clustering on (col1, col2) with scaling.
    Returns the cluster labels, scaled data, cluster centers (unscaled),
    and the silhouette score.
    Also produces an elbow plot (1..10) saved as 'elbow_plot.png'.
    """

    def plot_elbow_method(df_local):
        """Creates elbow plot for best K selection."""
        scaler_temp = StandardScaler()
        data_temp = scaler_temp.fit_transform(df_local[[col1, col2]])

        inertia_values = []
        for k in range(1, 11):
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_temp.fit(data_temp)
            inertia_values.append(kmeans_temp.inertia_)

        fig, ax = plt.subplots()
        ax.plot(range(1, 11), inertia_values, marker="o")
        plt.xlabel("Clusters")
        plt.ylabel("Inertia")
        plt.title("Elbow Method")
        plt.savefig("elbow_plot.png")
        plt.close()

    plot_elbow_method(df)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[[col1, col2]])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    silhouette_val = silhouette_score(data_scaled, labels)

    centers_scaled = kmeans.cluster_centers_
    centers_unscaled = scaler.inverse_transform(centers_scaled)

    return (
        labels,
        data_scaled,
        centers_unscaled[:, 0],
        centers_unscaled[:, 1],
        silhouette_val
    )


def plot_clustered_data(labels, data_scaled, xkmeans, ykmeans, silhouette_val):
    """
    Visualizes clusters with centroids in a scatter plot.
    'data_scaled' is in scaled domain, so be mindful of axis labeling
    if you want to show actual cm values. For demonstration, we keep
    it scaled & label accordingly.
    """
    fig, ax = plt.subplots()
    plt.scatter(
        data_scaled[:, 0],
        data_scaled[:, 1],
        c=labels,
        alpha=0.6,
        cmap="viridis"
    )
    plt.scatter(
        xkmeans,
        ykmeans,
        color="red",
        marker="X",
        s=200,
        label="Centroids"
    )
    plt.xlabel("Petal Length (scaled)")
    plt.ylabel("Petal Width (scaled)")
    plt.title(f"K-Means Clustering\nSilhouette Score: {silhouette_val:.2f}")
    plt.legend()
    plt.savefig("clustering.png")
    plt.close()


def perform_fitting(df, col_x, col_y):
    """
    Fits linear regression on col_x -> col_y.
    Scales col_x and returns:
        data_scaled       : scaled [col_x, col_y]
        x_pred_unscaled   : inverse-transformed x-range for plot
        y_pred            : predicted y values
        original_col      : original unscaled col_x data
    """
    df_local = df.copy()
    original_feature_data = df_local[col_x].values.copy()

    scaler = StandardScaler()
    df_local[[col_x]] = scaler.fit_transform(df_local[[col_x]])

    X = df_local[[col_x]].values
    y = df_local[col_y].values

    model = LinearRegression()
    model.fit(X, y)

    x_min, x_max = X.min(), X.max()
    x_pred_scaled = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_pred = model.predict(x_pred_scaled)

    x_pred_unscaled = scaler.inverse_transform(x_pred_scaled)
    data_scaled = df_local[[col_x, col_y]].values

    return data_scaled, x_pred_unscaled, y_pred, original_feature_data


def plot_fitted_data(
    data_scaled, x_unscaled, y_pred,
    original_feature_data, col_x, col_y
):
    """
    Plots the regression model using unscaled X and predictions.
    """
    fig, ax = plt.subplots()
    plt.scatter(
        original_feature_data,
        data_scaled[:, 1],
        alpha=0.6,
        label="Data (unscaled X)"
    )
    plt.plot(x_unscaled, y_pred, color="red", label="Fitted Line")
    plt.xlabel(f"{col_x} (cm) [unscaled]")
    plt.ylabel(f"{col_y} (cm)")
    plt.title(f"Linear Regression: {col_x} vs {col_y}")
    plt.legend()
    plt.savefig("fitting.png")
    plt.close()


def main():
    """Executes preprocessing, analysis, clustering, and fitting."""
    df = pd.read_csv("data.csv")
    df = preprocessing(df)

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    col = "petal_length"
    stats = statistical_analysis(df, col)
    writing(stats, col)

    cluster_out = perform_clustering(df, "petal_length", "petal_width")
    plot_clustered_data(*cluster_out)

    print(f"Silhouette Score: {cluster_out[-1]:.2f}")

    data_scaled, x_unscaled, y_pred, original_x_data = perform_fitting(
        df, "sepal_width", "sepal_length"
    )
    plot_fitted_data(
        data_scaled, x_unscaled, y_pred,
        original_x_data,
        col_x="sepal_width",
        col_y="sepal_length"
    )


if _name_ == "_main_":
    main()
