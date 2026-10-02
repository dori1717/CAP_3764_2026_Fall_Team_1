import pandas as pd


def get_categorical_columns(df):
    """
    Return the names of the categorical (non-numeric) columns.
    """
    return df.select_dtypes(exclude="number").columns.tolist()


def categorical_summary(df, columns):
    """
    Build one summary table for the categorical variables.

    For each column: number of unique values, the list of categories,
    the most frequent category (mode), its share in percent,
    and the percent of missing values.
    """
    rows = []
    for col in columns:
        counts = df[col].value_counts()
        rows.append({
            "variable": col,
            "n_unique": df[col].nunique(),
            "categories": ", ".join(sorted(counts.index.astype(str))),
            "most_frequent": counts.index[0],
            "most_frequent_pct": round(counts.iloc[0] / len(df) * 100, 1),
            "missing_pct": round(df[col].isna().mean() * 100, 1),
        })
    return pd.DataFrame(rows).set_index("variable")


def value_counts_table(df, col):
    """
    Return counts and percentages for every category of one column.
    """
    counts = df[col].value_counts(dropna=False)
    percent = (counts / len(df) * 100).round(1)
    return pd.DataFrame({"count": counts, "percent": percent})


def target_by_category(df, col, target="G3", pass_mark=10):
    """
    Summarize the target variable for each category of one column.

    Returns count, mean, median and the percent of students
    below the pass mark (G3 < 10 on the 0-20 scale).
    """
    grouped = df.groupby(col)[target]
    table = pd.DataFrame({
        "count": grouped.size(),
        "mean_G3": grouped.mean().round(2),
        "median_G3": grouped.median(),
        "fail_pct": grouped.apply(lambda s: (s < pass_mark).mean() * 100).round(1),
    })
    return table.sort_values("mean_G3", ascending=False)


def category_gap_ranking(df, columns, target="G3"):
    """
    Rank categorical variables by the gap between the highest and
    the lowest category mean of the target variable.
    """
    rows = []
    for col in columns:
        means = df.groupby(col)[target].mean()
        rows.append({
            "variable": col,
            "highest_category": means.idxmax(),
            "highest_mean_G3": round(means.max(), 2),
            "lowest_category": means.idxmin(),
            "lowest_mean_G3": round(means.min(), 2),
            "gap": round(means.max() - means.min(), 2),
        })
    return pd.DataFrame(rows).set_index("variable").sort_values("gap", ascending=False)
