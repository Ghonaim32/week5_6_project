import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2_contingency, zscore, shapiro, ttest_ind
from statsmodels.stats.proportion import proportions_ztest
pd.set_option('display.max_columns', None)


def convert_time_spent(df):
    df['time_spent'] = df['time_spent'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    return df

# ------------------------------------- Cleaning--------------------------------


# ---------- 1. LOAD DATA ----------

def load_data():
    """Load all raw data files and return as DataFrames."""
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))

    df_final_demo = pd.read_csv(os.path.join(base_path, "df_final_demo.txt"))
    df_final_experiment_clients = pd.read_csv(os.path.join(base_path, "df_final_experiment_clients.txt"))
    df_final_web_data_pt_1 = pd.read_csv(os.path.join(base_path, "df_final_web_data_pt_1.txt"))
    df_final_web_data_pt_2 = pd.read_csv(os.path.join(base_path, "df_final_web_data_pt_2.txt"))

    pd.set_option('display.max_columns', None)
    return df_final_demo, df_final_experiment_clients, df_final_web_data_pt_1, df_final_web_data_pt_2

# ---------- 2. MERGE + CONCAT ----------

def merge_data(df_demo, df_clients, df_web1, df_web2):
    """Concatenate web data parts and merge with experiment clients."""
    df_web_data = pd.concat([df_web1, df_web2], ignore_index=True)

    df_web_experiment = pd.merge(
        df_web_data,
        df_clients,
        on="client_id",
        how="inner"
    )
    return df_web_experiment

# ---------- 3. CLEAN WEB DATA ----------

def clean_web_data(df_web_experiment):
    """Clean and prepare web experiment data."""
    df = df_web_experiment.drop_duplicates().copy()

    # Replace step names with numeric codes
    df["process_step"] = df["process_step"].replace({
        "start": 1,
        "step_1": 2,
        "step_2": 3,
        "step_3": 4,
        "confirm": 5
    }).astype(int)

    # Fix datetime format
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

    # Sort logically
    df = df.sort_values(by=["client_id", "visit_id", "date_time"], ascending=[True, True, True])
    return df

# ---------- 4. FEATURE ENGINEERING ----------

def add_features(df):
    """Add time-based, step-based, and error-related columns."""
    df = df.copy()

    # Time from previous step
    df["time_from_prev_step"] = (
        df.groupby("visit_id")["date_time"]
        .diff()
        .dt.total_seconds()
        .astype("Int64")
        .fillna(0)
    )

    # Time from start
    df["time_from_start"] = (
        df.groupby("visit_id")["date_time"]
        .transform(lambda x: x - x.min())
        .dt.total_seconds()
        .astype("Int64")
    )

    # Steps and visits
    df["num_steps"] = df.groupby("visit_id")["process_step"].transform("count")
    df["num_visits"] = df.groupby("client_id")["visit_id"].transform("nunique")

    # Last step per visit
    last_step = (
        df.groupby("visit_id", as_index=False)["process_step"]
        .max()
        .rename(columns={"process_step": "last_step"})
    )
    df = df.merge(last_step, on="visit_id", how="left")

    # Completion and error flags
    df["completed"] = df["last_step"] == 5
    df["step_repeat_count"] = df.groupby(["visit_id", "process_step"])["process_step"].transform("count")
    df["total_steps"] = df.groupby("visit_id")["process_step"].transform("nunique")
    df["step_diff"] = df.groupby("visit_id")["process_step"].diff()
    df["error_flag"] = df["step_diff"] < 0

    return df

# ---------- 5. MERGE ALL ----------

def merge_all(df_web, df_demo):
    """Merge web data with demo table."""
    df_raw = pd.merge(df_web, df_demo, on="client_id", how="left")
    return df_raw

# ---------- 6. CLEAN FINAL DF ----------

def clean_final(df):
    """Clean data types, round values, and add age group."""
    df = df.copy()

    int_cols = ['client_id', 'clnt_tenure_yr', 'clnt_tenure_mnth', 'clnt_age',
                'num_accts', 'calls_6_mnth', 'logons_6_mnth']
    float_col = 'bal'

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')

    df[float_col] = pd.to_numeric(df[float_col], errors='coerce').round(2)

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df = df.sort_values(by=["client_id", "visit_id", "date_time"])

    bins = [18, 25, 35, 45, 55, 65, 100]
    labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    df["age_group"] = pd.cut(df["clnt_age"], bins=bins, labels=labels, right=False)

    df["gendr"] = df["gendr"].str.replace("X", "U")

    return df


# ------------------------------------- Analysis--------------------------------

def load_df_raw():
    df_raw = pd.read_csv(r"/Users/muayadhilamia/Desktop/Ironhack/Week-5/Project/week5_6_project/data/cleaned/df_cleand_raw_m.csv")
    return df_raw

# ---------- DATA CLEANING ----------
def clean_main_df(df_raw):
    """Clean the main dataset and prepare it for analysis."""
    df = df_raw.copy()

    # Remove 'Unnamed' column if exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Drop missing or empty Variation
    df = df[df['Variation'].notna() & (df['Variation'] != "")]

    # Integer conversion
    int_cols = [
        'client_id', 'clnt_tenure_yr', 'clnt_tenure_mnth', 
        'clnt_age', 'num_accts', 'calls_6_mnth', 'logons_6_mnth',
        'num_steps', 'num_visits', 'total_steps', 
        'step_repeat_count', 'step_diff', "time_from_prev_step"
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # Float conversion
    if 'bal' in df.columns:
        df['bal'] = pd.to_numeric(df['bal'], errors='coerce').round(2)

    # Fix date_time
    if 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')

    # Drop duplicates and sort
    df = df.drop_duplicates().sort_values(
        by=['client_id', 'visit_id', 'date_time'],
        ascending=[True, True, True]
    )

    # Step labels
    step_labels = {1: "Start", 2: "Step 1", 3: "Step 2", 4: "Step 3", 5: "Confirm"}
    df['step_label'] = df['process_step'].map(step_labels)

    return df


# ---------- KPI CALCULATIONS ----------
def calculate_error_rate(df):
    total_visits = df.groupby('Variation')['visit_id'].nunique()
    total_errors = df[df['error_flag'] == 1].groupby('Variation').size()
    error_rate = total_errors / total_visits
    return error_rate


def calculate_completion_rate(df):
    df_last_visit = df.drop_duplicates(subset="visit_id", keep="last")
    completion_rate = df_last_visit.groupby('Variation')['completed'].mean()
    return completion_rate, df_last_visit


# ---------- HYPOTHESIS TESTS ----------
def hypothesis_error_rate(df, alpha=0.05):
    """Run error rate hypothesis test and print full interpretation."""
    errors_per_visit = (
        df.groupby(['visit_id', 'Variation'])['error_flag']
        .sum()
        .reset_index(name='error_count')
    )
    errors_per_visit['has_error'] = errors_per_visit['error_count'] > 0

    test_group = errors_per_visit[errors_per_visit['Variation'] == 'Test']
    control_group = errors_per_visit[errors_per_visit['Variation'] == 'Control']

    error_counts = [test_group['has_error'].sum(), control_group['has_error'].sum()]
    error_n = [len(test_group), len(control_group)]

    z_stat, p_val = proportions_ztest(error_counts, error_n, alternative='smaller')

    print("\n******** ERROR RATE HYPOTHESIS ********")
    print(f"Z-statistic: {z_stat:.3f}")
    print(f"P-value: {p_val:.4f}")

    if p_val < alpha:
        print("✅ Test group has significantly LOWER error rate than Control.")
    else:
        print("❌ No statistical evidence that Test had fewer errors than Control.")
    print(f"Test group error rate: {test_group['has_error'].mean():.3%}")
    print(f"Control group error rate: {control_group['has_error'].mean():.3%}")
    print("***************************************")

    return z_stat, p_val


def hypothesis_completion_rate(df_last_visit, alpha=0.05):
    """Run completion rate hypothesis test and print full interpretation."""
    test_group = df_last_visit[df_last_visit['Variation'] == 'Test']
    control_group = df_last_visit[df_last_visit['Variation'] == 'Control']

    completion_counts = [test_group['completed'].sum(), control_group['completed'].sum()]
    completion_n = [len(test_group), len(control_group)]

    z_stat, p_val = proportions_ztest(completion_counts, completion_n, alternative='larger')

    print("\n******** COMPLETION RATE HYPOTHESIS ********")
    print(f"Z-statistic: {z_stat:.3f}")
    print(f"P-value: {p_val:.4f}")

    if p_val < alpha:
        print("✅ Test group has significantly HIGHER completion rate than Control.")
    else:
        print("❌ No statistical evidence that Test performed better in completion rate.")

    print(f"Test group completion rate: {test_group['completed'].mean():.3%}")
    print(f"Control group completion rate: {control_group['completed'].mean():.3%}")
    print("*******************************************")

    return z_stat, p_val


# ---------- OUTLIER CLEANING ----------
def remove_outliers_iqr(group):
    Q1 = group['time_from_prev_step'].quantile(0.25)
    Q3 = group['time_from_prev_step'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return group[(group['time_from_prev_step'] >= lower) & (group['time_from_prev_step'] <= upper)]


# ---------- T-TEST BY STEP ----------
def ttest_step_times(df):
    results = []
    for step in sorted(df['step_label'].dropna().unique()):
        test = df[(df['Variation'] == 'Test') & (df['step_label'] == step)]['time_from_prev_step']
        control = df[(df['Variation'] == 'Control') & (df['step_label'] == step)]['time_from_prev_step']
        t_stat, p_val = ttest_ind(test, control, alternative='less')

        results.append({
            'step': step,
            'test_mean': round(test.mean(), 2),
            'control_mean': round(control.mean(), 2),
            'mean_diff': round(control.mean() - test.mean(), 2),
            'p_value': round(p_val, 4),
            'significant': p_val < 0.05
        })
    return pd.DataFrame(results)


# ---------- COMPARE STEP DURATIONS ----------
def compare_step_durations(df):
    """Compare mean time per step between groups (with and without outliers), ordered by logical step sequence."""
    step_order = ["Start", "Step 1", "Step 2", "Step 3", "Confirm"]

    print("\n================ STEP DURATION COMPARISON ================")
    print(">>> With Outliers <<<")
    results_with = ttest_step_times(df)
    results_with['step'] = pd.Categorical(results_with['step'], categories=step_order, ordered=True)
    results_with = results_with.sort_values('step')
    print(results_with)

    # Remove outliers
    df_clean = (
        df.groupby(['Variation', 'step_label'], group_keys=False)
        .apply(remove_outliers_iqr)
        .reset_index(drop=True)
    )

    print(f"\n✅ Removed {len(df) - len(df_clean)} outliers using IQR filtering.")
    print("\n>>> After Outlier Removal <<<")
    results_clean = ttest_step_times(df_clean)
    results_clean['step'] = pd.Categorical(results_clean['step'], categories=step_order, ordered=True)
    results_clean = results_clean.sort_values('step')
    print(results_clean)

    print("\n🔍 Significant improvements after filtering:")
    sig = results_clean[results_clean['significant']]
    if not sig.empty:
        for _, row in sig.iterrows():
            print(f"✅ {row['step']}: Test faster by {row['mean_diff']}s (p = {row['p_value']})")
    else:
        print("❌ No significant improvements found after removing outliers.")
    print("==========================================================")

    return results_with, results_clean, df_clean



# ---------- KPI VISUALIZATIONS ----------
def plot_kpis(error_rate, completion_rate, save_dir="results/plots"):
    """Visualize Error Rate and Completion Rate as bar plots and save figures."""
    sns.set(style="whitegrid", palette="muted", font_scale=1.1)
    palette = {"Test": "#1f77b4", "Control": "#ff7f0e"}  # Blue = Test, Orange = Control

    os.makedirs(save_dir, exist_ok=True)

    # --- Error Rate ---
    plt.figure(figsize=(6, 5))
    sns.barplot(
        x=error_rate.index,
        y=error_rate.values * 100,
        palette=palette,
        order=["Test", "Control"]
    )
    plt.title("Error Rate per Visit by Variation")
    plt.xlabel("Variation")
    plt.ylabel("Error Rate (%)")
    plt.ylim(0, max(error_rate.values * 100) * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_rate.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # --- Completion Rate ---
    plt.figure(figsize=(6, 5))
    sns.barplot(
        x=completion_rate.index,
        y=completion_rate.values * 100,
        palette=palette,
        order=["Test", "Control"]
    )
    plt.title("Completion Rate by Variation")
    plt.xlabel("Variation")
    plt.ylabel("Completion Rate (%)")
    plt.ylim(0, max(completion_rate.values * 100) * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "completion_rate.png"), dpi=300, bbox_inches="tight")
    plt.show()


def plot_step_durations(df_with, df_clean, save_dir="results/plots"):
    """
    Visualize step duration (with and without outliers) using both bar plots and box plots.
    Includes additional comparison boxplots with consistent colors.
    """
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Settings
    sns.set(style="whitegrid", palette="muted", font_scale=1.1)
    palette = {"Test": "#1f77b4", "Control": "#ff7f0e"}  # Consistent colors
    step_order = ["Start", "Step 1", "Step 2", "Step 3", "Confirm"]

    os.makedirs(save_dir, exist_ok=True)

    # ---------- BAR PLOTS ----------
    mean_with = (
        df_with.groupby(["Variation", "step_label"])["time_from_prev_step"]
        .mean()
        .reset_index()
    )
    mean_clean = (
        df_clean.groupby(["Variation", "step_label"])["time_from_prev_step"]
        .mean()
        .reset_index()
    )

    mean_with["step_label"] = pd.Categorical(mean_with["step_label"], categories=step_order, ordered=True)
    mean_clean["step_label"] = pd.Categorical(mean_clean["step_label"], categories=step_order, ordered=True)

    # --- With Outliers (Bar) ---
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=mean_with,
        x="step_label",
        y="time_from_prev_step",
        hue="Variation",
        palette=palette
    )
    plt.title("Average Step Duration per Step")
    plt.xlabel("Step")
    plt.ylabel("Time from Previous Step (seconds)")
    plt.legend(title="Variation")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "step_duration_with_outliers_bar.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # --- Without Outliers (Bar) ---
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=mean_clean,
        x="step_label",
        y="time_from_prev_step",
        hue="Variation",
        palette=palette
    )
    plt.title("Average Step Duration per Step (After Outlier Removal)")
    plt.xlabel("Step")
    plt.ylabel("Time from Previous Step (seconds)")
    plt.legend(title="Variation")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "step_duration_no_outliers_bar.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # ---------- BOX PLOTS ----------
    sns.set(style="whitegrid", font_scale=1.1)

    # --- Boxplot (With Outliers Hidden, Raw Data) ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_with,
        x="step_label",
        y="time_from_prev_step",
        hue="Variation",
        order=step_order,
        hue_order=["Test", "Control"],
        palette=palette,
        showfliers=False
    )
    plt.title("Step Duration Distribution by Step")
    plt.xlabel("Step")
    plt.ylabel("Duration (seconds)")
    plt.legend(title="Variation")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "step_duration_with_outliers_box.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # --- Boxplot (After IQR Filtering) ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_clean,
        x="step_label",
        y="time_from_prev_step",
        hue="Variation",
        order=step_order,
        hue_order=["Test", "Control"],
        palette=palette,
        showfliers=False
    )
    plt.title("Step Duration Distribution (After IQR Outlier Removal)")
    plt.xlabel("Step")
    plt.ylabel("Duration (seconds)")
    plt.legend(title="Variation")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "step_duration_no_outliers_box.png"), dpi=300, bbox_inches="tight")
    plt.show()

   