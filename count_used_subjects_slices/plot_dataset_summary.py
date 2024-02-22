# %%
"""
Script that takes the summary scraped .csv file created using the scrape_dataset.py utility and creates summaries of the dataset:
- Visual summaries
- Text summaries
"""

import os
import json
import numpy as np
import pandas as pd
from itertools import chain, combinations
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib import cm
import copy


# %% UTILITIES
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def plot_matrix(df, ax, matrix_labels, *kargs):
    from matplotlib import colors

    # colors
    def _multiply_alpha(c, mult):
        r, g, b, a = colors.to_rgba(c)
        a *= mult
        return colors.to_hex((r, g, b, a), keep_alpha=True)

    facecolor = "auto"
    other_dots_color = 0.18
    with_lines = True
    orientation = "horizontal"

    if facecolor == "auto":
        bgcolor = matplotlib.rcParams.get("axes.facecolor", "white")
        r, g, b, a = colors.to_rgba(bgcolor)
        lightness = colors.rgb_to_hsv((r, g, b))[-1] * a
        facecolor = "black" if lightness >= 0.5 else "white"

    _other_dots_color = (
        _multiply_alpha(facecolor, other_dots_color)
        if isinstance(other_dots_color, float)
        else other_dots_color
    )

    upset = UpSet(df)
    data = upset.intersections
    n_cats = data.index.nlevels

    inclusion = data.index.to_frame().values

    # Prepare styling
    # define stiles
    subset_styles = [{"facecolor": facecolor} for i in range(len(data))]

    styles = [
        [
            (
                subset_styles[i]
                if inclusion[i, j]
                else {"facecolor": _other_dots_color, "linewidth": 0}
            )
            for j in range(n_cats)
        ]
        for i in range(len(data))
    ]

    styles = sum(styles, [])  # flatten nested list
    style_columns = {
        "facecolor": "facecolors",
        "edgecolor": "edgecolors",
        "linewidth": "linewidths",
        "linestyle": "linestyles",
        "hatch": "hatch",
    }
    styles = pd.DataFrame(styles).reindex(columns=style_columns.keys())
    styles["linewidth"].fillna(1, inplace=True)
    styles["facecolor"].fillna(facecolor, inplace=True)
    styles["edgecolor"].fillna(styles["facecolor"], inplace=True)
    styles["linestyle"].fillna("solid", inplace=True)
    del styles["hatch"]

    x = np.repeat(np.arange(len(data)), n_cats)
    y = np.tile(np.arange(n_cats), len(data))

    # Plot dots
    element_size = 35
    if element_size is not None:
        s = (element_size * 0.35) ** 2
    else:
        # TODO: make s relative to colw
        s = 200

    def _swapaxes(x, y, orientation="horizontal"):
        if orientation == "horizontal":
            return x, y
        return y, x

    ax.scatter(
        *_swapaxes(x, y),
        s=s,
        zorder=10,
        **styles.rename(columns=style_columns),
    )

    # plot lines
    if with_lines:
        idx = np.flatnonzero(inclusion)
        line_data = (
            pd.Series(y[idx], index=x[idx]).groupby(level=0).aggregate(["min", "max"])
        )
        colors = pd.Series(
            [
                style.get("edgecolor", style.get("facecolor", facecolor))
                for style in subset_styles
            ],
            name="color",
        )
        line_data = line_data.join(colors)
        ax.vlines(
            line_data.index.values,
            line_data["min"],
            line_data["max"],
            lw=3,
            colors="k",
            zorder=5,
        )

    # Ticks and axes
    tick_axis = ax.yaxis
    tick_axis.set_ticks(np.arange(n_cats))
    tick_axis.set_ticklabels(
        matrix_labels, rotation=0 if orientation == "horizontal" else -90
    )
    ax.xaxis.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    if not orientation == "horizontal":
        ax.yaxis.set_ticks_position("top")
    ax.set_frame_on(False)
    ax.set_xlim(-0.5, x[-1] + 0.5, auto=False)
    ax.grid(False)


# %% PATHS
PER_session_CSV_FILE_PATH = "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/count_used_subjects_slices/summary_for_plotting.csv"
SAVE_PATH = os.path.dirname(PER_session_CSV_FILE_PATH)

# open files
per_session_summary_csv = pd.read_csv(PER_session_CSV_FILE_PATH, low_memory=False)
per_session_summary_csv["gender"] = per_session_summary_csv.apply(
    lambda x: str(x.gender).title(), axis=1
)

# define which modalities and diagnosis to account for the VISUAL summaries
DIAGNOSIS_OF_INTEREST = [
    "ASTROCYTOMA",
    "EPENDYMOMA",
    "MEDULLOBLASTOMA",
]

# %% compact information (one row for each unique subject and session status - pre_op, post_op, unknown) with as many additional columns as the mr modalities
if not PER_SUBJECT_CSV_FILE_PATH:
    available_MR_SEQUENCES_OF_INTEREST = list(
        pd.unique(per_session_summary_csv["image_type"])
    )
    unique_subjects = list(pd.unique(per_session_summary_csv["subjectID"]))

    df_rows = []
    count_subj_no_diagnosis = {"pre_op": 0, "post_op": 0, "unknown": 0}

    for subj in unique_subjects:
        # get subject information
        subject_df = per_session_summary_csv.loc[
            per_session_summary_csv["subjectID"] == subj
        ]
        for session_status in ["pre_op", "post_op", "unknown"]:
            if any(subject_df["session_status"] == session_status):
                aus_dict = {
                    "subjectID": subj,
                    "gender": list(pd.unique(subject_df["gender"]))[0],
                    "survival": list(pd.unique(subject_df["survival"]))[0],
                    "ethnicity": list(pd.unique(subject_df["ethnicity"]))[0],
                    "scanner": list(pd.unique(subject_df["scanner"]))[0],
                    "session_status": session_status,
                }
                # work only on the ones with the appropriate session status
                subject_df = subject_df.loc[
                    subject_df["session_status"] == session_status
                ]
                # get diagnosis (take the one at the earliest age_at_diagnosis)
                # drop those that have no diagnosis (but count them)
                if any(
                    subject_df["diagnosis"]
                    != "AGE_AT_HISTOLOGY_DIAGNOSIS_NOT_AVAILABLE"
                ):
                    aus_dict["age_at_diagnosis"] = np.min(
                        subject_df["age_at_diagnosis"]
                    )
                    aus_dict["diagnosis"] = list(
                        pd.unique(
                            subject_df.loc[
                                subject_df["age_at_diagnosis"]
                                == aus_dict["age_at_diagnosis"]
                            ]["diagnosis"]
                        )
                    )[0]

                    # count the different modalities
                    for mr_modality in available_MR_SEQUENCES_OF_INTEREST:
                        aus_dict[mr_modality] = len(
                            subject_df.loc[subject_df["image_type"] == mr_modality]
                        )

                    # save information
                    df_rows.append(aus_dict)
                else:
                    count_subj_no_diagnosis[session_status] += 1

    # save so it is available for next time
    per_subject_summary_csv = pd.DataFrame(df_rows)
    per_subject_summary_csv.to_csv(
        os.path.join(SAVE_PATH, "radiology_per_subject_scraped_data.csv")
    )

# %% PLOT NUMBER OF SUBJECT PER DIAGNOSIS
SAVE = False
per_subject_diagnosis_summary = (
    per_session_summary_csv.groupby("subjectID")
    .apply(lambda x: pd.unique(x.diagnosis)[0])
    .reset_index()
    .rename(columns={"subjectID": "subjectID", 0: "diagnosis"})
)
df_diagnosis_sums = pd.DataFrame(
    data=[
        (
            d,
            len(
                per_subject_diagnosis_summary.loc[
                    per_subject_diagnosis_summary["diagnosis"] == d
                ]
            ),
        )
        for d in DIAGNOSIS_OF_INTEREST
    ],
    columns=["diagnosis", "count"],
)
# sort based on the count (largest to smallest)
df_diagnosis_sums = df_diagnosis_sums.sort_values(by=["count"])
# bar plot
plt.figure(figsize=(10, 10))
plt.barh(df_diagnosis_sums["diagnosis"], df_diagnosis_sums["count"], color="royalblue")
plt.ylabel("Diagnosis")
plt.xlabel("Number of Subjects")
plt.title("Number of Subjects per Diagnosis")
for i, v in enumerate(df_diagnosis_sums["count"]):
    plt.text(v + 0.5, i, f"{v}", color="black", va="center")

if SAVE:
    plt.savefig(
        fname=os.path.join(
            SAVE_PATH, f"radiology_ar_chart_per_diagnosis_number_of_subjects.pdf"
        ),
        dpi=200,
        format="pdf",
    )

# %% DIAGNOSIS AND GENDER COUNT
import seaborn as sns

SAVE = True
# smallest age for each unique combination of 'subjectID' and 'diagnosis'
df_violin = (
    per_session_summary_csv.groupby(["subjectID", "diagnosis", "gender"])[
        "age_at_diagnosis"
    ]
    .min()
    .reset_index()
)

colors = {"Male": "royalblue", "Female": "hotpink", "Nan": "grey"}

# violin plot
plt.figure(figsize=(12, 8))
ax = sns.violinplot(
    x="diagnosis",
    y="age_at_diagnosis",
    hue="gender",
    hue_order=list(df_violin.groupby(["gender"]).groups.keys()),
    order=list(df_violin.groupby(["diagnosis"]).groups.keys()),
    data=df_violin,
    palette=colors,
    cut=0,
)

# make pretty
plt.ylim(-1500, df_violin.age_at_diagnosis.max())
plt.legend(title="")
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.set_facecolor("whitesmoke")

plt.xlabel("Diagnosis")
plt.ylabel("Age at Diagnosis [days]")
plt.xticks(ha="center")

# add counts below
ax.axhline(y=-500, color="k", linestyle="--")

for idx, (diagnosis, gender) in enumerate(
    df_violin.groupby(["diagnosis", "gender"]).groups.keys()
):
    group_data = df_violin[
        (df_violin["diagnosis"] == diagnosis) & (df_violin["gender"] == gender)
    ]
    count = len(group_data)
    ypos = -1000
    xpos = per_session_summary_csv["diagnosis"].unique().tolist().index(
        diagnosis
    ) + 0.2 * (-1.3 if gender == "Male" else 0 if gender == "Female" else 1.4)
    ax.text(xpos, ypos, f"N={count}", ha="center", weight="bold")

if SAVE:
    plt.savefig(
        fname=os.path.join(SAVE_PATH, f"Diagnosis_gender_violin_plot.pdf"),
        dpi=200,
        format="pdf",
    )

    plt.savefig(
        fname=os.path.join(SAVE_PATH, f"Diagnosis_gender_violin_plot.png"),
        dpi=200,
        format="png",
    )
