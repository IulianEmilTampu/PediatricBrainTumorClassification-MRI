# %%
import os
import glob
import pandas as pd
from omegaconf import OmegaConf
import pathlib
from datetime import datetime

# %%
SOURCE = "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/train_model_archive_POST_20231208"
SAVE_PATH = "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/validation_results"
pathlib.Path(
    os.path.join(SAVE_PATH, f'Evaluation_{datetime.now().strftime("%Y%m%d")}')
).mkdir(parents=True, exist_ok=True)

# %% GATHER CVS FILES (summary evaluation and detailed evaluation)
summary_evaluation = []
detailed_evaluation = []
count = 0
for m in glob.glob(os.path.join(SOURCE, "*", "")):
    for t in glob.glob(os.path.join(m, "*", "")):
        for r in glob.glob(os.path.join(t, "REPETITION_*", "")):
            # load configuration file
            training_config = OmegaConf.load(os.path.join(r, "hydra_config.yaml"))
            for e in glob.glob(os.path.join(r, "evaluation_results", "")):
                for s in glob.glob(os.path.join(e, "*")):
                    # get set name
                    set_name = os.path.basename(s)
                    # load summary and detailed evaluation
                    for version, aggregadion in zip(
                        ("summary_performance.csv", "full_evaluation.csv"),
                        (summary_evaluation, detailed_evaluation),
                    ):
                        csv_file = os.path.join(s, version)
                        if os.path.isfile(csv_file):
                            # load CSV
                            df = pd.read_csv(csv_file)
                            # add the set information
                            df["evaluation_set"] = [set_name] * len(df)
                            # add extra information to the dataframe
                            df[
                                "model_version"
                            ] = training_config.model_settings.model_version
                            df["use_age"] = (
                                training_config.dataloader_settings.use_age
                                if "use_age"
                                in training_config.dataloader_settings.keys()
                                else False
                            )
                            df["test_date"] = (
                                training_config.logging_settings.start_day
                                if "start_day"
                                in training_config.logging_settings.keys()
                                else os.path.basename(
                                    os.path.dirname(pathlib.Path(m))
                                ).split("_")[-1]
                            )
                            df[
                                "time_stamp"
                            ] = training_config.logging_settings.start_time
                            df["pretraining"] = (
                                training_config.model_settings.use_SimCLR_pretrained_model
                                if "use_SimCLR_pretrained_model"
                                in training_config.model_settings.keys()
                                else False
                            )
                            df["pretraining_dataset"] = (
                                (
                                    training_config.model_settings.SimCLR_prettrained_model_setitngs.pretraining_dataset
                                    if training_config.model_settings.use_SimCLR_pretrained_model
                                    else None
                                )
                                if "use_SimCLR_pretrained_model"
                                in training_config.model_settings.keys()
                                else None
                            )
                            df[
                                "fine_tuning"
                            ] = training_config.model_settings.percentage_freeze_weights
                            df["repetition"] = int(
                                os.path.basename(pathlib.Path(r)).split("_")[-1]
                            )
                            df["nbr_classes"] = len(
                                list(training_config.dataset.classes_of_interest)
                            )
                            df["classes"] = ", ".join(
                                list(training_config.dataset.classes_of_interest)
                            )
                            df["MLP_nodes"] = len(
                                list(training_config.model_settings.mlp_nodes)
                            )
                            df["age_encoder_MLP_nodes"] = (
                                len(
                                    training_config.model_settings.age_encoder_mlp_nodes
                                )
                                if "age_encoder_mlp_nodes"
                                in training_config.model_settings.keys()
                                else 0
                            )
                            aggregadion.append(df)
                            count += 1
                        else:
                            print(
                                f"Missing {version} file for model {os.path.basename(t)}, {os.path.basename(r)}"
                            )
print(f"Found {count} files.")

# %% REORDER Column ORDER
colunms_ordered = [
    "test_date",
    "time_stamp",
    "model_version",
    "nbr_classes",
    "classes",
    "pretraining",
    "pretraining_dataset",
    "use_age",
    "age_encoder_MLP_nodes",
    "MLP_nodes",
    "fine_tuning",
    "evaluation_set",
    "dataset_version",
    "repetition",
    "fold_nbr",
    "performance_over",
    "matthews_correlation_coefficient",
    "overall_precision",
    "overall_recall",
    "overall_accuracy",
    "overall_f1-score",
    "overall_auc",
    "accuracy",
    "precision",
    "recall",
    "f1-score",
    "auc",
]
df = pd.concat(summary_evaluation)
df = df[colunms_ordered]
df.to_csv(
    os.path.join(SAVE_PATH, f"summary_evaluation_aggregated.csv"),
    index=False,
    index_label=False,
)

# and the fulle evaliation
colunms_ordered = [
    "test_date",
    "time_stamp",
    "model_version",
    "nbr_classes",
    "classes",
    "pretraining",
    "fine_tuning",
    "evaluation_set",
    "pretraining_dataset",
    "MLP_nodes",
    "use_age",
    "age_encoder_MLP_nodes",
    "repetition",
    "subject_IDs",
    "file_path",
    "tumor_relative_position",
    "target",
    "one_hot_encodig",
    "age_in_days",
    "age_normalized",
    "fold_1",
    "fold_2",
    "fold_3",
    "fold_4",
    "fold_5",
    "pred_fold_1",
    "pred_fold_5",
    "pred_fold_3",
    "pred_fold_4",
    "pred_fold_2",
    "per_slice_ensemble",
    "per_slice_per_class_uncertainty",
    "subject_ensemble_pred_fold_1",
    "subject_ensemble_pred_fold_5",
    "subject_ensemble_pred_fold_3",
    "subject_ensemble_pred_fold_4",
    "subject_ensemble_pred_fold_2",
    "subject_entropy_pred_fold_1",
    "subject_entropy_pred_fold_5",
    "subject_entropy_pred_fold_3",
    "subject_entropy_pred_fold_4",
    "subject_entropy_pred_fold_2",
    "overall_subject_ensemble",
    "overall_subject_entropy",
]

df = pd.concat(detailed_evaluation)
df = df[colunms_ordered]
df.to_csv(
    os.path.join(SAVE_PATH, f"detailed_evaluation_aggregated.csv"),
    index=False,
    index_label=False,
)
