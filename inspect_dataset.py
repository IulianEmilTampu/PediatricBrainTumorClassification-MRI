import glob
import os

total_file = glob.glob(
    os.path.join(
        "/flush/iulta54/Research/Data/CBTN/EXTRACTED_SLICES_TFR_MERGED_FROM_TB_20230320/T2",
        "*.tfrecords",
    )
)

print(f"Total files: {len(total_file)}")

unique_IDs = list(
    dict.fromkeys([os.path.basename(f).split("_")[2] for f in total_file])
)

print(f"Unique_IDS: {len(unique_IDs)}")

unique_IDs_infa = list(
    dict.fromkeys(
        [
            os.path.basename(f).split("_")[2]
            for f in total_file
            if os.path.basename(f).split("_")[1] == "infra"
        ]
    )
)

print(f"Unique_IDS infra: {len(unique_IDs_infa)}")

unique__infra_scans = list(
    dict.fromkeys(
        [
            "_".join(os.path.basename(f).split("_")[2:4])
            for f in total_file
            if os.path.basename(f).split("_")[1] == "infra"
        ]
    )
)

print(f"Unique_IDS infra scans: {len(unique__infra_scans)}")
