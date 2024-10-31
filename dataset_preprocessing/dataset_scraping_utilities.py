import os
import csv
import glob
import pathlib
import openpyxl
import numpy as np
import json
import pandas as pd
from copy import deepcopy
import nibabel as nib
import pickle


# %% UTILITIES
def get_diagnosis(primary_diagnosis, diagnosis_description):
    """
    Utility that given the list of primary diagnosis and description tries to
    identify the tumor type.

    Returns
    tumor_type : the identified tumor type. Set to None if it fails
    description : list of additional descriptions is case primaty diagnosis is Other
    grade : LGG or HGG if it can be infered by the diagnosis, else None
    """

    tumor_types = {
        "Atypical Teratoid Rhabdoid Tumor (ATRT)": "ATRT",
        "Not Reported": "None",
        "Other": "None",
        "Cavernoma": "CAVERNOMA",
        "Brainstem glioma- Diffuse intrinsic pontine glioma": "DIPG",
        "High-grade glioma/astrocytoma (WHO grade III/IV)": "ASTROCYTOMA",
        "Low-grade glioma/astrocytoma (WHO grade I/II)": "ASTROCYTOMA",
        "Germinoma": "GERMINOMA",
        "Ganglioglioma": "GANGLIOGLIOMA",
        "Teratoma": "TERATOMA",
        "Ependymoma": "EPENDYMOMA",
        "Neurofibroma/Plexiform": "NEUROFIBRILOMA-PLEXIFORM",
        "Low-grade glioma/astrocytoma (WHO grade I/II); Other": "ASTROCYTOMA-OTHERS",
        "Medulloblastoma": "MEDULLOBLASTOMA",
        "Ganglioglioma; Low-grade glioma/astrocytoma (WHO grade I/II)": "GANGLIOGLIOMA-ASTROCYTOMA",
        "Meningioma": "MENINGIOMA",
        "Embryonal Tumor with Multilayered Rosettes (ETMR); Other": "ETMR-OTHERS",
        "Embryonal Tumor with Multilayered Rosettes (ETMR)": "ETMR",
    }

    description_types = [
        "Glial neoplasm",
        "pilocytic/piloid features",
        "Rhabdoid Tumor" "Osteosarcoma" "Ependymoblastoma",
    ]

    description_types = {
        "Not Applicable": "Not-applicable",
        "Inflammation": "INFLAMATION",
        "Rhabdoid Tumor": "RHABDOID",
        "Proteinaceous debris and small fragment of adenohypophysis": "Proteinaceous debris and small fragment of adenohypophysis",
        "Lesional tissue with pilocytic/piloid features": "Lesional tissue with pilocytic/piloid features",
        "Portions of white matter with a few reactive astrocytes and rare cells with mild atypia": "Portions of white matter with a few reactive astrocytes and rare cells with mild atypia",
        "Glial neoplasm": "GLIAL-NEOPLASM",
        "Low-grade glioma": "LOW-GRADE-GIOMA",
        "Non-diagnostic tissue": "NON-DIAGNOSTIC-TISSUE",
        "Osteosarcoma": "OSTEOSARCOMA",
        "Ependymoblastoma": "EPENDYMOBLASTOMA",
    }

    # if there is no diagnosis left
    if len(primary_diagnosis) == 0:
        tumor_type = None
        description = None
        grade = "Not-available"
        return tumor_type, description, grade
    else:
        diagnosis_list = []
        description_list = []
        grade_list = []
        for idx, d in enumerate(primary_diagnosis):
            if "Other" in d:
                # check the diagnosis description
                check = [
                    diagnosis_description[idx] == t for t in description_types.keys()
                ]
                if any(check):
                    description_list.append(
                        description_types[diagnosis_description[idx]]
                    )
                    grade_list.append("Not available")
            elif not "Not Reported" in d:
                # check which of the tumor types
                check = [d == t for t in tumor_types.keys()]
                if any(check):
                    diagnosis_list.append(tumor_types[d])
                else:
                    diagnosis_list.append("Tumor type not in default")

                # check LGG or HGG
                if "Low-grade" in d:
                    grade_list.append("LGG")
                elif "High-grade" in d:
                    grade_list.append("HGG")
                else:
                    grade_list.append("Not available")

        # return a list of unique elements of diagnosis, descriptions and grades
        tumor_type = list(dict.fromkeys(diagnosis_list))
        description = list(dict.fromkeys(description_list))
        grade = list(dict.fromkeys(grade_list))

        return tumor_type, description, grade


def check_bval_bvec(volumes):
    """
    Given a list of volume that are .bvalues and .bvec, opens the files and
    checks that the information is useful (different from all ones or zeros).
    """
    # try to open .bvec file and check if information is useful
    try:
        bvec_file = [
            v
            for v in volumes
            if (".bvec" in os.path.basename(v) and ".json" not in os.path.basename(v))
        ]
        # there may be more tha one .bvec file, check all of them
        for f in bvec_file:
            aus_flag = []
            with open(f) as file:
                for line in file.readlines():
                    # print(line)
                    line = line.replace("\n", " ").replace("\t", " ")
                    aus_flag.append(
                        any(
                            [
                                float(c) != 0.0 and float(c) != 1.0
                                for c in line.strip().split(" ")
                            ]
                        )
                    )
        bvec_usability_flag = any(aus_flag)
    except:
        bvec_usability_flag = False

    # try to open .bval file and check if information is useful
    try:
        bval_file = [
            v
            for v in volumes
            if (".bval" in os.path.basename(v) and ".json" not in os.path.basename(v))
        ]
        # there may be more tha one .bvec file, check all of them
        for f in bval_file:
            aus_flag = []
            with open(f) as file:
                for line in file.readlines():
                    aus_flag.append(
                        any([c != "0" and c != "1" for c in line.strip().split(" ")])
                    )
        bval_usability_flag = any(aus_flag)
    except:
        bval_usability_flag = False

    # return the file name of the volume that is not .bval or .bvec and the usability
    diff_volume = [
        v
        for v in volumes
        if (".nii.gz" in os.path.basename(v) and ".json" not in os.path.basename(v))
    ]
    return diff_volume, any([bvec_usability_flag, bval_usability_flag])


def get_scan_type(file_path):
    """
    Heuristic that checks the file name and tries to identify the type of scan.
    Fooking for:
    - localizer: localizer
    - t1: se tw, se_t1, t1_se or t1
    - t2
    - t2 FLAIR: t2_flair
    - diffusion: diff, adc, ADC
    """

    # get only file name
    file_name = os.path.basename(file_path)
    # if 'se t1' in file_name or 'se_t1' in file_name or 't1_se' in file_name or 't1' in file_name or 'T1' in file_name:
    if any(target in file_name for target in ["se t1", "se_t1", "t1_se", "t1", "T1"]):
        scan_type = "T1"
    elif any(target in file_name for target in ["t2_flair", "T2_FLAIR"]):
        scan_type = "T2_FLAIR"
    elif any(target in file_name for target in ["t2", "T2"]):
        scan_type = "T2"
    elif any(target in file_name for target in ["flair", "FLAIR"]):
        scan_type = "FLAIR"
    elif any(
        target in file_name
        for target in ["diff", "DIFF", "trace", "TRACE", "adc", "ADC", "EXP", "FA"]
    ):
        scan_type = "DIFFUSION"
        # there are different types of diffusion files
        if any(target in file_name for target in ["TRACE"]):
            scan_type = "DIFFUSION_TRACE"
        elif any(target in file_name for target in ["adc", "ADC"]):
            scan_type = "DIFFUSION_ADC"
        elif any(target in file_name for target in ["EXP"]):
            scan_type = "DIFFUSION_EXP"
        elif any(target in file_name for target in ["FA"]):
            scan_type = "DIFFUSION_FA"
    elif any(target in file_name for target in ["localizer"]):
        scan_type = "LOCALIZER"
    elif any(target in file_name for target in ["Perfusion"]):
        scan_type = "PERFUSION"
    else:
        scan_type = "UNKNOWN"

    return scan_type


def get_file_extension(files):
    """
    Utility that given a list of files returs a list of extensions of the files
    in the input file list
    """

    extensions = []
    for f in files:
        aus = pathlib.Path(f).suffixes
        if len(aus) == 1:
            extensions.append(aus[0])
        elif len(aus) >= 2:
            if aus[-1] == ".json":
                extensions.append(aus[-1])
            else:
                extensions.append("".join(aus[-2::]))

    return extensions


def check_pre_or_post_operative_session(
    clinical_timeline, clinical_diagnosis_type, session_time
):
    """
    Utility function that uses the information in the clinical information file
    to understand if the session is pre or post operative. In particular, using
    the diagnostic_type and the age of the patient (in days) at which it was
    done, one can see if it is the first encounter of the tumor, recurrence
    (defined as tumor re-growth after total resection) and progressive (defined
    as tumor growth after partial resection). Thus, scans that are intermediate
    in time between the initial enounter and recurrence/progressive, are pre
    operative.

    It is important also to check that, if there is only one the status will be
    unknown if the session_time is antecedent the first encounter. If the
    session time is precedent the first encounter, the session status is set to
    pre_op.

    Steps
    - find the time of the Initial CNS Tumor (first encounter)
    - find the time of the first Recurrence or Progression
    - check where the session time locates compared to the initial encounter
        and the recurrence or progression
    """
    try:
        first_encounter_time_idx = max(
            loc
            for loc, val in enumerate(clinical_diagnosis_type)
            if val == "Initial CNS Tumor"
        )
    except:
        # if there is no record of the first encounter
        return "unknown"

    """ from previous version
    if (
        len(clinical_diagnosis_type) > 1
        and first_encounter_time_idx < len(clinical_diagnosis_type) - 1
    ):
        # there are multiple clinical_diagnosis_type and the first encounter is not the last encounter
        if session_time < clinical_timeline[first_encounter_time_idx + 1]:
            return "pre_op"
        else:
            return "post_op"
    elif session_time <= clinical_timeline[first_encounter_time_idx]:
        return "pre_op"
    else:
        return "post_op"
    """
    if session_time <= clinical_timeline[first_encounter_time_idx]:
        return "pre_op"
    else:
        return "post_op"

    """
    Utility that given a file path, tries to find out if the scan was performed
    pre or post contrast agent. Here is mostly important for T1 scans, since
    all the other acquisitions should be pre contrast
    """
    # get only file name
    file_name = os.path.basename(file_path)
    # if 'se t1' in file_name or 'se_t1' in file_name or 't1_se' in file_name or 't1' in file_name or 'T1' in file_name:
    if any(target in file_name.lower() for target in ["post", "p0st", "c+"]):
        contrast_status = "post_contrast"
    else:
        contrast_status = "pre_contrast"

    return contrast_status


def get_empty_overall_subject_info():
    return {
        "gender": "Not_available",
        "ethnicity": "Not_available",
        "age_at_diagnosis": "Not_available",
        "age_at_sample_acquisition": [],
        "diagnosis": "Not_available",
        "overall_survival": "Not_available",
        "progression_free_survival": "Not_available",
        "vital_status": "Not_available",
    }


def get_empty_radiology_session_dict():
    return {
        "T1W": [],
        "T1WGD": [],
        "T2w": [],
        "FLAIR": [],
        "DIFFUSION": [],
        "FSPGR": [],
        "UNKNOWN": [],
        "PERFUSION": [],
        "SWI": [],
        "MAG": [],
        "PHE": [],
        "ASL": [],
        "SWAN": [],
        "CBF": [],
        "pre_post_operation_status": None,
    }


class MRSequenceInvestigator:
    def __init__(self, acquisition_information_dict):
        self.acquisition_information_dict = acquisition_information_dict

    def is_t1w(self):
        # simple t1w sequyence with no T2 or fat suppression
        target = ["t1", "bravo", "mpr"]
        flag = False
        if any(
            [
                t in self.acquisition_information_dict["SeriesDescription"].lower()
                for t in target
            ]
        ):
            flag = True
        return (flag, "T1W")

    def is_t1w_se(self):
        # t1w spin echo
        flag = False
        if self.is_t1w()[0]:
            if self.acquisition_information_dict["ScanningSequence"]:
                if "SE" in self.acquisition_information_dict["ScanningSequence"]:
                    flag = True
        return (flag, "T1W_SE")

    def is_t1w_fl(self):
        # t1w flair
        flag = False
        if self.is_t1w()[0]:
            if (
                "flair"
                in self.acquisition_information_dict["SeriesDescription"].lower()
            ):
                flag = True
        return (flag, "T1W_FL")

    def is_t1w_fspgr(self):
        # t1w fast spoiled gradient
        target = ["fspgr", "bravo"]
        flag = False
        if any(
            [
                t in self.acquisition_information_dict["SeriesDescription"].lower()
                for t in target
            ]
        ):
            flag = True
        return (flag, "T1W_FSPGR")

    def is_fspgr(self):
        # t1w fast spoiled gradient
        target = ["fspgr", "bravo"]
        t1_tr_threshold = 0.02  # seconds
        t1_te_threshold = 0.007  # seconds
        flag = False
        if any(
            [
                t in self.acquisition_information_dict["SeriesDescription"].lower()
                for t in target
            ]
        ):
            # check tr and te times
            if all(
                [
                    self.acquisition_information_dict["EchoTime"],
                    self.acquisition_information_dict["RepetitionTime"],
                ]
            ):
                if all(
                    [
                        float(self.acquisition_information_dict["EchoTime"])
                        <= t1_te_threshold,
                        float(self.acquisition_information_dict["RepetitionTime"])
                        <= t1_tr_threshold,
                    ]
                ):
                    flag = True
        return (flag, "T1W_FSPGR")

    def is_t2w_cube(self):
        # t2w cube sequence (GE)
        target = ["cube"]
        t2_tr_threshold = 2.5  # seconds
        t2_te_threshold = 0.1  # seconds
        flag = False
        if any(
            [
                t in self.acquisition_information_dict["SeriesDescription"].lower()
                for t in target
            ]
        ):
            # check tr and te times
            if all(
                [
                    self.acquisition_information_dict["EchoTime"],
                    self.acquisition_information_dict["RepetitionTime"],
                ]
            ):
                if all(
                    [
                        float(self.acquisition_information_dict["EchoTime"])
                        <= t2_te_threshold,
                        float(self.acquisition_information_dict["RepetitionTime"])
                        >= t2_tr_threshold,
                    ]
                ):
                    flag = True
        return (flag, "T2W")

    def is_t2w_space(self):
        # t2w 3d sequence (Simence)
        target = ["space"]
        t2_tr_threshold = 2.5  # seconds
        t2_te_threshold = 0.5  # seconds
        flag = False
        if any(
            [
                t in self.acquisition_information_dict["SeriesDescription"].lower()
                for t in target
            ]
        ):
            # check tr and te times
            if all(
                [
                    self.acquisition_information_dict["EchoTime"],
                    self.acquisition_information_dict["RepetitionTime"],
                ]
            ):
                if all(
                    [
                        float(self.acquisition_information_dict["EchoTime"])
                        <= t2_te_threshold,
                        float(self.acquisition_information_dict["RepetitionTime"])
                        >= t2_tr_threshold,
                    ]
                ):
                    flag = True
        return (flag, "T2W")

    def is_t1w_mprage(self):
        # t1w magnetization prepared gradient echo
        flag = False
        if self.is_t1w()[0]:
            if self.acquisition_information_dict["ScanningSequence"]:
                if all(
                    [
                        "GR" in self.acquisition_information_dict["ScanningSequence"],
                        "IR" in self.acquisition_information_dict["ScanningSequence"],
                    ]
                ):
                    if self.acquisition_information_dict["SequenceName"]:
                        if (
                            "tfl3d1_16"
                            in self.acquisition_information_dict["SequenceName"]
                        ):
                            if (
                                "MP"
                                in self.acquisition_information_dict["SequenceVariant"]
                            ):
                                flag = True
        return (flag, "T1W_MPRAGE")

    def is_t2w(self):
        # fluid attenuated inversion recovery
        target = ["t2"]
        flag = False
        if any(
            [
                t in self.acquisition_information_dict["SeriesDescription"].lower()
                for t in target
            ]
        ):
            flag = True
        return (flag, "T2W")

    def is_t2w_trim(self):
        # t2w with fat attenuation
        flag = False
        if self.is_t2w()[0]:
            if "ScanOptions" in self.acquisition_information_dict.keys():
                if "FS" in self.acquisition_information_dict["ScanOptions"]:
                    flag = True
        return (flag, "T2W_TRIM")

    def is_t2w_flair(self):
        # t2w with fluid attentuation
        target = [
            "t2_flair",
            "flair",
        ]
        flag = False
        if any(
            [
                t in self.acquisition_information_dict["SeriesDescription"].lower()
                for t in target
            ]
        ):
            flag = True
        return (flag, "T2W_FLAIR")

    def is_flair(self):
        # FLAIR
        target = [
            "flair",
        ]
        flag = False
        if any(
            [
                t in self.acquisition_information_dict["SeriesDescription"].lower()
                for t in target
            ]
        ):
            flag = True
        return (flag, "FLAIR")

    def is_swi(self):
        # susceptibility weighted imaging
        target = ["swi"]
        flag = False
        if any(
            [
                t in self.acquisition_information_dict["SeriesDescription"].lower()
                for t in target
            ]
        ):
            flag = True
        return (flag, "SWI")

    def is_mag(self):
        # magnitude image
        target = ["mag"]
        flag = False
        if any(
            [
                t in self.acquisition_information_dict["SeriesDescription"].lower()
                for t in target
            ]
        ):
            flag = True
        return (flag, "MAG")

    def is_phe(self):
        # phase image
        flag = False
        if any(
            [
                "phase" in t.lower()
                for t in self.acquisition_information_dict["ImageType"]
            ]
        ):
            flag = True
        return (flag, "PHE")

    def is_asl(self):
        # atrial spin echo
        target = ["asl", "pasl"]
        flag = False
        if any(
            [
                any(
                    [
                        "asl" in t.lower()
                        for t in self.acquisition_information_dict["ImageType"]
                    ]
                ),
                any(
                    [
                        t
                        in self.acquisition_information_dict[
                            "SeriesDescription"
                        ].lower()
                        for t in target
                    ]
                ),
            ]
        ):
            flag = True
        return (flag, "ASL")

    def is_swan(self):
        # susceptibility-weighted angiography
        target = ["swan"]
        flag = False
        if any(
            [
                t in self.acquisition_information_dict["SeriesDescription"].lower()
                for t in target
            ]
        ):
            flag = True
        return (flag, "SWAN")

    def is_cbf(self):
        # cerebral blood flow
        target = ["cbf", "cerebral_blood_flow", "cerebral blood flow", "csf_flow"]
        flag = False
        if any(
            [
                t in self.acquisition_information_dict["SeriesDescription"].lower()
                for t in target
            ]
        ):
            flag = True
        return (flag, "CBF")

    def is_angio(self):
        # cerebral blood flow
        target = ["pjn", "tof", "mra", "mrv"]
        flag = False
        if any(
            [
                t in self.acquisition_information_dict["SeriesDescription"].lower()
                for t in target
            ]
        ):
            flag = True
        return (flag, "ANGIO")

    def is_chemical_shift(self):
        # cerebral blood flow
        target = [
            "csi",
        ]
        flag = False
        if any(
            [
                t in self.acquisition_information_dict["SeriesDescription"].lower()
                for t in target
            ]
        ):
            flag = True
        return (flag, "CSI")

    def is_vs3D(self):
        # cerebral blood flow
        target = [
            "vs3d",
        ]
        flag = False
        if any(
            [
                t in self.acquisition_information_dict["SeriesDescription"].lower()
                for t in target
            ]
        ):
            flag = True
        return (flag, "Vs3D")

    def is_diffusion(self):
        # diffusion sequence
        target = ["diffusion", "diff", "_diff_", "dwi", "dw", "dti"]
        flag = False
        # here checking the ImageType and the SeriesDescription
        if any(
            [
                any(
                    [
                        t.upper() in self.acquisition_information_dict["ImageType"]
                        for t in target
                    ]
                ),
                any(
                    [
                        t
                        in self.acquisition_information_dict[
                            "SeriesDescription"
                        ].lower()
                        for t in target
                    ],
                ),
            ]
        ):
            flag = True
        return (flag, "DIFFUSION")

    def is_diffusion_fa(self):
        # fractional anisotropy
        flag = False
        if self.is_diffusion()[0]:
            if any(
                [
                    "fa" in t.lower()
                    for t in self.acquisition_information_dict["ImageType"]
                ]
            ):
                flag = True
        return (flag, "FA")

    def is_diffusion_adc(self):
        # apparent diffucion coefficient
        flag = False
        if self.is_diffusion()[0]:
            if any(
                [
                    "adc" in t.lower()
                    for t in self.acquisition_information_dict["ImageType"]
                ]
            ):
                flag = True
        return (flag, "ADC")

    def is_diffusion_trace(self):
        # trace
        flag = False
        if self.is_diffusion()[0]:
            if any(
                [
                    "trace" in t.lower()
                    for t in self.acquisition_information_dict["ImageType"]
                ]
            ):
                flag = True
        return (flag, "TRACE")

    def is_diffusion_exp(self):
        flag = False
        target = ["Exponential_Apparent_Diffusio"]
        if self.is_diffusion()[0]:
            if any(
                [
                    any(
                        [
                            "exp" in t.lower()
                            for t in self.acquisition_information_dict["ImageType"]
                        ]
                    ),
                    any(
                        [
                            t
                            in self.acquisition_information_dict[
                                "SeriesDescription"
                            ].lower()
                            for t in target
                        ]
                    ),
                ]
            ):
                flag = True
        return (flag, "EXP")

    def is_diffusion_fd(self):
        # apparent fiber density
        flag = False
        target = ["fd"]
        if self.is_diffusion()[0]:
            if any(
                [
                    any(
                        [
                            "fd" in t.lower()
                            for t in self.acquisition_information_dict["ImageType"]
                        ]
                    ),
                    any(
                        [
                            t
                            in self.acquisition_information_dict[
                                "SeriesDescription"
                            ].lower()
                            for t in target
                        ]
                    ),
                ]
            ):
                flag = True
        return (flag, "FD")

    def is_perfusion(self):
        # perfusion
        target = ["perfusion"]
        flag = False
        if any(
            [
                t in self.acquisition_information_dict["SeriesDescription"].lower()
                for t in target
            ]
        ):
            flag = True
        return (flag, "PERFUSION")

    def get_mr_modality(self):
        mr_modality = "UNKNOWN"
        # check if T1w
        if self.is_t1w()[0]:
            mr_modality = self.is_t1w()[1]
            # check if any of the T1W variations
            for check in [
                self.is_t1w_fl(),
                self.is_t1w_se(),
                self.is_t1w_mprage(),
                self.is_t1w_fspgr(),
            ]:
                if check[0] == True:
                    mr_modality = check[1]
        elif self.is_fspgr()[0]:
            mr_modality = self.is_fspgr()[1]
        elif self.is_t2w()[0]:
            mr_modality = self.is_t2w()[1]
            # check if any of the T2W variations
            for check in [self.is_t2w_flair(), self.is_t2w_trim()]:
                if check[0] == True:
                    mr_modality = check[1]
        elif self.is_t2w_cube()[0]:
            mr_modality = self.is_t2w_cube()[1]
        elif self.is_t2w_space()[0]:
            mr_modality = self.is_t2w_space()[1]
        elif self.is_diffusion()[0]:
            mr_modality = self.is_diffusion()[1]
            # check if any of the diffusion variations
            for check in [
                self.is_diffusion_fa(),
                self.is_diffusion_adc(),
                self.is_diffusion_trace(),
                self.is_diffusion_exp(),
                self.is_diffusion_fd(),
            ]:
                if check[0] == True:
                    mr_modality = check[1]
        else:
            for check in [
                self.is_mag(),
                self.is_phe(),
                self.is_asl(),
                self.is_swi(),
                self.is_swan(),
                self.is_cbf(),
                self.is_perfusion(),
                self.is_angio(),
                self.is_chemical_shift(),
                self.is_vs3D(),
                self.is_flair(),
            ]:
                if check[0]:
                    mr_modality = check[1]

        return mr_modality


def get_radiology_acquisition_information(
    path_to_acquisition_json_file, from_dict=False, acquisition_info_dict=None
):
    """
    Utility that scraps the .json file information and tries to understand what type
    of MR acquisition we are looking at. Note that the heuristic to get the MR acquisition
    type is based on lookig at the .json files and trying to understand what the different
    fields mean. There are cases were the modality can not be distinguised sinde the
    information neede to tell things apart is missing (e.g. contrast information is not directly
    provided thus needs to be ifered from other bits of information).

    Steps
    - Copy the .json information and remove the 'irrelevant' information.
    - based on the remaining information, check which MR modality we are looking at.
        Independend class ModalityInvestigator that tries to undertand what modality we are looking at.
    - Return the information.
    """
    acquisition_info_fields = [
        "MagneticFieldStrength",
        "Manufacturer",
        "ManufacturersModelName",
        "BodyPartExamined",
        "MRAcquisitionType",
        "SeriesDescription",
        "ScanningSequence",
        "SequenceVariant",
        "ScanOptions",
        "SequenceName",
        "ImageType",
        "EchoTime",
        "RepetitionTime",
        "FlipAngle",
        "DiffusionScheme",
        "ReceiveCoilName",
        "ReceiveCoilActiveElements",
        "PulseSequenceDetails",
        "dim1",
        "dim2",
        "dim3",
    ]

    if not from_dict:
        acquisition_info_dict = {}
        # take out information
        try:
            with open(path_to_acquisition_json_file) as json_file:
                acquisition_information = json.load(json_file)
                for key in acquisition_info_fields:
                    if key in acquisition_information.keys():
                        acquisition_info_dict[key] = acquisition_information[key]
                    else:
                        acquisition_info_dict[key] = None

                # get mr_modality infromation

                mr_modality_investigator = MRSequenceInvestigator(
                    acquisition_information
                )
                mr_modality = mr_modality_investigator.get_mr_modality()
        except:
            # print(f"Json file not found: {path_to_acquisition_json_file}")
            acquisition_info_dict = dict.fromkeys(
                acquisition_info_fields, "NotAvailable"
            )
            mr_modality = "UNKNOWN"
    else:
        # get mr_modality infromation

        mr_modality_investigator = MRSequenceInvestigator(acquisition_info_dict)
        mr_modality = mr_modality_investigator.get_mr_modality()

    # add information about file name
    acquisition_info_dict["file_name"] = os.path.join(
        path_to_acquisition_json_file.split(os.path.sep)[-3],
        path_to_acquisition_json_file.split(os.path.sep)[-2],
        path_to_acquisition_json_file.split(os.path.sep)[-1],
    )

    def get_contrast_status(acqusition_name):
        """
        Utility that given a file path, tries to find out if the scan was performed
        pre or post contrast agent. Here is mostly important for T1 scans, since
        all the other acquisitions should be pre contrast
        """

        # if 'se t1' in file_name or 'se_t1' in file_name or 't1_se' in file_name or 't1' in file_name or 'T1' in file_name:
        if any(
            target in acqusition_name.lower() for target in ["post", "p0st", "C+", "gd"]
        ):
            contrast_status = "post_contrast"
        else:
            contrast_status = "pre_contrast"

        return contrast_status

    # check if pre or post contrast  (change T1w to T1wGD if contrast is used)
    contrast_status = get_contrast_status(acquisition_info_dict["SeriesDescription"])
    acquisition_info_dict["GD_contrast"] = contrast_status

    if all([contrast_status == "post_contrast", "T1W" in mr_modality]):
        mr_modality = mr_modality + "_GD"

    # return values
    return mr_modality, acquisition_info_dict


def get_radiology_information_from_volume(path_to_acquisition_nii_file):
    # # open nifti file
    vol = nib.load(path_to_acquisition_nii_file)
    # bring volume to canonical coordinates
    try:
        vol = nib.funcs.as_closest_canonical(vol)
    except:
        print(
            "Re-orientation to canonical failed. Using original volume orientation to get dimensions."
        )
    # get volume header
    vol_header = vol.header
    return {
        "dim_x": vol_header["pixdim"][2],
        "dim_y": vol_header["pixdim"][3],
        "dim_z": vol_header["pixdim"][1],
        "pixels_x": vol.header["dim"][2],
        "pixels_y": vol.header["dim"][3],
        "pixels_z": vol.header["dim"][1],
    }
