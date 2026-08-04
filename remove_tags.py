import os
from pathlib import Path
import pydicom
from pydicom.uid import generate_uid

INPUT_DIR = r"D:/AI-BASED-MEDICAL-IMAGING/Extracted from February 20 2026"
OUTPUT_DIR = r"D:/AI-BASED-MEDICAL-IMAGING/PID Removal from February 20 2026"

# Tags commonly containing PHI
PHI_TAGS = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientMotherBirthName",

    "OtherPatientIDs",
    "OtherPatientNames",

    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",

    "ReferringPhysicianName",
    "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers",

    "PerformingPhysicianName",
    "OperatorsName",
    "PhysiciansOfRecord",
    "NameOfPhysiciansReadingStudy",

    "AccessionNumber",
    "MedicalRecordLocator",

    "StudyID",
    "RequestingPhysician",

    "DeviceSerialNumber",
    "StationName",

    "ProtocolName",

    "StudyComments",
    "ImageComments",

    "PatientInsurancePlanCodeSequence",
]

# Date tags frequently containing identifying information
DATE_TAGS = [
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "PatientBirthDate",
]

TIME_TAGS = [
    "StudyTime",
    "SeriesTime",
    "AcquisitionTime",
    "ContentTime",
]


def anonymize_dataset(ds):
    """
    Apply research-grade de-identification.
    """

    # Remove all private tags
    ds.remove_private_tags()

    # Remove common PHI fields
    for tag_name in PHI_TAGS:
        if tag_name in ds:
            del ds[tag_name]

    # Remove dates
    for tag_name in DATE_TAGS:
        if tag_name in ds:
            ds[tag_name].value = ""

    # Remove times
    for tag_name in TIME_TAGS:
        if tag_name in ds:
            ds[tag_name].value = ""

    # Replace UIDs with generated values
    uid_tags = [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
    ]

    for tag_name in uid_tags:
        if tag_name in ds:
            ds[tag_name].value = generate_uid()

    # Replace patient identifiers
    if "PatientName" in ds:
        ds.PatientName = "ANON"

    if "PatientID" in ds:
        ds.PatientID = generate_uid()[-12:]

    # DICOM de-identification documentation
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = (
        "Removed PHI tags, private tags, dates, times, and regenerated UIDs"
    )

    return ds


def process_file(input_file, output_file):
    try:
        ds = pydicom.dcmread(input_file)

        ds = anonymize_dataset(ds)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(output_file)

        print(f"Processed: {input_file}")

    except Exception as e:
        print(f"Failed: {input_file}")
        print(e)


for dir in os.listdir(INPUT_DIR):
    if os.path.isfile(INPUT_DIR + "/" + dir):
        src = Path(INPUT_DIR + "/" + dir)

        try:
            ds = pydicom.dcmread(src, stop_before_pixels=True)

            rel_path = src.relative_to(INPUT_DIR)
            dst = Path(OUTPUT_DIR) / rel_path

            process_file(src, dst)

        except Exception:
            # Skip non-DICOM files
            pass

print("Done.")