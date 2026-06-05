# Multi-Test Structural Validation Project Foldering Guide and Action Plan

## 1. Purpose

This guide defines a practical foldering system for a structural analysis and validation project that contains multiple tests, finite element analysis work, raw and processed test data, subcontractor communication, presentations, reports, and final validation deliverables.

The goal is to make the folder easy to audit, easy to maintain, and easy to understand months or years later.

The main organizing principle is:

> Organize primarily by test, then by file type inside each test.

This is usually better than organizing only by file type because validation engineers normally ask questions such as:

- Where is everything related to Test T001?
- Which FEA model was used for this test correlation?
- Which raw data file produced this processed result?
- Which presentation or report used this result?
- What was received from the subcontractor?
- Which documents are final, released, or superseded?

A good foldering system should preserve traceability between requirements, tests, raw data, processed data, FEA models, FEA results, correlation files, presentations, reports, and communications.

---

## 2. Core Foldering Philosophy

The project folder should separate four major categories of information:

1. **Project-level information**  
   Information shared by all tests, such as requirements, standards, global FEA models, test matrix, action items, and final validation summary.

2. **Test-specific information**  
   Everything related to one specific test, such as raw data, processed data, test-specific FEA, correlation plots, photos, presentations, subcontractor communication, and test report.

3. **Cross-test information**  
   Comparison across multiple tests, global correlation matrices, lessons learned, model update tracking, and validation-level conclusions.

4. **Released deliverables and archive**  
   Final signed reports, official submissions, released presentations, obsolete files, superseded versions, and backup packages.

The basic logic is:

```text
Project-level folders = shared basis and final validation evidence
Test-level folders    = everything specific to one test
Cross-test folders    = comparison, summary, and engineering conclusions across tests
Archive folders       = superseded, obsolete, or frozen historical material
```

---

## 3. Recommended Top-Level Folder Structure

Use the following structure for a project containing multiple validation tests:

```text
Project_Validation/
│
├── 00_Project_Admin_and_Index/
├── 01_Requirements_and_Basis/
├── 02_Global_FEA_Models/
├── 03_Tests/
├── 04_Cross_Test_Correlation/
├── 05_Final_Validation_Deliverables/
├── 06_Scripts_and_Tools/
├── 07_Project_Communication/
├── 08_Working_Temporary/
└── 99_Archive/
```

### 3.1 Folder Descriptions

| Folder | Purpose |
|---|---|
| `00_Project_Admin_and_Index` | Master indexes, document register, test matrix, action item list, validation status tracker. |
| `01_Requirements_and_Basis` | Specifications, requirements, acceptance criteria, load cases, standards, assumptions. |
| `02_Global_FEA_Models` | Baseline/global FEA models used across multiple tests. |
| `03_Tests` | Main folder containing one subfolder per test. |
| `04_Cross_Test_Correlation` | Comparison of results across tests, global correlation matrix, lessons learned. |
| `05_Final_Validation_Deliverables` | Final validation reports, signed documents, official submissions. |
| `06_Scripts_and_Tools` | Common Python, APDL, MATLAB, Excel macro, and post-processing tools. |
| `07_Project_Communication` | Project-level emails, meeting minutes, decisions, subcontractor communication not specific to one test. |
| `08_Working_Temporary` | Temporary work, scratch files, files waiting to be sorted. |
| `99_Archive` | Superseded, obsolete, frozen, or backup material. |

---

## 4. Detailed Top-Level Folder Structure

```text
Project_Validation/
│
├── 00_Project_Admin_and_Index/
│   ├── Project_Index.xlsx
│   ├── Test_Matrix.xlsx
│   ├── Document_Register.xlsx
│   ├── Action_Item_List.xlsx
│   ├── Validation_Status_Tracker.xlsx
│   └── Folder_Readme.md
│
├── 01_Requirements_and_Basis/
│   ├── Specifications/
│   ├── Test_Requirements/
│   ├── Load_Cases/
│   ├── Acceptance_Criteria/
│   ├── Interface_Definitions/
│   ├── Material_Allowables/
│   ├── Standards_and_References/
│   └── Assumptions_and_Clarifications/
│
├── 02_Global_FEA_Models/
│   ├── Baseline_Model/
│   ├── Released_Model_Versions/
│   ├── Common_Material_Data/
│   ├── Common_Boundary_Conditions/
│   ├── Common_Load_Definitions/
│   ├── Common_Contacts_and_Connections/
│   ├── Mesh_Studies/
│   └── Model_Change_Log/
│
├── 03_Tests/
│   ├── T001_Static_Load_Test/
│   ├── T002_Pressure_Test/
│   ├── T003_Vibration_Test/
│   └── T004_Thermal_Test/
│
├── 04_Cross_Test_Correlation/
│   ├── Global_Correlation_Matrix.xlsx
│   ├── Common_Sensor_Comparison/
│   ├── Repeated_Load_Case_Comparison/
│   ├── FEA_Model_Update_Tracking/
│   ├── Validation_Margins_Summary/
│   └── Lessons_Learned/
│
├── 05_Final_Validation_Deliverables/
│   ├── Draft/
│   ├── Internal_Review/
│   ├── Customer_or_Authority_Review/
│   ├── Released/
│   ├── Signed/
│   └── Official_Submissions/
│
├── 06_Scripts_and_Tools/
│   ├── Python/
│   ├── APDL/
│   ├── Matlab/
│   ├── Excel_Macros/
│   ├── Common_Postprocessing_Tools/
│   └── Tool_Documentation/
│
├── 07_Project_Communication/
│   ├── Emails/
│   ├── Meeting_Minutes/
│   ├── Technical_Decisions/
│   ├── Subcontractor_Communication/
│   └── Customer_or_Authority_Communication/
│
├── 08_Working_Temporary/
│   ├── Scratch/
│   ├── To_Be_Sorted/
│   ├── Temporary_Exports/
│   └── Old_Working_Files/
│
└── 99_Archive/
    ├── Superseded/
    ├── Obsolete/
    ├── Frozen_Baselines/
    └── Backup_Packages/
```

---

## 5. Recommended Test-Level Folder Structure

Each test should have the same internal structure. This consistency is more important than having a perfect structure.

Example:

```text
03_Tests/
│
├── T001_Static_Load_Test/
│   ├── 00_Test_Index/
│   ├── 01_Test_Planning/
│   ├── 02_Raw_Test_Data/
│   ├── 03_Processed_Test_Data/
│   ├── 04_Test_Specific_FEA/
│   ├── 05_FEA_Results/
│   ├── 06_Test_vs_FEA_Correlation/
│   ├── 07_Photos_Videos/
│   ├── 08_Test_Communication/
│   ├── 09_Presentations/
│   ├── 10_Test_Report/
│   ├── 11_Nonconformities_and_Anomalies/
│   └── 12_Test_Closeout/
```

### 5.1 Test Folder Descriptions

| Folder | Purpose |
|---|---|
| `00_Test_Index` | Test-specific index, data register, sensor map, load case map. |
| `01_Test_Planning` | Test plan, procedure, instrumentation plan, setup drawings, risk assessments. |
| `02_Raw_Test_Data` | Original untouched data received from test rig or subcontractor. |
| `03_Processed_Test_Data` | Cleaned, filtered, converted, reduced, or calculated data. |
| `04_Test_Specific_FEA` | FEA files specific to this test setup, loading, boundary conditions, and correlation. |
| `05_FEA_Results` | Solver results, postprocessed results, exported plots, stress/strain/displacement images. |
| `06_Test_vs_FEA_Correlation` | Comparison tables, sensor-vs-FEA plots, error calculations, pass/fail summaries. |
| `07_Photos_Videos` | Test setup photos, sensor photos, failure photos, videos. |
| `08_Test_Communication` | Emails, meeting notes, subcontractor messages specific to this test. |
| `09_Presentations` | Internal review presentations, board presentations, customer presentations for this test. |
| `10_Test_Report` | Draft, reviewed, released, and signed versions of the test report. |
| `11_Nonconformities_and_Anomalies` | Test deviations, anomalies, invalid channels, model issues, NCRs. |
| `12_Test_Closeout` | Final checklist, action closures, approval evidence, closeout notes. |

---

## 6. Detailed Test-Level Folder Structure

```text
T001_Static_Load_Test/
│
├── 00_Test_Index/
│   ├── T001_Index.xlsx
│   ├── T001_Data_Register.xlsx
│   ├── T001_Sensor_Map.xlsx
│   ├── T001_Load_Case_Map.xlsx
│   └── T001_Readme.md
│
├── 01_Test_Planning/
│   ├── Test_Plan/
│   ├── Test_Procedure/
│   ├── Instrumentation_Plan/
│   ├── Sensor_Layout/
│   ├── Test_Setup_Drawings/
│   ├── Risk_Assessment/
│   └── Pre_Test_Review/
│
├── 02_Raw_Test_Data/
│   ├── Original_From_Test_Rig/
│   ├── Original_From_Subcontractor/
│   ├── Strain_Gauge_Data/
│   ├── Displacement_Data/
│   ├── Load_Cell_Data/
│   ├── Pressure_Data/
│   ├── Temperature_Data/
│   ├── DAQ_Configuration/
│   └── Metadata/
│
├── 03_Processed_Test_Data/
│   ├── Cleaned_Data/
│   ├── Filtered_Data/
│   ├── Calculated_Channels/
│   ├── Unit_Converted_Data/
│   ├── Time_Aligned_Data/
│   ├── Reduced_Data/
│   ├── Plots/
│   └── Processing_Notes/
│
├── 04_Test_Specific_FEA/
│   ├── Working_Models/
│   ├── Released_Models/
│   ├── Solver_Input_Files/
│   ├── Boundary_Conditions/
│   ├── Load_Application/
│   ├── Contact_Settings/
│   ├── Mesh_Files/
│   ├── Material_Data/
│   ├── Coordinate_Systems/
│   └── FEA_Assumptions/
│
├── 05_FEA_Results/
│   ├── Raw_Solver_Results/
│   ├── Postprocessed_Results/
│   ├── Stress_Results/
│   ├── Strain_Results/
│   ├── Displacement_Results/
│   ├── Reaction_Force_Results/
│   ├── Contact_Results/
│   ├── Sensor_Extraction_Results/
│   ├── Result_Images/
│   └── Result_Animations/
│
├── 06_Test_vs_FEA_Correlation/
│   ├── Correlation_Tables/
│   ├── Sensor_vs_FEA/
│   ├── Strain_Gauge_vs_FEA/
│   ├── Displacement_vs_FEA/
│   ├── Load_Response_Comparison/
│   ├── Error_Calculations/
│   ├── Pass_Fail_Assessment/
│   ├── Correlation_Plots/
│   └── Correlation_Notes/
│
├── 07_Photos_Videos/
│   ├── Test_Setup/
│   ├── Instrumentation/
│   ├── During_Test/
│   ├── Post_Test_Inspection/
│   ├── Damage_or_Failure/
│   └── Videos/
│
├── 08_Test_Communication/
│   ├── Emails/
│   ├── Meeting_Minutes/
│   ├── Subcontractor_Questions/
│   ├── Subcontractor_Answers/
│   ├── Data_Deliveries/
│   └── Decisions/
│
├── 09_Presentations/
│   ├── Working_Presentations/
│   ├── Internal_Review/
│   ├── Review_Board/
│   ├── Customer_or_Authority/
│   └── Released/
│
├── 10_Test_Report/
│   ├── Draft/
│   ├── Internal_Review/
│   ├── Review_Comments/
│   ├── Released/
│   └── Signed/
│
├── 11_Nonconformities_and_Anomalies/
│   ├── Test_Issues/
│   ├── Data_Issues/
│   ├── Sensor_Issues/
│   ├── FEA_Model_Issues/
│   ├── Deviations/
│   ├── NCRs/
│   └── Engineering_Assessments/
│
└── 12_Test_Closeout/
    ├── Action_Closure/
    ├── Approval_Evidence/
    ├── Final_Checklist/
    └── Lessons_Learned/
```

---

## 7. Test Naming Convention

Each test folder should start with a unique test ID.

Recommended format:

```text
T###_Component_TestType
```

Examples:

```text
T001_CompressorCasing_Static_Load_Test
T002_CompressorCasing_Pressure_Test
T003_CompressorCasing_Vibration_Test
T004_CompressorCasing_Thermal_Cycle_Test
T005_Bracket_Fatigue_Test
```

Use the test ID everywhere:

- Folder names
- File names
- Report names
- Presentation names
- Correlation files
- Raw data registers
- Processed data files
- FEA models
- Meeting notes

This makes searching much easier.

---

## 8. File Naming Convention

A good validation file name should answer the following questions without opening the file:

- Which project or component is this for?
- Which test is this related to?
- What is the content?
- Which version is it?
- What is the status?
- What is the date?

Recommended format:

```text
YYYYMMDD_TestID_Component_Content_Version_Status.ext
```

Alternative format if test ID is the most important identifier:

```text
TestID_Component_Content_YYYYMMDD_Version_Status.ext
```

Recommended examples:

```text
20260605_T001_CompressorCasing_RawSGData_v01_Original.csv
20260605_T001_CompressorCasing_FilteredSGData_v02_Working.xlsx
20260605_T001_CompressorCasing_FEA_LoadCase3_SEQVResults_v04_Reviewed.png
20260605_T001_CompressorCasing_TestVsFEA_SGCorrelation_v03_Final.xlsx
20260605_T001_CompressorCasing_ValidationReview_Presentation_v05_Released.pptx
20260605_T001_CompressorCasing_TestReport_v04_Released.pdf
```

Avoid names like:

```text
final.pptx
final_final.pptx
latest_results.xlsx
new_analysis.wbpz
copy_of_test_data.csv
updated_model.ans
presentation_last.pptx
```

---

## 9. Recommended Version and Status Tags

Use simple, consistent version numbers:

```text
v01
v02
v03
v04
```

Use clear status tags:

```text
Original
Working
Draft
InternalReview
CustomerReview
Reviewed
Approved
Final
Released
Signed
Superseded
Obsolete
```

Recommended examples:

```text
T001_CompressorCasing_StaticLoad_FEA_v01_Working.wbpz
T001_CompressorCasing_StaticLoad_FEA_v02_Correlated.wbpz
T001_CompressorCasing_StaticLoad_FEA_v03_Released.wbpz
T001_CompressorCasing_TestReport_v01_Draft.docx
T001_CompressorCasing_TestReport_v02_InternalReview.docx
T001_CompressorCasing_TestReport_v03_Released.pdf
T001_CompressorCasing_TestReport_v03_Signed.pdf
```

Suggested rule:

> Never overwrite a released file. Create a new version instead.

---

## 10. Raw Data Rules

Raw test data is engineering evidence. It should be protected.

Recommended rules:

1. Raw data must never be edited directly.
2. Raw data should be stored exactly as received.
3. If possible, keep the original compressed delivery package from the subcontractor.
4. Any cleaned, converted, filtered, or reduced data should go into `03_Processed_Test_Data`.
5. The processed file should reference the raw file used to create it.
6. If raw data is invalid or incomplete, do not delete it. Document the issue in the data register.

Example:

```text
02_Raw_Test_Data/
    Original_From_Subcontractor/
        20260605_T001_SubcontractorDelivery_v01_Original.zip

03_Processed_Test_Data/
    Filtered_Data/
        20260607_T001_FilteredSGData_v01_Working.xlsx

03_Processed_Test_Data/
    Processing_Notes/
        20260607_T001_SGDataProcessingNotes_v01_Working.md
```

---

## 11. Processed Data Rules

Processed data should be traceable and reproducible.

For every processed data file, try to record:

- Raw input file name
- Processing script or tool used
- Processing date
- Engineer responsible
- Filters applied
- Unit conversions applied
- Removed channels or invalid sensors
- Time alignment operations
- Calculated channels
- Output file name

Recommended processing note template:

```text
# Data Processing Note

Test ID: T001
Processed File: 20260607_T001_FilteredSGData_v01_Working.xlsx
Raw Input File: 20260605_T001_RawSGData_v01_Original.csv
Processing Script: SG_Postprocess_v03.py
Engineer: [Name]
Date: 2026-06-07

## Operations Applied
- Removed invalid channels: SG12, SG18
- Converted microstrain to strain
- Applied low-pass Butterworth filter, cutoff = 3 Hz, order = 2
- Time-aligned all channels to load-cell trigger
- Calculated principal strain and von Mises stress

## Notes
- SG12 was excluded due to unstable signal after 45 seconds.
- SG18 was disconnected before maximum load.
```

---

## 12. FEA Organization Rules

Separate global/common FEA models from test-specific FEA models.

Use:

```text
02_Global_FEA_Models/
```

for models shared by multiple tests.

Use:

```text
03_Tests/T001_Static_Load_Test/04_Test_Specific_FEA/
```

for FEA work specific to one test.

### 12.1 Global FEA Folder

Store the following in the global FEA folder:

- Baseline model
- Released global model versions
- Common material data
- Common boundary condition definitions
- Common load definitions
- Common contact definitions
- Mesh sensitivity studies
- Model change log

Example:

```text
02_Global_FEA_Models/
│
├── Baseline_Model/
│   └── CompressorCasing_GlobalModel_v01_Baseline.wbpz
│
├── Released_Model_Versions/
│   ├── CompressorCasing_GlobalModel_v02_Released.wbpz
│   └── CompressorCasing_GlobalModel_v03_Released.wbpz
│
└── Model_Change_Log/
    └── CompressorCasing_FEA_Model_Change_Log.xlsx
```

### 12.2 Test-Specific FEA Folder

Store the following in the test-specific FEA folder:

- Test-specific boundary conditions
- Test-specific load application
- Test fixture representation
- Local coordinate systems
- Sensor extraction points
- Test-specific mesh changes
- Contact setting changes
- Solver input files
- Working and correlated model versions

Example:

```text
03_Tests/T001_Static_Load_Test/04_Test_Specific_FEA/
│
├── Working_Models/
│   ├── T001_CompressorCasing_StaticLoad_FEA_v01_Working.wbpz
│   └── T001_CompressorCasing_StaticLoad_FEA_v02_Working.wbpz
│
├── Released_Models/
│   └── T001_CompressorCasing_StaticLoad_FEA_v03_Released.wbpz
│
├── Boundary_Conditions/
├── Load_Application/
├── Sensor_Extraction_Points/
└── FEA_Assumptions/
```

---

## 13. FEA Results Organization Rules

FEA results should be organized by result type and load case.

Recommended structure:

```text
05_FEA_Results/
│
├── Raw_Solver_Results/
├── Postprocessed_Results/
├── Stress_Results/
│   ├── LC01/
│   ├── LC02/
│   └── LC03/
├── Strain_Results/
│   ├── LC01/
│   ├── LC02/
│   └── LC03/
├── Displacement_Results/
├── Reaction_Force_Results/
├── Contact_Results/
├── Sensor_Extraction_Results/
├── Result_Images/
└── Result_Animations/
```

Each exported result should identify:

- Test ID
- Component
- Load case
- Result type
- FEA model version
- Date
- Status

Example:

```text
T001_CompressorCasing_LC03_SEQV_Modelv03_20260607_v01_Reviewed.png
T001_CompressorCasing_LC03_TotalDef_Modelv03_20260607_v01_Reviewed.png
T001_CompressorCasing_LC03_SGExtraction_Modelv03_20260607_v01_Reviewed.xlsx
```

---

## 14. Test-vs-FEA Correlation Folder

This is one of the most important folders in a validation project.

Recommended structure:

```text
06_Test_vs_FEA_Correlation/
│
├── Correlation_Tables/
├── Sensor_vs_FEA/
├── Strain_Gauge_vs_FEA/
├── Displacement_vs_FEA/
├── Load_Response_Comparison/
├── Error_Calculations/
├── Pass_Fail_Assessment/
├── Correlation_Plots/
└── Correlation_Notes/
```

Recommended files:

```text
T001_Correlation_Index.xlsx
T001_SG_vs_FEA_Correlation_v01_Working.xlsx
T001_Displacement_vs_FEA_Correlation_v01_Working.xlsx
T001_Correlation_Plots_v01_Working.pptx
T001_Correlation_Assessment_v01_Draft.md
```

The correlation folder should answer:

- Which sensors were compared with FEA?
- Which FEA node, element, path, or location was used?
- Which load case was compared?
- What was the measured value?
- What was the FEA value?
- What was the percentage difference?
- Was the difference acceptable?
- Was the test passed or failed?
- Which model version produced the correlation?

---

## 15. Recommended Registers and Index Files

A foldering system is not enough by itself. You also need index files.

### 15.1 Project Index

Location:

```text
00_Project_Admin_and_Index/Project_Index.xlsx
```

Recommended columns:

| Column | Description |
|---|---|
| File Name | Exact file name. |
| Folder Path | Relative folder path. |
| Description | Short description of the file. |
| Source | Internal, subcontractor, test lab, customer, supplier, etc. |
| Author | File author or responsible engineer. |
| Date Created | Original creation date. |
| Date Received | Date received, if external. |
| Version | v01, v02, etc. |
| Status | Working, Draft, Reviewed, Released, Signed, Superseded. |
| Related Test ID | T001, T002, etc. |
| Related Load Case | LC01, LC02, etc. |
| Related FEA Model | Model version used. |
| Used in Report? | Yes or No. |
| Notes | Important comments. |

### 15.2 Test Matrix

Location:

```text
00_Project_Admin_and_Index/Test_Matrix.xlsx
```

Recommended columns:

| Column | Description |
|---|---|
| Test ID | Unique test identifier. |
| Test Name | Descriptive test name. |
| Component | Tested component or assembly. |
| Purpose | Strength validation, stiffness validation, pressure proof, vibration, etc. |
| Requirement Reference | Requirement or specification paragraph. |
| Load Case | Related load case. |
| Acceptance Criteria | Pass/fail basis. |
| Test Date | Actual or planned test date. |
| Facility | Test facility or subcontractor. |
| Subcontractor | Company responsible for test execution, if applicable. |
| FEA Model Version | Model version used for prediction or correlation. |
| Report Status | Not started, draft, reviewed, released, signed. |
| Result Status | Pass, fail, conditional pass, open. |
| Comments | Notes and open items. |

Example row:

```text
T001 | Compressor Casing Static Load Test | Compressor Casing | Strength validation | REQ-145 | LC-03 | No yielding above limit load | 2026-06-05 | ABC Test Lab | XYZ Ltd. | FEA_v04 | Released | Pass | Correlation acceptable
```

### 15.3 Test-Specific Index

Location:

```text
03_Tests/T001_Static_Load_Test/00_Test_Index/T001_Index.xlsx
```

Recommended columns:

| Column | Description |
|---|---|
| File Name | Exact file name. |
| Description | What the file contains. |
| Folder | Relative folder location. |
| Source | Internal, subcontractor, test rig, FEA, postprocessing, etc. |
| Author | Responsible person. |
| Date | File date. |
| Version | v01, v02, etc. |
| Status | Working, Draft, Released, etc. |
| Used in Report? | Yes or No. |
| Related Load Case | LC01, LC02, etc. |
| Related Sensor | SG01, LVDT02, etc. |
| Related FEA Model | Model version. |
| Notes | Important comments. |

### 15.4 Correlation Index

Location:

```text
03_Tests/T001_Static_Load_Test/06_Test_vs_FEA_Correlation/T001_Correlation_Index.xlsx
```

Recommended columns:

| Column | Description |
|---|---|
| Test ID | T001, T002, etc. |
| Measurement Type | Strain, displacement, load, pressure, temperature. |
| Sensor ID | SG01, LVDT01, etc. |
| Physical Location | Description of sensor location. |
| Test Channel Name | Channel name in raw data. |
| FEA Extraction Location | Node, element, path, named selection, coordinate system, etc. |
| Load Case | LC01, LC02, etc. |
| Test Value | Measured result. |
| FEA Value | Analysis result. |
| Difference | Absolute difference. |
| Difference % | Percentage difference. |
| Acceptance Limit | Allowed difference or engineering criterion. |
| Pass/Fail | Correlation result. |
| FEA Model Version | Model used for extraction. |
| Processed Data Version | Data version used. |
| Comment | Explanation or notes. |

---

## 16. Communication Organization

Communication should be stored where it belongs.

Use project-level communication folder for general project decisions:

```text
07_Project_Communication/
```

Use test-level communication folder for test-specific messages:

```text
03_Tests/T001_Static_Load_Test/08_Test_Communication/
```

### 16.1 Recommended Communication Subfolders

```text
08_Test_Communication/
│
├── Emails/
├── Meeting_Minutes/
├── Subcontractor_Questions/
├── Subcontractor_Answers/
├── Data_Deliveries/
└── Decisions/
```

### 16.2 Email File Naming

Recommended email naming format:

```text
YYYYMMDD_TestID_Sender_Topic_Status.msg
```

Examples:

```text
20260605_T001_ABC_TestLab_RawDataDelivery_Received.msg
20260606_T001_ABC_TestLab_InvalidSG12_Clarification.msg
20260608_T001_Internal_CorrelationDecision_Approved.msg
```

If you export emails as PDF:

```text
20260605_T001_ABC_TestLab_RawDataDelivery_Received.pdf
```

---

## 17. Presentation Organization

Presentations should be separated into working, review, and released versions.

Recommended structure:

```text
09_Presentations/
│
├── Working_Presentations/
├── Internal_Review/
├── Review_Board/
├── Customer_or_Authority/
└── Released/
```

Recommended file names:

```text
T001_CompressorCasing_StaticLoad_InternalReview_v01_Draft.pptx
T001_CompressorCasing_StaticLoad_ReviewBoard_v02_Reviewed.pptx
T001_CompressorCasing_StaticLoad_FinalPresentation_v03_Released.pptx
```

Suggested rule:

> Presentations should not be treated as source evidence. They are summaries. The data, FEA results, and correlation files behind each slide should remain traceable.

---

## 18. Report Organization

Each test should have its own report folder.

Recommended structure:

```text
10_Test_Report/
│
├── Draft/
├── Internal_Review/
├── Review_Comments/
├── Released/
└── Signed/
```

Recommended report names:

```text
T001_CompressorCasing_StaticLoad_TestReport_v01_Draft.docx
T001_CompressorCasing_StaticLoad_TestReport_v02_InternalReview.docx
T001_CompressorCasing_StaticLoad_TestReport_v03_Released.pdf
T001_CompressorCasing_StaticLoad_TestReport_v03_Signed.pdf
```

For the overall validation report, use:

```text
05_Final_Validation_Deliverables/
```

Example:

```text
05_Final_Validation_Deliverables/
│
├── Draft/
│   └── CompressorCasing_FinalValidationReport_v01_Draft.docx
│
├── Internal_Review/
│   └── CompressorCasing_FinalValidationReport_v02_InternalReview.docx
│
├── Released/
│   └── CompressorCasing_FinalValidationReport_v03_Released.pdf
│
└── Signed/
    └── CompressorCasing_FinalValidationReport_v03_Signed.pdf
```

---

## 19. Archive Rules

Do not delete important engineering files unless your organization has a formal data-retention policy allowing it.

Instead, move old files to archive folders.

Recommended archive categories:

```text
99_Archive/
│
├── Superseded/
├── Obsolete/
├── Frozen_Baselines/
└── Backup_Packages/
```

Definitions:

| Folder | Meaning |
|---|---|
| `Superseded` | Replaced by a newer valid version. |
| `Obsolete` | No longer valid and should not be used. |
| `Frozen_Baselines` | Baseline package preserved for traceability. |
| `Backup_Packages` | Compressed backups or frozen copies. |

Suggested rule:

> If a file was used in a released report, do not delete it. Archive it with traceability.

---

## 20. Practical Action Plan for Cleaning a Messy Existing Folder

### Phase 1: Freeze the Current Folder

Before reorganizing anything, preserve the current state.

Actions:

1. Create a backup of the current messy folder.
2. Name it clearly.
3. Do not modify this backup.

Example:

```text
99_Archive/Backup_Packages/20260605_ProjectValidation_OriginalMessyFolder_Backup.zip
```

Reason:

You may later need to recover original context, timestamps, file locations, or subcontractor delivery packages.

---

### Phase 2: Create the New Folder Skeleton

Create the recommended top-level structure:

```text
Project_Validation/
│
├── 00_Project_Admin_and_Index/
├── 01_Requirements_and_Basis/
├── 02_Global_FEA_Models/
├── 03_Tests/
├── 04_Cross_Test_Correlation/
├── 05_Final_Validation_Deliverables/
├── 06_Scripts_and_Tools/
├── 07_Project_Communication/
├── 08_Working_Temporary/
└── 99_Archive/
```

Then create one folder per known test:

```text
03_Tests/T001_Static_Load_Test/
03_Tests/T002_Pressure_Test/
03_Tests/T003_Vibration_Test/
```

Inside each test folder, create the standard internal structure.

---

### Phase 3: Build a Test Matrix

Create:

```text
00_Project_Admin_and_Index/Test_Matrix.xlsx
```

Minimum required columns:

```text
Test ID
Test Name
Component
Purpose
Load Case
Requirement Reference
Acceptance Criteria
Test Date
Facility
Subcontractor
FEA Model Version
Report Status
Result Status
Comments
```

Use this matrix to define the official test IDs.

Do not start moving large amounts of data before the test IDs are clear.

---

### Phase 4: Sort Files by Test First

Go through the messy folder and sort files into one of these groups:

1. Belongs clearly to T001
2. Belongs clearly to T002
3. Belongs clearly to T003
4. Applies to multiple tests
5. Project-level file
6. Unknown or unclear

Put unclear files temporarily into:

```text
08_Working_Temporary/To_Be_Sorted/
```

Do not force unclear files into a test folder without evidence.

---

### Phase 5: Separate Raw Data from Processed Data

Inside each test folder, move original untouched test data to:

```text
02_Raw_Test_Data/
```

Move cleaned, filtered, converted, plotted, or calculated data to:

```text
03_Processed_Test_Data/
```

Important rule:

> Raw data should be preserved exactly as received.

If you are unsure whether a file is raw or processed, put it in `To_Be_Sorted` and mark it for review.

---

### Phase 6: Separate FEA Models from FEA Results

Move FEA models to:

```text
04_Test_Specific_FEA/
```

Move result exports to:

```text
05_FEA_Results/
```

Typical model files:

- `.wbpz`
- `.wbpj`
- `.inp`
- `.cdb`
- `.dat`
- `.ans`
- solver input decks
- mesh files

Typical result files:

- stress plots
- displacement plots
- strain plots
- exported CSV/XLSX result tables
- sensor extraction files
- animations
- solver output files

If a model is used across multiple tests, store it in:

```text
02_Global_FEA_Models/
```

and reference it from the test-specific folder or index.

---

### Phase 7: Organize Correlation Evidence

Move comparison files to:

```text
06_Test_vs_FEA_Correlation/
```

This includes:

- Test-vs-FEA Excel comparisons
- Error percentage calculations
- Sensor extraction comparisons
- Strain gauge correlation plots
- Displacement correlation plots
- Pass/fail summaries
- Engineering assessments

Create a correlation index for each test:

```text
T001_Correlation_Index.xlsx
```

---

### Phase 8: Organize Communication

Move test-specific emails and meeting notes to:

```text
08_Test_Communication/
```

Move project-level communication to:

```text
07_Project_Communication/
```

Recommended categories:

```text
Emails/
Meeting_Minutes/
Subcontractor_Questions/
Subcontractor_Answers/
Data_Deliveries/
Decisions/
```

For subcontractor data deliveries, keep both:

1. The communication record
2. The delivered data package

The email may go to:

```text
08_Test_Communication/Data_Deliveries/
```

The raw delivered data should go to:

```text
02_Raw_Test_Data/Original_From_Subcontractor/
```

---

### Phase 9: Organize Presentations and Reports

Move presentations to:

```text
09_Presentations/
```

Move test reports to:

```text
10_Test_Report/
```

Move final validation-level reports to:

```text
05_Final_Validation_Deliverables/
```

Do not use presentations as the only location for important figures. Store the underlying plots or result images in the relevant data, FEA result, or correlation folder.

---

### Phase 10: Create Index Files

After sorting files, create or update:

```text
00_Project_Admin_and_Index/Project_Index.xlsx
00_Project_Admin_and_Index/Test_Matrix.xlsx
03_Tests/T001_Static_Load_Test/00_Test_Index/T001_Index.xlsx
03_Tests/T001_Static_Load_Test/06_Test_vs_FEA_Correlation/T001_Correlation_Index.xlsx
```

Repeat for each test.

At minimum, the index should identify:

- File name
- Folder path
- Description
- Source
- Version
- Status
- Related test
- Related load case
- Related FEA model
- Whether it was used in the report

---

### Phase 11: Rename Critical Files

Do not rename everything immediately. Start with critical files only:

1. Raw data packages
2. Processed data used in reports
3. Released FEA models
4. Correlation files
5. Released presentations
6. Released reports
7. Signed documents

Use the naming convention:

```text
YYYYMMDD_TestID_Component_Content_Version_Status.ext
```

After the critical files are clean, rename lower-priority files only if necessary.

---

### Phase 12: Final Review and Closeout

For each test, check that the folder can answer these questions:

- Where is the test plan?
- Where is the raw data?
- Where is the processed data?
- Which script or method produced the processed data?
- Which FEA model was used?
- Where are the FEA results?
- Where is the test-vs-FEA correlation?
- Which report used these results?
- Which files are released or signed?
- Which files are obsolete or superseded?

If the answer is unclear, update the folder structure or index file.

---

## 21. Minimum Practical Version

If the full structure feels too heavy, use this simplified structure:

```text
Project_Validation/
│
├── 00_Index/
├── 01_Requirements/
├── 02_Global_FEA/
├── 03_Tests/
│   ├── T001_Static_Load_Test/
│   │   ├── 00_Index/
│   │   ├── 01_Test_Planning/
│   │   ├── 02_Raw_Data/
│   │   ├── 03_Processed_Data/
│   │   ├── 04_FEA/
│   │   ├── 05_Correlation/
│   │   ├── 06_Communication/
│   │   ├── 07_Presentations/
│   │   └── 08_Report/
│   │
│   ├── T002_Pressure_Test/
│   └── T003_Vibration_Test/
│
├── 04_Final_Deliverables/
├── 05_Scripts/
└── 99_Archive/
```

This simplified version is usually enough for everyday engineering work.

The detailed version is better when:

- The project may be audited.
- Multiple subcontractors are involved.
- The validation package will be submitted to a customer or authority.
- Several engineers are working in the same folder.
- There are many FEA versions and test iterations.
- Long-term traceability is important.

---

## 22. Recommended Implementation Order

Use this order to avoid getting stuck:

```text
1. Backup current messy folder
2. Create new folder skeleton
3. Create Test_Matrix.xlsx
4. Assign test IDs
5. Sort files by test
6. Separate raw data from processed data
7. Separate FEA models from FEA results
8. Move correlation files
9. Move communication records
10. Move presentations and reports
11. Create project and test indexes
12. Rename only critical files
13. Archive superseded or obsolete files
14. Review traceability
15. Freeze final validation package
```

---

## 23. Practical Rules to Follow

1. Do not edit raw data.
2. Do not overwrite released files.
3. Do not keep final reports only in email attachments.
4. Do not use vague names such as `final_final` or `latest`.
5. Do not mix different tests inside the same raw data folder.
6. Do not mix FEA models and exported result images without subfolders.
7. Do not use presentations as the only storage location for engineering evidence.
8. Every processed file should be traceable to a raw file.
9. Every correlation file should identify the FEA model version used.
10. Every released report should be traceable to the data and analysis versions behind it.
11. Every test should have a unique test ID.
12. Every test folder should use the same internal structure.
13. Superseded files should be archived, not silently deleted.
14. Unknown files should go into `To_Be_Sorted`, not randomly into final folders.
15. The folder structure should support auditability, not just personal convenience.

---

## 24. Suggested Folder Readme Template

Create this file:

```text
00_Project_Admin_and_Index/Folder_Readme.md
```

Template:

```markdown
# Project Validation Folder Readme

## Project
Project Name: [Project Name]  
Component: [Component Name]  
Responsible Engineer: [Name]  
Last Updated: [Date]

## Foldering Principle
This project is organized primarily by test ID. Each test folder contains its own raw data, processed data, FEA work, correlation files, communication, presentations, and reports.

## Test IDs
| Test ID | Test Name | Description |
|---|---|---|
| T001 | [Name] | [Description] |
| T002 | [Name] | [Description] |
| T003 | [Name] | [Description] |

## Important Rules
- Raw data must not be modified.
- Released files must not be overwritten.
- Processed data must be traceable to raw data.
- Correlation files must identify the FEA model version used.
- Final reports must be stored in `05_Final_Validation_Deliverables`.

## Key Index Files
- `00_Project_Admin_and_Index/Project_Index.xlsx`
- `00_Project_Admin_and_Index/Test_Matrix.xlsx`
- `03_Tests/[TestID]/00_Test_Index/[TestID]_Index.xlsx`
- `03_Tests/[TestID]/06_Test_vs_FEA_Correlation/[TestID]_Correlation_Index.xlsx`
```

---

## 25. Final Recommended Structure

For your use case, the recommended practical structure is:

```text
Project_Validation/
│
├── 00_Project_Admin_and_Index/
├── 01_Requirements_and_Basis/
├── 02_Global_FEA_Models/
├── 03_Tests/
│   ├── T001_Component_TestType/
│   │   ├── 00_Test_Index/
│   │   ├── 01_Test_Planning/
│   │   ├── 02_Raw_Test_Data/
│   │   ├── 03_Processed_Test_Data/
│   │   ├── 04_Test_Specific_FEA/
│   │   ├── 05_FEA_Results/
│   │   ├── 06_Test_vs_FEA_Correlation/
│   │   ├── 07_Photos_Videos/
│   │   ├── 08_Test_Communication/
│   │   ├── 09_Presentations/
│   │   ├── 10_Test_Report/
│   │   ├── 11_Nonconformities_and_Anomalies/
│   │   └── 12_Test_Closeout/
│   │
│   ├── T002_Component_TestType/
│   └── T003_Component_TestType/
│
├── 04_Cross_Test_Correlation/
├── 05_Final_Validation_Deliverables/
├── 06_Scripts_and_Tools/
├── 07_Project_Communication/
├── 08_Working_Temporary/
└── 99_Archive/
```

This structure is detailed enough for serious validation work but still practical enough to use daily.
