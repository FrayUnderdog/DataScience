HW4 Regression + KNN — How to Run
=================================

Files expected:
- hw4_train.csv  (required)
- hw4_test.csv   (optional, but recommended by the assignment)

Steps:
1) Put both CSVs in the SAME folder as this script.
2) Run:
   python hw4_reg_knn_solution.py
3) Outputs:
   - hw4_test_with_bp.csv     : TEST dataset with predicted 'BloodPressure' set.
   - knn_k_vs_accuracy.csv    : Table of k (1..19) vs accuracy on TEST.
   - hw4_report.txt           : Best k and accuracy summary.
   
Notes:
- If 'hw4_test.csv' is missing, the script will perform a random split from the TRAIN file
  so that you can still execute and verify the pipeline. For final submission, please provide
  an explicit TEST file as required by the assignment.
- All feature columns must be numeric for KNN. Any non-numeric columns will be dropped
  automatically before training/testing the KNN classifier.
