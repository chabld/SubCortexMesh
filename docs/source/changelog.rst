Updates
=========

v1.0.1 - (TBC)
--------------

NEW FEATURES
~~~~~~~~~~~~
- merge_all() to create a merged mesh even when ROIs are missing, replacing them with a corresponding template that has nan values along the non-missing ROIs. Note that if a subject has empty ROIs contained in their merged mesh, slm_analysis() will omit the ROI for the whole cohort.

FIXES
~~~~~
- slm_analysis management of sub_list: added a fix for subject IDs that are contained
  inside other subject IDs strings so they are not confused (now uses exact directory
  name instead of just pattern found in path)