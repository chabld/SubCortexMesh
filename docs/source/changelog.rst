Updates
=========

v1.0.1 - (TBC)
--------------

NEW FEATURES
~~~~~~~~~~~~
- merge_all() to create a merged mesh even when ROIs are missing, replacing them with a corresponding template that has nan values along the non-missing ROIs. Note that if a subject has empty ROIs contained in their merged mesh, slm_analysis() will omit the ROI for the whole cohort.
- New function cluster_summary() prints descriptions of significant clusters from SLM objects returned by slm_analysis(), detailing positive and negative clusters' corresponding regions-of-interest labels, their sizes (n_vert: number of vertices), XYZ coordinates, peak t-statistics and cluster-wise p-values. 

FIXES
~~~~~
- slm_analysis management of sub_list: added a fix for subject IDs that are contained
  inside other subject IDs strings so they are not confused (now uses exact directory
  name instead of just pattern found in path)