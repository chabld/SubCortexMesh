Updates
=========

v1.2.2 (TBC)
--------------

NEW FEATURES
~~~~~~~~~~~~
- Added a cmap argument for the plotter of cifti_metrics().

v1.2.1
--------------
FIXES
~~~~~
- cifti_metrics() no longer creates subdirectories but create unique directories for a subject's distinct session, run, acquisition or task, based on the BIDS structure of the input dscalars. This allows them to be treated as unique directories and not to be confused by slm_analysis().
- Added missing documentation for slm_analysis().

v1.2.0
--------------

NEW FEATURES
~~~~~~~~~~~~
- A new function, cifti_metrics(), is now available to convert CIFTI 91k subcortical data to surfaces.The function extracts the ~30k voxel-based subcortical grayordinates and projects their values to a the fsaverage surface template space. It is compatible with 91k outputs such as that of the Human Connectome Project preprocessing pipeline or pipelines that output cifti such as ASLprep, fMRIprep, XCP-D. It works similarly to mesh_metrics() and creates a surface_metrics_cifti/ output directory, containing surfaces with their CIFTI values (in separate subdirectories per session-,task-,run- and acq- when applicable). It also produces summary statistics in template space. These output surfaces can be analysed and plotted the same way the outputs of mesh_metrics() are.

FIXES
~~~~~
- Error messaging when no subject directory is found by merge_tools(), instead of silence.

v1.1.0
--------------

NEW FEATURES
~~~~~~~~~~~~
- An anatomical surface-based atlas in fsaverage template was created to help users identify subsections of region-of-interests including the Thalamus, Hippocampus, Amygdala, Cerebellum, Brain-Stem, Pallidum and Ventral Diencephalon. Details of the atlas and how to use it are described on the website in a new article ("Surface-based atlases and subsections"). Functions that can make use of the atlas are: mesh_metrics() (adds atlas-based summary statistics), vis_merged(), vis_merged_flat(), slm_plot() (adds colours and/or outlines button toggles) and cluster_summary() (adds atlas-based locations). 
- merge_all() to create a merged mesh even when ROIs are missing, replacing them with a corresponding template that has nan values along the non-missing ROIs. Note that if a subject has empty ROIs contained in their merged mesh, slm_analysis() will omit the ROI for the whole cohort.
- New function cluster_summary() prints descriptions of significant clusters from SLM objects returned by slm_analysis(), detailing positive and negative clusters' corresponding regions-of-interest labels, their sizes (n_vert: number of vertices), XYZ coordinates, peak t-statistics and cluster-wise p-values. 

FIXES
~~~~~
- slm_analysis management of sub_list: added a fix for subject IDs that are contained inside other subject IDs strings so they are not confused (now uses exact directory name instead of just pattern found in path)
- subseg_getvol previously ignored the session label for multi-session data (sub-XXX_ses-YY; fmriprep naming convention) and overwrote mutiple sessions of data into the same subject output folder. This is fixed