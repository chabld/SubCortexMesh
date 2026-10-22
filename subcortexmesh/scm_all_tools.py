#####################################################################################
##########Tools to run the full SuBCortexMesh pipeline with parallel jobs############

from typing import Optional, Union, Sequence
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from subcortexmesh.subseg_getvol import subseg_getvol
from subcortexmesh.vol2surf import vol2surf
from subcortexmesh.mesh_metrics import mesh_metrics
from subcortexmesh.merge_tools import merge_all
from subcortexmesh.cifti_metrics import cifti_metrics
from subcortexmesh.qc_tools import autoqc_outliers
from tqdm import tqdm
import time
import multiprocessing as mp
import contextlib
import traceback
from functools import partial
import os
import re

#####################################################################################
##############################Jobs to be ran in parallel#############################

#For structural data (FreeSurfer, FSL)
def _struct_job(subid, inputdir, outputdir, template, toolboxdata, roilabel,
                dilate_erode, shape_smooth, metric, metric_smooth,
                native_meshes, overwrite, silent):
    os.makedirs(f"{outputdir}/logs/", exist_ok=True)
    with open(f"{outputdir}/logs/job_{subid}.txt", "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        try:
            if not silent:
                print(f"Starting job for {subid}")
            
            subseg_getvol(inputdir=inputdir, outputdir=outputdir, template=template,
                        toolboxdata=toolboxdata, overwrite=overwrite,
                        sub_list=subid)
            
            vol2surf(inputdir=f"{outputdir}/sub_volumes", outputdir=outputdir,
                    roilabel=roilabel, dilate_erode=dilate_erode,
                    smoothing=shape_smooth, overwrite=overwrite,
                    sub_list=subid)
            
            mesh_metrics(inputdir=f"{outputdir}/sub_surfaces", outputdir=outputdir,
                        template=template, toolboxdata=toolboxdata, metric=metric,
                        roilabel=roilabel, smooth=metric_smooth,
                        native_meshes=native_meshes, overwrite=overwrite,
                        sub_list=subid)
            
            merge_all(inputdir=f"{outputdir}/surface_metrics/", template=template,
                      toolboxdata=toolboxdata, overwrite=overwrite, sub_list=subid)
            
            if not silent:
                print(f"Finished job {subid}")
            return (subid, None)
        except Exception:
            traceback.print_exc(file=f)
            return (subid, "failed")

#For CIFTI data (ASLprep, HCP pipeline etc.)
def _cifti_job(subid, inputdir, outputdir, cifti_dscalar, metric, roilabel,
               toolboxdata, overwrite, silent):
    os.makedirs(f"{outputdir}/logs/", exist_ok=True)
    with open(f"{outputdir}/logs/job_{subid}.txt", "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        try:
            if not silent:
                print(f"Starting job {subid}")
            
            cifti_metrics(dscalar=cifti_dscalar, metric=metric, inputdir=inputdir,
                        outputdir=outputdir, roilabel=roilabel,
                        toolboxdata=toolboxdata, overwrite=overwrite,
                        sub_list=subid)
                        
            #because cifti outputs BIDS levels as subject directory, identify those containing the subid
            cifti_sub_list = [
                d for d in os.listdir(inputdir)
                if os.path.isdir(os.path.join(inputdir, d))
                and subid in d
            ]
            merge_all(inputdir=f"{outputdir}/surface_metrics_cifti/", metric=metric,
                      template='fsaverage', toolboxdata=toolboxdata,
                      overwrite=overwrite, sub_list=cifti_sub_list)
                    
            if not silent:
                print(f"Finished job {subid}")
            return (subid, None)
        except Exception:
            traceback.print_exc(file=f)
            return (subid, "failed")
        

#####################################################################################
##############################Full pipeline function#################################

def scm_all(
    inputdir: Union[str, Path],
    outputdir: Union[str, Path],
    template: str,
    sub_list: Optional[Sequence[str]] = None,
    toolboxdata: Optional[Union[str, Path]] = None,
    roilabel: Union[str, Sequence[str]] = ['left-cerebellum-cortex', 'right-cerebellum-cortex', 
                                'left-pallidum', 'right-pallidum', 'left-putamen', 'right-putamen', 'left-thalamus', 'right-thalamus','left-amygdala', 'right-amygdala', 'left-hippocampus','right-hippocampus', 'left-accumbens-area','right-accumbens-area','left-caudate', 'right-caudate', 'left-ventraldc', 'right-ventraldc', 'brain-stem'],
    dilate_erode: bool = True,
    shape_smooth: int = 15,
    metric: Union[str, Sequence[str]] = ['thickness', 'curvature', 'surfarea'],
    metric_smooth: tuple[int, int, int] = (0, 5, 5),
    native_meshes: bool = False,
    autoqc_n_sd: float = 2, 
    cifti_dscalar: Optional[str] = None, 
    overwrite: bool = False,
    silent: bool = False,
    total_nthread: int = 1,
    sub_nthread: Optional[int] = 1
    ):
    
    """ Run the whole structural processing pipeline 
    
    This function runs the entire processing workflow of SubCortexMesh, separately on all or selected subjects of a FreeSurfer, FSL FIRST or CIFTI pipeline output directory, with options for parallel processing. Instead of sequential quality check viewers, it generates subject-wise reports in a qc/ subdirectory.
    
    scm_all() runs the following steps (wrapping key functions, please find their respective documentations for more details):
    
    - Extracting and coregistering subcortical volumes (subseg_getvol() function): subject segmentations are coregistered to their respective template space (fsaverage/MNI305 for FreeSurfer, MNI152 for FSL), and each subcortical volume extracted separately. 
    - Converting volumes to surfaces (vol2surf() function): volumes are converted to surface. This includes pre-conversion dilation/erosion of the volume and smoothing of the surface.
    - Reporting surface shape outliers (autoqc_outliers() function): saves a "autoqc_outliers.csv" sheet to flag, for each subject and subcortical region, whether they deviate from the cohort mean across five heuristic graphical features (volume, surface area, sphericity, elongation and symmetry).
    - Computing surface-based metrics (mesh_metrics() function): extracts thickness, curvature and surface area metrics from the subcortical surfaces outputted, and projects the metrics in template space.  
    - Merging all surfaces (merge_all() function): creates a new mesh merging all subcortical meshes outputted by mesh_metrics() in their corresponding template, for available metrics separately.
    
    For CIFTI data, which only requires two later stages of the worklow, the cifti_metrics() and merge_all() functions are run. cifti_metrics() extracts CIFTI subcortical grayordinates and projects their values to the fsaverage surface template space. 
    
    Parallel processing: scm_all() processes one subject per parallel process, each limited to the number of threads specified in sub_nthread (affects the volume coregistration). When scm_all() is run from a script file, the function must be placed after an `if __name__ == "__main__":` which makes sure Python won't raise a RuntimeError about starting new processes before bootstrapping has finished. That is not needed in an interactive session or Jupyter notebook.
    
    Parameters
    ----------
    inputdir : str, Path
        The path to a directory containing each individual subject's directory with
        preprocessing data inside. For FreeSurfer, typically the $SUBJECTS_DIR given 
        to the recon-all pipeline. For FSL, a directory with one subdirectory per 
        subject, each receiving its output from run_first_all. For CIFTI data output, a 
        directory where the CIFTI objects were outputted.
    outputdir : str, Path
        The path where all the subcortical data, including individual volumes, surfaces, 
        metrics,  statistics files, and logs will be created.
    sub_list : str, Sequence, optional
        An optional list of subject IDs to be processed. If not specified, all subjects 
        found will be included.
    template : str
        The name of the template the surfaces are supposed to be matching to. For FreeSurfer
        outputs, it is 'fsaverage'. For FSL FIRST, it is 'fslfirst'.
    toolboxdata : str, Path, optional
        The path of the "subcortexmesh_data" package data directory. The  default path 
        is assumed to be the user's home directory (pathlib's Path.home()). Users will 
        be prompted to download it if not found.
    roilabel : str, Sequence
        The name(s) of the region(s)-of-interest to be computed as strings. Default is all
        subcortices across all segmentation templates: 'left-cerebellum-cortex', 'right-cerebellum-cortex', 'left-pallidum', 'right-pallidum', 'left-putamen', 
        'right-putamen', 'left-thalamus', 'right-thalamus','left-amygdala',  'right-amygdala', 'left-hippocampus', 'right-hippocampus', 'left-accumbens-area','right-accumbens-area','left-caudate', 'right-caudate', 'left-ventraldc', 'right-ventraldc', and 'brain-stem'.
    dilate_erode : bool
        Uses vtkImageDilateErode3D to slightly erode and dilate the volume before conversion, 
        which helps unify sparse voxels and produce smooth vertices. Default is True.
    shape_smooth : int
        The number of iterations of the smoothing procedure (0 will apply no smoothing). The 
        settings are otherwise those of VTK's SmoothDiscreteMarchingCubes.  Default is 15.
    metric : str, Sequence
        The name(s) of the metric(s) to be computed as strings. Options are "thickness", 
        "curvature", "surfarea", and default is all of them. For CIFTI data processing, this can only be one string, i.e. the name of the CIFTI metric matching the cifti_dscalar to be computed. 
    metric_smooth : tuple
        The full-maximum half-width (FMHW) values that will be applied for smoothing 
        each surface-based measure along the surface. 0 means no smoothing is applied. 
        In the following order: thickness, surface area, curvature. There should be 
        always three floats corresponding to that order, even if some metrics are not 
        selected in the "metric" argument.
    native_meshes : bool, optional
        Whether to save the meshes computed in subject space before they are projected 
        to template space (they will contain one scalar for each metric). Default is 
        False. overwrite must be True for this process to be accounted for.
    autoqc_n_sd : float
        The number of standard deviations from the cohort mean to define subjects as outliers, for each graphical feature value. Default is 2.
    cifti_dscalar : str
        Only for CIFTI data processing: all non cifti-related arguments will be ignored if not None. The specific pattern of the CIFTI file names (.dscalar.nii). These files names differ with the values and processing pipelines, so a suffix e.g "_den-91k_stat-reho_boldmap.dscalar" specifies reho maps are to be extracted. Also requires one metric name (e.g., "reho") specified in the metric argument. The function will not extract multiple values ('reho', 'alff', 'cbf' etc.) so make sure the pattern contains one. Wildcards ".*" are allowed if someone wants to specify a BIDS pattern (e.g., ".*ses-2.*rest.*_den-91k_stat-reho_boldmap.dscalar" ).
    overwrite : bool
        Whether files are to be overwritten or skipped if already in outputdir. 
        Default is False.
    silent : bool
        Whether messages about the process are to be printed. Default is False.
    total_nthread : int
        The total number of CPU threads to be allocated to the pipeline (default is 1). 
    sub_nthread : int
        The number of CPU threads to be allocated for each subject separately (default is 1). This only affects the volume coregistration stage. 
    """
    
    #prepare outputdir
    os.makedirs(outputdir, exist_ok=True)
    
    if sub_list is None:
        #list subs (1st depth of inputdir, and only counted as sub folders with the right "sub-xxx" pattern)
        pattern = r'(sub-[\w-]+)'
        sub_list = [f for f in next(os.walk(inputdir))[1] if re.search(pattern, f)]
        sub_list.sort()    
    elif isinstance(sub_list, str):
        sub_list=[sub_list]
    
    #error if sub_list empty
    if sub_list==[]:
        raise FileNotFoundError(f"No subject folders identified. Ensure inputdir is the directory that contains sub-ID subfolders.")
    
    if not silent:
        print(f"Running the entire SubCortexMesh pipeline on {len(sub_list)} subject(s)...")
    
    #record duration
    start_time = time.perf_counter()
    
    if cifti_dscalar is not None and not isinstance(metric, str):
        raise ValueError("The cifti_dscalar argument was given. For CIFTI data, the metric argument must be a single string matching the cifti_dscalar (e.g. metric='reho').")
    
    if cifti_dscalar is None:
        job = partial(_struct_job, inputdir=inputdir, outputdir=outputdir,
                      template=template, toolboxdata=toolboxdata,
                      roilabel=roilabel, dilate_erode=dilate_erode,
                      shape_smooth=shape_smooth, metric=metric,
                      metric_smooth=metric_smooth, native_meshes=native_meshes,
                      overwrite=overwrite, silent=silent)
    else:
        if not silent:
            print("cifti_dscalar specified. Processing CIFTI data only")
        job = partial(_cifti_job, inputdir=inputdir, outputdir=outputdir,
                      cifti_dscalar=cifti_dscalar, metric=metric,
                      roilabel=roilabel, toolboxdata=toolboxdata,
                      overwrite=overwrite, silent=silent)

    #making sure ANTs, sciPy etc. only use evenly available threads per parallel job, then reinitialise at the end
    if sub_nthread < 1 or total_nthread < 1:
        raise ValueError("total_nthread and sub_nthread must be at least 1.")
    if sub_nthread > total_nthread: #making sure it can be divided
        raise ValueError("sub_nthread cannot be larger than total_nthread.") 
    nworkers = max(1, min(total_nthread // sub_nthread, len(sub_list))) 
    thread_vars = ("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "OMP_NUM_THREADS",
                   "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    old_env = {k: os.environ.get(k) for k in thread_vars}
    os.environ.update({k: str(sub_nthread) for k in thread_vars})
    if not silent:
        print(f"{total_nthread} CPU threads set: each subject will be assigned {sub_nthread} CPU threads, {nworkers} subject(s) at a time.")
    
    try:
        #run one subject per worker
        results = []
        with ProcessPoolExecutor(max_workers=nworkers,
                                 mp_context=mp.get_context("spawn")) as executor:
            futures = {executor.submit(job, s): s for s in sub_list}
            for fut in tqdm(as_completed(futures), total=len(sub_list),
                            disable=silent, desc="Processing"):
                subid = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:  #worker died, not a Python error in the job
                    res = (subid, f"worker crashed: {type(e).__name__}")
                results.append(res)
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    
    failed = [s for s, status in results if status]
    if failed:
        print(f"{len(failed)} subject(s) failed, see {outputdir}/logs/: {failed}")
     
    #####   
    #outliers report (structural only, since CIFTI data is all on the same template surface)
    if cifti_dscalar is None:
        flag_df=autoqc_outliers(inputdir=f"{outputdir}/sub_surfaces/",
                                n_sd=autoqc_n_sd, outputdir=f"{outputdir}/qc/",
                                silent=True)
        if not silent:
            for subid in flag_df.index:
                flagged_structures = [c for c in flag_df.columns if flag_df.loc[subid, c] >= 3]
                if not flagged_structures:
                    continue
                print(f"Outlier detected for {subid}: {flagged_structures}")
    
    #print total duration
    if not silent:
        hours, rem = divmod(int(time.perf_counter() - start_time), 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            duration = f"{hours}h {minutes:02d}min"
        elif minutes:
            duration = f"{minutes}min {seconds:02d}s"
        else:
            duration = f"{seconds}s"
        print(f"scm_all() completed in {duration}.")
                        