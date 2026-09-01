#####################################################################################
##################Converting CIFTI 91k subcortical data to surfaces################## 

import os
import re
import numpy as np
import nibabel as nib
import pandas as pd  
import vtk
import tempfile
import time
from typing import Optional, Union, Sequence
from pathlib import Path
from vtkmodules.util import numpy_support
import pyvista as pv
from subcortexmesh import template_data_fetch
from subcortexmesh.mesh_metrics import print_stats, print_stats_atlas
from scipy.ndimage import distance_transform_edt
from nibabel.processing import resample_from_to

def cifti_metrics(
    dscalar: str, 
    metric: str,
    inputdir: Union[str, Path],
    outputdir: Union[str, Path],
    roilabel: Union[str, Sequence[str]] = ['left-cerebellum-cortex', 'right-cerebellum-cortex', 
                                    'left-pallidum', 'right-pallidum', 'left-putamen', 'right-putamen', 'left-thalamus', 'right-thalamus','left-amygdala', 'right-amygdala', 'left-hippocampus','right-hippocampus', 'left-accumbens-area','right-accumbens-area','left-caudate', 'right-caudate', 'left-ventraldc', 'right-ventraldc', 'brain-stem'],
    toolboxdata: Optional[Union[str, Path]] = None,
    plot_projection: bool = False,
    overwrite: bool = True,
    silent: bool = False):
    
    """Subcortical grayordinates from CIFTI files
    
    This function extracts the ~30k voxel-based subcortical grayordinates, in CIFTI (Connectivity Informatics Technology Initiative) file format, and projects their values to a surface template space. This is done with SCM's fsaverage template surfaces as CIFTI subcortices are also based on the ASeg segmentations. It is compatible with 91k outputs such as that of the Human Connectome Project preprocessing pipeline or pipelines that output cifti such as ASLprep, fMRIprep, XCP-D. It works similarly to mesh_metrics() and creates a surface_metrics_cifti/ output directory, containing surfaces with their CIFTI values (in separate subdirectories per session-,task-,run- and acq- when applicable). It also produces summary statistics, but only in template space as no subject surface is created in this process.
    
    Parallel processes: to avoid conflicts, subjects will be skipped if a "isrunning" tmp 
    file exists to mark them as currently processing. The tmp file is removed at the end or 
    replaced if 1 hour old. If a process has been interrupted, remove the tmp manually to 
    rerun a subject before the 1 hour (its path is printed when flagged). 
    
    Parameters
    ----------
    dscalar: str
        The specific pattern of the CIFTI file names (.dscalar.nii). These files names differ with the values and processing pipelines, so a suffix e.g "_den-91k_stat-reho_boldmap.dscalar" specifies reho maps are to be extracted. The function will not extract multiple values ('reho', 'alff', 'cbf' etc.) so make sure the pattern contains one.
    metric: str
        The name of the metric to be computed as string. It will be used to name the surface files, scalars and identify them with the merging tools.
    inputdir : str, Path
        The directory where the CIFTI objects were outputted. The function will look for sub-IDs throughout and identify the files indicated by the dscalar argument.
    outputdir : str, Path
        The path where subcortical meshes with vertex-wise values assigned will be saved (will 
        create a "surface_metrics_cifti" directory, as well as session/run/acq specific outputs).
    roilabel: str, Sequence
        The name(s) of the region(s)-of-interest to be computed as strings. Default is all
        subcortices across all segmentation templates: 'left-cerebellum-cortex', 'right-cerebellum-cortex', 'left-pallidum', 'right-pallidum', 'left-putamen', 
        'right-putamen', 'left-thalamus', 'right-thalamus','left-amygdala',  'right-amygdala', 'left-hippocampus', 'right-hippocampus', 'left-accumbens-area','right-accumbens-area','left-caudate', 'right-caudate', 'left-ventraldc', 'right-ventraldc', and 'brain-stem'.
    toolboxdata : str, Path, optional
        The path of the "subcortexmesh_data" package data directory. The  default path 
        is assumed to be the user's home directory (pathlib's Path.home()). Users will 
        be prompted to download it if not found.
    plot_projection: bool
        Whether to plot the voxel-space values next to the template-space mesh the values are projected to. Default is False.
    overwrite : bool
        Whether files are to be overwritten or skipped if already in outputdir. 
        Default is True.
    silent : bool
        Whether messages about the process are to be printed. Default is False.
    """
    
    #template data is needed
    toolboxdata=template_data_fetch(datapath=toolboxdata, template = 'fsaverage')
    
    #paths check
    if not os.path.exists(f"{inputdir}"):
        raise FileNotFoundError("Input directory not found. Please verify the path provided as inputdir.")
    
    #creates folder where volumes will go 
    subvol_path = os.path.join(outputdir, "surface_metrics_cifti")
    if not silent and not os.path.exists(subvol_path):
        print(f"Writing the {outputdir}/surface_metrics_cifti directory...")
    os.makedirs(subvol_path, exist_ok=True)

    #mesh loader function
    def load_mesh(path):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput()
        
    #list subjects 
    sub_list = [f for f in next(os.walk(inputdir))[1] if re.search(r'(sub-[\w-]+)', f)]
    sub_list.sort()   
    
    subindex=0
    for subid in sub_list:
        subindex=subindex+1
        
        #unique tmp file to avoid parallel loop conflicts
        fname = os.path.join(tempfile.gettempdir(), f"{subid}_isrunning_cifti.tmp")
        #if exists already, and tmp file is younger than 1h, skip subject
        if os.path.exists(fname): 
            tmp_lifetime = (time.time() - os.path.getmtime(fname)) / 3600
            if tmp_lifetime < 1:
                if not silent:
                    print(f"{subid} already running (tmp file: {fname}).")
                continue
        #creates tmp if loop wasn't skipped
        with open(fname, "w"):
                pass
        
        if not silent: 
            print(f"Extracting CIFTI data for {subid}... [{subindex}/{len(sub_list)}]")
        #preparing subdir
        os.makedirs(f"{outputdir}/surface_metrics_cifti/", exist_ok=True)
        subdir = f"{outputdir}/surface_metrics_cifti/{subid}" 
        os.makedirs(subdir, exist_ok=True)
        
        #look for relevant dscalar file(s)
        sub_files = [os.path.join(dp, f)
                for dp, dn, fn in os.walk(f"{inputdir}/{subid}/")
                for f in fn if re.search(dscalar, f)]
        
        #check for duplicate matches within the same session/task/acq/run (can happen in dscalar is too broad)
        seen = {}
        for f in sub_files:
            m = re.search(r"(ses-\d+(?:_task-[^_]+)(?:_acq-[^_]+)?(?:_run-\d+)?)", f)
            key = m.group(1) if m else f
            if key in seen:
                raise ValueError(f"Multiple dscalar files matched the pattern '{dscalar}' for {subid} in the exact same session/task/run. Narrow the `dscalar` pattern to make sure one is selected.")
            seen[key] = f
        
        for ciftifile in sub_files:
            
            #checking for session, run etc. in file name
            m = re.search(r"(ses-\d+(?:_task-[^_]+)(?:_acq-[^_]+)?(?:_run-\d+)?)", ciftifile)
            #will save it in a ses/run/acq specific folders to avoid conflicts in later tools
            if m:
                sub_ses = m.group(1)
                subdir_spec=os.path.join(subdir, sub_ses)
                os.makedirs(subdir_spec, exist_ok=True)
                if not silent: 
                    print(f"=> Loading {sub_ses} scalar...")
            
            
            ###############################################################################
            ###############################FETCHING CIFTI DATA#############################
            
            img = nib.load(ciftifile)
            bm = img.header.get_axis(1)  #BrainModelAxis 
            is_subcort = bm.volume_mask  #only subcortices are volumes, cortices are surface
            
            vol_coords = bm.voxel[is_subcort]  
            #subcortical labels: 'CIFTI_STRUCTURE_THALAMUS_left' etc.
            vol_labels   = np.asarray(bm.name)[is_subcort] 
            affine  = bm.affine  #MNI affine
            vol_shape = bm.volume_shape #resolution
            vol_values = img.get_fdata()[0, is_subcort]  #voxel-wise functional values
            
            #CIFTI structure named as per SCM
            struct_to_region = {
                "CIFTI_STRUCTURE_BRAIN_STEM": "brain-stem",
                "CIFTI_STRUCTURE_ACCUMBENS_LEFT": "left-accumbens-area",
                "CIFTI_STRUCTURE_ACCUMBENS_RIGHT": "right-accumbens-area",
                "CIFTI_STRUCTURE_AMYGDALA_LEFT": "left-amygdala",
                "CIFTI_STRUCTURE_AMYGDALA_RIGHT": "right-amygdala",
                "CIFTI_STRUCTURE_CAUDATE_LEFT": "left-caudate",
                "CIFTI_STRUCTURE_CAUDATE_RIGHT": "right-caudate",
                "CIFTI_STRUCTURE_CEREBELLUM_LEFT": "left-cerebellum-cortex",
                "CIFTI_STRUCTURE_CEREBELLUM_RIGHT": "right-cerebellum-cortex",
                "CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT": "left-hippocampus",
                "CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT": "right-hippocampus",
                "CIFTI_STRUCTURE_PALLIDUM_LEFT": "left-pallidum",
                "CIFTI_STRUCTURE_PALLIDUM_RIGHT": "right-pallidum",
                "CIFTI_STRUCTURE_PUTAMEN_LEFT": "left-putamen",
                "CIFTI_STRUCTURE_PUTAMEN_RIGHT": "right-putamen",
                "CIFTI_STRUCTURE_THALAMUS_LEFT": "left-thalamus",
                "CIFTI_STRUCTURE_THALAMUS_RIGHT": "right-thalamus",
                "CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT": "left-ventraldc",
                "CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT": "right-ventraldc",
            }
            vol_regions = np.array([struct_to_region[lab] for lab in vol_labels])
                
            #extract per region
            for regionlabel in struct_to_region.values():
                
                if regionlabel not in roilabel:
                    continue
                
                out_path=f"{subdir_spec}/{regionlabel}_{metric}.vtk"
                if not os.path.exists(out_path) or overwrite:
                    #get volume straight from the cifti coordinates
                    region_mask = vol_regions == regionlabel
                    region_vol = np.zeros(vol_shape, dtype=np.float32)
                    region_vol[tuple(vol_coords[region_mask].T)] = vol_values[region_mask]
                    
                    if not silent: 
                        print(f"   => {regionlabel}...")
                    
                    #####################################################################################
                    ################################PREPARING VOL########################################
                    
                    #fsaverage's own conformed-space reference volume — this is the exact grid the
                    #mesh's raw voxel-index coordinates were built against
                    templatevol = f"{toolboxdata}/template_data/fsaverage/T1.mgz"
                    fsaverage_ref_img = nib.load(templatevol)
                    
                    #wrap region_vol (CIFTI/MNI space) as a proper nifti with its real affine, then
                    #resample onto fsaverage's grid (the mesh is based upon)
                    region_vol_img = nib.Nifti1Image(region_vol, affine)
                    region_vol_resampled_img = resample_from_to(region_vol_img, fsaverage_ref_img, order=0)
                    region_vol_resampled = np.asarray(region_vol_resampled_img.dataobj)
                    
                    region_mask_img = nib.Nifti1Image((region_vol > 0).astype(np.float32), affine)
                    region_mask_resampled_img = resample_from_to(region_mask_img, fsaverage_ref_img, order=0)
                    region_mask_resampled = np.asarray(region_mask_resampled_img.dataobj) > 0.5
                    
                    region_vol = region_vol_resampled  #keep old variable name for the rest of the pipeline
                    region_shape = region_vol.shape
                    
                    #####################################################################################
                    #######################################VOL-TO-MESH###################################
                    
                    #load meshes corresponding to volumes
                    mesh = load_mesh(f"{toolboxdata}/template_data/fsaverage/surfaces/{regionlabel}.vtk")
                    
                    #flip Z for mesh and volume coord mismatch
                    pts = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())  #vertex coord 
                    vert_idx = np.round(pts).astype(int) #rounding to nearest integer converts vertex coords to voxel indices
                    vert_idx[:, 2] = (region_shape[2] - 1) - vert_idx[:, 2] 
                    vert_idx[:, 0] = np.clip(vert_idx[:, 0], 0, region_shape[0] - 1)
                    vert_idx[:, 1] = np.clip(vert_idx[:, 1], 0, region_shape[1] - 1)
                    vert_idx[:, 2] = np.clip(vert_idx[:, 2], 0, region_shape[2] - 1)
                    
                    #Get centroids to affine alignment between mesh and volumes
                    #volume centroid is mean position of all labeled voxels/vertex coords
                    nz = np.array(np.where(region_mask_resampled)).T  # (N, 3)
                    vol_centroid = nz.mean(axis=0)
                    mesh_centroid = vert_idx.mean(axis=0)
                    #Offset to apply to vert_idx to align mesh toward volume
                    offset = vol_centroid - mesh_centroid
                    #shift vert_idx by offset before EDT lookup
                    vert_idx_aligned = vert_idx + np.round(offset).astype(int)
                    vert_idx_aligned[:, 0] = np.clip(vert_idx_aligned[:, 0], 0, region_shape[0] - 1)
                    vert_idx_aligned[:, 1] = np.clip(vert_idx_aligned[:, 1], 0, region_shape[1] - 1)
                    vert_idx_aligned[:, 2] = np.clip(vert_idx_aligned[:, 2], 0, region_shape[2] - 1)
                    vert_idx=vert_idx_aligned
                    
                    #Projecting values using Euclidean Distance Transform (EDT): every vertex is assigned its closest 
                    #voxel equivalent
                    labeled_mask = region_mask_resampled #remove background voxels
                    _, nearest_idx = distance_transform_edt(~labeled_mask, return_indices=True)
                    #every vertex is assigned the value of the nearest voxel
                    mesh_values = region_vol[
                        nearest_idx[0][vert_idx[:, 0], vert_idx[:, 1], vert_idx[:, 2]],
                        nearest_idx[1][vert_idx[:, 0], vert_idx[:, 1], vert_idx[:, 2]],
                        nearest_idx[2][vert_idx[:, 0], vert_idx[:, 1], vert_idx[:, 2]]
                    ]
                    
                    #final mesh with assigned scalars
                    vtk_values = numpy_support.numpy_to_vtk(mesh_values, deep=True, array_type=vtk.VTK_FLOAT)
                    vtk_values.SetName(metric)
                    mesh.GetPointData().AddArray(vtk_values)
                    mesh.GetPointData().SetActiveScalars(metric)
                    
                    #####################################################################################
                    #######################################PlOT##########################################
                    
                    if plot_projection is True:
                        #plot 3d voxel-based volume with continuous values
                        region_vol_flipped = np.flip(region_vol, axis=2).copy() #flip Z as in previous code
                        vol = pv.ImageData()
                        vol.dimensions = np.array(region_shape) + 1
                        vol.cell_data[metric] = region_vol_flipped.flatten(order="F")
                        
                        #remove back ground voxels using the mask
                        region_mask_flipped = np.flip(region_mask_resampled, axis=2).copy()
                        vol.cell_data["in_mask"] = region_mask_flipped.flatten(order="F").astype(np.uint8)
                        vol_thresh = vol.threshold(0.5, scalars="in_mask")   #thresholds on binary mask
                        
                        #shared color range across mesh + volume so they're comparable
                        clim = [
                            min(mesh_values.min(), region_vol[region_mask_resampled].min()),
                            max(mesh_values.max(), region_vol[region_mask_resampled].max()),
                        ]
                        
                        #make plot
                        mesh_pv = pv.wrap(mesh)
                        plotter = pv.Plotter()
                        
                        region_surf = vol_thresh.extract_surface(algorithm='dataset_surface')
                        plotter.add_mesh(region_surf, scalars=metric, cmap="viridis", clim=clim, opacity=1)
                        mesh_actor = plotter.add_mesh(mesh_pv, scalars=metric, cmap="viridis", clim=clim, backface_culling=False)
                        
                        #slider to allow users to separate mesh from volume
                        def update_y(value):
                            if abs(value) < 1:
                                mesh_actor.SetPosition(0, 0, 0)
                            else:
                                mesh_actor.SetPosition(0, -value, 0)
                        
                        plotter.add_slider_widget(update_y, rng=[-50, 50], title="Y-axis")
                        plotter.add_scalar_bar(title=metric)
                        plotter.add_axes(interactive=True, ylabel='Y')
                        plotter.show(title=f"{sub_ses} - {regionlabel}: volume vs mesh")
                    
                    #####################################################################################
                    ##############################STATS SUMMARY##########################################
                    
                    if not silent: 
                        print("   Printing descriptive statistics...")
                    #As opposed to mesh_metrics, these statistics are not in subject space
                    #as the voxel values are directly transferred to the template
                    print_stats(subdir_spec, mesh, regionlabel, metric)
                    print_stats_atlas(subdir_spec, mesh, regionlabel, metric, 'fsaverage', toolboxdata)
                    
                    #####################################################################################
                    #######################################SAVE##########################################
                    
                    #guarantee overwriting
                    if os.path.exists(out_path) and overwrite:
                        os.remove(out_path)
                    
                    writer = vtk.vtkPolyDataWriter()
                    writer.SetFileName(out_path)
                    writer.SetInputData(mesh)
                    _ = writer.Write()
                
                else:
                    if not silent: 
                        print(f"   {regionlabel} already extracted.")
        
        os.remove(fname)  #cleanup tmp file
    
    if not silent: 
            print(f"Surface metrics stored to {outputdir}/surface_metrics_cifti/")
