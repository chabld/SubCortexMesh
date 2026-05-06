#####################################################################################
######Creates a large mesh collating all aseg subcortices into one big surface#######

import pyvista as pv
import numpy as np
import pandas as pd 
import os
import vtk
import tempfile
import time
from typing import Optional, Union, Sequence
from pathlib import Path
from subcortexmesh import template_data_fetch

def merge_all(
    inputdir: Union[str, Path],
    template: str,
    toolboxdata: Optional[Union[str, Path]] = None,
    metric: Union[str, Sequence[str]] = ['thickness', 'curvature', 'surfarea'],
    plot_merged: bool = False,
    overwrite: bool = True,
    silent: bool = False,
):
    """Merging all subcortical outputs into a single surface object
    
    This function creates a new mesh merging all subcortical meshes outputted by 
    mesh_metrics() in a given template, for available metrics separately, keeping 
    their vertex-wise values. It will only work if all subcortices have been 
    processed. The merged mesh will be saved along the surface meshes in the 
    directories used as input.
    
    For fslfirst, the cerebella need to have been created in FSL FIRST inside the same
    output directory as run_first_all's, naming them "*R_Cereb_first" and "*L_Cereb_first",
    and processed with subseg_getvol() and vol2surf(). See subseg_getvol()'s description 
    for guidance.
    
    Parallel processes: to avoid conflicts, subjects will be skipped if a "isrunning" tmp 
    file exists to mark them as currently processing. The tmp file is removed at the end or 
    replaced if 1 hour old. If a process has been interrupted, remove the tmp manually to 
    rerun a subject before the 1 hour (its path is printed when flagged). 
    
    Parameters
    ----------
    inputdir : str, Path
        The path where the surface-based metrics were outputted (using mesh_metrics()).
        The outputdir will be the inputdir.
    template: str
        The name of the template the surfaces are supposed to be matching to. For FreeSurfer
        outputs, it is 'fsaverage'. For FSL FIRST, it is 'fslfirst'.
    toolboxdata : str, Path, optional
        The path of the "subcortexmesh_data" package data directory. The  default path 
        is assumed to be the user's home directory (pathlib's Path.home()). Users will 
        be prompted to download it if not found.
    metric: str, Sequence
        The name(s) of the metric(s) to be computed as strings. Options are "thickness", 
        "curvature", "surfarea", and default is all of them.
    plot_merged: bool
        Whether to plot the resulting merged mesh. Default is False.
    overwrite : bool
        Whether files are to be overwritten or skipped if already made. Default 
        is True.
    silent : bool
        Whether messages about the process are to be printed. Default is False.
    """
    
    ###################################################################
    ###################################################################
    
    #template data is needed
    toolboxdata=template_data_fetch(datapath=toolboxdata, template = template)
    if template=='fsaverage':
        mergedmesh='allaseg'
        nroi=19
    if template=='fslfirst':
        mergedmesh='allfslfirst'
        nroi=17
    
    #Subfunctions
    #mesh loader function
    def load_mesh(path):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput()
    
    #merge all meshes together, storing the ROI labels as an in-vtk array
    def mesh_merger():
        appendFilter = vtk.vtkAppendPolyData()
        for i, mesh in enumerate(mesh_list):
            meshfile=load_mesh(f"{inputdir}/{subid}/{mesh}")
            
            # Add string array with ROI name
            tagArray = vtk.vtkIntArray()
            tagArray.SetName("roi_id")
            tagArray.SetNumberOfTuples(meshfile.GetNumberOfPoints())
            tagArray.FillComponent(0, i)  # i = ROI index
            _ = meshfile.GetPointData().AddArray(tagArray)
            appendFilter.AddInputData(meshfile)
        
        appendFilter.Update()
        merged_mesh = appendFilter.GetOutput()
        
        if plot_merged:
            vis_merged(merged_vtk=merged_mesh)
        
        return merged_mesh
    
    ###################################################################
    ###################################################################
    
    #run on subject output directories
    #list subjects in the surface subjects directory
    sub_list =[    d for d in os.listdir(inputdir)
        if os.path.isdir(os.path.join(inputdir, d))]
    
    subindex=0
    for subid in sub_list:
        subindex=subindex+1
        
        #unique tmp file to avoid parallel loop conflicts
        fname = os.path.join(tempfile.gettempdir(), f"{subid}_isrunning_merge.tmp")
        if os.path.exists(fname): #if exists already, and tmp file is younger than 1h, skip subject
            tmp_lifetime = (time.time() - os.path.getmtime(fname)) / 3600
            if tmp_lifetime < 1:
                print(f"{subid} already running (tmp file: {fname}).")
                continue
        else: #creates tmp
            with open(fname, "w"):
                pass
        
        if not silent: 
            print(f"Creating all-aseg surfaces for {subid}... [{subindex}/{len(sub_list)}]")
        
        for measure in ['thickness', 'surfarea', 'curvature']:
            
            if measure not in metric:
                continue
            
            if not os.path.exists(f"{inputdir}/{subid}/{mergedmesh}_{measure}.vtk") or overwrite:
                
                #listing files for that metric
                mesh_list = [
                    f for f in os.listdir(f"{inputdir}/{subid}")
                    if f"{measure}" in f and not f.startswith("all") #explicitly do not list the merged vtk
                ]
                
                if len(mesh_list) > 0:
                    
                    if len(mesh_list) != nroi:
                        if not silent: 
                            print(f"{measure} ignored: all subcortices of the {template} template ({nroi}) must be available to create the template-wide surface.")
                    else:
                        if not silent: 
                            print(f"=> Merging {measure} ...")
                        
                        #force the mesh_list follow templates' ROI order as in the lookup table
                        roi_lookup = pd.read_csv(f"{toolboxdata}/template_data/{template}/surfaces/{mergedmesh}_roi_id.txt",sep="\t")
                        roi_order = roi_lookup['label'].tolist()
                        #reorder mesh_list
                        mesh_list_sorted = []
                        for roi in roi_order:
                            #find filename in mesh_list that starts with ROI label
                            matches = [f for f in mesh_list if f.startswith(roi) and f.endswith(".vtk")]
                            if matches: 
                                mesh_list_sorted.extend(matches)
                        mesh_list = mesh_list_sorted
                        
                        #merge mesh
                        merged_mesh=mesh_merger()
                            
                        #save it
                        #guarantee overwriting
                        out_path=f"{inputdir}/{subid}/{mergedmesh}_{measure}.vtk"
                        if os.path.exists(out_path):
                            os.remove(out_path)
                        
                        writer = vtk.vtkPolyDataWriter()
                        writer.SetFileName(out_path)
                        writer.SetInputData(merged_mesh)
                        _ = writer.Write()
                else:
                    if not silent: 
                        print(f"No mesh file (.vtk) found at all for {subid}'s surface {measure}.")
            else:
                if not silent: 
                    print(f"=> {measure} already merged")
        
        os.remove(fname)  #cleanup tmp file
                    
                
###################################################################
###################################################################

#interactive plot function that plots all surfaces together, and gives a slider to space them out from eachother
def vis_merged(
    merged_vtk: Union[str, Path, vtk.vtkPolyData], 
    cmap: str = "viridis", 
    smooth_mesh:  Optional[int] = 0,
    ):
    """Interactive 3D viewer for a merged subcortical surface.
    
    Loads a merged mesh produced by merge_all(), separates ROIs by
    roi_id, and displays them with a slider to spread the structures 
    apart from their global centroid.
    
    Authors: Charly H.A. Billaud, Nicolas P.M. Lavarde
    
    Parameters
    ----------
    merged_vtk : str, Path, vtk.vtkPolyData 
        Path to the merged .vtk file produced by merge_all() or the VTK polydata variable itself
    cmap: str
        Name of the color map to be assigned to the background volume, as listed in matplotlib's colormaps. Default is "viridis".
    smooth_mesh: int, optional
        Number of iterations of cosmetic smoothing to make the surface appear smoother. Default is 0.
    """
    if isinstance(merged_vtk, (str, Path)):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(merged_vtk))
        reader.Update()
        mesh = pv.wrap(reader.GetOutput())
    else:
        mesh = pv.wrap(merged_vtk)
    
    #appearance smoother
    if smooth_mesh is not None and smooth_mesh > 0:
        s = vtk.vtkWindowedSincPolyDataFilter()
        s.SetInputData(mesh)
        s.SetNumberOfIterations(smooth_mesh)
        s.SetPassBand(0.001)
        s.NonManifoldSmoothingOn()
        s.NormalizeCoordinatesOn()
        s.Update()
        mesh=pv.wrap(s.GetOutput())
    
    if 'roi_id' not in mesh.point_data.keys():
        raise ValueError(
            f"'roi_id' point array not found in {merged_vtk}. "
            "Make sure the file was produced by merge_all()."
        )
    
    roi_ids = np.array(mesh.point_data['roi_id']).astype(int)
    n_roi = int(roi_ids.max()) + 1
    
    wrapped_meshes = []
    for i in range(n_roi):
        mask = roi_ids == i
        if not mask.any():
            continue
        sub_ug = mesh.extract_points(mask, adjacent_cells=False)
        sub = sub_ug.extract_surface(algorithm='dataset_surface')
        wrapped_meshes.append(sub)
    
    all_points = np.vstack([wm.points for wm in wrapped_meshes])
    global_centroid = all_points.mean(axis=0)
    
    centroids = [wm.points.mean(axis=0) for wm in wrapped_meshes]
    original_points = [wm.points.copy() for wm in wrapped_meshes]
    
    plotter = pv.Plotter()
    measure = mesh.active_scalars_name
    if measure is None:
        raise ValueError(
            "No active scalar (surface-based metric) found in the mesh. Make sure the file was produced by merge_all()."
        )
    for wm in wrapped_meshes:
        _ = plotter.add_mesh(wm, scalars=measure, cmap=cmap)
    
    # Y flipped as VTK's coord syst not following RAS
    plotter.reset_camera()
    loc, foc, _ = plotter.camera_position
    plotter.camera_position = [loc, foc, (0, -1, 1)]
    
    def update_distance(distfactor):
        for wm, centroid, orig in zip(wrapped_meshes, centroids, original_points):
            direction = centroid - global_centroid
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            translation = direction * distfactor
            wm.points[:] = orig - centroid + (centroid + translation)
        plotter.render()
    
    plotter.add_slider_widget(update_distance, rng=[0, 50], value=0)
    if isinstance(merged_vtk, (str, Path)):
        plotter.show(title=f"{str(merged_vtk)} - {measure}")
    else:
        plotter.show(title=f"{measure}")

