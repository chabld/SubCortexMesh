#####################################################################################
######Creates a large mesh collating all aseg subcortices into one big surface#######

import pyvista as pv
import numpy as np
import pandas as pd 
import os
import vtk
import tempfile
import time
from typing import Optional, Union, Sequence, Tuple
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
    clim: Optional[Tuple[float, float]] = None,
    smooth_mesh:  Optional[int] = 0
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
    clim: Tuple, optional
        Sequence of float stating the minimum and maximum value of the color bar. Default is minimum and maximum value.
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
            "No active scalar (surface-based value) found in the mesh. Make sure the file was produced by merge_all()."
        )
    
    #compute clim from the full mesh before splitting
    scalars_data = mesh.point_data[measure]
    if clim is None:
        clim = [np.nanmin(scalars_data), np.nanmax(scalars_data)]
    for wm in wrapped_meshes:
        _ = plotter.add_mesh(wm, scalars=measure, cmap=cmap, clim=clim, nan_color='lightgrey')
    
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


#####################################################################################
######Flat 2D grid preview of all subcortical ROIs from a merged VTK surface########

#Function to automatically compute orientations  
def _compute_orientations(submeshes, cam_distance=300.0):
    """Compute a PCA-based camera position for each submesh.
    
    Author: Nicolas P.M. Lavarde
    
    Parameters
    ----------
    submeshes : list of pyvista.PolyData or None
        Each element is a centered submesh, or None for empty ROIs.
    cam_distance : float
        Distance from the origin at which the camera is placed along PC3.

    Returns
    -------
    list of tuple or None
        Same length as submeshes. Each non-None element is a PyVista
        camera_position tuple: ((cx, cy, cz), (0, 0, 0), (ux, uy, uz)).
    """
    orientations = []
    for sub in submeshes:
        if sub is None:
            orientations.append(None)
            continue
        
        pts = sub.points
        cov = np.cov(pts.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # eigh returns ascending order — reverse to get descending variance
        idx_sorted = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx_sorted]
        
        # PC1 = most elongated axis (col 0), PC2 = up vector (col 1), PC3 = camera axis (col 2)
        pc2 = eigenvectors[:, 1]
        pc3 = eigenvectors[:, 2]
        
        # Ensure camera is on the positive-Z side
        if pc3[2] < 0:
            pc3 = -pc3
        
        cam_pos = tuple(pc3 * cam_distance)
        focal_point = (0.0, 0.0, 0.0)
        up_vector = tuple(pc2)
        
        orientations.append((cam_pos, focal_point, up_vector))

    return orientations


# returns a single PNG file with a grid of all ROIs from the merged mesh, 
# each colored differently and oriented along its PCA axes. 

def vis_merged_flat(
    merged_vtk: Union[str, Path, vtk.vtkPolyData],
    output_path: Union[str, Path] = "flat_plot.png",
    ncols: int = 4,
    silent: bool = False,
    scalars: str = None,
    cmap: str = 'viridis', 
    clim: Optional[Tuple[float, float]] = None,
    smooth_mesh:  Optional[int] = 0
):
    """Flat 2D grid preview of all subcortical ROIs from a merged VTK surface
    
    Reads a merged .vtk file produced by merge_all() which contains all
    subcortical structures concatenated into a single mesh, where each ROI
    was assigned a number ID. For each ROI, the corresponding sub-mesh is
    extracted, centered, and placed in a 2D grid layout saved as PNG.
    
    Authors: Nicolas P.M. Lavarde, Charly H.A. Billaud
    
    Parameters
    ----------
    merged_vtk : str, Path, vtk.vtkPolyData
        Path to the merged .vtk file produced by merge_all().
    output_path : str, Path
        Path for the output PNG file. Default is 'flat_plot_preview.png'.
    ncols : int
        Number of columns in the grid layout. Default is 4.
    silent : bool
        Whether to suppress progress messages. Default is False.
    scalars: str
        Name of the vertex-wise value which was assigned. Default is whatever measure was assigned
        by mesh_metrics() ('thickness', 'curvature', or 'surfarea'). Can also be the 'roi_id' 
        assigned by merge_all().
    cmap: str
        Name of the color map to be assigned to the background volume, as listed in matplotlib's colormaps. Default is "viridis".
    clim: Tuple, optional
        Sequence of float stating the minimum and maximum value of the color bar. Default is minimum and maximum value.
    smooth_mesh: int, optional
        Number of iterations of cosmetic smoothing to make the surface appear smoother. Default is 0.
    """
    
    ###################################################################
    ##########################LOAD MERGED MESH#########################
    
    if isinstance(merged_vtk, (str, Path)):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(merged_vtk))
        reader.Update()
        mesh = pv.wrap(reader.GetOutput())
    else:
        mesh = pv.wrap(merged_vtk)
    
    if 'roi_id' not in mesh.point_data.keys():
        raise ValueError(
            f"'roi_id' point array not found in {merged_vtk}. "
            "Make sure the file was produced by merge_all()."
        )
    
    roi_ids = np.array(mesh.point_data['roi_id']).astype(int)
    
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
    
    ###################################################################
    ##############################DEFINE SCALARS#######################
    
    #default scalar is metric
    measure = mesh.active_scalars_name
    if measure is None:
        raise ValueError(
            "No active scalar (surface-based value) found in the mesh. Make sure the file was produced by merge_all().")
    if scalars is None:
        scalars=measure
        scalars_data = mesh.point_data[scalars]
    else:
        mesh.set_active_scalars(scalars)
        scalars_data = mesh.point_data[scalars]
    #compute clim from the full mesh before splitting
    if clim is None:
        clim = [np.nanmin(scalars_data), np.nanmax(scalars_data)]
    
    ###################################################################
    ######################EXTRACT ROI SUB-MESHES#######################
    
    submeshes = []
    #get number of ROIs based on the available roi_id scalar (assigned by merge_all())
    _N_ROI_MAP = {95718: 19, 82412: 17}
    n_roi = _N_ROI_MAP.get(mesh.n_points)
    if n_roi is None:
        raise ValueError(
            f"Cannot auto-detect n_roi for mesh with {mesh.n_points} vertices. "
            "Expected 95718 (fsaverage, 19 ROIs) or 82412 (fslfirst, 17 ROIs)."
        )
    for i in range(n_roi):
        mask = roi_ids == i
        n_pts = int(mask.sum())
        
        if n_pts == 0:
            if not silent:
                print(f"  ROI {i}: no vertices found, skipping.")
            submeshes.append(None)
            continue
        
        # extract_points keeps only faces whose ALL vertices are in the mask
        sub_ug = mesh.extract_points(mask, adjacent_cells=False)
        sub = sub_ug.extract_surface(algorithm='dataset_surface')
        
        # center each structure at origin so cells don't overlap
        centroid = sub.points.mean(axis=0)
        sub = sub.copy()
        sub.points -= centroid
        
        submeshes.append(sub)
    
    orientations = _compute_orientations(submeshes)
    
    ###################################################################
    ##########################GRID RENDERING###########################
    
    nrows = int(np.ceil(n_roi / ncols))
    plotter = None
    plotter = pv.Plotter(
        shape=(nrows, ncols),
        off_screen=True,
        window_size=(ncols * 300, nrows * 300),
    )
    plotter.set_background("white")
    
    for idx in range(n_roi):
        row = idx // ncols
        col = idx % ncols
        plotter.subplot(row, col)
        
        if submeshes[idx] is not None:
            plotter.add_mesh(
                submeshes[idx],
                show_edges=False,
                smooth_shading=True,
                ambient=0.3,
                diffuse=0.7,
                #settings inherited from stat_tools
                scalars=submeshes[idx].point_data[scalars],
                cmap=cmap,
                clim=clim,
                nan_color='lightgrey',
                show_scalar_bar=True
            )
            plotter.add_text(
                f"ROI {idx}",
                position="upper_left",
                font_size=8,
                color="black",
            )
        else:
            plotter.add_text(
                f"ROI {idx}\n(empty)",
                position="upper_left",
                font_size=8,
                color="gray",
            )

        # orthographic view oriented along the PCA camera axis
        plotter.camera.parallel_projection = True
        plotter.camera_position = orientations[idx]
        plotter.reset_camera()
    
    # fill unused cells (if n_roi not a multiple of ncols)
    for idx in range(n_roi, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        plotter.subplot(row, col)
        plotter.set_background("white")
    
    plotter.screenshot(str(output_path))
    if not silent:
        print(f"Saved flat plot to: {output_path}")
