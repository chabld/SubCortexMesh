#####################################################################################
######Creates a large mesh collating all aseg subcortices into one big surface#######

import pyvista as pv
import numpy as np
import pandas as pd 
import os
import re
import vtk
import tempfile
import time
import warnings
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
    their vertex-wise values. If not all subcortices have been processed, the function
    will add the missing subcortices with empty NaN values instead of the metrics. The 
    merged mesh will be saved along the surface meshes in the directories used as input.
    
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
    if template=='fslfirst':
        mergedmesh='allfslfirst'
    
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
        for i, (filename, vtk_obj) in enumerate(mesh_list):
            meshfile = load_mesh(f"{inputdir}/{subid}/{filename}") if filename else vtk_obj
            
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
                if not silent:
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
                    
                    if not silent:
                        print(f"=> Merging {measure} ...")

                    #force the mesh_list follow templates' ROI order as in the lookup table
                    roi_lookup = pd.read_csv(f"{toolboxdata}/template_data/{template}/surfaces/{mergedmesh}_roi_id.txt",sep="\t")
                    roi_order = roi_lookup['label'].tolist()

                    #detect missing structures
                    present_rois = set()
                    for f in mesh_list:
                        for roi in roi_order:
                            if f.startswith(roi) and f.endswith(".vtk"):
                                present_rois.add(roi)
                                break
                    missing_rois = [roi for roi in roi_order if roi not in present_rois]

                    if missing_rois and not silent:
                        print(f"=> Warning: missing structures for {subid}: {', '.join(missing_rois)}. Template meshes with NaN scalars will be used.")

                    #get scalar names from first available subject mesh
                    first_file = next((f for f in mesh_list if f.endswith(".vtk")), None)
                    scalar_names = []
                    if first_file is not None:
                        ref_mesh = load_mesh(f"{inputdir}/{subid}/{first_file}")
                        pd_ref = ref_mesh.GetPointData()
                        scalar_names = [pd_ref.GetArrayName(j) for j in range(pd_ref.GetNumberOfArrays())]

                    #reorder mesh_list as tuples (filename, vtk_obj); inject template NaN meshes for missing ROIs
                    mesh_list_sorted = []
                    for roi in roi_order:
                        matches = [f for f in mesh_list if f.startswith(roi) and f.endswith(".vtk")]
                        if matches:
                            mesh_list_sorted.append((matches[0], None))
                        else:
                            tmpl_mesh = load_mesh(f"{toolboxdata}/template_data/{template}/surfaces/{roi}.vtk")
                            n_pts = tmpl_mesh.GetNumberOfPoints()
                            for sname in scalar_names:
                                nan_array = vtk.vtkFloatArray()
                                nan_array.SetName(sname)
                                nan_array.SetNumberOfTuples(n_pts)
                                nan_array.Fill(float('nan'))
                                tmpl_mesh.GetPointData().AddArray(nan_array)
                            if scalar_names:
                                tmpl_mesh.GetPointData().SetActiveScalars(scalar_names[0])
                            mesh_list_sorted.append((None, tmpl_mesh))
                    mesh_list = mesh_list_sorted

                    #merge mesh
                    merged_mesh=mesh_merger()

                    #save it
                    #guarantee overwriting
                    out_path=f"{inputdir}/{subid}/{mergedmesh}_{measure}.vtk"
                    if os.path.exists(out_path):
                        os.remove(out_path)

                    writer = vtk.vtkPolyDataWriter()
                    writer.SetFileTypeToBinary()
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
    smooth_mesh:  Optional[int] = 0,
    atlas_map: Optional[bool] = False,
    toolboxdata: Optional[Union[str, Path]] = None
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
    atlas_map: bool, optional
        When True, adds toggle buttons to show anatomical atlas' colour labels and/or outlines on top of the mesh values. Requires the toolboxdata argument.
    toolboxdata : str, Path, optional
        The path of the "subcortexmesh_data" package data directory. The  default path 
        is assumed to be the user's home directory (pathlib's Path.home()). It is only needed
        if users want to plot the atlas toggle. Users will be prompted to download it if not found.
    """
    if isinstance(merged_vtk, (str, Path)):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(merged_vtk))
        reader.Update()
        mesh = pv.wrap(reader.GetOutput())
    else:
        mesh = pv.wrap(merged_vtk)
    
    ###################################################################
    ##############################DEFINE SCALARS#######################
    
    measure = mesh.active_scalars_name
    if measure is None:
        raise ValueError(
            "No active scalar (surface-based value) found in the mesh. Make sure the file was produced by merge_all()."
        )
    
    #If atlas needed, fetched from the template data and appended as a mesh scalar
    atlas_lut = None
    if atlas_map:
        if mesh.GetNumberOfPoints()==95718:
            template='fsaverage'
            toolboxdata = template_data_fetch(datapath=toolboxdata, template=template)
            arraypath=f'{toolboxdata}/template_data/{template}/atlas/anatomical_atlas_{template}.npy'
            if os.path.isfile(arraypath):
                #set colour and atlas mask to the mesh
                atlasarray=np.load(arraypath)
                atlas_lut=pd.read_csv(f'{toolboxdata}/template_data/{template}/atlas/anatomical_atlas_{template}_names.txt')
                #read RGB colour code in the LUT and make a 3d colour array
                lut_rgb = atlas_lut.set_index('new_id')[['R', 'G', 'B']]
                atlas_rgb = np.array(
                    [tuple(lut_rgb.loc[int(v)]) if int(v) in lut_rgb.index else (211, 211, 211) for v in atlasarray], dtype=np.uint8)
                #append to mesh
                mesh.point_data['atlas_rgb'] = atlas_rgb
                mesh.point_data['atlas_id'] = atlasarray.astype(int)   #id for hover lookup
            else:
                raise FileNotFoundError(f"No atlas was found in the template data directory ({toolboxdata}). To obtain the atlas, remove the directory and download the up-to-date data using template_data_fetch().")
        elif mesh.GetNumberOfPoints()==82412:
            warnings.warn(f"No atlas parcellation exists for the FSL FIRST merged mesh yet, so the atlas_map argument will be ignored.")
        
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
    #compute clim from the full mesh before splitting
    scalars_data = mesh.point_data[measure]
    #avoid warning if no value at all 
    if not np.isnan(scalars_data).all():
        if clim is None:
            clim = [np.nanmin(scalars_data), np.nanmax(scalars_data)]
    
    base_actors = [] #base mesh
    atlas_actors = [] #atlas mesh
    atlas_meshes = []  #copies to adapt to slider
    actor_mesh_map = {}
    outline_actors = []
    outline_meshes = [None] * len(wrapped_meshes) #atlas outline
    outline_original_points = [None] * len(wrapped_meshes) 
    for idx, wm in enumerate(wrapped_meshes):
        m_actor = plotter.add_mesh(wm, scalars=measure, cmap=cmap, clim=clim, nan_color='lightgrey')
        base_actors.append(m_actor)
        actor_mesh_map[m_actor] = wm
        if atlas_map:
            am = wm.copy()
            a_actor = plotter.add_mesh(am, scalars='atlas_rgb', rgb=True)
            a_actor.visibility = False
            atlas_actors.append(a_actor)
            atlas_meshes.append(am)
            actor_mesh_map[a_actor] = am
            
            #outline settings
            label_values = np.unique(wm.point_data['atlas_id'])
            if len(label_values) > 1:
                boundary_values = np.arange(label_values.min() - 0.5, label_values.max() + 1.5, 1.0)
                ol = wm.contour(isosurfaces=boundary_values, scalars='atlas_id')
                if ol.n_points > 0:
                    #adapt colour to nearest neighbor vertices
                    locator = vtk.vtkPointLocator()
                    locator.SetDataSet(wm)
                    locator.BuildLocator()
                    nearest_ids = np.array([locator.FindClosestPoint(pt) for pt in ol.points])
                    ol.point_data['atlas_rgb'] = wm.point_data['atlas_rgb'][nearest_ids]
                    o_actor = plotter.add_mesh(ol, scalars='atlas_rgb', rgb=True, line_width=2, render_lines_as_tubes=True)
                    o_actor.visibility = False
                    outline_actors.append(o_actor)
                    outline_meshes[idx] = ol
                    outline_original_points[idx] = ol.points.copy()
    
    # Y flipped as VTK's coord syst not following RAS
    plotter.reset_camera()
    loc, foc, _ = plotter.camera_position
    plotter.camera_position = [loc, foc, (0, -1, 1)]
    
    def update_distance(distfactor):
        for i, (wm, centroid, orig) in enumerate(zip(wrapped_meshes, centroids, original_points)):
            direction = centroid - global_centroid
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            translation = direction * distfactor
            new_points = orig - centroid + (centroid + translation)
            wm.points[:] = new_points
            if atlas_map:
                atlas_meshes[i].points[:] = new_points   
                if outline_meshes[i] is not None: 
                    outline_meshes[i].points[:] = outline_original_points[i] + translation
        
        plotter.render()
    
    #slider settings
    plotter.add_slider_widget(
        update_distance, 
        rng=[0, 50], 
        value=0,
        pointa=(0.25, 0.92),   # left end, higher up (closer to 1.0 = top)
        pointb=(0.75, 0.92),   # right end, same height as pointa
        tube_width=0.005,      # thinner track (default is usually ~0.008)
        slider_width=0.02,     # thinner knob (default is usually ~0.02)
    )
    
    if atlas_map:
        def toggle_atlas(flag):
            for a in atlas_actors:
                a.visibility = flag
            for m in base_actors:
                m.visibility = not flag
            plotter.render()
        
        #checkbox for atlas colours
        plotter.add_checkbox_button_widget(
            toggle_atlas,
            value=False, #not visible by default
            position=(10, plotter.window_size[1] - 150),
            size=50,
            border_size=3,
            color_on='mediumseagreen',
            color_off='grey',
        )
        
        #checkbox for outlines
        def toggle_outline(flag):
            for o in outline_actors:
                o.visibility = flag
            plotter.render()

        plotter.add_checkbox_button_widget(
            toggle_outline,
            value=False, #not visible by default
            position=(70, plotter.window_size[1] - 150), 
            size=50,
            border_size=3,
            color_on='dodgerblue',
            color_off='grey',
        )
        
        #hover settings
        hover_picker = vtk.vtkCellPicker()
        hover_picker.SetTolerance(0.0005)
        label_lookup = atlas_lut.set_index('new_id')['new_label'].to_dict()
        
        plotter.add_text("", position=(10, 10), font_size=10, color='black', name='hover_label')
        
        def on_mouse_move(caller, event):
            x, y = plotter.iren.interactor.GetEventPosition()
            hover_picker.Pick(x, y, 0, plotter.renderer)
            actor = hover_picker.GetActor()
            point_id = hover_picker.GetPointId()
            
            label_text = ""
            if actor is not None and point_id is not None and point_id >= 0:
                src_mesh = actor_mesh_map.get(actor)
                if src_mesh is not None and 'atlas_id' in src_mesh.point_data:
                    atlas_id = int(src_mesh.point_data['atlas_id'][point_id])
                    label_text = label_lookup.get(atlas_id, "")
            
            plotter.add_text(label_text, position=(10, 10), font_size=10, color='black', name='hover_label')
            plotter.render()
        
        plotter.iren.add_observer("MouseMoveEvent", on_mouse_move)
    
    #title
    if isinstance(merged_vtk, (str, Path)):
        plotter.show(title=f"{str(merged_vtk)} - {measure}")
    else:
        plotter.show(title=f"{measure}")


#####################################################################################
######Flat 2D grid preview of all subcortical ROIs from a merged VTK surface########

def vis_merged_flat(
    merged_vtk: Union[str, Path, vtk.vtkPolyData],
    output_path: Union[str, Path] = "flat_plot.png",
    silent: bool = False,
    scalars: str = None,
    cmap: str = 'viridis',
    clim: Optional[Tuple[float, float]] = None,
    smooth_mesh:  Optional[int] = 0,
    atlas_map = False,
    toolboxdata: Optional[Union[str, Path]] = None
):
    """Flat 2D grid preview of all subcortical ROIs from a merged VTK surface

    Reads a merged .vtk file produced by merge_all() which contains all
    subcortical structures concatenated into a single mesh, where each ROI
    was assigned a number ID. For each ROI, the corresponding sub-mesh is
    extracted, centered, and placed in a 2D grid layout saved as PNG.
    The layout is paired (left/right structures in adjacent columns); in
    a top and bottom views.

    Authors: Nicolas P.M. Lavarde, Charly H.A. Billaud

    Parameters
    ----------
    merged_vtk : str, Path, vtk.vtkPolyData
        Path to the merged .vtk file produced by merge_all().
    output_path : str, Path
        Path for the output PNG file. Default is 'flat_plot_preview.png'.
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
    atlas_map: bool, optional
        Whether to add a button which allows to add an anatomical atlas' colour outline. 
    toolboxdata : str, Path, optional
        The path of the "subcortexmesh_data" package data directory. The  default path 
        is assumed to be the user's home directory (pathlib's Path.home()). Users will 
        be prompted to download it if not found.
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
    #avoid warning if no value at all 
    if not np.isnan(scalars_data).all():
        if clim is None:
            clim = [np.nanmin(scalars_data), np.nanmax(scalars_data)]
    
    ###################################################################
    ######################EXTRACT ATLAS################################
    
    #If atlas needed, fetched from the template data and appended as a mesh scalar
    atlas_lut = None
    if atlas_map:
        if mesh.GetNumberOfPoints()==95718:
            template='fsaverage'
            toolboxdata = template_data_fetch(datapath=toolboxdata, template=template)
            arraypath=f'{toolboxdata}/template_data/{template}/atlas/anatomical_atlas_{template}.npy'
            if os.path.isfile(arraypath):
                #set colour and atlas mask to the mesh
                atlasarray=np.load(arraypath)
                atlas_lut=pd.read_csv(f'{toolboxdata}/template_data/{template}/atlas/anatomical_atlas_{template}_names.txt')
                #read RGB colour code in the LUT and make a 3d colour array
                lut_rgb = atlas_lut.set_index('new_id')[['R', 'G', 'B']]
                atlas_rgb = np.array(
                    [tuple(lut_rgb.loc[int(v)]) if int(v) in lut_rgb.index else (211, 211, 211) for v in atlasarray], dtype=np.uint8)
                #append to mesh
                mesh.point_data['atlas_rgb'] = atlas_rgb
                mesh.point_data['atlas_id'] = atlasarray.astype(int)   #id for hover lookup
            else:
                raise FileNotFoundError(f"No atlas was found in the template data directory ({toolboxdata}). To obtain the atlas, remove the directory and download the up-to-date data using template_data_fetch().")
        elif mesh.GetNumberOfPoints()==82412:
            warnings.warn(f"No atlas parcellation exists for the FSL FIRST merged mesh yet, so the atlas_map argument will be ignored.")
    
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

    roi_names = {i: f"ROI {i}" for i in range(n_roi)}
    
    if mesh.GetNumberOfPoints()==95718:
        template='fsaverage'
    elif mesh.GetNumberOfPoints()==82412:
        template='fslfirst'
    
    _N_ROI_MESH = {19: 'allaseg', 17: 'allfslfirst'}
    mergedmesh = _N_ROI_MESH[n_roi]
    toolboxdata = template_data_fetch(datapath=toolboxdata, template=template)
    roi_id_path = f"{toolboxdata}/template_data/{template}/surfaces/{mergedmesh}_roi_id.txt"
    roi_lookup = pd.read_csv(roi_id_path, sep='\t')
    roi_names = dict(zip(roi_lookup['id'].astype(int), roi_lookup['label']))
    
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
    
    ###################################################################
    ######################EXTRACT OUTLINES##############################
    
    outline_submeshes = [None] * n_roi
    if atlas_map and atlas_lut is not None:
        for i, sub in enumerate(submeshes):
            if sub is None or 'atlas_id' not in sub.point_data:
                continue
            label_values = np.unique(sub.point_data['atlas_id'])
            label_values = label_values[label_values > 0]
            if len(label_values) < 2:
                continue  # nothing to contour, only one label present in this ROI
            boundary_values = np.arange(label_values.min() - 0.5, label_values.max() + 1.5, 1.0)
            ol = sub.contour(isosurfaces=boundary_values, scalars='atlas_id')
            if ol.n_points == 0:
                continue
            locator = vtk.vtkPointLocator()
            locator.SetDataSet(sub)
            locator.BuildLocator()
            nearest_ids = np.array([locator.FindClosestPoint(pt) for pt in ol.points])
            ol.point_data['atlas_rgb'] = sub.point_data['atlas_rgb'][nearest_ids]
            outline_submeshes[i] = ol
    
    ###################################################################
    ####################BUILD PAIR/SINGLETON ROWS######################
    
    # Pass 1: map stripped names to their left/right roi_id
    left_map = {}
    right_map = {}
    for roi_id, label in roi_names.items():
        ll = label.lower()
        if ll.startswith("left-"):
            left_map[ll[5:]] = roi_id
        elif ll.startswith("right-"):
            right_map[ll[6:]] = roi_id
    
    # Pass 2: build ordered row list — pairs first (roi_id order), singletons last
    seen = set()
    rows = []
    singleton_ids = []
    
    for roi_id in sorted(roi_names.keys()):
        if roi_id in seen:
            continue
        label = roi_names[roi_id]
        ll = label.lower()

        if ll.startswith("left-"):
            key = ll[5:]
            right_id = right_map.get(key)
            if right_id is not None:
                rows.append(('pair', roi_id, right_id))
                seen.add(roi_id)
                seen.add(right_id)
            else:
                singleton_ids.append(roi_id)
                seen.add(roi_id)
        elif ll.startswith("right-"):
            key = ll[6:]
            left_id = left_map.get(key)
            # only fires when right precedes left in roi_id order (atypical)
            if left_id is not None and left_id not in seen:
                rows.append(('pair', left_id, roi_id))
                seen.add(roi_id)
                seen.add(left_id)
            else:
                singleton_ids.append(roi_id)
                seen.add(roi_id)
        else:
            singleton_ids.append(roi_id)
            seen.add(roi_id)
    
    for roi_id in singleton_ids:
        rows.append(('singleton', roi_id))
    
    ###################################################################
    ##########################GRID RENDERING###########################
    # Y flipped as VTK's coord syst not following RAS
    
    nrows = len(rows)
    plotter = pv.Plotter(
        shape=(nrows, 4),
        off_screen=True,
        window_size=(4 * 300, nrows * 300),
    )
    plotter.set_background("white")
    
    # fixed orthographic cameras: Y is superior in VTK coords
    cam_top    = ((0, -500, 0), (0, 0, 0), (0, -1, -1))
    cam_bottom = ((0, 500, 0), (0, 0, 0), (0, -1, -1))
    
    def _place_mesh(sub_idx, col, row_idx, camera, text_label, view_label, is_empty_label=False):
        plotter.subplot(row_idx, col)
        sub = submeshes[sub_idx] if sub_idx < len(submeshes) else None
        if sub is not None:
            plotter.add_mesh(
                sub,
                show_edges=False,
                smooth_shading=True,
                ambient=0.3,
                diffuse=0.7,
                scalars=sub.point_data[scalars],
                cmap=cmap,
                clim=clim,
                nan_color='lightgrey',
                show_scalar_bar=False,
            )
            #add matching outline
            if atlas_map and sub_idx < len(outline_submeshes) and outline_submeshes[sub_idx] is not None:
                plotter.add_mesh(
                    outline_submeshes[sub_idx],
                    scalars='atlas_rgb', rgb=True,
                    line_width=2, render_lines_as_tubes=True,
            )
            plotter.add_text(text_label, position="upper_edge", font_size=8, color="black")
            plotter.add_text(view_label, position="lower_edge", font_size=8, color="black")
        else:
            plotter.add_text(
                f"{text_label}\n(empty)" if not is_empty_label else text_label,
                position="upper_edge",
                font_size=8,
                color="gray")
        plotter.camera.parallel_projection = True
        plotter.camera_position = camera
        plotter.reset_camera()
    
    for row_idx, row_def in enumerate(rows):
        
        if row_def[0] == 'pair':
            _, left_id, right_id = row_def
            left_label  = roi_names.get(left_id,  f"ROI {left_id}")
            right_label = roi_names.get(right_id, f"ROI {right_id}")

            _place_mesh(left_id,  0, row_idx, cam_top,    left_label, "top view")
            _place_mesh(right_id, 1, row_idx, cam_top,    right_label, "top view")
            _place_mesh(right_id, 2, row_idx, cam_bottom, right_label, "bottom view")
            _place_mesh(left_id,  3, row_idx, cam_bottom, left_label, "bottom view")
        
        else:  # singleton
            _, roi_id = row_def
            label = roi_names.get(roi_id, f"ROI {roi_id}")

            _place_mesh(roi_id, 1, row_idx, cam_top,    label, "top view")
            plotter.subplot(row_idx, 1)
            plotter.set_background("white")
            _place_mesh(roi_id, 2, row_idx, cam_bottom, label, "bottom view")
            plotter.subplot(row_idx, 3)
            plotter.set_background("white")
    
    #add scalar bar to last empty square
    plotter.subplot(nrows - 1, 3)
    plotter.add_text(scalars, position="upper_edge", font_size=8, color="black")
    plotter.add_scalar_bar(
        vertical=True,
        position_x=0.3,
        position_y=0.15,
        width=0.8,
        height=0.6,
        fmt="%.2f",
        label_font_size=10
    )
    
    ###################################################################
    ##########################ATLAS LEGEND##############################
    
    if atlas_map and atlas_lut is not None:
        plotter.subplot(nrows - 1, 0)
        plotter.add_text('Labels', position="upper_edge", font_size=8, color="black")
        plotter.set_background("white")
        #merge L and R as it's the same colours
        def _base_label(label):
            return re.sub(r'_(L|R)$', '', label)
        #also abbreviate ROIs so they fit in the box:
        _SHORTEN_MAP = {'Cerebellum': 'Cerebel','Hippocampus': 'Hippo','Globus_Pallidus': 'Pallidum',
            'Brain_Stem': 'BStem','Amygdala': 'Amyg','amygdaloid': 'amyg','Suprageniculate': 'Supragen',
            'Centromedian': 'Centmed','Peduncle': 'Pedunc','Nucleus': 'Nuc','_Area':'','Thalamus': 'Thal',
            'Cerebellar': 'Cerebel','Superior': 'Sup','Lateral': 'Lat','Latero':'Lat','Ventral': 'Ventr',
            'Ventral_DC': 'VDC', 'Cortico': 'Cort', 'Parafascicular': 'Parafasc'
        }
        def _shorten_label(label):
            for full, short in _SHORTEN_MAP.items():
                label = label.replace(full, short)
            return label
        
        legend_lut = (
            atlas_lut.assign(base_label=atlas_lut['new_label'].apply(_base_label).apply(_shorten_label))
            .drop_duplicates(subset='base_label')
        )
        n_labels = len(legend_lut)
        n_cols = 2
        rows_per_col = int(np.ceil(n_labels / n_cols))
        col_x_positions = [0.02, 0.50, 0.70]
        row_height = 0.88 / rows_per_col
        
        for j, row in enumerate(legend_lut.itertuples()):
            col = j // rows_per_col
            row_in_col = j % rows_per_col
            color = (row.R / 255, row.G / 255, row.B / 255)
            x = col_x_positions[col]
            y = 0.88 - row_in_col * row_height
            plotter.add_text(
                row.base_label,
                position=(x, y),
                viewport=True,
                font_size=5,
                color=color)
    
    plotter.screenshot(str(output_path))
    if not silent:
        print(f"Saved flat plot to: {output_path}")
