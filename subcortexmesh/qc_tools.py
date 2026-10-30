#####################################################################################
#####################################################################################
###########Plotting native surfaces on top of their volume ##########################

import os
import re
import vtk
import io
import html
import base64
import pyvista as pv
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.decomposition import PCA
from subcortexmesh.mesh_metrics import extract_medial_curve
import matplotlib
import matplotlib.pyplot as mplpyplot
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Button, Slider
matplotlib.use('TkAgg') #matplotlib rendering backend for windows
from typing import Optional, Union, Sequence
from pathlib import Path
from PIL import Image

def surf_qcplot(
    volpath: Union[str, Path],
    surfdir: Union[str, Path],
    vol_color_map: Optional[str] = "gray",
    outline_color_map: Optional[str] = "tab20",
    roilabel: Union[str, Sequence[str]] = ['left-cerebellum-cortex', 'right-cerebellum-cortex', 
                                    'left-pallidum', 'right-pallidum', 'left-putamen', 'right-putamen', 'left-thalamus', 'right-thalamus','left-amygdala', 'right-amygdala', 'left-hippocampus','right-hippocampus', 'left-accumbens-area','right-accumbens-area','left-caudate', 'right-caudate', 'left-ventraldc', 'right-ventraldc', 'brain-stem'],
    default_mesh: int = 5
    ):
    """Plotting subject surface boundaries on top of matching volume
    
    This function function will plot all surfaces generated with vol2surf() for a given subject (normally stored in "sub_surfaces/sub-[id]/") and plot their boundaries on top of a selected .nii volume Because the surfaces are based on a volume rigidly coregistered to a template, the surfaces will match a volume generated with subseg_getvol(), i.e.  "sub_volumes/sub-[id]/ants_coreg/T1_[template]_rigid_coreg.nii.gz" (or any volume likewise coregistered).   
    
    Note that as per subseg_getvol(), individual regions-of-interest (ROIs) will have been inflated and smoothed by default to minimise graphical artefacts, which will naturally be reflected in the plot: regions will appear with slightly wider boundaries than their original anatomy. Another logical result is that ROI boundaries will appear overlapping, but since the surfaces are processed by SubCortexMesh entirely separately, this has no effect on the metrics values (ROIs overlapping here cannot be mixed up in their calculation by mesh_metrics() or merge_all()).
    
    Parameters
    ----------
    volpath : str, Path
        The path to the volume to be plotted in the background (e.g., the T1 volume coregistered to fsaverage or fslfirst templates as part of subseg_getvol() in sub_volume/sub-[id]/ants_coreg/).
    surfdir : str, Path
        The path to the directory where subcortical .vtk meshes have been saved (normally, "sub_surfaces/sub-[id]".
    vol_color_map : str
        Name of the color map to be assigned to the background volume, as listed in matplotlib's colormaps. Default is "gray".
    outline_color_map : str
        Name of the color map to be assigned (in alphabetical order) to the subcortical surface outlines, as listed in matplotlib's colormaps. Default is "tab20".
    roilabel: str, Sequence
        The name(s) of the region(s)-of-interest to be included in the plot. Default is all
        subcortices across all segmentation templates: 'left-cerebellum-cortex', 
        'right-cerebellum-cortex', 'left-pallidum', 'right-pallidum', 'left-putamen', 
        'right-putamen', 'left-thalamus', 'right-thalamus','left-amygdala',  'right-amygdala', 'left-hippocampus', 'right-hippocampus', 'left-accumbens-area','right-accumbens-area','left-caudate', 'right-caudate', 'left-ventraldc', 'right-ventraldc', and 'brain-stem'.
        Users can also omit the hemispheres to include both sides of a given ROI.
    default_mesh : int
        The index number corresponding to the region-of-interest to by plotted
        by default. 5 (left-hippocampus, for fsaverage) is the default.
    """
    
    ###################################################################
    ##################VOLUME/BOUNDARIES RENDER#########################
    
    #Load volume
    vol_reader = vtk.vtkNIFTIImageReader()
    vol_reader.SetFileName(volpath)
    vol_reader.Update()
    vol = pv.wrap(vol_reader.GetOutput())
    #convert volume to numpy array for later manipulation
    vol_arr = vol.point_data[vol.active_scalars_name].reshape(vol.dimensions, order="F")
    
    #Load surfaces and store them in an array
    surf_files = [f for f in sorted(os.listdir(surfdir)) if f.endswith(".vtk")]
    if len(surf_files) == 0:
        raise FileNotFoundError(f"No surface files were found in {surfdir}.")
    #only selected ROIs as per roilabel
    if isinstance(roilabel, str):
        roilabel = [roilabel] #if a single string given, turns to list
    surf_files=[s for s in surf_files if any(re.search(p, s) for p in roilabel)]
    #default mesh can only be 5 if at least 5 surfs
    if len(surf_files) <= 5:
        default_mesh=0 
        
    surfaces = []
    for fname in surf_files:
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(os.path.join(surfdir, fname))
        reader.Update()
        #downsampling vertex count to make it fasters
        decimator = vtk.vtkQuadricDecimation()
        decimator.SetInputConnection(reader.GetOutputPort())
        #decimation stronger for bigger meshes
        n_pts = reader.GetOutput().GetNumberOfPoints()
        reduction = 0.95 if n_pts > 8000 else 0.9
        decimator.SetTargetReduction(reduction)
        #decimator.PreserveTopologyOn()
        decimator.Update()
        surfaces.append(pv.wrap(decimator.GetOutput()))
    
    
    # limit sliders to the slice range actually occupied by any surface
    all_points = np.vstack([surf.points for surf in surfaces])
    sag_lo, sag_hi = int(np.floor((all_points[:,0].min() - vol.origin[0]) / vol.spacing[0])), \
                     int(np.ceil ((all_points[:,0].max() - vol.origin[0]) / vol.spacing[0]))
    ax_lo,  ax_hi  = int(np.floor((all_points[:,1].min() - vol.origin[1]) / vol.spacing[1])), \
                     int(np.ceil ((all_points[:,1].max() - vol.origin[1]) / vol.spacing[1]))
    cor_lo, cor_hi = int(np.floor((all_points[:,2].min() - vol.origin[2]) / vol.spacing[2])), \
                     int(np.ceil ((all_points[:,2].max() - vol.origin[2]) / vol.spacing[2]))
    
    # clamp to valid array indices
    sag_lo, sag_hi = max(sag_lo, 0), min(sag_hi, vol.dimensions[0]-1)
    ax_lo,  ax_hi  = max(ax_lo,  0), min(ax_hi,  vol.dimensions[1]-1)
    cor_lo, cor_hi = max(cor_lo, 0), min(cor_hi, vol.dimensions[2]-1)
    
    #assign colors (tab20) to each mesh (adapted to available meshes)
    cmap = mplpyplot.get_cmap(outline_color_map, len(surfaces))
    mesh_colors = [cmap(i) for i in range(len(surfaces))]
    
    #prepare plot
    #space for slices
    fig = mplpyplot.figure(figsize=(8, 8))
    ax_sag = fig.add_subplot(2, 2, 1)   # top-left
    ax_ax  = fig.add_subplot(2, 2, 2)   # top-right
    ax_cor = fig.add_subplot(2, 2, 3)   # bottom-left
    axes = {'sag': ax_sag, 'ax': ax_ax, 'cor': ax_cor}
    #base slice values, will change across slider
    x_mid, y_mid, z_mid  = ((sag_lo + sag_hi) // 2, (ax_lo  + ax_hi) // 2, (cor_lo + cor_hi) // 2) #middle slices
    slice_idx = {'sag':x_mid, 'ax':y_mid, 'cor':z_mid}
    #default rotation angle, will change at every click
    current_rotation = {'sag':1, 'ax':2, 'cor':2} 
    
    #mesh rotator: cannot simply be done with rot90 due to array structure (no empty space to rotate along with it)
    #a button callback will later modify its parameters for users to click
    def rotate_points(meshcoords, current_rotation, shape):
        height, width = shape #slice size for rotation reference
        if current_rotation == 1:
            return np.column_stack([meshcoords[:,1], width - meshcoords[:,0]]) #turn 90° leftward (moving axes)
        elif current_rotation == 2:
            return np.column_stack([width - meshcoords[:,0], height - meshcoords[:,1]]) #180°
        elif current_rotation == 3:
            return np.column_stack([height - meshcoords[:,1], meshcoords[:,0]]) #270°
        else:   
            return meshcoords #no rotation yet or back to default angle
    
    #draw slice
    def draw_slice(axis):
        subplot = axes[axis]
        
        #current zoom kept across slices if not the initial draw
        xlim = subplot.get_xlim()
        ylim = subplot.get_ylim()
        is_default = xlim == (0.0, 1.0) and ylim == (0.0, 1.0)  #default
        
        subplot.clear()
        
        #draft volume slice once
        if axis == 'sag': #x axis
            #load slice at that view
            img = vol_arr[slice_idx[axis],:,:].T #slice converted to 2D Numpy array 
            img_rot = np.rot90(img, current_rotation[axis]) #rotate coordinates 90 degrees for volume slice
            subplot.imshow(img_rot, cmap=vol_color_map, origin="lower") #update plotter
            #increase slider value from origin on that specific axis, relative to voxel space
            origin=(vol.origin[0]+slice_idx['sag']*vol.spacing[0],0,0) 
        elif axis == 'ax': #y axis
                img = vol_arr[:,slice_idx[axis],:].T
                img_rot = np.rot90(img, current_rotation[axis]) 
                subplot.imshow(img_rot, cmap=vol_color_map, origin="lower") 
                origin=(0,vol.origin[1]+slice_idx['ax']*vol.spacing[1],0)
        elif axis == 'cor': #z axis
                img = vol_arr[:,:,slice_idx[axis]].T
                img_rot = np.rot90(img, current_rotation[axis]) 
                subplot.imshow(img_rot, cmap=vol_color_map, origin="lower") 
                origin=(0,0,vol.origin[2]+slice_idx['cor']*vol.spacing[2])
        
        #plot every ROI surface on volume slice
        for surf, color in zip(surfaces, mesh_colors):
            #extract coordinates of intersecting mesh points
            if axis == 'sag': #x axis
                #identify mesh coordinates at that slice and rotate separately
                outline = surf.slice((1,0,0),origin=origin) #intersect mesh at this slice
                if outline.n_points == 0: #if empty, don't care
                    continue  
                meshcoords = outline.points[:, [1, 2]] #get updated coords for the moving axes
            elif axis == 'ax': #y axis
                outline = surf.slice((0,1,0),origin=origin)
                if outline.n_points == 0: #if empty, don't care
                    continue
                meshcoords = outline.points[:, [0, 2]] 
            elif axis == 'cor': #z axis
                outline = surf.slice((0,0,1),origin=origin)
                if outline.n_points == 0: #if empty, don't care
                    continue
                meshcoords = outline.points[:, [0, 1]]
            
            #rotate mesh if present inside volume slice
            if outline.n_points > 0:
                meshcoords = rotate_points(meshcoords, current_rotation[axis], img.shape)
                subplot.plot(meshcoords[:,0], meshcoords[:,1], ',', color=color, markersize=1)
            lines = outline.lines.reshape(-1, 3)[:, 1:]  # strip cell count, get pairs of point indices
            for i0, i1 in lines:
                p0 = meshcoords[i0]
                p1 = meshcoords[i1]
                subplot.plot([p0[0], p1[0]], [p0[1], p1[1]], '-', color=color, linewidth=1)
        
        subplot.set_title({'sag':"Sagittal",'ax':"Axial",'cor':"Coronal"}[axis])
        subplot.axis("off")
        
        # Restore zoom if user had zoomed in
        if not is_default:
            subplot.set_xlim(xlim)
            subplot.set_ylim(ylim)
        
        fig.canvas.draw_idle()
    
    #initial slice draw
    draw_slice('sag')
    draw_slice('ax')
    draw_slice('cor')
    
    ####################################################################
    ###########################3D VIEWER RENDER#########################
    
    ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
    
    current_mesh_idx = [default_mesh]  
    
    def draw_3d_mesh(idx):
        ax_3d.clear()
        surf = surfaces[idx]
        color = mesh_colors[idx] #same color as boundaries
        #define mesh
        faces = surf.faces.reshape(-1, 4)[:, 1:]
        verts = surf.points
        poly = Poly3DCollection(verts[faces], alpha=0.6, linewidth=0)
        poly.set_facecolor(color[:3])
        ax_3d.add_collection3d(poly)
           
        #Force aspect ratio across all 3 axes to prevent mpl from 
        #twisting shape
        bounds = [verts[:, i].min() for i in range(3)] + [verts[:, i].max() for i in range(3)]
        center = [(bounds[i] + bounds[i+3]) / 2 for i in range(3)]
        max_range = max(bounds[i+3] - bounds[i] for i in range(3)) / 2
        ax_3d.set_xlim3d(center[0] - max_range, center[0] + max_range)
        ax_3d.set_ylim3d(center[1] - max_range, center[1] + max_range)
        ax_3d.set_zlim3d(center[2] - max_range, center[2] + max_range)
        ax_3d.set_title(os.path.splitext(surf_files[idx])[0], fontsize=8)
        ax_3d.axis('off')
        fig.canvas.draw_idle()
    
    draw_3d_mesh(default_mesh) #startup mesh (left hippocampus)
    
    ####################################################################
    ############################ZOOM SCROLL#############################
    
    def on_scroll(event):
        ax = event.inaxes
        if ax not in axes.values():
            return
        
        scale = 0.85 if event.button == 'up' else 1.15
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        cx = (xlim[0] + xlim[1]) / 2
        cy = (ylim[0] + ylim[1]) / 2
        
        ax.set_xlim(cx - (cx - xlim[0]) * scale, cx + (xlim[1] - cx) * scale)
        ax.set_ylim(cy - (cy - ylim[0]) * scale, cy + (ylim[1] - cy) * scale)
        fig.canvas.draw_idle()
    
    class FakeEvent:
        button = 'up'
        inaxes = None
    
    fake = FakeEvent()
    for ax in axes.values():
        fake.inaxes = ax
        for _ in range(3):  # number of scroll steps
            on_scroll(fake)
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    ####################################################################
    ##############################BUTTONS###############################
    
    #rotation button 
    def make_rotate_callback(axis):
        def rotate(event):
            current_rotation[axis] = (current_rotation[axis] + 1) % 4 #4 angles, so value can only be cyclic
            draw_slice(axis)
        return rotate
    
    #slide slice along axis
    def make_slider_callback(axis):
        def update(val):
            slice_idx[axis] = int(val)
            draw_slice(axis)
        return update
    
    #render actual buttons for each subplot (hard-coded coordinates)
    #sagittal
    button_sag = Button(mplpyplot.axes([0.08, 0.52, 0.04, 0.03]), "↻")
    slider_gui_sag = mplpyplot.axes([0.18, 0.525, 0.13, 0.025], facecolor="lightgrey")
    button_sag.on_clicked(make_rotate_callback('sag'))
    slider_sag = Slider(slider_gui_sag, 'sag', sag_lo, sag_hi, valinit=x_mid, valfmt='%d')
    slider_sag.on_changed(make_slider_callback('sag'))
    #axial
    button_ax = Button(mplpyplot.axes([0.40, 0.52, 0.04, 0.03]), "↻")
    slider_gui_ax = mplpyplot.axes([0.50, 0.525, 0.13, 0.025], facecolor="lightgrey")
    button_ax.on_clicked(make_rotate_callback('ax'))
    slider_ax = Slider(slider_gui_ax, 'ax', ax_lo,  ax_hi, valinit=y_mid, valfmt='%d')
    slider_ax.on_changed(make_slider_callback('ax'))
    #coronal
    button_cor = Button(mplpyplot.axes([0.08, 0.025, 0.04, 0.03]), "↻")
    slider_gui_cor = mplpyplot.axes([0.18, 0.030, 0.13, 0.025], facecolor="lightgrey")
    button_cor.on_clicked(make_rotate_callback('cor'))
    slider_cor = Slider(slider_gui_cor, 'cor', cor_lo, cor_hi, valinit=z_mid, valfmt='%d')
    slider_cor.on_changed(make_slider_callback('cor'))
    #3d viewer
    mesh_idx_text = fig.text(0.50, 0.042, f"{default_mesh}/{len(surfaces)-1}", ha='center', fontsize=9)
    button_mesh_prev = Button(mplpyplot.axes([0.54, 0.025, 0.04, 0.04]), "◀")
    button_mesh_next = Button(mplpyplot.axes([0.59, 0.025, 0.04, 0.04]), "▶")
    
    def prev_mesh(event):
        current_mesh_idx[0] = (current_mesh_idx[0] - 1) % len(surfaces)
        mesh_idx_text.set_text(f"{current_mesh_idx[0]}/{len(surfaces)-1}")
        draw_3d_mesh(current_mesh_idx[0])
    
    def next_mesh(event):
        current_mesh_idx[0] = (current_mesh_idx[0] + 1) % len(surfaces)
        mesh_idx_text.set_text(f"{current_mesh_idx[0]}/{len(surfaces)-1}")
        draw_3d_mesh(current_mesh_idx[0])
    
    button_mesh_prev.on_clicked(prev_mesh)
    button_mesh_next.on_clicked(next_mesh)
    
    ###################################################################
    #############################LEGEND################################
    
    #Add subcortical legend on the right side
    legend_handles = []
    legend_labels = []
    for fname, color in zip(surf_files, mesh_colors):
        handle = mplpyplot.Line2D([0],[0], color=color, marker='.', linestyle='', markersize=10)
        legend_handles.append(handle)
        legend_labels.append(os.path.splitext(fname)[0])
    
    fig.legend(legend_handles, legend_labels,
            loc='center left', bbox_to_anchor=(0.74, 0.47),
            ncol=1, fontsize='small')
    
    fig._widgets = [button_sag, slider_sag, button_ax, slider_ax, button_cor, slider_cor, button_mesh_prev, button_mesh_next]
    
    #window title
    fig.canvas.manager.set_window_title(f'Surface QC Plot - {surfdir}')
    mplpyplot.subplots_adjust(left=0.088, bottom=0.077, right=0.712, top=0.95, wspace=0.017)
    mplpyplot.show()





#####################################################################################
#####################################################################################
####################Detecting surface outliers#######################################



def autoqc_outliers(
    inputdir: Union[str, Path], 
    n_sd: float = 2,
    outputdir: Optional[Union[str, Path]] = None,
    silent: bool = False,
    ):
    """Automated algorithm to flag surface outliers
    
    This function reads all surface meshes outputted by vol2surf in a "sub_surfaces" directory,
    and estimates five heuristic graphical features: 
    
    - Volume: total space enclosed in the mesh (VTK's mass properties)
    - Surface area: total area of all triangles on the mesh surface (VTK's mass properties)
    - Sphericity: how spherical the mesh is (VTK's mass properties)
    - Elongation: how elongated the mesh is (vertex divergences between axes)
    - Symmetry: how spread are vertices between the mesh's two halves
    
    For each ROI and subject, it will flag, relatively to the same ROI across the 
    cohort in sub_surfaces/, whether a subject's mesh diverges in 1, 2, 3, 4 or 
    all 5 features (± N standard deviations from the mean).
    
    Parameters
    ----------
    inputdir : str, Path
        The sub_surfaces/ directory where the surface objects were outputted (using vol2surf()).
    n_sd : float
        The number of standard deviations from the mean feature value, above or under 
        which a subject's label is to be flagged as an outlier. Default is 2.
    outputdir : str, Path, optional
        A path to the directory where the outlier table can be saved as "autoqc_outliers.csv" sheet.
    silent : bool
        Whether messages about the process are to be printed. Default is False.
    
    Returns
    -------
    flag_df : data.frame
        A data.frame array listing each subject per row, and each ROI per column. Each ROI is
        assigned 0 to 5 values, indicating whether the ROI for that subject was an outlier on
        none,any or all four heuristic features.
    """
    
    #Cohort-wide table to store all the dice files
    labels = ["left-accumbens-area", "right-accumbens-area", "left-amygdala", "right-amygdala",
                    "left-caudate", "right-caudate", "left-cerebellum-cortex", "right-cerebellum-cortex",
                    "left-hippocampus", "right-hippocampus", "left-pallidum", "right-pallidum",
                    "left-putamen", "right-putamen", "left-thalamus", "right-thalamus",
                    "left-ventraldc", "right-ventraldc", "brain-stem"]
    sub_list =[    d for d in os.listdir(inputdir)
            if os.path.isdir(os.path.join(inputdir, d))]
    
    #Four heuristic features
    features = ['volume', 'surface_area', 'sphericity', 'elongation', 'symmetry']
    #for each feature separately, collate their numeric value for each subject and each ROI in a 
    #global df
    dfs = {d: pd.DataFrame(index=sub_list, columns=labels) for d in features}
    
    #mesh loader function
    def load_mesh(path):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput()
    
    #data feature functions
    def mesh_symmetry(mesh):
        #sklearn's principal component analysis, helps find the directional axis across the mesh
        #get vertices that vary the most in the mesh (i.e., the direction across which it extends 
        # the most)
        pts = np.array([mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())])
        pca = PCA(n_components=3).fit(pts)
        distances = pts @ pca.components_[0] #dot product between every vertex and direction
        
        #split in half and get the variance in distance for half 1 and half 2
        #split at the average of distances in order to split roughly where a
        #change in shape largely appears
        spread_1= pts[distances <  distances.mean()].std()
        spread_2 = pts[distances >= distances.mean()].std()
        
        #get ratio of a half's distances compared to the other one (from 
        #perfectly symmetry 1.0 to asymmetry >1.0)
        return max(spread_1, spread_2) / min(spread_1, spread_2)
    
    def shape_features(mesh):
        #VTK mass properties filter gets closed mesh total "volume" (the total space enclosed) and "surface area" (total area of all triangles on the mesh surface) via divergence theorem 
        volume = vtk.vtkMassProperties()
        volume.SetInputData(mesh)
        volume.Update()
        V = volume.GetVolume()
        A = volume.GetSurfaceArea()
        
        #ratio of how sphere-like the shape is (0 to 1: perfect sphere is 1, increases with complexity)
        sphericity = volume.GetNormalizedShapeIndex()
        
        #Bounding box extents, i.e. the difference between the maximum and minimum coordinate 
        # (for each of the 3 axes). 
        #elongation the ratio of maximum divergence to minimum divergence among the axes
        pts = np.array([mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())])
        bbox = pts.max(axis=0) - pts.min(axis=0)
        elongation = np.sort(bbox)[-1] / np.sort(bbox)[0]
        
        #Asymmetry: ratio of 
        symmetry=mesh_symmetry(mesh)
        
        return dict(volume=V, surface_area=A, sphericity=sphericity, elongation=elongation, symmetry=symmetry)
    
    subindex=0
    for subid in sub_list:
        subindex=subindex+1
        
        if not silent: 
                print(f"{subid} [{subindex}/{len(sub_list)}]")
        
        #listing files available for subject
        mesh_list = [f for f in os.listdir(f"{inputdir}/{subid}") if f.endswith('.vtk')]
        
        if len(mesh_list) > 0:
            for meshfile in mesh_list:  
                #get base name to name surface the same way
                base = re.sub(r'\.vtk?$', '', meshfile)
                
                if not silent: 
                            print(f"   Checking {base}...")
                #load subject mesh and extract feature values
                subject_mesh = load_mesh(f"{inputdir}/{subid}/{meshfile}")
                desc = shape_features(subject_mesh)
                for d in features:
                    dfs[d].loc[subid, base] = desc[d]
        
        else:
            if not silent: 
                print(f"No mesh file (.vtk) found at all for {subid}.")
    
    #flag outliers: for each ROI and feature, flag subjects > mean ± 2 sd
    flag_df = pd.DataFrame(0, index=sub_list, columns=labels)
    for roi in labels:
        for d in features:
            vals = dfs[d][roi].dropna().astype(float)
            mean, std = vals.mean(), vals.std()
            outliers = vals[(vals < mean - n_sd*std) | (vals > mean + n_sd*std)].index
            flag_df.loc[outliers, roi] += 1  # accumulate flag count across features
    
    #save to CSV if needed
    if outputdir is not None:
        os.makedirs(outputdir, exist_ok=True)
        flag_df.to_csv(os.path.join(outputdir, "autoqc_outliers.csv"), index=True, index_label="subject")
    
    #flag_df contains 0-4 per subject/structure (0 means the subject is outlying in no feature,
    #4 means subject is outlying on all four features)
    return flag_df




#####################################################################################
#####################################################################################
##########Visual quality report: volumes, surfaces, medial curves and metrics########




def visual_qc_report(
    inputdir: Union[str, Path],
    template: str,
    metric_proj: Union[str, Sequence[str]] = 'thickness',
    roilabel: Union[str, Sequence[str]] = ['left-cerebellum-cortex', 'right-cerebellum-cortex', 
                                    'left-pallidum', 'right-pallidum', 'left-putamen', 'right-putamen', 'left-thalamus', 'right-thalamus','left-amygdala', 'right-amygdala', 'left-hippocampus','right-hippocampus', 'left-accumbens-area','right-accumbens-area','left-caudate', 'right-caudate', 'left-ventraldc', 'right-ventraldc', 'brain-stem'],
    sub_list: Optional[Sequence[str]] = None,
    outliers: Optional[Union[str, Path, pd.DataFrame]] = None,
    outlier_threshold: int = 3,
    cmap: str = "jet_r",
    columns: int = None,
    rows: int = None,
    overwrite: bool = True,
    silent: bool = False,
    cifti: bool = False
    ):
    """Creating a summary visual report of a subject's processing 
    
    The visual report is a single .html file per subject with a page containing a screenshot
    of every processing stage for every subcortical region, helping doing quick quality checks.
    This function can be run after a whole pipeline of processing has been generated for a 
    subject (volume extraction, volume to surface conversion, surface-based measures 
    computation; or CIFTI extraction). For mesh_metrics(), the subject-space surfaces must have 
    been saved (option native_meshes=True). For cifti_metrics(), the volumes must also be saved 
    (option keep_volumes=True). 
    
    For the base pipeline, the .html contains plots of:
    1. the subject's 3D volume (voxels, created by subseg_getvol()),
    2. the subject's 3D surface mesh (converted from the volume by vol2surf()),
    3. the surface (transparent) with its medial curve (computed by mesh_metrics()),
    4. the metric data on the subject's surface (native/subject space, computed by mesh_metrics()),
    5. the metric data projected on the template surface (projected by mesh_metrics()).
    
    If outlier flags generated by autoqc_outliers() are provided (in the outliers argument), 
    regions flagged as outliers will also be highlighted in the report.
    
    For CIFTI data (cifti=True): 
    1. the subject's 3D volume and metrics (voxel grayordinates with assigned values).
    2. the metrics values projected to the fsaverage template surface (projected with cifti_metrics()).
    
    Parameters
    ----------
    inputdir : str, Path
        A directory containing a sub_volumes/ directory where the subcortical volumes were 
        outputted (using subseg_getvol()), and a surface_metrics/ directory where the 
        surface-based measures were outputted (using mesh_metrics(), with native_meshes=True).
        If cifti is True, the outputdir given to cifti_metrics(), containing the 
        surface_metrics_cifti/ directory (with volumes/ subdirectories, i.e. keep_volumes=True).
    template: str
        The name of the template used in mesh_metrics(): for FreeSurfer
        outputs, it is 'fsaverage', for FSL FIRST, it is 'fslfirst'. 'fsaverage' for
        CIFTI data.
    metric_proj: str, Sequence
        The name(s) of the metric(s) to be plotted. Options are "thickness", 
        "curvature", "surfarea". Default is thickness. If several are given, each region 
        has one image per metric, and the metric is written next to its title.
    roilabel: str, Sequence
        The name(s) of the region(s)-of-interest to be computed as strings. Default is all
        subcortices across all segmentation templates: 'left-cerebellum-cortex', 'right-cerebellum-cortex', 'left-pallidum', 'right-pallidum', 'left-putamen', 
        'right-putamen', 'left-thalamus', 'right-thalamus','left-amygdala',  'right-amygdala', 'left-hippocampus', 'right-hippocampus', 'left-accumbens-area','right-accumbens-area','left-caudate', 'right-caudate', 'left-ventraldc', 'right-ventraldc', and 'brain-stem'.
    sub_list: str, Sequence, optional
        An optional list of subject IDs to be processed. If not specified, all subjects found will be included.
    outliers : str, Path, DataFrame, optional
        The table flagging outliers per subject and region, outputted by autoqc_outliers(): a 
        DataFrame object or the path to the .csv. Outliers will be flagged in the visual
        report if given. Default is "inputdir/qc/autoqc_outliers.csv" if present.
    outlier_threshold : int
        The minimum number of shape features a region must be an outlier in
        to be flagged (see autoqc_outliers() for more details on the features). Default is 3.
    cmap : str
        Color map of the metric surfaces. Default is "jet_r".
    columns : int
        Number of image columns in the window.  Default is 3 for FreeSurfer/FSL
        processing output, 4 for CIFTI output.
    rows : int
        Number of image rows in the window. Default is 3 for FreeSurfer/FSL
        processing output, 2 for CIFTI output.
    overwrite : bool
        Whether reports are to be overwritten or skipped if already existing. 
        Default is True.
    silent : bool
        Whether messages about the process are to be silenced. Default is False.
    cifti : bool
        Whether the report is made from cifti_metrics() outputs (surface_metrics_cifti/ 
        directory) instead of the structural pipeline's. template and outliers are then 
        ignored, and metric_proj must be the metric name given to cifti_metrics() (e.g. 
        "reho"). Default is False.
    
    Returns
    -------
    list of Path
        The reports written.
    
    Note: on headless nodes (HPC), run pyvista.start_xvfb() beforehand, or use an 
    OSMesa/EGL build of VTK.
    """
    
    figdir = os.path.join(inputdir, "figures")
    datadir = os.path.join(inputdir, "surface_metrics_cifti" if cifti else "sub_volumes")
    
    #set single string names to lists for flexible reuse
    metric_proj = [metric_proj] if isinstance(metric_proj, str) else list(metric_proj)
    roilabel = [roilabel] if isinstance(roilabel, str) else roilabel
    roilabel = [region.lower() for region in roilabel]
    
    #subject IDs 
    if not os.path.isdir(datadir):
        raise FileNotFoundError(f"No {os.path.basename(datadir)}/ directory found in inputdir. If your inputdir contains CIFTI outputs, add cifti=True.")
    subjects = sorted(d for d in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, d)))
    if sub_list is not None:
        wanted = [sub_list] if isinstance(sub_list, str) else sub_list
        subjects = [d for d in subjects if any(d == s or d.startswith(f"{s}_") for s in wanted)]
    if not subjects:
        raise FileNotFoundError(f"No subject folders identified. Ensure inputdir has a {os.path.basename(datadir)}/ directory with sub-ID subfolders.")
    
    #outlier flags (structural pipeline only): one row per subject, one column per region
    default_flags = os.path.join(inputdir, "qc", "autoqc_outliers.csv")
    if cifti:
        outliers = None
    else:
        if outliers is None and os.path.exists(default_flags):
            outliers = default_flags
        if outliers is not None:
            outliers = (outliers if isinstance(outliers, pd.DataFrame) else pd.read_csv(outliers, index_col=0)).copy()
            outliers.columns = outliers.columns.astype(str).str.lower()
            outliers.index = outliers.index.astype(str)
    
    #Renderer
    os.makedirs(figdir, exist_ok=True)
    load_region = _load_cifti_region if cifti else _load_region
    written = []
    for number, subid in enumerate(subjects, 1):
        report = os.path.join(figdir, f"{subid}_visual_quality_report.html")
        if os.path.exists(report) and not overwrite:
            if not silent:
                print(f"{subid}: report already in {report}, skipped.")
            continue
        
        if not silent:
            print(f"Creating the visual report for {subid}... [{number}/{len(subjects)}]")
        
        flagged = set()
        if outliers is not None and subid in outliers.index:
            flagged = set(outliers.columns[outliers.loc[subid] >= outlier_threshold])
        
        #regions are loaded first, and rendered once all of them are in memory, so that 
        #left and right counterparts can share an image
        regions = {region: load_region(inputdir, subid, region, metric_proj, template, silent) for region in roilabel}
        regions = {region: data for region, data in regions.items() if data}
        
        #one image per structure and metric (bilateral structures share one), rendered in memory
        images = []
        for metric in metric_proj:
            structures = {}
            for region, data in regions.items():
                if metric in data["panels"]:
                    structures.setdefault(re.sub(r"^(left|right)-", "", region), []).append(region)
            
            for structure, members in structures.items():
                #one row per view: the first row of each hemisphere carries its title
                figure_rows = []
                for region in members:
                    side = region.split("-")[0].capitalize() if region.startswith(("left-", "right-")) else None
                    mirror = -1 if region.startswith("right-") else 1
                    for index, (azimuth, elevation) in enumerate(VIEWS[structure]):
                        figure_rows.append((regions[region], mirror * azimuth, elevation, side if index == 0 else None))
                
                #outlier flag: "left flagged as outlier", "both flagged as outliers"...
                hits = [region for region in members if region in flagged]
                flag = None
                if hits:
                    side = "both" if len(hits) == 2 else hits[0].split("-")[0]
                    flag = "flagged as outlier" if members == ["brain-stem"] else f"{side} flagged as outlier" + ("s" if len(hits) == 2 else "")
                
                images.append((structure, metric, _render_figure(figure_rows, metric, (300, 300), "blue", cmap), flag))
                if not silent:
                    print(f"  => Rendered {structure} ({metric})")
        
        if not images:
            if not silent:
                print(f"  => Nothing plotted for {subid}: check inputdir's content ({os.path.basename(datadir)}/ with its subdirectories), metric_proj and roilabel.")
            continue
        
        #Rows and columns settings
        if not cifti:
            if columns is None:
                columns=3
            if rows is None:
                rows=3
        else:
            if columns is None:
                columns=4
            if rows is None:
                rows=1
        
        _write_report(report, subid, images, columns, rows, cifti)
        written.append(report)
        if not silent:
            print(f"  => Visual report written to {report}")
    
    return written


#####################################################################################
#################Loading a region: the panels of its figure##########################

#Views of each structure, as (azimuth, elevation) in degrees, one image row per view. Written for
#the left hemisphere: the right one is mirrored automatically. 
#azimuth: 0 = camera anterior, 90 = camera on the right, -90 = on the left, 180 = posterior
#elevation: 0 = horizontal, 90 = from above, -90 = from below
VIEWS = {
    "cerebellum-cortex": [(180, 0)],
    "pallidum": [(90, 10)],
    "putamen": [(90, 10)],
    "thalamus": [(-90, 10)],
    "amygdala": [(45, 10)],
    "hippocampus": [(0, 90)],
    "accumbens-area": [(-80, 10)],
    "caudate": [(90, 10)],
    "ventraldc": [(0, -90)],
    "brain-stem": [(-90, 0), (180, 0)],
}

#A region is loaded as {"R": rotation from anatomical (RAS) directions to the meshes' frame, 
#"panels": {metric: [(mesh, style, camera group), ...]}, ...}: one panel per image column. 
#Panels of the same camera group are framed together (they overlap, or should look alike).

def _voxel_surface(flat_mask, dimensions, spacing, origin, flat_values=None, name=None):
    """Blocky surface of the voxels in the mask (arrays flattened with x varying fastest), 
    with their values as cell data if given."""
    spacing = np.array(spacing)
    grid = pv.ImageData(dimensions=np.array(dimensions) + 1, spacing=spacing, origin=np.array(origin) - spacing / 2)
    grid.cell_data["in_mask"] = flat_mask.astype(np.uint8)
    if flat_values is not None:
        grid.cell_data[name] = flat_values
    return grid.threshold(0.5, scalars="in_mask").extract_surface(algorithm=None)

def _load_region(inputdir, subid, region, metrics, template, silent):
    """Structural pipeline: volume, subject surface, medial curve and projected metrics, 
    in the frame of mesh_metrics' meshes. None if the region has no volume or surface."""
    volume_dir = f"{inputdir}/sub_volumes/{subid}"
    volume_file = next((f for f in os.listdir(volume_dir) if f.lower() == f"{region}.nii.gz"), None)
    if volume_file is None:
        return None
    metric_dir = f"{inputdir}/surface_metrics/{subid}"
    native_path = f"{metric_dir}/native_space/{region}_native.vtk"
    if not os.path.exists(native_path):
        if not silent:
            print(f"  => No native-space mesh for {region} ({native_path}), skipped. Run mesh_metrics() with native_meshes=True.")
        return None
    
    #rotation applied by mesh_metrics to the meshes of the template
    rotation = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]) if template == "fslfirst" else np.eye(3)
    
    #voxels of the region, rotated like mesh_metrics rotates meshes
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(f"{volume_dir}/{volume_file}")
    reader.Update()
    image = pv.wrap(reader.GetOutput())
    labels = np.rint(np.asarray(image.active_scalars)).astype(int).ravel()
    volume = _voxel_surface(labels == labels.max(), image.dimensions, image.spacing, image.origin)
    volume.points = np.asarray(volume.points) @ rotation.T
    volume.clear_data()
    
    #subject surface (already rotated by mesh_metrics, carries the metrics) and its medial curve
    native = pv.read(native_path)
    curve = pv.wrap(extract_medial_curve(native, region, n_slices=100))
    plain = native.copy()
    plain.clear_data()  #no scalars, for the plain colour panels
    
    #one set of panels per metric, when available on the template surface too
    panels = {}
    for metric in metrics:
        template_path = f"{metric_dir}/{region}_{metric}.vtk"
        if metric in native.point_data and os.path.exists(template_path):
            panels[metric] = [(volume, "flat", 0), (plain, "solid", 0), (plain, "ghost", 0),
                              (native, "metric", 0), (pv.read(template_path), "metric", 1)]
    
    #RAS directions to the meshes' frame
    qform = reader.GetQFormMatrix() or reader.GetSFormMatrix()
    ras_to_mesh = np.eye(3) if qform is None else np.linalg.inv(np.array([[qform.GetElement(i, j) for j in range(3)] for i in range(3)]))
    return dict(R=rotation @ ras_to_mesh, curve=curve, panels=panels) if panels else None

def _load_cifti_region(inputdir, subid, region, metrics, template, silent):
    """CIFTI data: volume with its metric values (saved by cifti_metrics() with keep_volumes=True) 
    and the metric projected on the template surface. None if the region has none."""
    subject_dir = f"{inputdir}/surface_metrics_cifti/{subid}"
    panels, ras_to_mesh = {}, None
    for metric in metrics:
        mesh_path = f"{subject_dir}/{region}_{metric}.vtk"
        volume_path = f"{subject_dir}/volumes/{region}_{metric}_vol.nii.gz"
        if not os.path.exists(mesh_path):
            continue
        if not os.path.exists(volume_path):
            if not silent:
                print(f"  => No volume for {region} ({volume_path}), skipped. Run cifti_metrics() with keep_volumes=True.")
            continue
        mesh = pv.read(mesh_path)
        if metric not in mesh.point_data:
            continue
        
        #metric values (NaN outside the region) on the volume's grid, flipped along z like the template meshes
        image = nib.load(volume_path)
        values = np.flip(np.asarray(image.dataobj, dtype=np.float32), axis=2)
        volume = _voxel_surface(np.isfinite(values).flatten(order="F"), values.shape, (1, 1, 1), (0, 0, 0),
                                np.nan_to_num(values).flatten(order="F"), metric)
        panels[metric] = [(volume, "metric_flat", 0), (mesh, "metric", 1)]
        
        #RAS directions to voxel-index directions, then flipped along z as well
        if ras_to_mesh is None:
            axes = image.affine[:3, :3] / np.linalg.norm(image.affine[:3, :3], axis=0)
            ras_to_mesh = np.diag([1., 1., -1.]) @ np.linalg.inv(axes)
    return dict(R=ras_to_mesh, panels=panels) if panels else None


#####################################################################################
#################Rendering a figure##################################################

def _fit_view(points, R, azimuth, elevation, aspect):
    """Camera looking at the points from (azimuth, elevation) (anatomical directions, converted 
    to the meshes' frame by R), centred on them, and the scale (half-height, in mm) that just fits them."""
    az, el = np.radians(azimuth), np.radians(elevation)
    direction = R @ np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    up = R @ (np.array([0., 0., 1.]) if abs(elevation) < 89 else np.array([0., 1., 0.]))
    direction = direction / np.linalg.norm(direction)
    up = up - np.dot(up, direction) * direction
    up = up / np.linalg.norm(up)
    right = np.cross(up, direction)
    across, vertical = points @ right, points @ up
    focal = right * (across.max() + across.min()) / 2 + up * (vertical.max() + vertical.min()) / 2 \
            + direction * np.dot(points.mean(0), direction)
    need = max((vertical.max() - vertical.min()) / 2, (across.max() - across.min()) / 2 / aspect)
    return dict(direction=direction, up=up, focal=focal, need=need,
                distance=10 * np.linalg.norm(points.max(0) - points.min(0)))

def _render_figure(figure_rows, metric, size, color, cmap):
    """One image (WebP bytes, nothing is written to disk): one row per view, one column per 
    panel. figure_rows is a list of (region, azimuth, elevation, hemisphere title or None)."""
    nrow = len(figure_rows)
    aspect = size[0] / size[1]
    panels = [region["panels"][metric] for region, _, _, _ in figure_rows]
    ncol = len(panels[0])
    metric_columns = [c for c, (_, style, _) in enumerate(panels[0]) if style.startswith("metric")]
    
    #colour range shared by the metric panels
    values = np.concatenate([row[c][0].get_array(metric).ravel() for row in panels for c in metric_columns])
    clim = (float(np.nanmin(values)), float(np.nanmax(values)))
    if clim[1] <= clim[0]:
        clim = (clim[0], clim[0] + 1e-6)
    
    #one camera per row and camera group, framed on the points of its panels; same scale throughout
    cameras = []
    for (region, azimuth, elevation, _), row in zip(figure_rows, panels):
        group_points = {}
        for mesh, _, group in row:
            group_points.setdefault(group, []).append(np.asarray(mesh.points))
        cameras.append({group: _fit_view(np.vstack(points), region["R"], azimuth, elevation, aspect)
                        for group, points in group_points.items()})
    scale = 1.02 * max(camera["need"] for row in cameras for camera in row.values())  #2% margin
    
    #extra thin row at the bottom for the colour bar, spanning the metric columns
    bar_height = 0.15
    plotter = pv.Plotter(shape=(nrow + 1, ncol), row_weights=[1] * nrow + [bar_height], groups=[(nrow, metric_columns)],
                         off_screen=True, window_size=(ncol * size[0], int(round((nrow + bar_height) * size[1]))), border=False)
    plotter.set_background("white")
    
    metric_style = dict(scalars=metric, cmap=cmap, clim=clim, show_scalar_bar=False)
    styles = {"flat": dict(color=color, smooth_shading=False),
              "solid": dict(color=color, smooth_shading=True),
              "ghost": dict(color=color, smooth_shading=True, opacity=0.3),
              "metric": dict(metric_style, smooth_shading=True),
              "metric_flat": dict(metric_style, smooth_shading=False)}
    bar_mapper = None
    for r, ((region, _, _, side), row) in enumerate(zip(figure_rows, panels)):
        for c, (mesh, style, group) in enumerate(row):
            plotter.subplot(r, c)
            actor = plotter.add_mesh(mesh, show_edges=False, **styles[style])
            if style == "ghost":
                plotter.add_mesh(region["curve"], color="black", line_width=3, render_lines_as_tubes=True)  #medial curve
            if bar_mapper is None and style.startswith("metric"):
                bar_mapper = actor.mapper
            
            camera, fit = plotter.camera, cameras[r][group]
            camera.focal_point = fit["focal"]
            camera.position = fit["focal"] + fit["direction"] * fit["distance"]
            camera.up = fit["up"]
            camera.parallel_projection = True  #guarantees identical framing across the panels of a camera group
            camera.parallel_scale = scale
            plotter.reset_camera_clipping_range()
            if c == 0 and side:
                plotter.add_text(side, position="upper_left", font_size=12, color="black")  #Left / Right title
    
    plotter.subplot(nrow, metric_columns[0])
    plotter.add_scalar_bar(title="", mapper=bar_mapper, vertical=False, color="black", n_labels=3, fmt="%.3g",
                           position_x=0.1, position_y=0.4, width=0.8, height=0.55, label_font_size=9)
    plotter.add_text(metric, position="lower_edge", font_size=9, color="black")  #name of the metric
    
    #rendered and encoded in memory as WebP (light images)
    screenshot = plotter.screenshot(return_img=True)
    plotter.close()
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(screenshot[..., :3])).save(buffer, format="WEBP", quality=90, method=6)
    return buffer.getvalue()


#####################################################################################
##################HTML page: the subject's images in one report######################

_CSS = """
:root { color-scheme: dark; --vh: 100vh; --pad: 4px; --gap: 4px; --top: 26px; --title: 18px;
        /* height available to one row of cards, so that --rows rows fit the window exactly */
        --cellh: calc((var(--vh) - 2 * var(--pad) - var(--top) - (var(--rows) - 1) * var(--gap)) / var(--rows)); }
@supports (height: 100dvh) { :root { --vh: 100dvh; } }
* { box-sizing: border-box; }
body { margin: 0; padding: var(--pad); background: #2b2b2b; color: #fff;
       font-family: "Segoe UI", Helvetica, Arial, sans-serif; }
.top { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; margin: 0 0 4px 2px; }
h1 { font-weight: 600; font-size: 16px; line-height: 1.2; margin: 0; color: #fff; white-space: nowrap; }
h1 span { font-weight: 400; font-size: 12px; color: #aaa; margin-left: 8px; }
.legend { font-size: 11px; line-height: 19.2px; color: #bbb; text-align: right; white-space: nowrap;
          overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.legend b { font-weight: 600; color: #fff; }
.grid { display: grid; gap: var(--gap); align-items: start; }
.card { margin: 0; }
.card h3 { font-weight: 500; font-size: 12px; line-height: 1.2; margin: 0 0 2px 2px; color: #fff; }
.card h3 span { font-weight: 400; font-size: 11px; color: #aaa; margin-left: 6px; }
.card h3 span.flag { font-weight: 600; color: #ff9f43; margin-left: 8px; }
/* scaled to the column width, and capped in height so that --rows rows fit the window: 
   the whole image is always shown (object-fit: contain), never stretched or cropped */
.card img { display: block; width: 100%; height: auto; max-height: calc(var(--cellh) - var(--title));
            object-fit: contain; object-position: left top; cursor: zoom-in; }
#lb { display: none; position: fixed; inset: 0; background: rgba(20,20,20,.96); overflow: auto;
      cursor: zoom-out; z-index: 10; }
#lb img { display: block; margin: 0 auto; max-width: none; max-height: none; object-fit: none; }
"""

_JS = """
const lb = document.getElementById('lb'), lbimg = document.getElementById('lbimg');
document.querySelectorAll('.card img').forEach(img => img.addEventListener('click', () => {
  lbimg.src = img.src; lb.style.display = 'block'; }));
lb.addEventListener('click', () => { lb.style.display = 'none'; });
document.addEventListener('keydown', e => { if (e.key === 'Escape') lb.style.display = 'none'; });
"""

def _write_report(outputfile, subid, images, columns, rows, cifti):
    """Writes the subject's report. images is a list of (structure, metric, WebP bytes, outlier 
    flag or None). The metric is written next to the titles if several are present."""
    legend = ("Columns legend: <b>1:</b> Voxel-based metric volume, <b>2:</b> Metric projected on the template surface" if cifti else
              "Columns legend: <b>1:</b> Voxel-based volume, <b>2:</b> Converted surface, <b>3:</b> Medial curve, "
              "<b>4:</b> Metric (subject space), <b>5:</b> Metric (template space)")
    show_metric = len({metric for _, metric, _, _ in images}) > 1
    
    cards = ""
    for structure, metric, data, flag in sorted(images, key=lambda image: image[:2]):
        name = html.escape(structure.replace("-", " ").capitalize().replace("Ventraldc", "Ventral DC"))
        flag_tag = f'<span class="flag">&#9888; {html.escape(flag)}</span>' if flag else ""
        metric_tag = f"<span>{html.escape(metric)}</span>" if show_metric else ""
        cards += (f'<figure class="card"><h3>{name}{flag_tag}{metric_tag}</h3>'
                  f'<img src="data:image/webp;base64,{base64.b64encode(data).decode("ascii")}" alt="{name}"></figure>\n')
    
    title = "Visual quality report"
    Path(outputfile).write_text(
        f'<!DOCTYPE html>\n<html lang="en" style="--rows: {int(rows)}">\n<head>\n<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{title} - {html.escape(subid)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n'
        f'<div class="top"><h1>{title}<span>{html.escape(subid)}</span></h1><div class="legend">{legend}</div></div>\n'
        f'<div class="grid" style="grid-template-columns: repeat({int(columns)}, minmax(0, 1fr));">\n{cards}</div>\n'
        f'<div id="lb"><img id="lbimg" alt=""></div>\n<script>{_JS}</script>\n</body>\n</html>\n', encoding="utf-8")