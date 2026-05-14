#####################################################################################
###########Plotting native surfaces on top of their volume ##########################

import os
import vtk
import pyvista as pv
import numpy as np
import matplotlib
import matplotlib.pyplot as mplpyplot
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Button, Slider
matplotlib.use('TkAgg') #matplotlib rendering backend for windows
from typing import Optional, Union
from pathlib import Path

def surf_qcplot(
    volpath: Union[str, Path],
    surfdir: Union[str, Path],
    vol_color_map: Optional[str] = "gray",
    outline_color_map: Optional[str] = "tab20",
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
    #default mesh can only be 5 if at least 5 surfs
    if len(surf_files) < 5:
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
