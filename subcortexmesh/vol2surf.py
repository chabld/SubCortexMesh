#####################################################################################
##################Converting subcortical volumes to surfaces#########################

import vtk
import numpy as np
import nibabel as nib
import re
import os
import tempfile
import time
import pyvista as pv
from typing import Union, Sequence
from pathlib import Path

def vol2surf(
    inputdir: Union[str, Path],
    outputdir: Union[str, Path],
    dilate_erode: bool = True,
    roilabel: Union[str, Sequence[str]] = ['left-cerebellum-cortex', 'right-cerebellum-cortex', 
                                    'left-pallidum', 'right-pallidum', 'left-putamen', 'right-putamen', 'left-thalamus', 'right-thalamus','left-amygdala', 'right-amygdala', 'left-hippocampus','right-hippocampus', 'left-accumbens-area','right-accumbens-area','left-caudate', 'right-caudate', 'left-ventraldc', 'right-ventraldc', 'brain-stem'],
    smoothing: int = 15,
    plot_volnext2surf: bool = False,
    overwrite: bool = True,
    silent: bool = False
    ):
    """Converting subcortical volumes to surface meshes
    
    This function function converts the separate volumes outputted by subseg_getvol, 
    using the discrete marching cube method. This includes pre-conversion dilation/erosion 
    of the volume and smoothing of the surface (both can be disabled) to ensure meshes are 
    not too coarse and noisy. Note that the dilation/smootinh process was applied to the 
    meshes used as the standard space in mesh_metrics().
    
    An optional argument can be provided to visualise the volume and its rendered mesh
    (plot_volnext2surf), with an interactive slider, in order to check the quality of the 
    rendering of every mesh.
    
    Parallel processes: to avoid conflicts, subjects will be skipped if a "isrunning" tmp 
    file exists (in tempfile.gettempdir()) to mark them as currently processing. The tmp file 
    is removed at the end or replaced if 1 hour old. If a process has been interrupted, remove 
    the tmp manually to rerun a subject before the 1 hour (its path is printed when flagged). 
    
    Parameters
    ----------
    inputdir : str, Path
        The sub_volumes/ directory where the subcortical volumes were outputted 
        (using subseg_getvol()).
    outputdir : str, Path
        The path where subcortical meshes will be saved (creates the "sub_surfaces" 
        directory inside the path).
    roilabel: str, Sequence
        The name(s) of the region(s)-of-interest to be computed as strings. Default is all
        subcortices across all segmentation templates: 'left-cerebellum-cortex', 'right-cerebellum-cortex', 'left-pallidum', 'right-pallidum', 'left-putamen', 
        'right-putamen', 'left-thalamus', 'right-thalamus','left-amygdala',  'right-amygdala', 'left-hippocampus', 'right-hippocampus', 'left-accumbens-area','right-accumbens-area','left-caudate', 'right-caudate', 'left-ventraldc', 'right-ventraldc', and 'brain-stem'.
    dilate_erode : bool
        Uses vtkImageDilateErode3D to slightly erode and dilate the volume before conversion, 
        which helps unify sparse voxels and produce smooth vertices. Default is True.
    smoothing: int
        The number of iterations of the smoothing procedure (0 will apply no smoothing). The 
        settings are otherwise those of VTK's SmoothDiscreteMarchingCubes.  Default is 15.
    plot_volnext2surf: bool
        Whether to plot the volume and the surface together and check for their correspondance. 
        Default is False.
    overwrite : bool
        Whether files are to be overwritten or skipped if already in outputdir. Default is 
        True.
    silent : bool
        Whether messages about the process are to be printed. Default is False.
    """
    
    #list subjects 
    sub_list =[    d for d in os.listdir(inputdir)
        if os.path.isdir(os.path.join(inputdir, d))]
    
    #creates folder where surfaces will go 
    subsurf_path = os.path.join(outputdir, "sub_surfaces")
    if not silent and not os.path.exists(subsurf_path):
        print(f"Writing the {outputdir}/sub_surfaces directory...")
    os.makedirs(subsurf_path, exist_ok=True)
    
    subindex=0
    for subid in sub_list:
        subindex=subindex+1
        
        #unique tmp file to avoid parallel loop conflicts
        fname = os.path.join(tempfile.gettempdir(), f"{subid}_isrunning_surf.tmp")
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
            print(f"Creating surface meshes for {subid}... [{subindex}/{len(sub_list)}]") 
        
        #make subject folder
        os.makedirs(os.path.join(f"{outputdir}/sub_surfaces/{subid}"), exist_ok=True)
        subdir=f"{outputdir}/sub_surfaces/{subid}"
        
        vol_list=[f for f in os.listdir(f"{inputdir}/{subid}")
                if f.endswith(".nii.gz")]
            
        if len(vol_list) > 0:
            for volfile in vol_list:
                #path to volume
                nifti_file=(f"{inputdir}/{subid}/{volfile}") 
                #get base name to name surface the same way
                base = re.sub(r'\.nii(\.gz)?$', '', volfile)
                
                if base.lower() not in roilabel:
                    continue
                
                if not os.path.exists(f"{subdir}/{base.lower()}.vtk") or overwrite:
                    #Load volume with VTK
                    reader = vtk.vtkNIFTIImageReader()
                    reader.SetFileName(nifti_file)
                    reader.Update()
                    
                    #read volume label (binary masks so only ones)
                    roi_label = int(np.max(nib.load(nifti_file).get_fdata()))
                    
                    #dilate and erode volume to eliminate artefacts
                    if dilate_erode:
                        dilate = vtk.vtkImageDilateErode3D()
                        dilate.SetInputConnection(reader.GetOutputPort())
                        dilate.SetDilateValue(roi_label)   # grow ROI voxels
                        dilate.SetErodeValue(0)            # background
                        dilate.SetKernelSize(5,5,5)
                        dilate.Update()
                        #erosion
                        erode = vtk.vtkImageDilateErode3D()
                        erode.SetInputConnection(dilate.GetOutputPort())
                        erode.SetDilateValue(0) 
                        erode.SetErodeValue(roi_label) 
                        erode.SetKernelSize(4,4,4)
                        erode.Update()
                        vol=erode
                    else:
                        vol=reader
                    
                    #Turn volume to surface mesh, keeping voxels' label
                    dmc = vtk.vtkDiscreteMarchingCubes() 
                    dmc.SetInputConnection(vol.GetOutputPort())
                    dmc.SetValue(0, roi_label)
                    dmc.Update() 
                    
                    #Smooth the surface as pixellated 3D volume will make surface coarse 
                    #https://examples.vtk.org/site/Cxx/Modelling/SmoothDiscreteMarchingCubes/ default settings
                    if smoothing > 0:
                        passBand = 0.001
                        featureAngle = 120.0
                        smooth = vtk.vtkWindowedSincPolyDataFilter() #smoothing function
                        smooth.SetInputConnection(dmc.GetOutputPort()) 
                        smooth.SetNumberOfIterations(smoothing) 
                        smooth.BoundarySmoothingOff()
                        smooth.FeatureEdgeSmoothingOff()
                        smooth.SetFeatureAngle(featureAngle)
                        smooth.SetPassBand(passBand)
                        smooth.NonManifoldSmoothingOn()
                        smooth.NormalizeCoordinatesOn()
                        smooth.Update()
                        mesh=smooth.GetOutput()
                    else:
                        mesh=dmc.GetOutput()
                    
                    #remove any remaining floating bits separate from main mesh
                    #these are artefact that can appear upon vol to mesh conversion, erosion,dilation.
                    connectivity = vtk.vtkConnectivityFilter()
                    connectivity.SetInputData(mesh)
                    connectivity.SetExtractionModeToLargestRegion()
                    connectivity.Update()
                    #drop vertex point of bits dropped
                    cleaner = vtk.vtkCleanPolyData() 
                    cleaner.SetInputConnection(connectivity.GetOutputPort()) 
                    cleaner.Update()
                    mesh = cleaner.GetOutput()
                    
                    ###################################################################
                    ############################plotter################################
                    
                    #to check surf/vol alignment visually, comment out add_volume to just check mesh's shape
                    if plot_volnext2surf:
                        plotter = pv.Plotter()
                        #get size to adapt camera zoom
                        bounds = mesh.bounds
                        size = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
                        center = mesh.center
                        plotter.camera_position = [
                            (center[0], center[1], center[2] + 3*size),  # camera location
                            center,  (0, -1, 0)] #Y flipped as VTK's coord syst not following RAS
                        #render volume and surface 
                        actor_mesh = plotter.add_mesh(pv.wrap(mesh), color="red")
                        _ = plotter.add_volume(pv.wrap(vol.GetOutput()), cmap="Blues")
                        #slider to allow users to separate them
                        def update_y(value): 
                            if abs(value) < 1: 
                                actor_mesh.SetPosition(0, 0, 0) 
                            else: 
                                actor_mesh.SetPosition(0, -value, 0)
                        
                        plotter.add_slider_widget(update_y, rng=[-50, 50], title="Y-axis")
                        plotter.add_axes(interactive=True, ylabel='Y') #axis pointer helper
                        plotter.show(title=f"{subid} - {base}")
                    
                    ###################################################################
                    ###################################################################
                    
                    #save smoothed mesh
                    #guarantee overwriting
                    out_path = os.path.join(subdir, f"{base.lower()}.vtk")
                    if os.path.exists(out_path):
                        os.remove(out_path)
                    
                    writer = vtk.vtkPolyDataWriter() 
                    writer.SetFileName(out_path)
                    writer.SetInputData(mesh)
                    _ = writer.Write()
                    if not silent: 
                        print(f"  => Saved {base} to {out_path}")
                else:
                    out_path = os.path.join(subdir, f"{base.lower()}.vtk")
                    if not silent: 
                        print(f"  => {base} mesh already in: {out_path}")
        else:
            if not silent: 
                print(f"  => No volume file (.nii) found at all for {subid}.")
        
        os.remove(fname)  #cleanup tmp file
                