#####################################################################################
##########Statistical analyses using random field theory cluster correction##########

import vtk
from vtkmodules.util import numpy_support
import re
import numpy as np
import pyvista as pv
import warnings
from typing import Optional, Union, Sequence, Tuple
from pathlib import Path
from itertools import chain
from itertools import groupby
from brainstat.stats.SLM import SLM
from brainstat._typing import ArrayLike
from brainstat.mesh.data import mesh_smooth
from brainspace.vtk_interface.wrappers import BSPolyData

def slm_analysis(
    inputdir: Union[str, Path],
    metric: Union[str, Sequence[str]],
    roilabel: Union[str, Sequence[str]],
    model: ArrayLike, 
    contrast: ArrayLike, 
    mask: ArrayLike = None,
    correction: Optional[Union[str, Sequence[str]]] = None,
    thetalim: float = 0.01,
    drlim: float = 0.1,
    two_tailed: bool = True,
    cluster_threshold: float = 0.001,
    smooth: Optional[float] = None,
    sub_list: Optional[Sequence[str]] = None
    ):
    
    """Vertex-wise analysis with random field theory cluster correction
    
    This function is a wrapper for BrainStat's SLM (Standard Linear regression models)
    modelling functions, which fit linear or linear mixed models with surface-based 
    values, with random field theory cluster correction. You may refer to their 
    tutorial to understand the model and contrast arguments:
    https://brainstat.readthedocs.io/en/master/python/tutorials/tutorial_1.html
    
    slm_analysis() automatically collates surface-based measures in template space from a 
    surface_metrics/ folder produced by mesh_metrics() (or equivalent folder), in
    alphanumeric order of the subjects' subfolders, and fits them as the "surf" 
    argument in BrainStat. The model returns t-statistics, p-values,and Random field theory 
    clusters map for a selected contrast. 
    
    Note for SLM: The first surface is used as "template" given it works for template-space
    surfaces.
    
    Parameters
    ----------
    inputdir : str, Path
        The surface_metrics/ directory where the surface-based metrics were outputted 
        (using mesh_metrics()) or a directory with the same tree structure (subject folders,
        each with metrics .vtk files inside).
    roilabel: str, Sequence
        The name or array of names for the region(s)-of-interest to analyse: 'left-cerebellum-cortex', 'right-cerebellum-cortex', 'left-pallidum', 'right-pallidum', 'left-putamen', 
        'right-putamen', 'left-thalamus', 'right-thalamus','left-amygdala',  'right-amygdala', 'left-hippocampus', 'right-hippocampus', 'left-accumbens-area','right-accumbens-area','left-caudate', 'right-caudate', 'left-ventraldc', 'right-ventraldc', and 'brain-stem'.
    model : brainstat.stats.terms.FixedEffect, brainstat.stats.terms.MixedEffect
        The linear model to be fitted of dimensions (observations, predictors).
    contrast : array-like
        Vector of contrasts from the observations.
    mask: array-like, optional
        A mask containing True for vertices to include in the analysis, by
        default None.
    correction: str, Sequence, optional 
        String or sequence of strings. If it contains “rft” a random field theory multiple comparisons correction will be run. If it contains “fdr” a false discovery rate multiple comparisons correction will be run. Both may be provided. By default None.
    thetalim : float, optional
        Lower limit on variance coefficients in standard deviations, by
        default 0.01.
    drlim : float, optional
        Step of ratio of variance coefficients in standard deviations, by
        default 0.1.
    two_tailed : bool, optional
        Determines whether to return two-tailed or one-tailed p-values. Note
        that multivariate analyses can only be two-tailed, by default True.
    cluster_threshold : float, optional
        P-value threshold or statistic threshold for defining clusters in
        random field theory, by default 0.001.
    smooth : float, optional
        The full-maximum half-width (FMHW) value that will be applied for smoothing 
        of the surface-based measure along the surface. Default is None.
    sub_list: str, Sequence
        An optional list of subject IDs to determine whose surface is to be included in the
        analysis. Surfaces whose path does not contain a subject ID from the list will be 
        ignored. If not specified, all surfaces found will be included.
    
    Returns
    -------
    slm_model : 
        BrainStat SLM object containing the statistical outputs.
    """
    
    #collate surface matrices from the outputdir at that metric and specific metric
    #sorted alphanumerically via sub-ID
    labels = [roilabel] if isinstance(roilabel, str) else roilabel
    mesh_list = sorted(
    chain.from_iterable(
        Path(inputdir).rglob(f'*{label}_{metric}.vtk')
        for label in labels
    ),
    key=lambda p: (m := re.search(r'sub-\w+', str(p))) and m.group() or ''
    )
     
    #exclude non applicable subjects
    if sub_list is not None:
        mesh_list = [m for m in mesh_list if any(sub in str(m) for sub in np.asarray(sub_list))]
    
    if len(mesh_list) <= 0:
        raise FileNotFoundError(f"No surface file for {roilabel} {metric} found.")
    
    #group meshes from mesh_list by subject to merge multiple rois
    def getsubid(p):
        m = re.search(r'sub-\w+', str(p))
        return m.group() if m else ''
    
    surf_data = []
    appender = vtk.vtkAppendPolyData()
    for subject, subject_meshes in groupby(sorted(mesh_list, key=getsubid), key=getsubid):
        subject_scalars = []
        for mesh in subject_meshes:
            #read mesh and extract metrics scalar
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(str(mesh))
            reader.Update()
            surf = reader.GetOutput()
            _ = surf.GetPointData().SetActiveScalars(metric)
            scalar = numpy_support.vtk_to_numpy(surf.GetPointData().GetArray(metric))
            subject_scalars.append(scalar)
            
            #template surface can simply be 1st subject's mesh(es) (as all surfaces are template-based)
            if subject == getsubid(mesh_list[0]):
                appender.AddInputData(surf)
                
        
        surf_data.append(np.concatenate(subject_scalars))
    
    appender.Update()
    surf_template = BSPolyData(appender.GetOutput()) #template mesh
    surf_data = np.vstack(surf_data) #all subjects scalars
    
    #apply smoothing if applicable
    if smooth is not None and smooth > 0:
        surf_data=mesh_smooth(surf_data.copy(), surf_template, smooth)
    
    #mask out zero values (vertices with no data across subjects)
    if mask is None:
        mask = np.ones(surf_data.shape[1], dtype=bool)
        mask[np.all(surf_data == 0, axis=0)] = False
    
    #apply model on surf data
    slm_model = SLM(
        model=model,
        contrast=contrast,
        surf=surf_template,
        mask=mask,
        correction=correction,
        thetalim=thetalim,
        drlim=drlim,
        two_tailed=two_tailed,
        cluster_threshold=cluster_threshold,
        data_dir=None
    )
    slm_model.fit(surf_data)
    
    return slm_model
    
#####################################################################################
##########Statistical analyses using random field theory cluster correction##########

def slm_plot(
    slm, 
    stat: str, 
    threshold: Optional[float] = None, 
    cmap: str = None, 
    clim: Optional[Tuple[float, float]] = None,
    smooth_mesh: Optional[int] = 0, 
    mode: str = 'anatomical',
    flatmode_filepath: Union[str, Path] = 'flat_plot.png',
    ):
    """
    Plot SLM result maps on their respective surface.
    
    Authors: Charly H.A. Billaud, Nicolas P.M. Lavarde
    
    Parameters
    ----------
    slm: SLM object
        BrainStat SLM object containing the statistical outputs.
    stat: str
        Map to plot from the SLM object:
        
        - 't' for the t-statistics map
        - 't_fdr' for the t-statistics map filtered based on significance after false discovery rate (FDR) (use threshold argument)
        - 't_rft' for the t-statistics map filtered based on significance of random field theory (RFT) cluster-wise p-values (use threshold argument)
        - 'p_fdr' for the FDR-adjusted p-value map
        - 'p_rft' for the p-value map filtered based on significance of random field theory (RFT) cluster-wise p-values (use threshold argument)
        - 'clusters' for the identified clusters
    threshold: float, optional
        Thresholding value in which vertices are to be plotted (plots values below the set threshold for t_fdr, t_rft, p_fdr_, p_rft, and clusters; above the threshold for t).
    cmap: str
        Name of the color map to be assigned to the background volume, as listed in matplotlib's colormaps. Default is "RdBu_r" if the plot shows both negative and positive effects, 'Blues_r' 
        or negative only, 'Reds' for positive only.
    clim: Tuple, optional
        Sequence of float stating the minimum and maximum value of the color bar. Default is minimum and maximum value. It is not applied for the clusters maps as colour values follow cluster IDs.
    smooth_mesh: int, optional
        Number of iterations of cosmetic smoothing to make the surface appear smoother. Default is 0.
    mode: str, optional
        Rendering mode for all regions merged (see merge_tools() for the example output): 'flat' (2D PNG grid via vis_merged_flat), or 'slider' (interactive spread viewer via vis_merged). Argument ignored if individual ROIs are provided.
    flatmode_filepath: str, Path, optional
        Output path for the PNG when all ROIs merged are provided and mode='flat'. Default is 'flat_plot.png'.
    """
    
    #load mesh
    mesh = pv.wrap(slm.surf.VTKObject)  
    
    #appearance smoother
    if smooth_mesh is not None and smooth_mesh > 0:
        mesh=slm.surf.VTKObject 
        s = vtk.vtkWindowedSincPolyDataFilter()
        s.SetInputData(mesh)
        s.SetNumberOfIterations(smooth_mesh)
        s.SetPassBand(0.001)
        s.NonManifoldSmoothingOn()
        s.NormalizeCoordinatesOn()
        s.Update()
        mesh=pv.wrap(s.GetOutput())
      
    #add scalar values from SLM's maps:
    #t statistics
    if stat == 't':
        scalars = slm.t.squeeze()
        #apply threshold
        if threshold is not None:
            mask = scalars < threshold #t: mask under threshold
            scalars = scalars.copy().astype(float)
            scalars[mask] = np.nan
        
        if cmap is None:
            if np.nanmin(scalars) < 0 and np.nanmax(scalars) > 0:
                cmap = 'RdBu_r'
            elif np.nanmin(scalars) > 0:
                cmap = 'Reds'
            elif np.nanmax(scalars) < 0:
                cmap = 'Blues_r'
        else:
            cmap = cmap
        
        if clim is None:
            if np.nanmin(scalars) > 0:
                clim = [0, np.nanmax(scalars)]
            elif np.nanmax(scalars) < 0:
                clim = [np.nanmin(scalars), 0]
            else:
                clim = [np.nanmin(scalars), np.nanmax(scalars)]
        title = 't-statistics'
    
    #t statistics (filtered through FDR significance)
    elif stat == 't_fdr':
        scalars = slm.t.squeeze()
        if slm.Q is None:
            raise ValueError('The "fdr" correction was not applied to this SLM model.')
        else:
            #apply threshold according to FDR value
            if threshold is None or threshold == 0:
                warnings.warn('No threshold was indicated. t-values will be plotted regardless of FDR significance')
            else:
                scalars = slm.t.squeeze()
                mask = np.abs(slm.Q) > threshold #p: mask above threshold
                scalars = scalars.copy().astype(float)
                scalars[mask] = np.nan
            
            if cmap is None:
                if np.nanmin(scalars) < 0 and np.nanmax(scalars) > 0:
                    cmap = 'RdBu_r'
                elif np.nanmin(scalars) > 0:
                    cmap = 'Reds'
                elif np.nanmax(scalars) < 0:
                    cmap = 'Blues_r'
            else:
                cmap = cmap
            
            if clim is None:
                if np.nanmin(scalars) > 0:
                    clim = [0, np.nanmax(scalars)]
                elif np.nanmax(scalars) < 0:
                    clim = [np.nanmin(scalars), 0]
                else:
                    clim = [np.nanmin(scalars), np.nanmax(scalars)]
            title = 't-statistics (FDR)'
    
    #t statistics (filtered through RFT significance)
    elif stat == 't_rft':
        if threshold is None:
            threshold=1.01 #all clusters
        
        if slm.P['clus'][0].empty and slm.P['clus'][1].empty:
            raise ValueError('No RFT clusters were generated in this SLM model.')
        else:
            #get vertex values belonging to identified clusters
            scalars = np.full(slm.t.squeeze().shape, np.nan) #placeholder
            #loop across positive and negative cluvets
            for sign_idx, clus_df in enumerate([slm.P['clus'][0], slm.P['clus'][1]]):
                if clus_df.empty:
                    continue
                clusid_map = slm.P['clusid'][sign_idx].squeeze()
                t_map = slm.t.squeeze()
                #remove clusters below threshold
                for _, row in clus_df.iterrows():
                    if row['P'] < threshold:
                        scalars[clusid_map == row['clusid']] = t_map[clusid_map == row['clusid']]
            
            if cmap is None:
                #based on presence of negative or positive clusters (if not empty)
                has_pos = not slm.P['clus'][0].empty and (slm.P['clus'][0]['P'] < threshold).any()
                has_neg = not slm.P['clus'][1].empty and (slm.P['clus'][1]['P'] < threshold).any()
                if has_pos and has_neg:
                    cmap = 'RdBu_r'
                elif has_pos:
                    cmap = 'Reds'
                elif has_neg:
                    cmap = 'Blues_r'
                else:
                    cmap = 'RdBu_r' 
            else:
                cmap = cmap
            
            if clim is None:
                if np.nanmin(scalars) > 0:
                    clim = [0, np.nanmax(scalars)]
                elif np.nanmax(scalars) < 0:
                    clim = [np.nanmin(scalars), 0]
                else:
                    clim = [np.nanmin(scalars), np.nanmax(scalars)]
            title = 't-statistics (RFT)'
            
    #p-vals (FDR)
    elif stat == 'p_fdr':
        if slm.Q is None:
            raise ValueError('The "fdr" correction was not applied to this SLM model.')
        else:
            scalars = slm.Q 
            #apply threshold
            if threshold is not None:
                mask = scalars > threshold
                scalars = scalars.copy().astype(float)
                scalars[mask] = np.nan
            
            if cmap is None:
                cmap='RdBu_r'
            else:
                cmap = cmap
            
            if clim is None:
                if threshold is not None:
                    clim = [0, threshold]
                else:
                    clim = [0, np.nanmax(scalars)]
            title = 'p-values (FDR)'
    
    #Cluster corrected p-values
    elif stat == 'p_rft':
        if threshold is None:
            threshold=1.01 #all clusters
        
        if slm.P['clus'][0].empty and slm.P['clus'][1].empty:
            raise ValueError('No RFT clusters were generated in this SLM model.')
        else:
            #get vertex values belonging to identified clusters
            scalars = np.full(slm.P['pval']['C'].shape, np.nan) #placeholder
            #loop across positive and negative cluvets
            for sign_idx, clus_df in enumerate([slm.P['clus'][0], slm.P['clus'][1]]):
                if clus_df.empty:
                    continue
                clusid_map = slm.P['clusid'][sign_idx].squeeze()
                p_map=slm.P['pval']['C']
                #remove clusters below threshold
                for _, row in clus_df.iterrows():
                    if row['P'] < threshold:
                        scalars[clusid_map == row['clusid']] = p_map[clusid_map == row['clusid']]
            
            if cmap is None:
                #based on presence of negative or positive clusters (if not empty)
                has_pos = not slm.P['clus'][0].empty and (slm.P['clus'][0]['P'] < threshold).any()
                has_neg = not slm.P['clus'][1].empty and (slm.P['clus'][1]['P'] < threshold).any()
                if has_pos and has_neg:
                    cmap = 'RdBu_r'
                elif has_pos:
                    cmap = 'Reds'
                elif has_neg:
                    cmap = 'Blues_r'
                else:
                    cmap = 'RdBu_r' 
            else:
                cmap = cmap
            
            if clim is None:
                clim = [0, np.nanmax(scalars)]
            title = 'p-values (RFT)'
    
    elif stat == 'clusters':
        #get significant cluster IDs for both pos and neg contrasts
        if threshold is None:
            threshold=1.01 #all clusters
        
        sig_pos_ids = slm.P['clus'][0].loc[slm.P['clus'][0]['P'] < threshold, 'clusid'].values if not slm.P['clus'][0].empty else []
        sig_neg_ids = slm.P['clus'][1].loc[slm.P['clus'][1]['P'] < threshold, 'clusid'].values if not slm.P['clus'][1].empty else []
        # default to zeros if no negative or positive cluster at all
        pos_map = slm.P['clusid'][0].squeeze().copy() if slm.P['clusid'][0] is not None else np.zeros(slm.t.squeeze().shape)
        neg_map = slm.P['clusid'][1].squeeze().copy() if slm.P['clusid'][1] is not None else np.zeros(slm.t.squeeze().shape)
        
        #zero out non-significant clusters
        pos_map[~np.isin(pos_map, sig_pos_ids)] = 0
        neg_map[~np.isin(neg_map, sig_neg_ids)] = 0
        
        #combine IDs
        scalars = pos_map.copy()
        scalars[neg_map != 0] = -neg_map[neg_map != 0]
        scalars[scalars == 0] = np.nan
        
        if cmap is None:
            cmap='tab20'
        clim = None
        title = 'Significant clusters'
    else:
        raise ValueError(f"'{stat}' is not applicable. Choose from: 't', 't_fdr', 't_rft', 'p_fdr', 'p_rft', 'clusters'")
    
    #PyVista plotter
    mesh[title] = scalars
    
    # the default slm_plot output
    if mesh.GetNumberOfPoints() not in [95718,82412]:
        mesh.point_data[title] = scalars
        mesh.set_active_scalars(title)
        plotter = pv.Plotter()
        plotter.add_mesh(
            mesh,
            scalars=title,
            cmap=cmap,
            clim=clim,
            nan_color='lightgrey',
            show_scalar_bar=True,
            smooth_shading=True,
        )
        plotter.add_text(title, font_size=12)
        plotter.show()
    elif mesh.GetNumberOfPoints() in [95718,82412]: 
        #the 3D interactive plotter generated by vis_merged
        if mode == 'anatomical':
            from subcortexmesh.merge_tools import vis_merged
            mesh.point_data[title] = scalars
            mesh.set_active_scalars(title)
            vis_merged(
                merged_vtk=mesh,
                cmap=cmap,
                clim=clim,
                smooth_mesh=smooth_mesh)
        #the flat plot output generated by vis_merged_flat 
        elif mode == 'flat':
            from subcortexmesh.merge_tools import vis_merged_flat
            vis_merged_flat(
                merged_vtk=mesh,
                output_path=flatmode_filepath,
                scalars=title,
                cmap=cmap,
                clim=clim,
                smooth_mesh=smooth_mesh,
            )