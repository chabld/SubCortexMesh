#####################################################################################
##############getting subcortical volumes from subcortical segmentations#############

#Currently compatible with FreeSurfer's ASeg and FSL's FIRST segmentation outputs

import os
import ants
import re
import glob
import tempfile
import time
from typing import Optional, Union
from pathlib import Path
from subcortexmesh import template_data_fetch
import nibabel as nib
import numpy as np
                            
def subseg_getvol(
    inputdir: Union[str, Path],
    outputdir: Union[str, Path],
    template: str,
    toolboxdata: Optional[Union[str, Path]] = None,
    overwrite: bool = True,
    silent: bool = False,
    ):
    """Extracting and coregistering subcortical volumes from a cohort's subcortical volumes
    
    This function reads through subjects of a FreeSurfer or FSL FIRST output preprocessing 
    directory and identifies subcortical segmentation volumes. Typically, such volumes are 
    "aseg.mgz" for FreeSurfer, and "*all_fast_firstseg" for FSL FIRST (plus, if applicable, 
    "*Cereb_first.nii.gz"). For each subject, subseg_getvol() uses antspy to coregister the 
    segmentation volumes to their respective template space  (fsaverage/MNI305 for FreeSurfer, 
    MNI152 for FSL), and then extracts each subcortical region-of-interest separately. 
    
    T1 volumes are required for the coregistration, stored as [sub-ID]/mri/T1.mgz for 
    FreeSurfer and expected to be stored in the same "inputdir" path for FSL FIRST: 
    the function will search for files containing the sub-ID and the "*T1w.nii*" pattern for 
    each subject.
    
    Parallel processes: to avoid conflicts, subjects will be skipped if a "isrunning" tmp 
    file exists to mark them as currently processing. The tmp file is removed at the end or 
    replaced if 1 hour old. If a process has been interrupted, remove the tmp manually to 
    rerun a subject before the 1 hour (its path is printed when flagged). 
    
    Note for (optional) FSL cerebellar segmentation: the cerebella need to be created inside 
    the same output directory as run_first_all's, naming them "*R_Cereb_first" and 
    "*L_Cereb_first". It can be done as follows (do the same for `L_Cereb`):
    
    .. code-block:: bash
        
        first_flirt [subject T1file] "[subject_directory]/[sub-id]" -cort
        run_first -i [subject T1file] \\
            -t "[subject_directory]/[sub-id_cort.mat]" \\
            -n 40 \\
            -o "[subject_directory]/[sub-id]-R_Cereb_first" \\
            -m "${FSLDIR}/data/first/models_336_bin/intref_puta/R_Cereb.bmv" \\
            -intref "${FSLDIR}/data/first/models_336_bin/05mm/R_Puta_05mm.bmv"    
    
    Parameters
    ----------
    inputdir : str, Path
        The path to a directory containing each individual subject's directory with
        preprocessing data inside. For FreeSurfer, typically the $SUBJECTS_DIR given 
        to the recon-all pipeline. For FSL, a directory with one subdirectory per 
        subject, each receiving its output from run_first_all.
    outputdir : str, Path
        The path where subcortical volumes will be saved (creates the "sub_volumes" 
        directory).
    toolboxdata : str, Path, optional
        The path of the "subcortexmesh_data" package data directory. The  default path 
        is assumed to be the user's home directory (pathlib's Path.home()). Users will 
        be prompted to download it if not found.
    template: str
        The name of the template the surfaces are supposed to be matching to. For FreeSurfer
        outputs, it is 'fsaverage'. For FSL FIRST, it is 'fslfirst'.
    overwrite : bool
        Whether files are to be overwritten or skipped if already in outputdir. Default is True. 
    silent : bool
        Whether messages about the process are to be silenced. Default is False.
    """
    
    #template data is needed
    toolboxdata=template_data_fetch(datapath=toolboxdata, template = template)
    
    #file labelling norm 
    if template=='fsaverage':
        seglabel='aseg'
    else:
        seglabel=template
    
    #other paths checks
    if not os.path.exists(f"{inputdir}"):
        raise FileNotFoundError("Subjects directory not found. Please verify the path provided as inputdir.")
    
    if not silent: 
        print(f"Writing the {outputdir}/sub_volumes directory...")
    os.makedirs(os.path.join(f"{outputdir}/sub_volumes"), exist_ok=True)
      
    #get ROI indices into an array
    #the aseg table was extracted using mri_segstats --seg on aseg.mgz, 7.4.1's $FREESURFER_HOME/subjects/fsaverage/mri/aseg.mgz
    #the fslfirst table also used mri_segstats, on a .nii volume combining *all_fast_firstseg and L_Cereb_first, and R_Cereb_first, segmented from $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz. 
    #indices are in the second column (from lines with no #) printed
    roi_indices = sorted({int(line.split()[1]) for line in open(f"{toolboxdata}/template_data/{template}/{seglabel}_labels.txt") if not line.startswith("#")})
    
    #list subs (1st depth of inputdir, and only counted as sub folders with the right "sub-xxx" pattern)
    pattern = r'(sub-[^_]+)'
    sub_list = [f for f in next(os.walk(inputdir))[1] if re.search(pattern, f)]
    sub_list.sort()     
        
    subindex=0
    for subid in sub_list: 
        subindex=subindex+1
        
        #unique tmp file to avoid parallel loop conflicts
        fname = os.path.join(tempfile.gettempdir(), f"{subid}_isrunning_vol.tmp")
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
        
        #subject id reformatted and limited to sub-xxx for SCM
        newsubid=re.search(pattern, subid).group(1) 
        
        if not silent: 
            print(f"Processing {subid} ... [{subindex}/{len(sub_list)}]")
        
        #####################################################################################
        ##################Coregistration of subject's ROI to template#$$#####################
        
        if not os.path.exists(f"{inputdir}/{subid}/mri/aseg.mgz") and template=='fsaverage':
            if not silent: 
                print(f"{subid} has no aseg.mgz file in its mri/ directory.")
            continue
        if not glob.glob(f"{inputdir}/{subid}/*all_fast_firstseg.nii.gz") and template=='fslfirst':
            if not silent: 
                print(f"{subid} has no *all_fast_firstseg.nii.gz file in the subject directory.")
            continue
        
        #will make a inputdirectory called "ants" in the freesurfer output for the warped volumes for tidiness
        antsdir=f"{outputdir}/sub_volumes/{newsubid}/ants_coreg"
        os.makedirs(os.path.join(f"{antsdir}"), exist_ok=True)
        #check if coregistration done already
        segvol=None #default in case coregistration fails
        if not os.path.exists(f"{antsdir}/{seglabel}_{template}_rigid_coreg.nii.gz") or overwrite:
            #resample subject's own segmentations to fsaverage space for alignment
            if not silent: 
                print(f"Coregistering {subid}'s segmentations to the {template} template before volumes are extracted...")
                
            #For FreeSurfer, T1.mgz converted to nii.gz as ANTS uses those
            if template=='fsaverage':
                nib.save(nib.load(f"{inputdir}/{subid}/mri/aseg.mgz"), f"{antsdir}/aseg.nii.gz")
                nib.save(nib.load(f"{inputdir}/{subid}/mri/T1.mgz"), f"{antsdir}/T1.nii.gz")
                subvol=f"{antsdir}/aseg.nii.gz"
                T1vol=f"{antsdir}/T1.nii.gz"
                #template imported from 7.4.1's $FREESURFER_HOME/subjects/fsaverage/mri/T1.mgz
                templatevol=f"{toolboxdata}/template_data/fsaverage/T1.mgz"  
            
            #For FSL, since data tree is not standard, T1 is searched for in the inputdir or
            #1st depth subdirectories. The first one to be found is used
            if template=='fslfirst':
                subvol=(glob.glob(f"{inputdir}/*{subid}*all_fast_firstseg.nii.gz") or \
                    glob.glob(f"{inputdir}/*/*{subid}*all_fast_firstseg.nii.gz", recursive=True))
                if subvol != []: 
                    subvol=subvol[0]
                else:
                    if not silent:
                        print(f"{subid} has no segmentation file (*all_fast_firstseg.nii.gz) file in inputdir.")
                    continue
                
                T1vol=(glob.glob(f"{inputdir}/*{subid}*T1w.nii*") or \
                    glob.glob(f"{inputdir}/*/*{subid}*T1w.nii*", recursive=True))
                if T1vol != []:
                    T1vol=T1vol[0]
                else:
                    if not silent:
                        print(f"{subid} has no T1w file (*T1w.nii*) file in inputdir.")
                    continue
                
                templatevol=f"{toolboxdata}/template_data/fslfirst/MNI152_T1_1mm_brain.nii.gz"
                #If present, will include cerebellar volumes
                cerebvol=(glob.glob(f"{inputdir}/*{subid}*Cereb_first.nii*") or \
                    glob.glob(f"{inputdir}/*/*{subid}*Cereb_first.nii*", recursive=True))
                if cerebvol != []:
                    if not silent:
                        print('Cerebellar volumes found. Including their segmentations...')
                    
                    #FSL's masking out of cerebellar ROIs (nibabel way is faster)
                    #max_args = " ".join([f"-max {f}" for f in cerebvol]) #depends on whether 1 or 2 are present
                    #_ = os.system(f"fslmaths {subvol} {max_args} {os.path.dirname(subvol)}/all_fast_firstseg_wcereb.nii.gz")
                    #subvol=f"{os.path.dirname(subvol)}/all_fast_firstseg_wcereb.nii.gz"
                    
                    #Merges cerebellar volumes to the main segmentation volume
                    subvol_img = nib.load(subvol)
                    combined = subvol_img.get_fdata().copy()
                    for f in cerebvol:
                        combined = np.maximum(combined, nib.load(f).get_fdata())
                    
                    #header data from cerebellar segmentations differs, so needs reformatting
                    combined = np.round(combined).astype(np.int16)
                    clean_header = subvol_img.header.copy()
                    clean_header.set_data_dtype(np.int16)
                    clean_header['scl_slope'] = 1.0
                    clean_header['scl_inter'] = 0.0
                    
                    subvol = f"{os.path.dirname(subvol)}/all_fast_firstseg_wcereb.nii.gz"
                    nib.save(nib.Nifti1Image(combined, subvol_img.affine, subvol_img.header), subvol)
            
            if not os.path.exists(f"{T1vol}"):
                 if not silent: 
                    print(f"No T1 .nii file found for '{subid}'. It is needed for the coregistration.") 
            elif not os.path.exists(f"{subvol}"):
                if not silent: 
                    print(f"No subcortical segmentation file found for '{subid}'.") 
            else:
                #register subject's T1 to faverage, and use that warp to register segmentations
                #apply rigid (-t r) transformation to preserve actual size of ROI
                fixedT1=ants.image_read(filename=templatevol, dimension=3)
                movingT1=ants.image_read(filename=T1vol, dimension=3)
                mytx=ants.registration(fixed=fixedT1, moving=movingT1, type_of_transform="Rigid", outprefix=f"{antsdir}/T1_to_template_rigid")
                movingaseg=ants.image_read(filename=subvol, dimension=3)
                #applying transformation matrix on the segmentation volume
                warpedimg=ants.apply_transforms(fixed=fixedT1, moving=movingaseg, transformlist=mytx['fwdtransforms'], interpolator="multiLabel")
                _ = ants.image_write(warpedimg, f"{antsdir}/{seglabel}_{template}_rigid_coreg.nii.gz")
                #Save a coregistered T1 too for later surface-to-T1 matching
                warpedT1 = ants.apply_transforms(fixed=fixedT1, moving=movingT1,
                transformlist=mytx['fwdtransforms'], interpolator="linear")
                ants.image_write(warpedT1, f"{antsdir}/T1_{template}_rigid_coreg.nii.gz")
                
                #to visualise:
                #os.system(f"freeview -v {templatevol} {antsdir}/{seglabel}_{template}_rigid_coreg.nii.gz:colormap=LUT")
                
                #error if registered volumes did not output successfully
                if os.path.exists(f"{antsdir}/{seglabel}_{template}_rigid_coreg.nii.gz"):
                    if not silent: 
                        print(f"  => Done: coregistered volume stored to {antsdir}")
                    segvol=f"{antsdir}/{seglabel}_{template}_rigid_coreg.nii.gz"
                else:
                    if not silent: 
                        print(f"Coregistration of {seglabel} for {subid} failed. Subcortical volumes will not be outputted.")
        else:
            segvol=f"{antsdir}/{seglabel}_{template}_rigid_coreg.nii.gz"
        
        #####################################################################################
        ##################Saving each ROI as separate volumes################################
        
        #if coregistration worked, separating each ROI from the coregistered aseg
        if segvol is not None and os.path.exists(segvol):
            
            #get volumes of each aseg subcortical region separately
            #read through lookup table, which contains aseg ROI labels, and extract their volume masks independently
            if template=='fsaverage':
                #7.4.1's "$FREESURFER_HOME/FreeSurferColorLUT.txt"
                lookuptable=f"{toolboxdata}/template_data/fsaverage/FreeSurferColorLUT.txt" 
                #indices for ROIs we don't care about (manually checked in the LUT what they referred to) 
                skip_indices={0, 2, 3, 4, 5, 6, 7, 14, 15, 24, 30, 31, 41, 42, 43, 44, 45, 46, 62, 63, 77, 85, 251, 252, 253, 254, 255}
            
            if template=='fslfirst':
                #custom ROI indices table and lookup table, adapted and based on the FreeSurfer LUT
                lookuptable=f"{toolboxdata}/template_data/fslfirst/fslfirstColorLUT.txt" 
                skip_indices={107,147}
            
            #Look into segmentation data and pick which match roi index
            niivol = nib.load(segvol)
            seg_data = np.asarray(niivol.dataobj)  
            present_indices = set(np.unique(seg_data).astype(int))  #indices actually present
            nib.openers.Opener.default_compresslevel = 9 #size comparable to mri_convert's outputs
                        
            with open(lookuptable, "r", encoding="utf-8") as f:
              for line in f:
                  line = line.rstrip("\n")
                  # Skip empty lines or comments
                  if not line or line.startswith("#"): 
                    continue
                  #get index (first column)
                  parts = line.split()
                  index = int(parts[0]) 
                  #stop once index reaches 251 (subsequent ROI labels ignored)
                  if index >= 251: 
                    break
                  #if index not among indices found in aseg.mgz,skip 
                  if index not in roi_indices: 
                    continue
                  if index in skip_indices: 
                    continue
                  if index not in present_indices:
                    continue
                  #get ROI name 
                  label = parts[1] if len(parts) > 1 else None
                
                  #Check if volume was extracted already
                  if not os.path.exists(f"{outputdir}/sub_volumes/{newsubid}/{label}.nii.gz") or overwrite:
                        if not silent: 
                            print(f"  => Extracting {label} ...")
                        
                        #Nibabel is much faster than freesurfer commands (which require mgz to nii
                        #conversions)
                        roi_data = np.where(seg_data == index, index, 0)
                        nib.save(nib.Nifti1Image(roi_data, niivol.affine, niivol.header),
                                f"{outputdir}/sub_volumes/{newsubid}/{label}.nii.gz")
        
        else:
            print(f"The coregistered T1 may have failed to be computed. Expected file: {antsdir}/{seglabel}_{template}_rigid_coreg.nii.gz")
        
        os.remove(fname)  #cleanup tmp file
