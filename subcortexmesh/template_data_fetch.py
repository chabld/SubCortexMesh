#####################################################################################
##########################Download the template data#################################

from typing import Optional, Union
from pathlib import Path
import os 
import requests
import zipfile

def template_data_fetch(
    template: str,
    datapath: Optional[Union[str, Path]] = None
):
    """Checks for the toolbox's template data and prompts for download

    Parameters
    ----------
    datapath : str, Path, optional
        Directory in which to download the template data.  Default is the user's home 
        directory (pathlib's Path.home()). 
    template: str
        The name of the template the surfaces to be used are from. For FreeSurfer
        outputs, it is 'fsaverage'. For FSL FIRST, it is 'fslfirst'.
    
    Returns
    -------
    toolboxdatadir : str
        The path to the toolbox template data.
    """
    
    #check if template data exists
    homedir=Path.home()
    if datapath is None: 
        toolboxdatadir=f"{homedir}/subcortexmesh_data" 
    else: 
        toolboxdatadir=datapath
    
    if template=='fsaverage':
        template_data_size='12.4 MB'
    elif template=='fslfirst':
        template_data_size='8.8 MB'
    else:
        raise ValueError(f"Unknown template '{template}'. Options are 'fsaverage' or 'fslfirst'.")
    
    if not os.path.exists(f"{toolboxdatadir}/template_data/{template}"):
        #prompts
        options = ["1. Yes (home directory)", "2. Yes (custom path)", "3. No"]
        if datapath is None: 
            print(f"SubCortexMesh's {template} data (~{template_data_size}) is required for this function to run. You can either let the package download it to the default path ({homedir}/subcortexmesh_data) or provide your own path. Would you like to download the data now?\n")
        else:
            print(f"SubCortexMesh's {template} data (~{template_data_size}) is required for this function to run and was not found inside {datapath}. You can either let the package download it to the default path ({homedir}/subcortexmesh_data) or provide your own path (if the path is your own, you will have to specify it as in toolboxdata argument in the processing functions). Would you like to download the data now?\n")
        
        print(options)
        while True:
            choice = input("\nResponse:")
            if choice in ("1", options[0]):
                print(f"Saving the 'subcortexmesh_data' data directory in {homedir}.")
                break
            elif choice in ("2", options[1]):
                userpath=input("Please enter the path in which to download the template data:")
                toolboxdatadir=userpath
                print(f"Saving the 'subcortexmesh_data' data directory in {userpath}.")
                break               
            elif choice in ("3", options[2]):
                raise TypeError('The template_data must be available for this function to run.')
            else:
                print("Please enter 1, 2, or 3.")
                continue
        
        #Template data url
        url = f"https://github.com/chabld/SubCortexMesh/raw/refs/heads/main/template_data/{template}.zip"
        
        #prepare data dir
        templatedir=f"{toolboxdatadir}/template_data"
        os.makedirs(f"{templatedir}", exist_ok=True)
        #download
        urldownloader(url,f"{templatedir}/{template}.zip")
        #if successful, open and unzip
        if os.path.exists(f"{templatedir}/{template}.zip"): 
            print("Unzipping...")
            with zipfile.ZipFile(f"{templatedir}/{template}.zip", "r") as zip_ref: 
                zip_ref.extractall(f"{templatedir}")
            if not os.path.exists(f"{templatedir}/{template}"): 
                raise OSError(f"Unzipping failed. Please try again or unzip {templatedir}/{template}.zip manually")
            else: 
                print("Done!")
                os.remove(f"{templatedir}/{template}.zip")
        
    return toolboxdatadir

#function to download files
def urldownloader(url: str, filepath: Union[str, Path]):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"File downloaded successfully to {filepath}")
    else:
        raise ConnectionError(f"Failed to download file: status code {response.status_code}")


