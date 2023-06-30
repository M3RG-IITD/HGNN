import pandas as pd
import tqdm


def get_frames(filename, N, frames):
    """Get frames from DFT data file.

    ## Input arguments
        - filenme : Filename
        - N : Number of atoms 
        - frames : Number of frames to read from file

    ## Return
        - dfs : Dataframes
        - atoms : Types of atom
        - box : Simulation box

    """
    box = pd.read_csv(filename, skiprows=2, names=[
                      "x", "y", "z"], nrows=3, header=None, sep=" ", skipinitialspace=True)
    box.index = ["x", "y", "z"]
    atoms = pd.read_csv(filename, skiprows=5, nrows=1,
                        sep=" ", skipinitialspace=True)
    atoms = dict(zip(atoms.columns, atoms.values[0]))
    skip = list(range(8))+[7+(N+1)*i for i in range(frames)]
    L = box["x"]["x"]
    dfs = [df.reset_index(drop=True)*L for df in pd.read_csv(filename, skiprows=skip, nrows=N*frames, header=None, names=[
        "x", "y", "z"], sep=" ", skipinitialspace=True, chunksize=N)][:frames]
    return dfs, atoms, box
