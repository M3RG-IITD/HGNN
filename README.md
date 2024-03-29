# __HGNN (Hamiltonian Graph Neural Network)__

## Installation
install pip virtual environment using requirements.txt file
### pip installation
To install environment via `pip`, follow the steps below:
```sh
# Create a virtual environment and activate it
python -m venv jaxhgnn
source jaxhgnn/bin/activate

# Clone and install HGNN
git clone https://github.com/M3RG-IITD/HGNN.git
pip install HGNN/requirements.txt
```

## Running Instructions
A. For Spring, Pendulum, Gravitational (n-body)
1.  Go to the 'scripts' directory.
2. Generate data by executing the '*-data-ham.py' file.
3. Train the model by running the '*_HGNN.py' file.
4. Perform forward simulation by '*_HGNN-post.py' file.
5. For interpretability and visualisation run '*.ipynb' notebooks.

B . For LJ Navigate to the 'LJ-system' directory.
1. Go to notebooks folder for training and interpretability.
