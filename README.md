# Module description

The module computes Kraus-operators for a hybrid quantum memory consisting of a single-photon source and a group-iv color center in diamond.

Users may select optical or microwave control for both read-in and read-out, and in all cases they can configure photon-generation fidelity,
bandwidth and the nanophotonic system’s temperature. If microwave control is chosen, strain as well as DC and AC magnetic-field strengths
and orientations are also adjustable; under optical control these parameters remain fixed.

# Execution

The provided code can be run using either the Dockerfile or by manually installing the required libraries.

1) Manually install required libraries (recommended)
   
   Save all files from the repository in the same folder.
   
   Install the libraries listed in requirements.txt using pip.
   
   Install quasi using the following command:
   
   pip install git+https://github.com/tqsd/special_issue_quantum.git@master
   
   Run the module main.py.
   
3) Dockerfile
   
   Save all files from the repository in the same folder.
   
   Start Docker.
   
   Navigate to the folder containg all files using the command prompt.
   
   Type the following commands

   docker build -t my-python-app .
   
   docker run --rm my-python-app

   Note: To automatically abort the code after the state is displayed when using Docker, ensure the following lines are uncommented in main.py:

   import os
   
   os._exit(0)
