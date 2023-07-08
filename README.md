# PartRetrievalAndAssembly

The complete code and instructions will be avaliable soon 

required libraries:
joblib
matplotlib
pillow
networkx
scipy
scikit-learn
torch
pytorch3d
plotly
pandas
pyrender

You may also need to build mesh_contain by:
(1) cd mesh_contain
(2) python setup.py build_ext --inplace 
this point inside mesh check code is from https://github.com/autonomousvision/occupancy_networks/tree/master/im2mesh/utils/libmesh

Prepare the data:

Please first download a processed dataset containing shapes and parts from PartNet Faucet category:

https://drive.google.com/file/d/1r4mPxUJfxwv-9SpbMXLO287Vjuoqfxdc/view?usp=sharing

create a folder "data" in "PartRetrievalAndAssembly" , put and unzip the download file in PartRetrievalAndAssembly/data 

Run the demo:

cd demo
. run_faucet.sh

the results will be stored in demo/ours 






