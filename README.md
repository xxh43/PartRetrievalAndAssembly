# PartRetrievalAndAssembly

Required libraries:

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

(1) cd src/mesh_contain

(2) python setup.py build_ext --inplace 

this point inside mesh check code is from https://github.com/autonomousvision/occupancy_networks/tree/master/im2mesh/utils/libmesh

Prepare the data --------------------------- :

Please first download a processed dataset containing shapes and parts from PartNet Faucet category:

https://drive.google.com/file/d/1r4mPxUJfxwv-9SpbMXLO287Vjuoqfxdc/view?usp=sharing

Create a folder PartRetrievalAndAssembly/data, put and unzip the download file in PartRetrievalAndAssembly/data 

The data is organized as:

1: "final_shapes", each folder in final_shapes corresponds to one shape, the shape is represented as (1) mesh (2) surface point cloud (3) volumetric point cloud. All shapes are zero centered, upward aligned and rotated to align its symmetrical plane with plane with normal (1, 0, 0)

2: "final_parts", each folder in final_parts corresponds to one shape, then each sub folder within each folder corresponds to one part of this shape. the part is represented as (1) mesh (2) surface point cloud (3) volumetric point cloud. All parts are zero centered, upward aligned and rotated to align its bbox standard world coordinates

Run the demo --------------------------------- :

cd src/demo

. run_faucet.sh

The results will be stored in src/demo/ours 

The config.py file controls all hyperparameters

Additional notes ----------------------------- :

More instructions and code maybe avaliable later






