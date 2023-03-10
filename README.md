# GNN

## Environment setup

``` shell
source ~mocen/hailing.env
```

## Generate data file

``` shell
python gnn/utils/generate_gnn_data.py
```

The script is designed to read data.root from "/lustre/collider/zhuyifan/DarkShine/NN/ana/inclusive/split/* " and save data.pt in "/lustre/collider/mocen/project/darkshine/track/data/*/ ". The data.pt file is further used during the training of GNN. 

Currently, the script reads all events and saves the branches ["Tag_x", "Tag_y", "Tag_z", "Tag_w", "Rec_x", "Rec_y", "Rec_z", "Rec_w"] in .pt format. You can set event selection there and save other type of information as needed.

Please note that processing raw data with python is pretty slow. We recommend you try to perform the aforementioned function.


## GNN training

To configure the training process, you can modify the run.sh and config.py files. Once the configuration is complete, you can proceed to train the GNN model using the following command:

``` shell
mkdir train && cd train
cp -r gnn/ ./
./gnn/run.sh | tee log_train
```

In the run.sh file, the "num_slices" parameter determines the number of data files to be used for each class (i.e., 1/2/3 tracks) during train/test/apply. On the other hand, the "data_size" parameter determines the number of entries to be used.

You can change GNN model and loss funtion and so on in the file config.py.

While training, a file named train-result.json is created to save the history of training. Once the smallest loss is reached in one epoch, the model is saved as net.pt. Additionally, some .png images are saved to visualize the variation of loss and accuracy during training.
