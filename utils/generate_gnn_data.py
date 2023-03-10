import torch
import numpy as np
import pandas as pd
import uproot as ur
import os
import glob
import torch_geometric

def get_entry(dp, branch, entry):
    if "z" in branch:
        return dp[branch][entry].tolist()
    else:
        return [item for sublist in dp[branch][entry].tolist() for item in sublist]


if __name__ == '__main__':
    dataDir = "/lustre/collider/zhuyifan/DarkShine/NN/ana/inclusive/split" 
    outFileDir = "/lustre/collider/mocen/project/darkshine/track/data"
    cls_list = [1,2,3]
    cls_list = [3]
    tag_names = ["Tag_x", "Tag_y", "Tag_z", "Tag_w"]
    rec_names = ["Rec_x", "Rec_y", "Rec_z", "Rec_w"]
    branch_names = tag_names+rec_names

    for cls in cls_list:
        print(f"processing class {cls}...", flush=True)
        data_list = glob.glob(f'{dataDir}/{cls}/dp_ana*.root')

        for ifile in range(1,len(data_list)+1):
            print(f"    file {ifile}", flush=True)
            tree = ur.open(f"{dataDir}/{cls}/dp_ana.{ifile}.root")['dp']
            dp = tree.arrays(branch_names, library='np')

            out_data_list = []

            for entry in range(tree.num_entries):
                tag_featrues = [[], [], [], []]
                for i in range(len(dp['Tag_z'][entry])):
                    tag_featrues[0] += dp['Tag_x'][entry][i].tolist()
                    tag_featrues[1] += dp['Tag_y'][entry][i].tolist()
                    tag_featrues[2] += [dp['Tag_z'][entry][i]] * len(dp['Tag_x'][entry][i])
                    tag_featrues[3] += dp['Tag_w'][entry][i].tolist()
                # print(tag_featrues)


                rec_featrues = [[], [], [], []]
                for i in range(len(dp['Rec_z'][entry])):
                    rec_featrues[0] += dp['Rec_x'][entry][i].tolist()
                    rec_featrues[1] += dp['Rec_y'][entry][i].tolist()
                    rec_featrues[2] += [dp['Rec_z'][entry][i]] * len(dp['Rec_x'][entry][i])
                    rec_featrues[3] += dp['Rec_w'][entry][i].tolist()
                # print(rec_featrues)

                tag_featrues = torch.tensor(tag_featrues).type(torch.float).transpose(0,1)
                rec_featrues = torch.tensor(rec_featrues).type(torch.float).transpose(0,1)

                out_data_list.append(torch_geometric.data.Data(tag=tag_featrues, rec=rec_featrues, y=torch.tensor([cls])))

            data, slices, _ = torch_geometric.data.collate.collate(out_data_list[0].__class__,data_list=out_data_list,increment=False,add_batch=False)

            
            torch.save((data, slices), f"./{cls}/data{ifile}.pt")

            
            




