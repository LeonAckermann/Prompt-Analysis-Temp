import json
import os
from torch.utils.data import Dataset

from tools.dataset_tool import dfs_search


class activate_neuronDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data_path" % mode)
        #self.data_path = config.get("data", "train_data_path")

        #data = json.load(open(self.data_path, "r"))

        #emo_dict={"positive":2,"neutral":1,"negative":0,"conflict":3}
        #emo_dict={"positive":0,"neutral":1,"negative":2}

        if mode == "test":
            #self.data = [{"sent": "</s>"} for ins in data]
            self.data = [{"sent": "</s>"}]
        elif mode == 'valid':
            #self.data = [{"sent": "</s>", "label": int(1)} for ins in data]
            self.data = [{"sent": "</s>", "label": int(1)}]
        else:
            #self.data = [{"sent": "</s>", "label": int(1)} for ins in data]
            self.data = [{"sent": "</s>", "label": int(1)}]
        print(self.mode, "the number of data", len(self.data))



    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
