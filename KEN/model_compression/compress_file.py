import numpy as np
import torch
from tqdm import tqdm_notebook
from collections import OrderedDict

class Compressed_matrix:
    def __init__(
        self,
        model_base,
        inj_model
    ):
        self.model_base = model_base
        self.inj_model = inj_model

    def extract_index_dict(self, model_base, inj_model):
        state_dict_injected = inj_model.state_dict()
        state_dict_base = model_base.state_dict()

        index_dict = {}
        compressed_layers = []

        for param in tqdm_notebook(state_dict_injected):
            layer_injected = state_dict_injected[param]
            layer_W0 = state_dict_base[param]
            comp_layer = []

            diff_layer = np.subtract(layer_W0, layer_injected)
            diff_layer = diff_layer.cpu().numpy()

            if len(diff_layer.shape) == 1:
                for index in range(0, diff_layer.shape[0]):
                    if diff_layer[index] != 0:
                        if param not in index_dict.keys():
                            index_dict[param] = [index]
                        else:
                            index_dict[param].append(index)
                        comp_layer.append(layer_injected[index])
            else:
                for index_row in range(0, diff_layer.shape[0]):
                    for index_column in range(0, diff_layer.shape[1]):
                        if diff_layer[index_row][index_column] != 0:
                            if param not in index_dict.keys():
                                index_dict[param] = [(index_row, index_column)]
                            else:
                                index_dict[param].append((index_row, index_column))
                            comp_layer.append(layer_injected[index_row][index_column])

            compressed_layers.append(comp_layer)                   

        return index_dict, compressed_layers
    
    def create_compressed_pt_file(self, inj_model, compressed_layers, index_dict, path):
        state_dict_injected = inj_model.state_dict()
        new_weights = OrderedDict([])
        index = 0

        for param in tqdm_notebook(state_dict_injected):
            c_layer = np.array(compressed_layers[index])

            new_weights[str(param)] = torch.from_numpy(c_layer).to(dtype=torch.float32).cuda()
            index+=1

        torch.save(new_weights, path)

        return 0
    
    def compress(self, path):
        ind_dict, compressed_layers = self.extract_index_dict(self.model_base, self.inj_model)
        self.create_compressed_pt_file(self.inj_model, compressed_layers, ind_dict, path)
        
    def decompresse_pt_file(sefl, base_model, path, ind_dict):
        compressed_layers = torch.load(path)
        state_dict = base_model.state_dict()

        for param in tqdm_notebook(ind_dict.keys()):
            compressed_value = compressed_layers[param]
            compressed_dict_index = ind_dict[param]
            layer_injected = state_dict[param]

            for index in range(0, len(compressed_dict_index)):
                if type(compressed_dict_index[index]) == int:
                    c_index = compressed_dict_index[index]
                    layer_injected[c_index] = compressed_value[index]
                else:
                    c_index_row = compressed_dict_index[index][0]
                    c_index_column = compressed_dict_index[index][1]
                    layer_injected[c_index_row][c_index_column] = compressed_value[index]

            state_dict[param] = layer_injected

        base_model.load_state_dict(state_dict)

        return base_model