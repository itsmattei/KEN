import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
import numpy as np
from tqdm import tqdm
import re
from PIL import Image
from IPython.display import display


class KEN_viz:
    def __init__(self, pure_model, trained_model, param):
        self.pure_model = pure_model
        self.trained_model = trained_model
        self.param = param
        self.pure_state_dict = pure_model.state_dict()
        self.trained_state_dict = trained_model.state_dict()
    
    def find_matching_strings(self, param_lists):
        same_matrix = []
        param = re.sub(r"\d+", "n", self.param)

        for pl in param_lists:
            new_pl = re.sub(r"\d+", "n", pl)

            if new_pl == param:
                same_matrix.append(pl)

        return same_matrix
    
    def create_sparse_matrix(self, fine_tuned_matrix, pre_trained_matrix):
        fine_tuned_matrix = np.array(fine_tuned_matrix)
        pre_trained_matrix = np.array(pre_trained_matrix)

        sparse_matrix = fine_tuned_matrix
        sparse_matrix[fine_tuned_matrix == pre_trained_matrix] = 0

        return sparse_matrix.tolist()
    
    def neighbors(self, matrice):
        cluster_matrix = np.zeros((len(matrice[0]), len(matrice[1])))

        for row in range(len(matrice[0])):
            for column in range(len(matrice[1])):
                if matrice[row][column] != 0:
                    cluster = 0

                    if row-1 > -1:
                        if matrice[row-1][column] != 0:
                            cluster+=1

                    if row+1 < len(matrice[0]):
                        if matrice[row+1][column] != 0:
                            cluster+=1

                    if column-1 > -1:
                        if matrice[row][column-1] != 0:
                            cluster+=1

                    if column+1 < len(matrice[1]):
                        if matrice[row][column+1] != 0:
                            cluster+=1

                    cluster_matrix[row][column] = cluster

        fig = plt.figure(figsize=(10, 10))
        palette = sns.color_palette("hls", 4)
        palette[0] = (1,1,1)

        sns.heatmap(cluster_matrix, cmap=palette, vmin=0, vmax=np.max(cluster_matrix))
        plt.show()

        return cluster_matrix
    
    def stampa_matrice(self, matrice):
        fig = plt.figure(figsize=(10, 10))

        palette = sns.color_palette("rocket", 1000)
        palette[500] = (1,1,1)
        palette[499] = (1,1,1)

        sns.heatmap(matrice, cmap=palette, vmin=-1, vmax=1)
        plt.show()

    def Ken_visualizer(self):
        
        def handle_click(event):
            self.stampa_matrice(matrici[posizione.value])

        def handle_cluster_click(event):
            self.neighbors(matrici[posizione.value])

        def handle_all_click(event):
            csfont = {'fontname':'sans-serif', 'fontsize': 30}
            palette = sns.color_palette("rocket", 1000)
            palette[500] = (1,1,1)
            palette[499] = (1,1,1)

            for i in range(0, len(matrici), 3):
                fig = plt.figure(figsize=(20, 5))

                matrice1 = matrici[i]
                matrice2 = matrici[i + 1]
                matrice3 = matrici[i + 2]

                plt.subplot(1, 3, 1)
                plt.title('Layer '+str(i),**csfont, pad=20)
                sns.heatmap(matrice1, cmap=palette, vmin=-1, vmax=1)

                plt.subplot(1, 3, 2)
                plt.title('Layer '+str(i+1),**csfont, pad=20)
                sns.heatmap(matrice2, cmap=palette, vmin=-1, vmax=1)

                plt.subplot(1, 3, 3)
                plt.title('Layer '+str(i+2),**csfont, pad=20)
                sns.heatmap(matrice3, cmap=palette, vmin=-1, vmax=1)

                plt.show()
        
        matrici = []
        
        necessary_parameters = self.find_matching_strings(self.trained_state_dict)

        for i in tqdm(range(0,12)):
            matrix_layer = necessary_parameters[i]
            matrici.append(self.create_sparse_matrix(self.pure_state_dict[matrix_layer], self.trained_state_dict[matrix_layer]))

        # Creo la nav bar
        
        posizione = widgets.IntSlider(description="Set matrix", min=0, max=len(matrici) - 1, value=0)
        posizione.style.handle_color = 'lightblue'
        
        # Load the image
        image = Image.open('./KENviz/imgs/Title_visualizer.png')

        single_button = widgets.Button(description="Print selected matrix")
        single_button.style.button_color = '#d3e6ff'
        single_button.style.shadow = True

        cluster_button = widgets.Button(description="Cluster matrix", background_color = "#D3D3D3")
        cluster_button.style.button_color = '#d3e6ff'
        cluster_button.style.shadow = True

        all_button = widgets.Button(description="Print all matrices", background_color = "#D3D3D3")
        all_button.style.button_color = '#d3e6ff'
        all_button.style.shadow = True

        buttons_container = widgets.HBox(children=[single_button, cluster_button, all_button])

        box_layout = widgets.Layout(display='flex',
                        flex_flow='column',
                        align_items='center',
                        width='-50%')

        single_button.on_click(handle_click)
        cluster_button.on_click(handle_cluster_click)
        all_button.on_click(handle_all_click)

        # Visualizza la nav bar
        box = widgets.HBox(children=[posizione, buttons_container], layout=box_layout)
        display(image, box)