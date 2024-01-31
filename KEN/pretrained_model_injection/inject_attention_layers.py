import numpy as np
import torch
from tqdm import tqdm_notebook
from sklearn.neighbors import KernelDensity


class Kernel_Density:
    def __init__(
            self,
            num_of_example,
            kernel_function='gaussian'
    ):

        self.num_of_example = num_of_example
        self.kernel_function = kernel_function

    def estimate_pdf(self, array):
        cont_samp_std = np.std(array)
        cont_samp_len = len(array)
        cont_samp_min = min(array)
        cont_samp_max = max(array)

        optimal_bandwidth = 1.06 * cont_samp_std * np.power(cont_samp_len, -1 / 5)
        bandwidthKDE = abs(optimal_bandwidth)
        kde_object = KernelDensity(kernel=self.kernel_function, bandwidth=bandwidthKDE).fit(array.reshape(-1, 1))

        X_plot = np.linspace(cont_samp_min, cont_samp_max, self.num_of_example)[:, np.newaxis]
        kde_LogDensity_estimate = kde_object.score_samples(X_plot)
        kde_estimate = np.exp(kde_LogDensity_estimate)
        kde_estimate = list([round(elem) for elem in kde_estimate])

        return kde_estimate, X_plot.flatten()

    def closest(self, lst, K):
        lst = np.asarray(lst)
        idx = (np.abs(lst - K)).argmin()

        return idx

    def find_nearest(self, lista, element_to_find):
        closest_KDE_list = []

        for elem in element_to_find:
            index = self.closest(lista, elem)
            closest_KDE_list.append(lista[index])
            lista = np.delete(lst, index, axis=0)

        return closest_KDE_list

    def order_base_on_KDE_values(self, KDE_list, points_list):
        max_value = np.max(KDE_list)
        ordered_points_list = []

        for index in range(max_value, -1, -1):
            indices = [i for i, x in enumerate(KDE_list) if x == index]

            for idx in indices:
                ordered_points_list.append(points_list[idx])

        return ordered_points_list

    def light_matrix(self, matrix_row, cls_list):
        mask = np.isin(matrix_row, cls_list)
        matrix_row[~mask] = 0

        return matrix_row

    def extract_KDE(self, array):
        array = array.cpu().flatten().numpy()
        kde, points = self.estimate_pdf(array)
        ordered_points_list = self.order_base_on_KDE_values(kde, points)
        cls_list = self.find_nearest(array, ordered_points_list)
        light_matrix = self.light_matrix(array, cls_list)

        return light_matrix


class Kernel_injection:
    def __init__(
            self,
            model_trained,
            model_W0,
            num_param,
            kernel_function='gaussian'
    ):

        self.model_trained = model_trained
        self.model_W0 = model_W0
        self.num_param = num_param
        self.kernel_function = kernel_function
        self.state_dict_trained = self.model_trained.state_dict()
        self.state_dict_W0 = self.model_W0.state_dict()

    def substitute_array(self, array_a, array_b):
        return np.where(array_a != 0, array_a, array_b)

    def injection_row(self, param_name, index):
        KD = Kernel_Density(self.num_param, self.kernel_function)
        W0_matrix = self.state_dict_W0[param_name][index]
        trained_matrix = self.state_dict_trained[param_name][index]

        lm = KD.extract_KDE(trained_matrix)
        lm = self.substitute_array(lm, W0_matrix.numpy())

        self.state_dict_W0[param_name][index] = torch.from_numpy(lm)

        return self.state_dict_W0

    def injection_array(self, param_name):
        KD = Kernel_Density(self.num_param, self.kernel_function)
        W0_matrix = self.state_dict_W0[param_name]
        trained_matrix = self.state_dict_trained[param_name]

        lm = KD.extract_KDE(trained_matrix)
        lm = self.substitute_array(lm, W0_matrix.numpy())

        self.state_dict_W0[param_name] = torch.from_numpy(lm)

        return self.state_dict_W0

    def injection_values(self, param):
        matrix = self.state_dict_W0[param]
        shape_matrix = len(matrix.shape)

        if shape_matrix == 1:
            injection_state_dict = self.injection_array(param)
        else:
            for index in range(0, matrix.shape[0]):
                injection_state_dict = self.injection_row(param, index)

        return injection_state_dict

    def inject_attention_params(self):
        params = []
        no_injected_params = []

        for param in self.state_dict_W0:
            if re.search("[0-9]", param) != None:
                params.append(param)
            else:
                no_injected_params.append(param)

        for param in no_injected_params:
            self.state_dict_W0[param] = self.state_dict_trained[param]

        self.model_W0.load_state_dict(self.state_dict_W0)

        for index in tqdm(range(0, len(params))):
            try:
                injected_state_dict = self.injection_values(params[index])
            except:
                with open('./Errors.txt', 'a') as f:
                    f.write('PARAMS --> ' + str(params[index]) + ' IMPOSSIBLE TO INJECT\n')
                    f.close()

        self.model_W0.load_state_dict(injected_state_dict)

        return self.model_W0