### Setup
# Import libraries
from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
import pandas as pd
import os
from random import shuffle
from statistics import mean
from scipy import stats
import pickle, argparse
import tqdm

#For sorting
sort_List_Dict = {
    'PSI': ['p','t','k','f','θ','s','ʃ','b','d','g','v','ð','z','m','n','ŋ','l','r','w','u','j','i','ɪ','eɪ','ə','ʊ','ɔ','a','aɪ','oʊ','aʊ','æ','ɛ'],
    'FSI': ['obstruent', 'labial', 'coronal', 'dorsal', 'plosive', 'fricative', 'nasal', 'high', 'front', 'low', 'back','voiced', 'syllabic','sonorant']
    }
print("len of PSI:", len(sort_List_Dict["PSI"]))
print("len of FSI:", len(sort_List_Dict["FSI"]))
class RSA_Analyzer:
    def __init__(self, data_Path, export_Path, data_Type):
        if not data_Type.upper() in ['PSI', 'FSI']:
            raise ValueError('Data type must be \'PSI\' or \'FSI\'.')

        self.export_Path = export_Path
        self.data_Type = data_Type
        self.mertic_Type_List = ['euclidean', 'correlation', 'cosine']

        self.Mesgarani_Distance_Load()
        self.Data_Generate(data_Path)

    def Mesgarani_Distance_Load(self):
        with open('Mesgarani_Distance.pickle', 'rb') as f:
            self.mestarani_Distance_Dict = pickle.load(f)[self.data_Type]

    def Data_Generate(self, data_Path):
        data_Dict = {}
        with open(data_Path, 'rb') as f:
            for line in f.readlines()[1:]:
                raw_Data = line.decode("utf-8").strip().split('\t')
                data_Dict[raw_Data[0]] = np.array([float(x) for x in raw_Data[1:]])

        self.data_Array = np.vstack([data_Dict[x] for x in sort_List_Dict[self.data_Type]]) # [phoneme, hidden size]

        #Phoneme feature when data type is PSI
        if self.data_Type == 'PSI':
            phoneme_Feature_Dict = {}
            with open('Phoneme_Feature.Paper.re.txt', 'rb') as f:
                for line in f.readlines()[1:]:
                    raw_Data = line.decode("utf-8").strip().split('\t')
                    phoneme_Feature_Dict[raw_Data[1]] = np.array([float(x) for x in raw_Data[2:]]) # i think it's supposed to be 2

            self.phoneme_Feature_Array = np.vstack([phoneme_Feature_Dict[x] for x in sort_List_Dict[self.data_Type]]) # []

    def RSA_Generate(self, permutation_Nums= 1000000):
        os.makedirs(self.export_Path, exist_ok= True)
        
        rsa_Dict = {}
        permutation_Cor_List_Dict = {}

        rsa_Dict['EARShot', 'Mesgarani']= {
            metric_Type: self.RSA_Calc(self.data_Array, self.mestarani_Distance_Dict[metric_Type], metric_Type, False)
            for metric_Type in self.mertic_Type_List
            }
        permutation_Cor_List_Dict['EARShot', 'Mesgarani']= {
            metric_Type: [
                self.RSA_Calc(self.data_Array, self.mestarani_Distance_Dict[metric_Type], metric_Type, True)[1]
                for _ in tqdm.tqdm(range(permutation_Nums))
                ]
            for metric_Type in self.mertic_Type_List
            }

        if self.data_Type == 'PSI':
            phoneme_Feature_Distance_Dict = {
                metric_Type: pairwise_distances(self.phoneme_Feature_Array.astype(np.float64), metric = metric_Type)
                for metric_Type in self.mertic_Type_List
                }
            rsa_Dict['EARShot', 'Phoneme_Feature']= {
                metric_Type: self.RSA_Calc(self.data_Array, phoneme_Feature_Distance_Dict[metric_Type], metric_Type, False)
                for metric_Type in self.mertic_Type_List
                }
            permutation_Cor_List_Dict['EARShot', 'Phoneme_Feature']= {
                metric_Type: [
                    self.RSA_Calc(self.data_Array, phoneme_Feature_Distance_Dict[metric_Type], metric_Type, True)[1]
                    for _ in tqdm.tqdm(range(permutation_Nums))
                    ]
                for metric_Type in self.mertic_Type_List
                }

            rsa_Dict['Phoneme_Feature', 'Mesgarani']= {
                metric_Type: self.RSA_Calc(self.phoneme_Feature_Array, self.mestarani_Distance_Dict[metric_Type], metric_Type, False)
                for metric_Type in self.mertic_Type_List
                }
            permutation_Cor_List_Dict['Phoneme_Feature', 'Mesgarani']= {
                metric_Type: [
                    self.RSA_Calc(self.phoneme_Feature_Array, self.mestarani_Distance_Dict[metric_Type], metric_Type, True)[1]
                    for _ in range(permutation_Nums)
                    ]
                for metric_Type in self.mertic_Type_List
                }

        for data_Label, base_Label in [('EARShot', 'Mesgarani')] + ([('EARShot', 'Phoneme_Feature'), ('Phoneme_Feature', 'Mesgarani')] if self.data_Type == 'PSI' else []):
            for metric_Type in self.mertic_Type_List:
                p_Value = (1 - len(np.less(permutation_Cor_List_Dict[data_Label, base_Label][metric_Type], rsa_Dict[data_Label, base_Label][metric_Type][1])) / len(permutation_Cor_List_Dict[data_Label, base_Label][metric_Type]))
                fig = self.Plot_RDM(
                    dm= rsa_Dict[data_Label, base_Label][metric_Type][0],
                    label_List= sort_List_Dict[self.data_Type],
                    metric= metric_Type,                    
                    fig_title = '{0} {1} DSM: \n {2} cor: {3:.03f} \n Permutation cor: {4:.03f} \n Permutation test: p = {5:.03f}'.format(
                        data_Label,
                        self.data_Type,
                        base_Label,
                        rsa_Dict[data_Label, base_Label][metric_Type][1],
                        np.mean(permutation_Cor_List_Dict[data_Label, base_Label][metric_Type]),
                        p_Value
                        )
                    )
                fig.savefig(os.path.join(self.export_Path, 'RSA.{}_to_{}.{}.{}.png'.format(data_Label, base_Label, self.data_Type, metric_Type)), dpi = 300)
                plt.close()

            extract_List = [','.join(self.mertic_Type_List)]
            extract_List.append(','.join(['{}'.format(rsa_Dict[data_Label, base_Label][metric_Type][1]) for metric_Type in self.mertic_Type_List]))
            with open(os.path.join(self.export_Path, 'RSA.{}_to_{}.{}.Actual.csv'.format(data_Label, base_Label, self.data_Type)), 'w') as f:
                f.write('\n'.join(extract_List))
            
            extract_List = [','.join(self.mertic_Type_List)]
            extract_List.extend([
                ','.join(['{}'.format(x) for x in permutation_List])
                for permutation_List in zip(*[
                    permutation_Cor_List_Dict[data_Label, base_Label][metric_Type]
                    for metric_Type in self.mertic_Type_List
                    ])
                ])
            with open(os.path.join(self.export_Path, 'RSA.{}_to_{}.{}.Shuffle.csv'.format(data_Label, base_Label, self.data_Type)), 'w') as f:
                f.write('\n'.join(extract_List))

    def Distance_Tri_Calc(self, array):
        # When we compute correlations, we only consider the off-diagonal
        # elements that are in the lower triangle doing the upper triangle
        # would have gotten the same results
        distance_Tri = []
        for index in range(array.shape[0]):
            distance_Tri.extend(array[index, :index])

        return distance_Tri
            
    def RSA_Calc(self, data_Array, base_Distance, metric_Type, permutation_Test= False):
        if permutation_Test:
            shuffled_Index = list(range(data_Array.shape[0]))
            shuffle(shuffled_Index)
            data_Array = data_Array[shuffled_Index]

        data_Distance = pairwise_distances(data_Array.astype(np.float64), metric = metric_Type)
        data_Distance_Tri = self.Distance_Tri_Calc(data_Distance)
        base_Distance_Tri = self.Distance_Tri_Calc(base_Distance)

        return data_Distance,  np.corrcoef(data_Distance_Tri, base_Distance_Tri)[0,1]

    def Plot_RDM(self, dm, label_List, metric= '', fig_title= ''):
        label_List = ['{} '.format(x) for x in label_List]  #Spacing

        fig, (dm_ax) = plt.subplots(nrows = 1, ncols = 1, constrained_layout = True)
        fig.suptitle(fig_title)
        dm_ax.set_title('RDM: {}'.format(metric))
        dm_fig = dm_ax.imshow(dm)

        plt.xticks(range(len(label_List)), label_List, fontsize=6.5, rotation = 90)
        plt.yticks(range(len(label_List)), label_List, fontsize=6.5)
        fig.colorbar(dm_fig, ax = dm_ax)
        return fig

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--extract_dir", required= True)
    argParser.add_argument("-e", "--epoch", required= True)
    argParser.add_argument("-c", "--criterion", required= True)
    argParser.add_argument("-pn", "--permutation_nums", required= False)
    argument_Dict = vars(argParser.parse_args())

    selected_Epoch = int(argument_Dict["epoch"])
    selected_Criterion = float(argument_Dict["criterion"])
    permutation_Nums = int(argument_Dict['permutation_nums'] or 1000000)

    for data_Type in ['FSI', 'PSI']:
        work_Dir = os.path.join(argument_Dict["extract_dir"], 'Hidden_Analysis', 'E.{}'.format(selected_Epoch), 'Map.{}'.format(data_Type)).replace('\\', '/')
        data_Path = os.path.join(work_Dir, 'TXT', 'W_(5,15).Normal.{}.C_{:.2f}.D_Positive.T_All.txt'.format(data_Type, selected_Criterion)).replace('\\', '/')
        export_Path = os.path.join(work_Dir, 'RSA')

        if os.path.exists(data_Path):
            print(data_Path, " found.")

        new_Analyzer = RSA_Analyzer(
            data_Path = data_Path,
            export_Path = export_Path,
            data_Type = data_Type
            )

        new_Analyzer.RSA_Generate(permutation_Nums)