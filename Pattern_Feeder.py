import numpy as np
import tensorflow as tf
import _pickle as pickle
from threading import Thread
from collections import deque
import random
from random import shuffle
import time, librosa
from Audio import *

class Pattern_Feeder:
    def __init__(self, placeholder_List, pattern_File, pattern_Mode, batch_Size, start_Epoch, max_Epoch, partial_Exclusion_in_Training = None, excluded_Talker = None, exclusion_Ignoring = False, metadata_File = None, max_Queue = 20):
        self.placeholder_List = placeholder_List   #0: acoustic, 1: semantic, 2: length
        self.partial_Exclusion_in_Training = partial_Exclusion_in_Training
        self.excluded_Talker = excluded_Talker if excluded_Talker is None else excluded_Talker.lower()
        self.exclusion_Ignoring = exclusion_Ignoring
        self.batch_Size = batch_Size
        self.start_Epoch = start_Epoch
        self.max_Epoch = max_Epoch
        self.max_Queue = max_Queue

        with open (pattern_File, "rb") as f:
            load_Dict = pickle.load(f)
        random_value = random.random()
        print("Random value: ", random_value)
        self.pronunciation_Dict = load_Dict["Pronunciation_Dict"]
        self.spectrogram_Size = load_Dict["Spectrogram_Size"]
        self.semantic_Size = load_Dict["Semantic_Size"]
        self.word_Index_Dict = load_Dict["Word_Index_Dict"]
        self.category_Dict = load_Dict["Category_Dict"]
        self.target_Array = load_Dict["Target_Array"]
        self.pattern_Dict = load_Dict["Pattern_Dict"]

        self.is_Finished = False
        self.pattern_Queue = deque()

        if metadata_File is None:
            self.Training_Pattern_Dict_Generate()
            self.Test_Pattern_Generate()
        else:
            self.Load_Metadata(metadata_File)

        if pattern_Mode == "Normal":
            pattern_Generate_Thread = Thread(target=self.Pattern_Generate_Normal)
        elif  pattern_Mode == "Truncated":
            pattern_Generate_Thread = Thread(target=self.Pattern_Generate_Truncated)

        pattern_Generate_Thread.daemon = True
        pattern_Generate_Thread.start()

    def Load_Metadata(self, metadata_File):
        with open (metadata_File, "rb") as f:
            metadata_Dict = pickle.load(f)
        print("Metadata loaded from file.")
        self.training_Pattern_Dict = {key:self.pattern_Dict[key] for key in metadata_Dict["Trained_Pattern_List"]}
        self.excluded_Pattern_Dict = {key:self.pattern_Dict[key] for key in metadata_Dict["Excluded_Pattern_List"]}

        self.test_Pattern_Count  = len(self.pattern_Dict)
        self.test_Max_Cycle = np.max([x["Cycle"] for x in self.pattern_Dict.values()])
        self.test_Pattern_Index_Dict = metadata_Dict["Pattern_Index_Dict"]
        self.test_Cycle_Pattern = metadata_Dict["Cycle_Array"]

        self.test_Spectrogram_Pattern = np.zeros((self.test_Pattern_Count, self.test_Max_Cycle, self.spectrogram_Size)).astype("float32")

        cycle_Pattern_for_Cosistency_Check = np.zeros((self.test_Pattern_Count)).astype("float32")
        for (word, talker), index in self.test_Pattern_Index_Dict.items():
            self.test_Spectrogram_Pattern[index, :self.pattern_Dict[word, talker]["Cycle"], :] = self.pattern_Dict[word, talker]["Spectrogram"]
            cycle_Pattern_for_Cosistency_Check[index] = self.pattern_Dict[word, talker]["Cycle"]

        assert all(cycle_Pattern_for_Cosistency_Check == self.test_Cycle_Pattern)  #If cycle pattern does not match, model loaded a wrong metadata file.

    def Training_Pattern_Dict_Generate(self):
        '''
        When 'self.partial_Exclusion_in_Training' is 'P'(Pattern based), each talker's partial pattern will not be trained.
        When 'self.partial_Exclusion_in_Training' is 'T'(Talker based), a talker's all pattern will not be trained.
        When 'self.partial_Exclusion_in_Training' is 'M'(Mix based), each talker's partial pattern will not be trained and a talker's all pattern will not be trained.
        When 'self.partial_Exclusion_in_Training' is None, all pattern will be trained.
        '''
        self.training_Pattern_Dict = {}
        self.excluded_Pattern_Dict = {}

        if self.partial_Exclusion_in_Training is None:
            self.training_Pattern_Dict = self.pattern_Dict
            return

        talker_List = sorted(list(set([talker for word, talker in self.pattern_Dict.keys()])))
        shuffle(talker_List)
        word_List = sorted(list(self.word_Index_Dict.keys()))
        shuffle(word_List)
        print("A random number:", random.random())

        exclude_Size = len(word_List) // len(talker_List)

        if self.partial_Exclusion_in_Training.lower() == 'p':
            for talker_Index, talker in enumerate(talker_List):
                for word in word_List[:talker_Index * exclude_Size] + word_List[(talker_Index + 1) * exclude_Size:]:
                    self.training_Pattern_Dict[word, talker] = self.pattern_Dict[word, talker]
                for word in word_List[talker_Index * exclude_Size:(talker_Index + 1) * exclude_Size]:
                    self.excluded_Pattern_Dict[word, talker] = self.pattern_Dict[word, talker]
            return

        #Select excluded talker
        if not self.excluded_Talker is None:
            if not self.excluded_Talker.lower() in talker_List:
                raise Exception("The specified talker is not in list.")
        else:
            self.excluded_Talker = talker_List[-1]
        talker_List.remove(self.excluded_Talker)

        if self.partial_Exclusion_in_Training.lower() == 't':
            for word in self.word_Index_Dict.keys():
                for talker in talker_List:
                    self.training_Pattern_Dict[word, talker] = self.pattern_Dict[word, talker]
                self.excluded_Pattern_Dict[word, self.excluded_Talker] = self.pattern_Dict[word, self.excluded_Talker]
            return

        if self.partial_Exclusion_in_Training.lower() == 'm':
            for talker_Index, talker in enumerate(talker_List): #Pattern exclusion
                for word in word_List[:talker_Index * exclude_Size] + word_List[(talker_Index + 1) * exclude_Size:]:
                    self.training_Pattern_Dict[word, talker] = self.pattern_Dict[word, talker]
                for word in word_List[talker_Index * exclude_Size:(talker_Index + 1) * exclude_Size]:
                    self.excluded_Pattern_Dict[word, talker] = self.pattern_Dict[word, talker]
            for word in self.word_Index_Dict.keys():    #Talker exclusion
                self.excluded_Pattern_Dict[word, self.excluded_Talker] = self.pattern_Dict[word, self.excluded_Talker]
            return

        raise ValueError("Unsupported pattern exclusion mode")

    #Pattern making and inserting to queue
    def Pattern_Generate_Normal(self):
        #Batched Pattern Making
        if self.exclusion_Ignoring:
            pattern_Dict = self.pattern_Dict
        else:
            pattern_Dict = self.training_Pattern_Dict

        pattern_Count  = len(pattern_Dict)
        max_Cycle = np.max([x["Cycle"] for x in pattern_Dict.values()])

        spectrogram_Pattern = np.zeros((pattern_Count, max_Cycle, self.spectrogram_Size)).astype("float32")
        semantic_Pattern = np.zeros((pattern_Count, self.semantic_Size)).astype("float32")
        cycle_Pattern = np.zeros((pattern_Count)).astype("float32")

        for index, (word, talker) in enumerate(pattern_Dict.keys()):
            spectrogram_Pattern[index, :pattern_Dict[word, talker]["Cycle"], :] = pattern_Dict[word, talker]["Spectrogram"]
            semantic_Pattern[index, :] = pattern_Dict[word, talker]["Semantic"]
            cycle_Pattern[index] = pattern_Dict[word, talker]["Cycle"]

        #Queue
        for epoch in range(self.start_Epoch, self.max_Epoch):
            pattern_Index_List = list(range(pattern_Count))
            shuffle(pattern_Index_List)
            pattern_Index_Batch_List = [pattern_Index_List[x:x+self.batch_Size] for x in range(0, len(pattern_Index_List), self.batch_Size)]

            current_Index = 0
            is_New_Epoch = True
            while current_Index < len(pattern_Index_Batch_List):
                if len(self.pattern_Queue) >= self.max_Queue:
                    time.sleep(0.1)
                    continue

                selected_Spectrogram_Pattern = spectrogram_Pattern[pattern_Index_Batch_List[current_Index]]
                selected_Semantic_Pattern = semantic_Pattern[pattern_Index_Batch_List[current_Index]]
                selected_Cycle_Pattern = cycle_Pattern[pattern_Index_Batch_List[current_Index]]

                #For compatibility with the truncated mode
                tiled_Semantic_Pattern = np.tile(
                    np.expand_dims(selected_Semantic_Pattern, axis=1),
                    reps=[1, int(np.max(selected_Cycle_Pattern)), 1]
                    )

                new_Feed_Dict= dict(zip(
                    self.placeholder_List,
                    [
                        selected_Spectrogram_Pattern,
                        tiled_Semantic_Pattern,
                        selected_Cycle_Pattern
                        ]
                    ))
                self.pattern_Queue.append([epoch, is_New_Epoch, new_Feed_Dict])

                current_Index += 1
                is_New_Epoch = False

        self.is_Finished = True

    def Pattern_Generate_Truncated(self, truncation_Cycle = 10):
        if self.exclusion_Ignoring:
            pattern_Dict = self.pattern_Dict
        else:
            pattern_Dict = self.training_Pattern_Dict


        total_Cycle = np.sum([x["Spectrogram"].shape[0] for x in pattern_Dict.values()])
        chunk_Size = self.batch_Size * truncation_Cycle

        pattern_Key_List = [x for x in pattern_Dict.keys()]

        for epoch in range(self.start_Epoch, self.max_Epoch):
            shuffle(pattern_Key_List)
            series_Spectrogram_Pattern = np.vstack(
                [pattern_Dict[key]["Spectrogram"] for key in pattern_Key_List] +
                [np.zeros((chunk_Size - (total_Cycle % chunk_Size), self.spectrogram_Size))]
                )

            series_Semantic_Pattern = np.vstack(
                [np.tile(np.expand_dims(pattern_Dict[key]["Semantic"], axis=0), reps=[pattern_Dict[key]["Cycle"], 1])for key in pattern_Key_List] +
                [np.zeros((chunk_Size - (total_Cycle % chunk_Size), self.semantic_Size))]
                )

            current_Index = 0
            is_New_Epoch = True
            while current_Index < total_Cycle:
                if len(self.pattern_Queue) >= self.max_Queue:
                    time.sleep(0.1)
                    continue

                selected_Spectrogram_Pattern = np.reshape(
                    series_Spectrogram_Pattern[current_Index:current_Index + chunk_Size, :],
                    newshape = [self.batch_Size, truncation_Cycle, -1]
                    )
                selected_Semantic_Pattern = np.reshape(
                    series_Semantic_Pattern[current_Index:current_Index + chunk_Size, :],
                    newshape = [self.batch_Size, truncation_Cycle, -1]
                    )
                selected_Cycle_Pattern = np.array([10] * truncation_Cycle).astype("int32")

                new_Feed_Dict= dict(zip(
                    self.placeholder_List,
                    [
                        selected_Spectrogram_Pattern,
                        selected_Semantic_Pattern,
                        selected_Cycle_Pattern
                        ]
                    ))
                self.pattern_Queue.append([epoch, is_New_Epoch, new_Feed_Dict])

                current_Index += chunk_Size
                is_New_Epoch = False

        self.is_Finished = True

    #Pop a training pattern
    def Get_Pattern(self):
        while len(self.pattern_Queue) == 0: #When training speed is faster than making pattern, model should be wait.
            time.sleep(0.01)
        return self.pattern_Queue.popleft()

    #This function will be called only one time.
    def Test_Pattern_Generate(self):
        self.test_Pattern_Count  = len(self.pattern_Dict)
        self.test_Max_Cycle = np.max([x["Cycle"] for x in self.pattern_Dict.values()])

        self.test_Pattern_Index_Dict = {}
        self.test_Spectrogram_Pattern = np.zeros((self.test_Pattern_Count, self.test_Max_Cycle, self.spectrogram_Size)).astype("float32")
        self.test_Cycle_Pattern = np.zeros((self.test_Pattern_Count)).astype("float32")

        for index, (word, talker) in enumerate(self.pattern_Dict.keys()):
            self.test_Pattern_Index_Dict[word, talker] = index
            self.test_Spectrogram_Pattern[index, :self.pattern_Dict[word, talker]["Cycle"], :] = self.pattern_Dict[word, talker]["Spectrogram"]
            self.test_Cycle_Pattern[index] = self.pattern_Dict[word, talker]["Cycle"]

    #Return all patterns.
    def Get_Test_Pattern_List(self):
        pattern_Index_List = list(range(self.test_Pattern_Count))
        pattern_Index_Batch_List = [pattern_Index_List[x:x+self.batch_Size] for x in range(0, len(pattern_Index_List), self.batch_Size)]

        new_Feed_Dict_List = []

        for pattern_Index_Batch in pattern_Index_Batch_List:
            #Semantic pattern is not used in the test.
            new_Feed_Dict= {
                self.placeholder_List[0]: self.test_Spectrogram_Pattern[pattern_Index_Batch],
                self.placeholder_List[2]: np.ones((len(pattern_Index_Batch))) * self.test_Max_Cycle
                }
            new_Feed_Dict_List.append(new_Feed_Dict)

        return new_Feed_Dict_List

    #Voice file -> pattern.
    #In current study, hidden analysis use this function.
    def Get_Test_Pattern_from_Voice(self, voice_File_Path_List, window_Length = 10):
        spectrogram_List = []
        cycle_List = []

        for voice_File in voice_File_Path_List:
            sig = librosa.core.load(voice_File, sr = 8000)[0]
            spectrogram_Array = np.transpose(spectrogram(sig, frame_shift_ms = window_Length, frame_length_ms = window_Length))

            spectrogram_List.append(spectrogram_Array)
            cycle_List.append(spectrogram_Array.shape[0])

        spectrogram_Pattern = np.zeros((len(spectrogram_List), max(cycle_List), self.spectrogram_Size)).astype("float32")
        for index, spectrogram_Array in enumerate(spectrogram_List):
            spectrogram_Pattern[index, :cycle_List[index], :] = spectrogram_Array
        cycle_Pattern = np.hstack(cycle_List).astype("float32")

        #Semantic pattern is not used in the test.
        new_Feed_Dict= {
            self.placeholder_List[0]: spectrogram_Pattern,
            self.placeholder_List[2]: cycle_Pattern
            }

        return new_Feed_Dict
