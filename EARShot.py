import numpy as np
import tensorflow as tf
import random
import _pickle as pickle
from tensorflow.contrib.seq2seq import BasicDecoder, TrainingHelper, InferenceHelper, dynamic_decode
from tensorflow.contrib.rnn import LSTMCell, GRUCell, BasicRNNCell, LSTMStateTuple
from threading import Thread
import time, os, sys, argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from Pattern_Feeder import Pattern_Feeder
from SCRNCell import SCRNCell, SCRNStateTuple
from Customized_Functions import Correlation2D, Batch_Correlation2D, Batch_Cosine_Similarity2D, MDS
#import ctypes #If you run this script in the linux or mac, remove this line.
import fwr13y.d9m.tensorflow as tf_determinism

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    # https://github.com/NVIDIA/framework-reproducibility/blob/master/doc/d9m/tensorflow_status.md#TF_CUDNN_DETERMINISTIC
    tf_determinism.enable_determinism()  # Apply NVIDIA determinism patch
    
class Model:
    #Initialize the model
    def __init__(
        self,
        hidden_Size,
        learning_Rate,
        pattern_File,
        pattern_Mode,
        batch_Size,
        start_Epoch,
        max_Epoch,
        partial_Exclusion_in_Training,
        excluded_Talker,
        exclusion_Ignoring,
        metadata_File,
        hidden_Type,
        hidden_Reset,
        extract_Dir
        ):
        #ctypes.windll.kernel32.SetConsoleTitleW(extract_Dir.replace("_", ":").replace(".", " "))   #If you run this script in the linux or mac, remove this line.

        self.tf_Session = tf.Session()        

        self.acoustic_Size = 256
        self.semantic_Size = 300
        self.hidden_Size = hidden_Size
        self.learning_Rate = learning_Rate
        self.hidden_Type = hidden_Type
        self.hidden_Reset = hidden_Reset
        self.extract_Dir = extract_Dir
        
        self.Placeholder_Generate()
        
        #Pattern data is generated from other thread.
        self.pattern_Feeder = Pattern_Feeder(
            placeholder_List = [self.acoustic_Placeholder, self.semantic_Placeholder, self.length_Placeholder],
            pattern_File = pattern_File,
            pattern_Mode = pattern_Mode,
            partial_Exclusion_in_Training = partial_Exclusion_in_Training,
            excluded_Talker = excluded_Talker,
            exclusion_Ignoring = exclusion_Ignoring,
            batch_Size = batch_Size,
            start_Epoch = start_Epoch,
            max_Epoch = max_Epoch,
            metadata_File = metadata_File
            )

        self.Tensor_Generate()

        self.tf_Saver = tf.train.Saver(max_to_keep=0)

    #Initialize the tensor placeholder
    def Placeholder_Generate(self):
        with tf.variable_scope('placeHolders') as scope:
            self.acoustic_Placeholder = tf.placeholder(tf.float32, shape=(None, None, self.acoustic_Size), name = "acoustic_Placeholder") #(batch, length, size)            
            self.semantic_Placeholder = tf.placeholder(tf.float32, shape=(None, None, self.semantic_Size), name = "semantic_Placeholder") #(batch, size)
            self.length_Placeholder = tf.placeholder(tf.int32, shape=(None,), name = "length_Placeholder")   #(batch)

    #Tensor making for training and test
    def Tensor_Generate(self):
        with tf.variable_scope('model') as scope:
            batch_Size = tf.shape(self.acoustic_Placeholder)[0]

            #This model use only training helper.
            helper = TrainingHelper(
                inputs=self.acoustic_Placeholder,
                sequence_length = self.length_Placeholder,
                time_major = False
                )

            #RNN. Model can select four types hidden.
            #Previous RNN state is for the no reset.       
            if self.hidden_Type == "LSTM":
                rnn_Cell = LSTMCell(self.hidden_Size)
                previous_RNN_State = tf.Variable(
                    initial_value = LSTMStateTuple(
                        c = tf.zeros(shape=(self.pattern_Feeder.batch_Size, self.hidden_Size)), 
                        h = tf.zeros(shape=(self.pattern_Feeder.batch_Size, self.hidden_Size))
                        ),
                    trainable = False
                    )
                decoder_Initial_State = LSTMStateTuple(
                    c=previous_RNN_State[0][:batch_Size],
                    h=previous_RNN_State[1][:batch_Size]
                    )
            elif self.hidden_Type == "SCRN":
                rnn_Cell = SCRNCell(self.hidden_Size)
                previous_RNN_State = tf.Variable(
                    initial_value = SCRNStateTuple(
                        s = tf.zeros(shape=(self.pattern_Feeder.batch_Size, self.hidden_Size)), 
                        h = tf.zeros(shape=(self.pattern_Feeder.batch_Size, self.hidden_Size))
                        ),
                    trainable = False
                    )
                decoder_Initial_State = SCRNStateTuple(
                    s=previous_RNN_State[0][:batch_Size],
                    h=previous_RNN_State[1][:batch_Size]
                    )
            elif self.hidden_Type in ["GRU", "BPTT"]:
                if self.hidden_Type == "GRU":
                    rnn_Cell = GRUCell(self.hidden_Size)
                elif self.hidden_Type == "BPTT":
                    rnn_Cell = BasicRNNCell(self.hidden_Size)
                previous_RNN_State = tf.Variable(
                    initial_value = tf.zeros(shape=(self.pattern_Feeder.batch_Size, self.hidden_Size)),
                    trainable = False
                    )
                decoder_Initial_State = previous_RNN_State[:batch_Size]

            decoder = BasicDecoder(
                cell=rnn_Cell, 
                helper=helper, 
                initial_state=decoder_Initial_State
                )
            outputs, final_State, _ = dynamic_decode(
                decoder = decoder,
                output_time_major = False,
                impute_finished = True
                )
            
            hidden_Activation = outputs.rnn_output

            #Semantic   (hidden_size -> semantic_size)
            semantic_Logits = tf.layers.dense(
                inputs = hidden_Activation,
                units = self.semantic_Size,                
                use_bias=True,
                name = "semantic_Logits"
                )

        #Back-prob.
        with tf.variable_scope('training_Loss') as scope:
            loss_Calculation = tf.nn.sigmoid_cross_entropy_with_logits(                
                labels = self.semantic_Placeholder,
                logits = semantic_Logits
                )

            loss = tf.reduce_mean(loss_Calculation)
            loss_Display = tf.reduce_mean(loss_Calculation)    #This is for the display. There is no meaning.
            
            global_Step = tf.Variable(0, name='global_Step', trainable = False)

            #Noam decay of learning rate
            step = tf.cast(global_Step + 1, dtype=tf.float32)
            warmup_Steps = 4000.0
            learning_Rate = self.learning_Rate * warmup_Steps ** 0.5 * tf.minimum(step * warmup_Steps**-1.5, step**-0.5)

            #Weight update. We use the ADAM optimizer
            optimizer = tf.train.AdamOptimizer(learning_Rate)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            clipped_Gradients, global_Norm = tf.clip_by_global_norm(gradients, 1.0)
            optimize = optimizer.apply_gradients(zip(clipped_Gradients, variables), global_step=global_Step)

            #For no reset. Model save the rnn states.
            if self.hidden_Type == "LSTM":
                rnn_State_Assign = tf.assign(
                    ref= previous_RNN_State,
                    value = LSTMStateTuple(
                       c = tf.concat([final_State[0][:batch_Size], previous_RNN_State[0][batch_Size:]], axis = 0),
                       h = tf.concat([final_State[1][:batch_Size], previous_RNN_State[1][batch_Size:]], axis = 0)
                       )
                    )
            if self.hidden_Type == "SCRN":
                rnn_State_Assign = tf.assign(
                    ref= previous_RNN_State,
                    value = SCRNStateTuple(
                       s = tf.concat([final_State[0][:batch_Size], previous_RNN_State[0][batch_Size:]], axis = 0),
                       h = tf.concat([final_State[1][:batch_Size], previous_RNN_State[1][batch_Size:]], axis = 0)
                       )
                    )
            elif self.hidden_Type in ["GRU", "BPTT"]:
                rnn_State_Assign = tf.assign(
                    ref= previous_RNN_State,
                    value = tf.concat([final_State[:batch_Size], previous_RNN_State[batch_Size:]], axis = 0)
                    )

        with tf.variable_scope('test') as scope:
            #In test, previous hidden state should be zero. Thus, the saved values should be backup and become zero.
            if self.hidden_Type == "LSTM":
                backup_RNN_State = tf.Variable(
                    initial_value = LSTMStateTuple(
                        c = tf.zeros(shape=(self.pattern_Feeder.batch_Size, self.hidden_Size)), 
                        h = tf.zeros(shape=(self.pattern_Feeder.batch_Size, self.hidden_Size))
                        ),
                    trainable = False
                    )
            elif self.hidden_Type == "SCRN":
                backup_RNN_State = tf.Variable(
                    initial_value = SCRNStateTuple(
                        s = tf.zeros(shape=(self.pattern_Feeder.batch_Size, self.hidden_Size)), 
                        h = tf.zeros(shape=(self.pattern_Feeder.batch_Size, self.hidden_Size))
                        ),
                    trainable = False
                    )
            elif self.hidden_Type in ["GRU", "BPTT"]:
                backup_RNN_State = tf.Variable(
                    initial_value = tf.zeros(shape=(self.pattern_Feeder.batch_Size, self.hidden_Size)),
                    trainable = False
                    )

            backup_RNN_State_Assign = tf.assign(
                ref= backup_RNN_State,
                value = previous_RNN_State
                )
            with tf.control_dependencies([backup_RNN_State_Assign]):
                if self.hidden_Type == "LSTM":
                    zero_RNN_State_Assign = tf.assign(
                        ref= previous_RNN_State,
                        value = LSTMStateTuple(
                            c = tf.zeros(shape=(self.pattern_Feeder.batch_Size, self.hidden_Size)), 
                            h = tf.zeros(shape=(self.pattern_Feeder.batch_Size, self.hidden_Size))
                            )
                        )
                elif self.hidden_Type == "SCRN":
                    zero_RNN_State_Assign = tf.assign(
                        ref= previous_RNN_State,
                        value = SCRNStateTuple(
                            s = tf.zeros(shape=(self.pattern_Feeder.batch_Size, self.hidden_Size)), 
                            h = tf.zeros(shape=(self.pattern_Feeder.batch_Size, self.hidden_Size))
                            )
                        )
                elif self.hidden_Type in ["GRU", "BPTT"]:
                    zero_RNN_State_Assign = tf.assign(
                        ref= previous_RNN_State,
                        value = tf.zeros(shape=(self.pattern_Feeder.batch_Size, self.hidden_Size))
                        )
            
            restore_RNN_State_Assign = tf.assign(
                ref= previous_RNN_State,
                value = backup_RNN_State
                )

            semantic_Activation = tf.nn.sigmoid(semantic_Logits)
                    
        self.training_Tensor_List = [global_Step, learning_Rate, loss_Display, optimize]
        if not self.hidden_Reset:   #If hidden is not reset, model use the save function.
            self.training_Tensor_List.append(rnn_State_Assign)
        
        self.test_Mode_Turn_On_Tensor_List = [backup_RNN_State_Assign, zero_RNN_State_Assign]  #Hidden state backup
        self.test_Mode_Turn_Off_Tensor_List = [restore_RNN_State_Assign]   #Hidden state restore

        self.test_Tensor_List = [global_Step, semantic_Activation] #In test, we only need semantic activation
        
        self.hidden_Plot_Tensor_List = [tf.transpose(hidden_Activation, perm=[0, 2, 1])]   #In hidden analysis, we only need hidden activation.

        self.tf_Session.run(tf.global_variables_initializer()) #Initialize the weights
        
    #Checkpoint load
    def Restore(self, force_Overwrite = False):
        if not os.path.exists(self.extract_Dir + "/Checkpoint"):
            print("There is no checkpoint.")
            return
        if not force_Overwrite:
            latest_Checkpoint = tf.train.latest_checkpoint(self.extract_Dir + "/Checkpoint")
            print("Lastest checkpoint:", latest_Checkpoint)
            if latest_Checkpoint is not None:
                latest_Trained_Epoch = int(latest_Checkpoint[latest_Checkpoint.index("Checkpoint-") + 11:])
                if latest_Trained_Epoch > self.pattern_Feeder.start_Epoch:
                    try:
                        input("\n".join([
                        "WARNING!",
                        "THE START EPOCH IS SMALLER THAN THE TRAINED MODEL.",
                        "CHANGE THE START EPOCH OR THE FOLDER NAME OF PREVIOUS MODEL TO PREVENT TO OVERWRITE.",
                        "TO STOP, PRESS 'CTRL + C'.",
                        "TO CONTINUE, PRESS 'ENTER'.\n"
                        ]))
                    except KeyboardInterrupt:
                        print("Stopped.")
                        sys.exit()

        checkpoint = self.extract_Dir + "/Checkpoint/Checkpoint-" + str(self.pattern_Feeder.start_Epoch)
        try:
            self.tf_Saver.restore(self.tf_Session, checkpoint)
        except tf.errors.NotFoundError:
            print("here is no checkpoint about the start epoch. Stopped.")
            sys.exit()
        print("Checkpoint '", checkpoint, "' is loaded.")

    def TEST(self):
        # Get the list of trainable variables
        trainable_vars = tf.trainable_variables()

        # Evaluate the variables to get their values
        weights_values = self.tf_Session.run(trainable_vars)

        # Print the weights
        for var, val in zip(trainable_vars, weights_values):
            print(f"Variable: {var.name}, Value: {val}")
        
    #Training
    def Train(self, test_Timing, checkpoint_Timing = 1000):
        if not os.path.exists(self.extract_Dir + "/Checkpoint"):
            os.makedirs(self.extract_Dir + "/Checkpoint")
        checkpoint_Path = self.extract_Dir + "/Checkpoint/Checkpoint"

        while not self.pattern_Feeder.is_Finished or len(self.pattern_Feeder.pattern_Queue) > 0:    #When there is no more training pattern, the train function will be done.
            current_Epoch, is_New_Epoch, feed_Dict = self.pattern_Feeder.Get_Pattern()            

            #Initial test and save
            if is_New_Epoch and current_Epoch % test_Timing == 0:
                self.Test(epoch=current_Epoch)
            if is_New_Epoch and current_Epoch % checkpoint_Timing == 0:
                self.tf_Saver.save(self.tf_Session, checkpoint_Path, global_step = current_Epoch)
                print("Checkpoint saved")

            start_Time = time.time()
            global_Step, learning_Rate, training_Loss = self.tf_Session.run(
                fetches = self.training_Tensor_List,
                feed_dict = feed_Dict
                )[:3]

            print(
                "Spent_Time:", np.round(time.time() - start_Time, 3), "\t",
                "Global_Step:", global_Step, "\t",
                "Epoch:", current_Epoch, "\t",
                "Learning_Rate:", learning_Rate, "\n",
                "Training_Loss:", np.mean(training_Loss)#" ".join(["%0.5f" % x for x in training_Loss])
                )

        #Final test and save
        test_Thread = self.Test(epoch=current_Epoch + 1)
        self.tf_Saver.save(self.tf_Session, checkpoint_Path, global_step = current_Epoch + 1)
        print("Checkpoint saved")

        test_Thread.join() #Wait unitl finishing the test and extract the data.

    #Test
    def Test(self, epoch):
        self.tf_Session.run(self.test_Mode_Turn_On_Tensor_List) #Backup the hidden state

        semantic_Activation_List = []

        test_Feed_Dict_List = self.pattern_Feeder.Get_Test_Pattern_List()        

        for feed_Dict in test_Feed_Dict_List:
            global_Step, semantic_Activation = self.tf_Session.run(
                fetches = self.test_Tensor_List,
                feed_dict = feed_Dict
                )                        
            padding_Array = np.zeros((semantic_Activation.shape[0], self.pattern_Feeder.test_Max_Cycle, semantic_Activation.shape[2])) #Padding is for stacking the result data.
            padding_Array[:, :semantic_Activation.shape[1], :] = semantic_Activation
            semantic_Activation_List.append(padding_Array)

        self.tf_Session.run(self.test_Mode_Turn_Off_Tensor_List)     #Restore the hidden state
         
        extract_Thread = Thread(target=self.Extract, args=(np.vstack(semantic_Activation_List).astype("float32"), epoch))
        extract_Thread.start()

        return extract_Thread
             
    #Data extract
    def Extract(self, semantic_Activation, epoch):
        if not os.path.exists(self.extract_Dir + "/Result"):
            os.makedirs(self.extract_Dir + "/Result")

        #If there is no metadata, save the metadata
        #In metadata, there are several basic hyper parameters, and the pattern information for result analysis.        
        if not os.path.isfile(self.extract_Dir + "/Result/Metadata.pickle"):
            metadata_Dict = {}
            metadata_Dict["Acoustic_Size"] = self.acoustic_Size
            metadata_Dict["Semantic_Size"] = self.semantic_Size
            metadata_Dict["Hidden_Size"] = self.hidden_Size
            metadata_Dict["Learning_Rate"] = self.learning_Rate
            metadata_Dict["Hidden_Type"] = self.hidden_Type
            
            metadata_Dict["Pronunciation_Dict"] = self.pattern_Feeder.pronunciation_Dict
            metadata_Dict["Word_Index_Dict"] = self.pattern_Feeder.word_Index_Dict
            metadata_Dict["Category_Dict"] = self.pattern_Feeder.category_Dict
            
            metadata_Dict["Pattern_Index_Dict"] = self.pattern_Feeder.test_Pattern_Index_Dict
            metadata_Dict["Target_Array"] = self.pattern_Feeder.target_Array   #[Pattern, 300]
            metadata_Dict["Cycle_Array"] = self.pattern_Feeder.test_Cycle_Pattern  #[Pattern]

            metadata_Dict["Trained_Pattern_List"] = sorted(list(self.pattern_Feeder.training_Pattern_Dict.keys()), key=lambda x: x[0]+"-"+x[1]) #'Trained' category patterns
            metadata_Dict["Excluded_Pattern_List"] = sorted(list(self.pattern_Feeder.excluded_Pattern_Dict.keys()), key=lambda x: x[0]+"-"+x[1])    #'Excluded words' and 'excluded talkers' patterns
            metadata_Dict["Excluded_Talker"] = self.pattern_Feeder.excluded_Talker    #'Excluded words' and 'excluded talkers' patterns
            
            with open(self.extract_Dir + "/Result/Metadata.pickle", "wb") as f:
                pickle.dump(metadata_Dict, f, protocol=0)

        result_Dict = {}
        result_Dict["Epoch"] = epoch
        result_Dict["Result"] = semantic_Activation
        result_Dict["Exclusion_Ignoring"] = self.pattern_Feeder.exclusion_Ignoring
                
        with open(self.extract_Dir + "/Result/{:06d}.pickle".format(epoch), "wb") as f:
            pickle.dump(result_Dict, f, protocol=0)

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-ht", "--hidden_type", required=True)  #LSTM, GRU, SCRN, BPTT
    argParser.add_argument("-hu", "--hidden_unit", required=True)  #int
    argParser.add_argument("-tt", "--test_timing", required=True)  #int
    argParser.add_argument("-se", "--start_epoch", required=False) #When you want to load the model, you should assign this parameter with 'metadata_file'. Basic is 0.
    argParser.set_defaults(start_epoch = "0")
    argParser.add_argument("-me", "--max_epoch", required=False)   #Finishing epoch
    argParser.set_defaults(max_epoch = "20000")
    argParser.add_argument("-em", "--exclusion_mode", required=False)  #P: Pattern based, T: Talker based, M: Mix, or None.
    argParser.set_defaults(exclusion_mode = None)
    argParser.add_argument("-et", "--exclusion_talker", required=False) #The assigned talker's all patterns were excluded. This is only for the T or M mode. If you does not assign and model is 'T' or 'M', Model select randomly one talker.
    argParser.set_defaults(exclusion_talker = None)
    argParser.add_argument("-ei", "--exclusion_ignoring", action='store_true') #When this parameter is True, model will train all pattern including the excluded patterns.
    argParser.set_defaults(exclusion_ignoring = False)
    argParser.add_argument("-mf", "--metadata_file", required=False)   #If you want to load the model, you should assign this parameter with 'start_epoch'.
    argParser.set_defaults(metadata_file = None)
    argParser.add_argument("-idx", "--index", required=False)  #This is just for identifier. This parameter does not affect the model's performance
    argParser.set_defaults(idx = None)
    argument_Dict = vars(argParser.parse_args())
    
    hidden_Type = argument_Dict["hidden_type"].upper()
    hidden_Unit = int(argument_Dict["hidden_unit"])
    test_Timing = int(argument_Dict["test_timing"])
    start_Epoch = int(argument_Dict["start_epoch"])
    max_Epoch = int(argument_Dict["max_epoch"])
    exclusion_Mode = argument_Dict["exclusion_mode"]
    exclusion_Talker = argument_Dict["exclusion_talker"].upper()
    exclusion_Ignoring = argument_Dict["exclusion_ignoring"]

    metadata_File = argument_Dict["metadata_file"]
    simulation_Index = argument_Dict["index"]

    set_seed(42)
    
    #Pattern file is including spectrogram, semantic, cycle, pattern index, pronunciation dict, and phonetic competator information.
    #This method improves pattern generating speed, but it cannot be applied to the big lexicon project.
    file_Name = "Pattern_Dict.IM_Spectrogram.OM_SRV.AN_10.Size_10000.WL_10.pickle"

    extract_Dir_List = ["/scratch/jsm04005/fie24002/paper2020/Results/IDX_{}/HM_{}".format(simulation_Index, hidden_Type)]
    extract_Dir_List.append("H_{}".format(hidden_Unit))
    extract_Dir_List.append("EM_{}".format(exclusion_Mode))
    if not exclusion_Talker is None:
        extract_Dir_List.append("ET_{}".format(exclusion_Talker))
    if not simulation_Index is None:
        extract_Dir_List.append("IDX_{}".format(simulation_Index))
    extract_Dir = ".".join(extract_Dir_List)

    new_Model = Model(
        hidden_Size= hidden_Unit,
        learning_Rate= 0.002, 
        pattern_File=file_Name, 
        pattern_Mode = "Normal", #"Normal" or "Truncated",
        partial_Exclusion_in_Training = exclusion_Mode,
        excluded_Talker = exclusion_Talker,
        exclusion_Ignoring = exclusion_Ignoring,
        batch_Size= 1000, 
        start_Epoch=start_Epoch,    #For restore
        max_Epoch=max_Epoch,
        metadata_File= metadata_File,
        hidden_Type = hidden_Type,
        hidden_Reset = False,
        extract_Dir=extract_Dir
        )
    # new_Model.TEST()
    new_Model.Restore(force_Overwrite=True)
    new_Model.Train(test_Timing=test_Timing, checkpoint_Timing=test_Timing)
    #new_Model.Test(start_Epoch)