
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import random
import time
import torch
import torch.nn as TorchNeuralNetwork
import matplotlib.pyplot as plot
from math import sqrt as SquareRoot
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoTokenizer
import gc
gc.enable()
CUDA = "cuda"
CPU = "cpu"
warnings.filterwarnings('ignore')

RBS_MODEL = "transformers_roberta_base_squad2"
RBS_MODEL_TOKENIZER = "transformers_roberta_base_squad2"

RLS_MODEL = "transformers_roberta_large_squad2"
RLS_MODEL_TOKENIZER = "transformers_roberta_large_squad2"

Deberta_MODEL = "transformers_deberta_large"
Deberta_MODEL_TOKENIZER = "transformers_deberta_large"

Electra_MODEL = "transformers_electra_large_discriminator"
Electra_MODEL_TOKENIZER = "transformers_electra_large_discriminator"

MODEL_BATCH = 16

MODEL_LENGTH = 248
MODEL_PAD = "max_length"
MODEL_NORM = 1e-7
MODEL_PROB = 0.0
MODEL_HIDDEN_1 = 1
MODEL_HIDDEN_2 = 512
MODEL_SEED_VALUE = 1000
MODEL_CPU_WORKERS = 2
DIMENSION_1 = 768
DIMENSION_2 = 1024

#roberta-base-squad2 tokenizer
rbs_model_token = AutoTokenizer.from_pretrained(RBS_MODEL_TOKENIZER)

#roberta-large-squad2 tokenizer
rls_model_token = AutoTokenizer.from_pretrained(RLS_MODEL_TOKENIZER)

#deberta tokenizer
deberta_model_token = AutoTokenizer.from_pretrained(Deberta_MODEL_TOKENIZER)

#electra tokenizer
electra_model_token = AutoTokenizer.from_pretrained(Electra_MODEL_TOKENIZER)

@st.cache()

class CommonLit_Architecture(TorchNeuralNetwork.Module): 
    def __init__(self,commonlit_model):
        super().__init__()

        #checking for transformer model
        if (commonlit_model == "rbs") :
          MODEL = RBS_MODEL
          MODEL_HIDDEN_3 = DIMENSION_1

        elif (commonlit_model == "rls") :
          MODEL = RLS_MODEL
          MODEL_HIDDEN_3 = DIMENSION_2

        elif (commonlit_model == "deberta") :
          MODEL = Deberta_MODEL
          MODEL_HIDDEN_3 = DIMENSION_2

        elif (commonlit_model == "electra") :
          MODEL = Electra_MODEL
          MODEL_HIDDEN_3 = DIMENSION_2

        #Download configuration from huggingface.co
        commonlit_cfg = AutoConfig.from_pretrained(MODEL)

        #Chaning parameters of configuration file
        commonlit_cfg.update({"output_hidden_states":True, "hidden_dropout_prob": MODEL_PROB,"layer_norm_eps": MODEL_NORM}) 

        #The architecture we want to use can be get from the name or the path of the pretrained model we are supplying to the from_pretrained method.
        #AutoClasses are here to do this job for us so that we can automatically retrieve the relevant model given the name/path to the pretrained weights/config/vocabulary:
        self.model = AutoModel.from_pretrained(MODEL, config=commonlit_cfg) 

        #Attention_Head(2nd Fine Tuning Strategy)     
        self.attn_head = TorchNeuralNetwork.Sequential(            
            TorchNeuralNetwork.Linear(MODEL_HIDDEN_3, MODEL_HIDDEN_2),            
            TorchNeuralNetwork.Tanh(),                       
            TorchNeuralNetwork.Linear(MODEL_HIDDEN_2, MODEL_HIDDEN_1),
            TorchNeuralNetwork.Softmax(dim=MODEL_HIDDEN_1)
        )

        #regression layer
        self.linear_reg = TorchNeuralNetwork.Sequential(                        
            TorchNeuralNetwork.Linear(MODEL_HIDDEN_3, MODEL_HIDDEN_1)                        
        ) 
        
    #forward function
    def forward(self, commonlit_encode, commonlit_attn,Concatenate_Last_4_Layers):

        #output of last layer of transformer
        model_hidden = self.model(input_ids=commonlit_encode,attention_mask=commonlit_attn)  #shape: Batch_Size*Sequence_Length*Hidden_Size     

        if not Concatenate_Last_4_Layers:
          #Use only output of last layer of transformer
          model_hidden_stack_mean = model_hidden.hidden_states[-1]
        else:
          #Concatenate Last 4 layers(1st Fine Tuning Strategy)
          model_hidden_stack = torch.stack([model_hidden.hidden_states[-1],model_hidden.hidden_states[-2],model_hidden.hidden_states[-3],model_hidden.hidden_states[-4]]) 
                                                                                                                #shape: 4*Batch_Size*Sequence_Length*Hidden_Size 
          model_hidden_stack_mean =  torch.mean(model_hidden_stack, 0) #shape:Batch_Size*Sequence_Length*Hidden_Size

        #Getting weigths from Attention_Head Network
        model_hidden_weight = self.attn_head(model_hidden_stack_mean) #shape:*Batch_Size*Sequence_Length*1
                
        # Multiplying weigths(model_hidden_weight) from Attention_Head Network by Output of last layer of network(model_hidden_stack_mean). 
        # model_hidden_weight * model_hidden_stack_mean
        # Then averaging the tensor across sequence Length Dimension.
        # The output(model_hidden_vec) will be passed to Regression Layer to get prediction score.
        model_hidden_vec = torch.sum(model_hidden_weight * model_hidden_stack_mean, dim=1) #shape:*Batch_Size*Hidden_Size     
    
        return self.linear_reg(model_hidden_vec)

def commonlit_test(commonlit_arch, commonlit_iterable,Concatenate_Last_4_Layers):

  #sets model in evaluation mode
  commonlit_arch.eval()

  commonlit_iter_len = len(commonlit_iterable.dataset)
  commonlit_pred = np.zeros(commonlit_iter_len)    
  commonlit_row = 0

  #context-manager that disabled gradient calculation  
  with torch.no_grad():

    #enumerating over test data
    for _, (commonlit_encode, commonlit_attn) in enumerate(commonlit_iterable):
      commonlit_attn = commonlit_attn.to(CUDA)
      commonlit_encode = commonlit_encode.to(CUDA)        

      #getting return from commonlit architecture                  
      commonlit_return = commonlit_arch(commonlit_encode, commonlit_attn,Concatenate_Last_4_Layers)                        

      # Flattens input by reshaping it into a one-dimensional tensor. 
      commonlit_flatten = commonlit_return.flatten()
      commonlit_pred[commonlit_row : commonlit_row + commonlit_return.shape[0]] = commonlit_flatten.to(CPU)
      commonlit_row = commonlit_return.shape[0] + commonlit_row

  return commonlit_pred


class CommonLit_Item(Dataset):
  #run once when instantiating the Dataset object
  def __init__(self, data,commonlit_model,io=False,):
    super().__init__()

    #checking for transformer model
    if (commonlit_model == "rbs") :
      model_token = rbs_model_token

    elif (commonlit_model == "rls") :
      model_token = rls_model_token

    elif (commonlit_model == "deberta") :
      model_token = deberta_model_token

    elif (commonlit_model == "electra") :
      model_token = electra_model_token

    self.io = io
    if not self.io:
      self.read_ease = torch.tensor(data.target.values, dtype=torch.float32) 

    self.excerpt = data.excerpt.tolist()
    self.ec = model_token.batch_encode_plus(self.excerpt,padding = MODEL_PAD,max_length = MODEL_LENGTH,truncation = True,return_attention_mask=True)

    self.item_data = data        

  #loads and returns a sample from the dataset at the given index.                                   
  def __getitem__(self, index): 
    commonlit_attn = torch.tensor(self.ec['attention_mask'][index])       
    commonlit_encode = torch.tensor(self.ec['input_ids'][index])
        
    if not self.io:
      commonlit_read_ease = self.read_ease[index]
      return (commonlit_encode, commonlit_attn, commonlit_read_ease)       
    else:
      return (commonlit_encode, commonlit_attn)   

  #returns the number of samples in our dataset.  
  def __len__(self):
    return len(self.item_data)

def commonlit_ensemble_predictions(X,commonlit_model) :
  test_data = X
  commonlit_result = np.zeros((5, len(test_data)))

  #creating datasets
  commonlit_item_pred = CommonLit_Item(test_data,commonlit_model,io=True)

  #creating iterable datasets
  commonlit_iterable_pred = DataLoader(commonlit_item_pred, batch_size=MODEL_BATCH,drop_last=False, shuffle=False, num_workers=MODEL_CPU_WORKERS)

  #going through each of saved models
  for arch_save_model in range(5):   
    arch_save_model_one = arch_save_model + 1          
    arch_dir = "cs_"+commonlit_model+f"_models/model_{arch_save_model_one}.pth"
    print("{0} Save Model Path {1}".format(commonlit_model,arch_dir))

    #initialize commonlit architecture                    
    commonlit_arch = CommonLit_Architecture(commonlit_model)

    #Loads an object saved with torch.save() from a file.
    commonlit_load_dir = torch.load(arch_dir)

    #Loads a modelÃ¢â‚¬â„¢s parameter dictionary
    commonlit_arch.load_state_dict(commonlit_load_dir) 

    #Sending to CUDA Device   
    commonlit_arch.to(CUDA)
    
    if(commonlit_model == "electra"):
      Concatenate_Last_4_Layers = False
    else:
      Concatenate_Last_4_Layers = True

    commonlit_result[arch_save_model] = commonlit_test(commonlit_arch, commonlit_iterable_pred,Concatenate_Last_4_Layers)
    
    del commonlit_arch
    gc.collect()
  
  predictions_transformers = commonlit_result.mean(axis=0)    
  return predictions_transformers

def final_fun_1(X):

  #Roberta-Base-Squad2
  ensemble_rbs_pred = commonlit_ensemble_predictions(X,"rbs")

  #Roberta-Large-Squad2
  ensemble_rls_pred = commonlit_ensemble_predictions(X,"rls")

  #Deberta
  ensemble_deberta_pred = commonlit_ensemble_predictions(X,"deberta")

  #Electra
  ensemble_electra_pred = commonlit_ensemble_predictions(X,"electra")

  ensemble_final_pred = (ensemble_rbs_pred + ensemble_rls_pred + ensemble_deberta_pred + ensemble_electra_pred)/4

  return ensemble_final_pred
  
def commonlit_reading_score(text):
  streamlit_data = pd.DataFrame()
  streamlit_data["id"] = ["c12129c31"]
  streamlit_data["url_legal"] = [None]
  streamlit_data["license"] = [None]
  streamlit_data["excerpt"] = [text]
  commonlit_predictions =  final_fun_1(streamlit_data)
  return round(commonlit_predictions[0],2)
    
def main():       

    commonlit_box_1 = """ 
    <div style ="background-color:red;padding:5px"> 
    <h1 style ="color:black;text-align:center;">Project 2</h1> 
    <h1 style ="color:black;text-align:center;">CommonLit Reading Ease Score</h1>
    </div> 
    """
    st.markdown(commonlit_box_1, unsafe_allow_html = True) 
    
    commonLit_reading_passage = st.text_input("Please enter text", max_chars=1500, help="The reading ease score will be generated after wait of 7-8mins")

    if st.button("Get_Reading_Score"): 
        reading_score = commonlit_reading_score(commonLit_reading_passage) 
        st.success('Reading Ease Score is {}'.format(reading_score))
     
if __name__=='__main__': 
    main()