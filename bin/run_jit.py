
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import os
from kws_model import *
from encoder import *
from decoder import *
from CTC import *

import numpy as np

def getword_from_id(wordindex,hyps,vocab_size):
    word_seq=[]
    for i in hyps:
        if(i>1 and i<vocab_size-1):
            word_seq.append(wordindex[i])
    print(word_seq)


encoder = conformerencoder(80,2,80,train_flag=False)
ctc=CTC(16,80)
decoder=transformerdecoder(vocab_size=16,encoder_output_size=80,num_blocks=2,linear_units=80)

model=KWSModel(vocab_size=16,encoder=encoder,ctc=ctc,decoder=decoder)
checkpoit=torch.load('./kwsaed.pt',map_location='cpu')
model.load_state_dict(checkpoit,strict=False)
# store word id
word_index={}
# Count decode result

with open('.//conf//wordid.txt') as char_map:
    for line in char_map:
        word,index=line.split()
        word_index[int(index)]=word
       
wavdir='.//testwav//eight'

file_paths=[]

for root, directories, files in os.walk(wavdir):
    for filename in files:
        filepath = os.path.join(root, filename)
        file_paths.append(filepath)


for wave in file_paths:
    audio_data,samplerate=torchaudio.load(wave)
    print(wave)
   
    mfdata=kaldi.fbank(audio_data,num_mel_bins=80,
                        dither=0.0,energy_floor=0.0,
                        sample_frequency=samplerate)   
    m,n=mfdata.size()
    
    inputdata=mfdata.reshape(1,m,n)
    input_len=torch.zeros((1))
    input_len[0]=m
    ctcpro=model.ctc_greedy_search(inputdata,input_len).squeeze()
    maxindex=ctcpro.max(1,keepdim=False)[1]
   # print(maxindex)

    hyp,_=model.attention_rescoring(inputdata,input_len,beam_size=2,ctc_weight=0.5)
    getword_from_id(word_index,hyp,16)
    #print(outputdata[0]
   

