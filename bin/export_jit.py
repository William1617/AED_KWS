
from kws_model import *
from encoder import *
from decoder import *
from CTC import *
import torch
import numpy as np

#cmvn_mean=torch.Tensor(np.loadtxt('cmvn_mean.txt'))
#cmvn_var=torch.Tensor(np.loadtxt('cmvn_var.txt'))
#istd=1/cmvn_var
encoder = conformerencoder(80,2,80)
ctc=CTC(40,80)
decoder=transformerdecoder(vocab_size=40,encoder_output_size=80,num_blocks=2,linear_units=80)

model=KWSModel(vocab_size=39,encoder=encoder,ctc=ctc,decoder=decoder)
checkpoit=torch.load('./aedchnmodel.pt',map_location='cpu')
model.load_state_dict(checkpoit,strict=False)

quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
print(quantized_model)
script_quant_model = torch.jit.script(quantized_model)
script_quant_model.save('./aedmodel.zip')


