import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pytorch_lightning as pl

class EmbeddingSummation(pl.LightningModule):
    def __init__(self, modelName="albert-base-v2", fromDownload=True):
        super().__init__()
        self.modelName = modelName
        self.filename = f'embeddings_{modelName}.pt' 
#         self.tokenizer = AutoTokenizer.from_pretrained(modelName)
        self.dropout  = torch.nn.Dropout(.25)
        if fromDownload:
            self.embeddings = AutoModel.from_pretrained(modelName).embeddings #11683584 params
        else:
            try:
                state_dict = torch.load(self.filename, map_location='cpu')
                config = AutoConfig.from_pretrained(self.modelName)
                self.embeddings = state_dict['className'](config)
                del(state_dict['className'])
                self.embeddings.load_state_dict(state_dict)
            except FileNotFoundError:
                raise ValueError("""Could not find {self.filename}. Pass fromDownload=True to EmbeddingSummation's __init__ to download the weights into you machine. You can use save_embeddings and then you fromDownload=False""")

        
    def save_embeddings(self, filename=None):
        filename = self.filename if filename is None else filename
        state_dict = self.embeddings.state_dict()
        state_dict['className'] = type(self.embeddings)
        torch.save(state_dict, filename)
        print(f"Saved embeddings at {filename}")

#     def setInputsToDevice(self, inputs):
#         device = self.device
#         inputs = {key: inputs[key].to(device) for key in inputs}
#         return inputs      

    def forward(self, inputs, mask)->torch.Tensor:
        """Inputs and mask have same shape. They are outputs of a tokenizer."""

        # forward pass on embeddings, out is b t u
        out = self.embeddings(inputs)
        
        # multiply attention masks to get the sum
        mask = torch.unsqueeze(mask, dim=1) * 1. # b, 1, t
        out = torch.matmul(mask, out) # b 1 u
        out = torch.squeeze(out, dim=1) # b u
        return self.dropout(out)