

from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
from functools import partial
from typing import Optional
from warnings import warn

class myLightningModule(LightningModule):
    '''
    This training code follows the standard structure of Pytorch - lighthning. It's worth looking at their docs for a more in depth dive as to why it is this was
    '''
    
    def __init__(self,
                learning_rate,
                **kwargs,
                ):

        super().__init__()
        self.save_hyperparameters()
        self.loss=torch.nn.CrossEntropyLoss()
        self.MSELoss=torch.nn.MSELoss()
        #Define your own model here, 
        self.model=torch.nn.Sequential(*[
            torch.nn.Linear(50000,512),
            torch.nn.Linear(512,256),
            torch.nn.Linear(256,512),
            torch.nn.Linear(512,40000)
        ])
    def forward(self,input):
        #This inference steps of a foward pass of the model 
        return self.model(input)

    def training_step(self, batch, batch_idx,optimizer_idx=0):
        #The batch is collated for you, so just seperate it here and calculate loss. 
        #By default, PTL handles optimization and scheduling and logging steps. so All you have to focus on is functionality. Here's an example...
        input,truth=batch[0],batch[1]
        out=self.forward(input)
        MSE=self.MSELoss(out,truth)
        
        logitsA= torch.bmm(input,out.permute(0,2,1)) #shape, B, H,H
        logitsB= torch.bmm(input.permute(0,2,1),out) # shape B,W,W
        logitsA=logitsA.permute(1,2,0)
        logitsB=logitsB.permute(1,2,0)
        
        lossA=self.loss(logitsA,torch.arange(logitsA.shape[0]))
        lossB=self.loss(logitsB,torch.arange(logitsB.shape[0]))
        loss=lossA+lossB
        loss=loss/2
        self.log("MSE",MSE, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
      
      
    def validation_step(self, batch, batch_idx):
      
        input,desired=batch[0],batch[1]
        out=self.forward(input)
        #You could log here the val_loss, or just print something. 
        
    def configure_optimizers(self):
        #Automatically called by PL. So don't worry about calling it yourself. 
        #you'll notice that everything from the init function is stored under the self.hparams object 
        optimizerA = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, eps=1e-8)
        
        #Define scheduler here too if needed. 
        return [optimizerA]
