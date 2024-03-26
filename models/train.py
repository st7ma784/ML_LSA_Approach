

from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
from functools import partial
from typing import Optional
from warnings import warn
class Permute(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.permute(0,2,1)

class GUMBELSoftMax(nn.Module):
    def __init__(self,dim=-1):
        super().__init__()
        self.dim=dim
    def forward(self, x):
        return torch.nn.functional.gumbel_softmax(x,dim=self.dim,hard=True)

class myLightningModule(LightningModule):
    '''
    This training code follows the standard structure of Pytorch - lighthning. It's worth looking at their docs for a more in depth dive as to why it is this was
    '''
    
    def __init__(self,
                activation="relu",
                size=(51,53),
                layers=3,
                **kwargs,
                ):

        super().__init__()
        self.save_hyperparameters()
        self.loss=torch.nn.CrossEntropyLoss()
        self.MSELoss=torch.nn.MSELoss()
        #Define your own model here, 
        self.layers=layers
        self.h,self.w=size
        self.activation=torch.nn.ReLU if activation=="relu" else torch.nn.GELU
        self.WModel=torch.nn.Sequential(*[
            torch.nn.Linear(self.w,512),
            self.activation(),
            torch.nn.Linear(512,self.w),
            self.activation(),

        ])
        self.HModel=torch.nn.Sequential(*[
            torch.nn.Linear(self.h,512),
            self.activation(),
            torch.nn.Linear(512,self.h),
            self.activation(),

        ])

        self.layer=torch.nn.Sequential(*[
            self.WModel,
            Permute(),
            self.HModel,
            Permute()])   
        self.modellayers= torch.nn.Sequential(*[self.layer for _ in range(self.layers)])                        
        self.model=torch.nn.Sequential(*[self.modellayers,GUMBELSoftMax(dim=1 if self.w<self.h else 2)])

        print("h,w is {}".format((self.h,self.w)))
        
        self.lossfn,self.auxlossfn=(self.hloss,self.wloss) if self.h>self.w else (self.wloss,self.hloss)
        #this is done so the main loss fn has a definite one in every column, aux loss has some ones in some columns. 






    def forward(self,input):
        #This inference steps of a foward pass of the model 
        return self.model(input)
    def hloss(self,A,B,Batchsize):
        P=torch.mean(torch.diagonal(B,dim1=1,dim2=2))
        return self.loss(B,torch.arange(B.shape[0],device=self.device).unsqueeze(1).repeat(1,Batchsize)),P
    def wloss(self,A,B,Batchsize):
        P=torch.mean(torch.diagonal(A,dim1=1,dim2=2))

        return self.loss(A,torch.arange(A.shape[0],device=self.device).unsqueeze(1).repeat(1,Batchsize)),P

    def training_step(self, batch, batch_idx):
        #The batch is collated for you, so just seperate it here and calculate loss. 
        #By default, PTL handles optimization and scheduling and logging steps. so All you have to focus on is functionality. Here's an example...
        input,truth=batch[0],batch[1]
        out=self.forward(input)
        #print(torch.sum(out).item()/input.shape[0])
        #print(min(self.h,self.w))
        assert int(torch.sum(out).item()/input.shape[0])==min(self.h,self.w)
        logitsB= torch.bmm(out.permute(0,2,1),truth) #shape, B, H,H
        logitsA= torch.bmm(out,truth.permute(0,2,1)) # shape B,W,W
        logitsA=logitsA.permute(1,2,0)
        logitsB=logitsB.permute(1,2,0)
        # there IS merit to using both... but one will be far noisier than tother! 
        #maybe use scaler? 
        with torch.no_grad():
            auxloss,auxP=self.auxlossfn(logitsA.clone().detach(),logitsB.clone().detach(),input.shape[0])
            MSE=self.MSELoss(out.clone().detach(),truth)
            self.log("auxp",auxP,prog_bar=True)

            self.log("auxloss",auxloss,prog_bar=True)
            self.log("MSE",MSE, prog_bar=True)


        loss,P=self.lossfn(logitsA,logitsB,input.shape[0])
        return {'loss': loss,'P':P}
      

        
    def configure_optimizers(self):
        #Automatically called by PL. So don't worry about calling it yourself. 
        #you'll notice that everything from the init function is stored under the self.hparams object 
        optimizerA = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, eps=1e-8)
        
        #Define scheduler here too if needed. 
        return [optimizerA]
