

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

class MyModel(nn.Module):
    def __init__(self,h,w,activation,layers):
        super().__init__()

        self.transformerW=torch.nn.TransformerEncoderLayer(w,1,dim_feedforward=4*w,activation=activation,batch_first=True)
        self.transformerH=torch.nn.TransformerEncoderLayer(h,1,dim_feedforward=4*h,activation=activation,batch_first=True)
        self.perm=Permute()
        self.sm=GUMBELSoftMax(dim=1 if w<h else 2)
        self.softmax=nn.Softmax(dim=1)
        layer=[torch.nn.TransformerEncoderLayer(w,1,dim_feedforward=4*w,activation=activation,batch_first=True),
               Permute(),
               torch.nn.TransformerEncoderLayer(h,1,dim_feedforward=4*h,activation=activation,batch_first=True),
               Permute(),
               nn.Softmax(dim=1 if w<h else 2)
               ]*layers
        self.model=nn.Sequential(*layer)
    def forward(self,x):
        x=self.model(x)
        return self.sm(x)        


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
               
        self.sm=nn.Softmax(dim=1 if self.w<self.h else 2)#(dim=1 if self.w<self.h else 2)

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
        self.model=torch.nn.Sequential(*[self.modellayers,self.sm])
        # self.model=MyModel(self.h,self.w,activation,layers=layers)



        print("h,w is {}".format((self.h,self.w)))
        
        self.lossfn,self.auxlossfn=(self.hloss,self.wloss) if self.h>self.w else (self.wloss,self.hloss)
        #this is done so the main loss fn has a definite one in every column, aux loss has some ones in some columns. 






    def forward(self,input):
        #This inference steps of a foward pass of the model 
        return self.model(input)
    def hloss(self,A,B,Batchsize):
        P=torch.mean(torch.diagonal(B,dim1=0,dim2=1))
        return self.loss(B,torch.diag_embed(torch.ones(B.shape[0],device=self.device)).unsqueeze(-1).repeat(1,1,Batchsize)),P
    def wloss(self,A,B,Batchsize):
        P=torch.mean(torch.diagonal(A,dim1=0,dim2=1))

        return self.loss(A,torch.diag_embed(torch.ones(A.shape[0],device=self.device)).unsqueeze(-1).repeat(1,1,Batchsize)),P

    def training_step(self, batch, batch_idx):
        #The batch is collated for you, so just seperate it here and calculate loss. 
        #By default, PTL handles optimization and scheduling and logging steps. so All you have to focus on is functionality. Here's an example...
        input,truth=batch[0],batch[1]
        out=self.forward(input)
        #print(torch.sum(out).item()/input.shape[0])
        #print(min(self.h,self.w))
        #assert int(torch.sum(out).item()/input.shape[0])==min(self.h,self.w)
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
        self.log('precision',P,prog_bar=True )
        self.log('train_loss', loss,enable_graph=False, prog_bar=True)
        return {'loss': loss,}
      

        
    def configure_optimizers(self):
        #Automatically called by PL. So don't worry about calling it yourself. 
        #you'll notice that everything from the init function is stored under the self.hparams object
        if self.hparams.optimizer=="AdamW": 
            optimizerA = torch.optim.AdamW( 
            self.parameters(), lr=self.hparams.learning_rate, eps=1e-8)
        elif self.hparams.optimizer=="RAdam":
            optimizerA = torch.optim.RAdam( 
            self.parameters(), lr=self.hparams.learning_rate,)# eps=1e-8)
        else:
            raise("Optim not implemented")
        #Define scheduler here too if needed. 
        return [optimizerA]
