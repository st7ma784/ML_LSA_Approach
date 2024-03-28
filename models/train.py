

from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
from functools import partial
from typing import Optional
from warnings import warn
from lsafunctions import *
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

class MyTrModel(nn.Module):
    def __init__(self,h,w,softmax,activation,layers):
        super().__init__()
        self.h,self.w=h,w
        self.layers=layers
        self.transformerW=torch.nn.TransformerEncoderLayer(w,1,dim_feedforward=4*w,activation=activation,batch_first=True)
        self.transformerH=torch.nn.TransformerEncoderLayer(h,1,dim_feedforward=4*h,activation=activation,batch_first=True)
        self.perm=Permute()
        self.sm=nn.Softmax(dim=1 if self.w<self.h else 2) if softmax=="softmax" else GUMBELSoftMax(dim=1 if self.w<self.h else 2)
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

class MyDNNModel(nn.Module):
    def __init__(self,h,w,softmax,activation,layers):
        super().__init__()
        self.h,self.w=h,w
        self.layers=layers
        self.activation=torch.nn.ReLU if activation=="relu" else torch.nn.GELU
        self.sm=nn.Softmax(dim=1 if self.w<self.h else 2) if softmax=="softmax" else GUMBELSoftMax(dim=1 if self.w<self.h else 2)
        self.model= torch.nn.Sequential(*[torch.nn.Sequential(*[
            torch.nn.Sequential(*[
                torch.nn.Linear(self.w,512),
                self.activation(),
                torch.nn.Linear(512,self.w),
                self.activation(),

            ]),
            Permute(),
            torch.nn.Sequential(*[
                torch.nn.Linear(self.h,512),
                self.activation(),
                torch.nn.Linear(512,self.h),
                self.activation(),
                ]),
            Permute()]) for _ in range(self.layers)])                        
        
    def forward(self,x):
        x=self.model(x)
        return self.sm(x)        

class MyLSAModel(nn.Module):
    def __init__(self,h,w,model,softmax):
        super().__init__()
        self.h,self.w=h,w
        self.alg=get_all_LSA_fns()[model] 
        self.bias=nn.Parameter(torch.ones(self.h,self.w))
        self.sm=nn.Softmax(dim=1 if self.w<self.h else 2) if softmax=="softmax" else GUMBELSoftMax(dim=1 if self.w<self.h else 2)

    def forward(self,x):
        
        out=torch.stack([self.alg(i,maximize=True) for i in x],dim=0)
        out=out/out.norm(p=2,keepdim=True)
        out=out*self.bias
        return self.sm(out)

class myLightningModule(LightningModule):
    '''
    This training code follows the standard structure of Pytorch - lighthning. It's worth looking at their docs for a more in depth dive as to why it is this was
    '''
    
    def __init__(self,
                activation="relu",
                size=(51,53),
                softmax="gumbel",
                model="transformer",
                layers=3,
                precision="None",
                **kwargs,
                ):

        super().__init__()
        self.save_hyperparameters()
        self.loss=torch.nn.CrossEntropyLoss()
        self.MSELoss=torch.nn.MSELoss()
        #Define your own model here, 
        self.layers=layers
        self.h,self.w=size
        self.model=MyTrModel(self.h,self.w,softmax=softmax,activation=activation,layers=layers)
        
        self.max_epochs=100
        if model=="linear":
            self.model=MyDNNModel(self.h,self.w,softmax=softmax,activation=activation,layers=layers)
        elif model in get_all_LSA_fns():
            self.model=MyLSAModel(self.h,self.w,softmax=softmax,model=model)
            self.max_epochs=1
        self.precisionfn=self.convert_null
        if precision=="e5m2":
            self.precisionfn=self.convert_to_fp8_e5m2
        elif precision=="e4m3":
            self.precisionfn=self.convert_to_fp8_e4m3



        print("h,w is {}".format((self.h,self.w)))
        
        self.lossfn,self.auxlossfn=(self.hloss,self.wloss) if self.h>self.w else (self.wloss,self.hloss)
        #this is done so the main loss fn has a definite one in every column, aux loss has some ones in some columns. 





    def convert_to_fp8_e5m2(self,T):
        return T.to(torch.float8_e5m2).to(torch.float32)
    def convert_to_fp8_e4m3(self,T):
        return T.to(torch.float8_e4m3fn).to(torch.float32)
    def convert_null(self,T):
        return T
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

        input=self.precisionfn(input)
     

        out=self.forward(input)
     
        logitsB= torch.bmm(out.permute(0,2,1),truth) #shape, B, H,H
        logitsA= torch.bmm(out,truth.permute(0,2,1)) # shape B,W,W
        logitsA=logitsA.permute(1,2,0)
        logitsB=logitsB.permute(1,2,0)
        # there IS merit to using both... but one will be far noisier than tother! 
        #maybe use scaler? 
        with torch.no_grad():
            auxloss,R=self.auxlossfn(logitsA.clone().detach(),logitsB.clone().detach(),input.shape[0])
            MSE=self.MSELoss(out.clone().detach(),truth)
            self.log("auxp",R,prog_bar=True)
            self.log("auxloss",auxloss,prog_bar=True)
            self.log("MSE",MSE, prog_bar=True)


        loss,P=self.lossfn(logitsA,logitsB,input.shape[0])
        F1=2*( P*R) /(P+R)
        self.log('F1',F1,prog_bar=True)
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
