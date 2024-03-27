
import torch     
import os
import pytorch_lightning as pl
from scipy.optimize import linear_sum_assignment
from functools import partial

def outputconversion(func): #converts the output of a function back to 1-hot tensor
    def wrapper(*args, **kwargs):
        func=kwargs.pop("func")
        args=list(args)
        x=args.pop(0)
        output=torch.zeros_like(x) + 1e-8

        x1,y1=func(x, *args, **kwargs)
        try:
            output[x1,y1]=1
        except:
            output[y1,x1]=1
        return output
    return partial(wrapper,func=func)
LSA=outputconversion(linear_sum_assignment)
## An Example dataset, needs to implement a torch.data.utils.Dataset. This one automatically loads COCO for us from MSCOCO annotations, which we extend to include our own tokenizer

class myDataset(torch.utils.data.Dataset):
    def __init__(self, size=(5,10),*args, **kwargs):
        #check if root and annfile exist
        self.w,self.h=size
        super().__init__( *args, **kwargs)
    def __len__(self):
        return 100000
    def __getitem__(self, index: int):
        array=torch.rand(self.w,self.h)
        truth=LSA(array,maximize=True)
        return array,truth


# Dataset

class myDataModule(pl.LightningDataModule):
    ## This dataModule takes care of downloading the data per node and then PL may replace the sampler if doing distributed multi-node training. 
    ## Some settings here may be worth editing if on a machine where Pin memory, or workers are limited. 
    def __init__(self, Cache_dir='./', size=(53,51), batch_size=256,*args, **kwargs):
        super().__init__()
        self.data_dir = Cache_dir
        self.size=size
        self.ann_dir=os.path.join(self.data_dir,"annotations")
        self.batch_size = batch_size
        
    def train_dataloader(self):

        # IF you know that you're only ever using 1 gpu (HEC /local runs only...) then consider using https://lightning-bolts.readthedocs.io/en/latest/dataloaders/async.html
        return torch.utils.data.DataLoader(myDataset(self.size), batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
    def val_dataloader(self):
     
        return torch.utils.data.DataLoader(myDataset(self.size), batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
    def test_dataloader(self):

        return torch.utils.data.DataLoader(myDataset(self.size), batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
    def prepare_data(self):
        pass
    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        print("Entered COCO datasetup")
       
