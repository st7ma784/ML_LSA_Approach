from test_tube import HyperOptArgumentParser

class baseparser(HyperOptArgumentParser):
    def __init__(self,*args,strategy="random_search",**kwargs):

        super().__init__( *args,strategy=strategy, add_help=False) # or random search
        # self.add_argument("--dir",default="/nobackup/projects/<YOURBEDEPROJECT>/$USER/data",type=str)
        # self.add_argument("--log_path",default="/nobackup/projects/<YOURBEDEPROJECT>/$USER/logs/",type=str)
        self.opt_list("--learning_rate", default=0.0001, type=float, options=[2e-4,1e-4,5e-5,1e-5,4e-6], tunable=True)
        self.opt_list("--batch_size", default=100, type=int)
        
        #INSERT YOUR OWN PARAMETERS HERE

        self.opt_list("--accelerator", default='gpu', type=str, options=['gpu'], tunable=False)
        self.opt_list("--num_trials", default=0, type=int, tunable=False)
        #self.opt_range('--neurons',default=50, type=int, tunable=True, low=100, high=800, nb_samples=8, log_base=None)
        self.opt_list("--activation",default="relu",type=str,options=["relu","gelu"],tunable=True)
        self.opt_list("--layers",default=2,type=int,options=[0,1,2,3,4,5],tunable=True)
        self.opt_list("--h",default=5,type=int,options=[11,22,33,44,55,66],tunable=True)
        self.opt_list("--w",default=7,type=int,options=[12,24,36,48,60,84],tunable=True)
        self.opt_list("--optimizer",default="AdamW",type=str,options=["AdamW","RAdam"],tunable=True)
        #This is important when passing arguments as **config in launcher
        self.argNames=["dir","log_path","w","h","learning_rate","batch_size","optimizer","modelname","activation","layers","accelerator","num_trials"]
    def __dict__(self):
        return {k:self.parse_args().__dict__[k] for k in self.argNames}

import wandb
from tqdm import tqdm



class parser(baseparser):
    def __init__(self,*args,strategy="random_search",**kwargs):

        super().__init__( *args,strategy=strategy, add_help=False,**kwargs) # or random search
        self.run_configs=set()
        self.keys=set()
        self.keys_of_interest=set(["learning_rate","batch_size","optimizer","modelname","activation","layers","w","h",])


    def generate_wandb_trials(self,entity,project):
        wandb.login(key='9cf7e97e2460c18a89429deed624ec1cbfb537bc')
        api = wandb.Api()

        runs = api.runs(entity + "/" + project)
        print("checking prior runs")
        for run in tqdm(runs):
            config=run.config
            sortedkeys=list([str(i) for i in config.keys() if i in self.keys_of_interest])
            sortedkeys.sort()
            values=list([str(config[i]) for i in sortedkeys])
            code="_".join(values)
            self.run_configs.add(code)
        hyperparams = self.parse_args()
        NumTrials=hyperparams.num_trials if hyperparams.num_trials>0 else 1
        trials=hyperparams.generate_trials(NumTrials)
        print("checking if already done...")
        trial_list=[]
        for trial in tqdm(trials):

            sortedkeys=list([str(i) for i in trial.__dict__.keys() if i in self.keys_of_interest])
            sortedkeys.sort()
            values=list([str(trial.__dict__[k]) for k in sortedkeys])
            
            code="_".join(values)
            while code in self.run_configs:
                trial=hyperparams.generate_trials(1)[0]
                sortedkeys=list([str(i) for i in trial.__dict__.keys() if i in self.keys_of_interest])
                sortedkeys.sort()
                values=list([str(trial.__dict__[k]) for k in sortedkeys])
            
                code="_".join(values)
            trial_list.append(trial)
        return trial_list
        
# Testing to check param outputs
if __name__== "__main__":
    
    #If you call this file directly, you'll see the default ARGS AND the trials that might be generated. 
    myparser=parser()
    hyperparams = myparser.parse_args()
    print(hyperparams.__dict__)
    for trial in hyperparams.generate_trials(10):
        print(trial)
        
