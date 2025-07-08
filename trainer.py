from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
import lightning as L
from data.utils import DataModule
from models.cross_seggpt import CrossSegGPT
from configs.model_cfg import CrossSegCFG
from configs.data_cfg import DataConfig

def train_cross_seggpt(args):
    mlf_logger = MLFlowLogger(experiment_name=args.exp_name, tracking_uri=args.mlflow_uri)
    
    trainer_strategy = DDPStrategy(find_unused_parameters=True)

    trainer_cfg = {
        "accelerator": "gpu",
        "devices": args.devices,
        "precision": "auto",
        "strategy": trainer_strategy,
        "max_epochs": args.epochs,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "logger": mlf_logger
    }

    trainer = L.Trainer(**trainer_cfg)
    
    dm = DataModule(DataConfig())
    model = CrossSegGPT(CrossSegCFG())

    trainer.fit(model, dm)