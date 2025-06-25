import lightning as L
from torch.utils.data import DataLoader

class IntegratedDataset(L.LightningDataModule):
    def __init__(self, cfg):
        pass
        """
        lightning 의 dataset 의 initialization 순서 
        prepare_data 호출 -> setup 호춢
        
        """
    def prepare_data(self):
        return super().prepare_data()
    
    def setup(self, stage: str):
        # stage 는 train, validation, 뭐 이런거. Enum 형태로 해도 됨.
        # 여기에서 일반적인 torch dataset 불러온다고 보면 됨
        return super().setup(stage)
    
    def train_dataloader(self):
        return super().train_dataloader()
    
    def val_dataloader(self):
        return super().val_dataloader()
    
    def test_dataloader(self):
        return super().test_dataloader()