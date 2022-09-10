import zipfile
from pathlib import Path

import timm
from fastai.vision.all import (CategoryBlock, CrossEntropyLossFlat, DataBlock,
                               ImageBlock, Learner, RandomSplitter, Resize,
                               SaveModelCallback, accuracy, aug_transforms,
                               get_image_files, parent_label)


# set the directories
BASE_DIR = Path().absolute()
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
DS_DIR = Path('data/dataset')
DS_DIR.mkdir(parents=True, exist_ok=True)

# to unzip file 
with zipfile.ZipFile(DATA_DIR / 'dataset.zip', 'r') as zip_ref:
    zip_ref.extractall(DS_DIR)

# set the DataBlock 
printer = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                 get_items=get_image_files, 
                 splitter=RandomSplitter(valid_pct=0.15),
                 get_y=parent_label,
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224))

# set the dataloaders
dls = printer.dataloaders(DS_DIR)
# if you want to see batch
# dls.show_batch()

# if you want to see models list
# timm.list_models()

model = timm.create_model(model_name='convnext_tiny', pretrained=True, num_classes=2)

cbs = SaveModelCallback('accuracy', fname='3d_print_convnext_tiny')
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=[accuracy], cbs=cbs)

#* to find learning rate 
#* learn.lr_find()

# fine-tune model with fit one cycle
# 5e-4 is learning rate, you can change it if you want
learn.fit_one_cycle(8, 5e-4)
learn.show_results()

# for predict 
# learn.predict(<file_path>)

