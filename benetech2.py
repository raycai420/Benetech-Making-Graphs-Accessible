import os
import sys
import cv2
import json
import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from random import shuffle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

from transformers import (
    AutoProcessor,
    Pix2StructConfig,
    Pix2StructForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

from polyleven import levenshtein # a faster version of levenshtein
import albumentations as A
from albumentations.pytorch import ToTensorV2
import re

import warnings
warnings.simplefilter("ignore")

Config = {
    'IMAGE_DIR': '/home/scratch/yifuc/checkpoints/benetech/data/train/images/',
    'MAX_PATCHES': 1024,
    'MODEL_NAME': "/home/scratch/yifuc/checkpoints/benetech/model/matcha_e4", #"google/matcha-chartqa",
    'IMG_SIZE': (256, 256),
    'MAX_LEN': 512,
    'LR': 2e-5,
    'NB_EPOCHS': 5,
    'TRAIN_BS': 8,
    'VALID_BS': 2,
    'ALL_SAMPLES': int(1e+100),
    'DEVICE':'cuda:7'
    #'_wandb_kernel': 'tanaym',
}

BOS_TOKEN = "<|BOS|>"
X_START = "<x_start>"
X_END = "<x_end>"
Y_START = "<y_start>"
Y_END = "<y_end>"

new_tokens = [
    "<line>",
    "<vertical_bar>",
    "<scatter>",
    "<dot>",
    "<horizontal_bar>",
    X_START,
    X_END,
    Y_START,
    Y_END,
    BOS_TOKEN,
]

CHART_TYPE_TOKENS = ["<line>",
    "<vertical_bar>",
    "<scatter>",
    "<dot>",
    "<horizontal_bar>"]

def augments():
    return A.Compose([
        A.Resize(width=Config['IMG_SIZE'][0], height=Config['IMG_SIZE'][1]),
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,
        ),
        ToTensorV2(),
    ])

class BeneTechDataset(Dataset):
    def __init__(self, dataset, processor, augments=None):
        self.dataset = dataset
        self.processor = processor
        self.augments = augments

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = cv2.imread(item['image'])
        if self.augments:
            image = self.augments(image=image)['image']
        encoding = self.processor(
            images=image,
            return_tensors="pt", 
            add_special_tokens=True, 
            max_patches=Config['MAX_PATCHES']
        )
        
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = item["label"]
        return encoding
    
def get_model(extra_tokens=new_tokens):
    path = Config['MODEL_NAME']
    print(f'loading {path}')
    processor = AutoProcessor.from_pretrained(Config['MODEL_NAME'], is_vqa=False)
    model = Pix2StructForConditionalGeneration.from_pretrained(Config['MODEL_NAME'])
    processor.image_processor.size = {
        "height": Config['IMG_SIZE'][0],
        "width": Config['IMG_SIZE'][1],
    }

    processor.tokenizer.add_tokens(extra_tokens)
    model.resize_token_embeddings(len(processor.tokenizer))
    return processor, model

def collator(batch):
    new_batch = {"flattened_patches":[], "attention_mask":[]}
    texts = [item["text"] for item in batch]
    text_inputs = processor(
        text=texts, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt", 
        add_special_tokens=True, 
        max_length=Config['MAX_LEN']
    )
    new_batch["labels"] = text_inputs.input_ids
    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])
    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch

#-------------valid score
def rmse(y_true, y_pred) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between the true and predicted values.

    Args:
        y_true (List[float]): The true values.
        y_pred (List[float]): The predicted values.

    Returns:
        float: The Root Mean Square Error.
    """
    return np.sqrt(np.mean(np.square(np.subtract(y_true, y_pred))))


def sigmoid(x: float) -> float:
    """
    Calculate the sigmoid function for the given value.

    Args:
        x (float): The input value.

    Returns:
        float: The result of the sigmoid function.
    """
    return 2 - 2 / (1 + np.exp(-x))


def normalized_rmse(y_true, y_pred) -> float:
    """
    Calculate the normalized Root Mean Square Error (RMSE) between the true and predicted values.

    Args:
        y_true (List[float]): The true values.
        y_pred (List[float]): The predicted values.

    Returns:
        float: The normalized Root Mean Square Error.
    """
    numerator = rmse(y_true, y_pred)
    denominator = rmse(y_true, np.mean(y_true))

    # https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/396947
    if denominator == 0:
        if numerator == 0:
            return 1.0
        return 0.0

    return sigmoid(numerator / denominator)


def normalized_levenshtein_score(y_true, y_pred) -> float:
    """
    Calculate the normalized Levenshtein distance between two lists of strings.

    Args:
        y_true (List[str]): The true values.
        y_pred (List[str]): The predicted values.

    Returns:
        float: The normalized Levenshtein distance.
    """
    total_distance = np.sum([levenshtein(yt, yp) for yt, yp in zip(y_true, y_pred)])
    length_sum = np.sum([len(yt) for yt in y_true])
    return sigmoid(total_distance / length_sum)


def score_series(
    y_true, y_pred
) -> float:
    """
    Calculate the score for a series of true and predicted values.

    Args:
        y_true (List[Union[float, str]]): The true values.
        y_pred (List[Union[float, str]]): The predicted values.

    Returns:
        float: The score for the series.
    """
    if len(y_true) != len(y_pred):
        return 0.0
    if isinstance(y_true[0], str):
        return normalized_levenshtein_score(y_true, y_pred)
    else:
        # Since this is a generative model, there is a chance it doesn't produce a float.
        # In that case, we return 0.0.
        try:
            return normalized_rmse(y_true, list(map(float, y_pred)))
        except:
            return 0.0


def benetech_score(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """Evaluate predictions using the metric from the Benetech - Making Graphs Accessible.

    Parameters
    ----------
    ground_truth: pd.DataFrame
        Has columns `[data_series, chart_type]` and an index `id`. Values in `data_series`
        should be either arrays of floats or arrays of strings.

    predictions: pd.DataFrame
    """
    if not ground_truth.index.equals(predictions.index):
        raise ValueError(
            "Must have exactly one prediction for each ground-truth instance."
        )
    if not ground_truth.columns.equals(predictions.columns):
        raise ValueError(f"Predictions must have columns: {ground_truth.columns}.")
    pairs = zip(
        ground_truth.itertuples(index=False), predictions.itertuples(index=False)
    )
    scores = []
    for (gt_series, gt_type), (pred_series, pred_type) in pairs:
        if gt_type != pred_type:  # Check chart_type condition
            scores.append(0.0)
        else:  # Score with RMSE or Levenshtein as appropriate
            scores.append(score_series(gt_series, pred_series))

    ground_truth["score"] = scores

    grouped = ground_truth.groupby("chart_type", as_index=False)["score"].mean()

    chart_type2score = {
        chart_type: score
        for chart_type, score in zip(grouped["chart_type"], grouped["score"])
    }

    return np.mean(scores), chart_type2score


def string2triplet(pred_string: str):
    """
    Convert a prediction string to a triplet of chart type, x values, and y values.

    Args:
        pred_string (str): The prediction string.

    Returns:
        Tuple[str, List[str], List[str]]: A triplet of chart type, x values, and y values.
    """
    
    chart_type = "line"
    for tok in CHART_TYPE_TOKENS:
        if tok in pred_string:
            chart_type = tok.strip("<>")

    pred_string = re.sub(r"<one>", "1", pred_string)

    x = pred_string.split(X_START)[1].split(X_END)[0].split(";")
    y = pred_string.split(Y_START)[1].split(Y_END)[0].split(";")

    if len(x) == 0 or len(y) == 0:
        return chart_type, [], []

    min_length = min(len(x), len(y))

    x = x[:min_length]
    y = y[:min_length]

    return chart_type, x, y


def validation_metrics(val_outputs, val_ids, gt_df: pd.DataFrame):
    """
    Calculate validation metrics for a set of outputs, ids, and ground truth dataframe.

    Args:
        val_outputs (List[str]): A list of validation outputs.
        val_ids (List[str]): A list of validation ids.
        gt_df (pd.DataFrame): The ground truth dataframe.

    Returns:
        Dict[str, float]: A dictionary containing the validation scores.
    """
    pred_triplets = []

    for example_output in val_outputs:

        if not all([x in example_output for x in [X_START, X_END, Y_START, Y_END]]):
            pred_triplets.append(("line", [], []))
        else:
            pred_triplets.append(string2triplet(example_output))

    pred_df = pd.DataFrame(
        index=[f"{id_}_x" for id_ in val_ids] + [f"{id_}_y" for id_ in val_ids],
        data={
            "data_series": [x[1] for x in pred_triplets]
            + [x[2] for x in pred_triplets],
            "chart_type": [x[0] for x in pred_triplets] * 2,
        },
    )

    overall_score, chart_type2score = benetech_score(
        gt_df.loc[pred_df.index.values], pred_df
    )

    return {
        "val_score": overall_score,
        **{f"{k}_score": v for k, v in chart_type2score.items()},
    }

def score(truth, valid_output):
  assert(len(truth) == len(valid_output))

  pred_triplets, true_triplets = [], []

  for example_output in valid_output:
    if not all([x in example_output for x in [X_START, X_END, Y_START, Y_END]]):
        pred_triplets.append(("line", [], []))
    else:
        pred_triplets.append(string2triplet(example_output))

  for true_output in truth:
    true_triplets.append(string2triplet(true_output))

  pairs = zip(true_triplets, pred_triplets)
  scores = []
  for (true_label, true_x, true_y), (pred_label, pred_x, pred_y) in pairs:
    if true_label != pred_label:
      scores.append(0.0)
    else:
      scores.append(score_series(true_x, pred_x))
      scores.append(score_series(true_y, pred_y))
  return np.mean(scores)
#-------------------------------

def train_one_epoch(model, processor, train_loader, optimizer, scaler, scheduler=None):
    """
    Trains the model on all batches for one epoch with NVIDIA's AMP
    """
    model.train()
    avg_loss = 0
    # with autocast():
    prog_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for idx, batch in prog_bar:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
          labels = batch['labels'].to(Config['DEVICE'])
          flattened_patches = batch.pop("flattened_patches").to(Config['DEVICE'])
          attention_mask = batch.pop("attention_mask").to(Config['DEVICE'])

          outputs = model(
              flattened_patches=flattened_patches,
              attention_mask=attention_mask,
              labels=labels
          )

          loss = outputs.loss
          # scaler.scale(loss).backward()
          # scaler.step(optimizer)
          # scaler.update()
          # loss.backward()
          # optimizer.step()
           #set_to_none=True
          prog_bar.set_description(f"loss: {loss.item():.4f}")
          #wandb_log(train_step_loss=loss.item())
          avg_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        if idx % 100 == 0 and idx != 0:
            # val_outputs, true_outputs = [], []
            # predictions = model.generate(flattened_patches=flattened_patches,
            #                              attention_mask=attention_mask,
            #                              max_length = 512,
            #                              num_beams=1,
            #                              temperature=.7,
            #                              top_k=1,
            #                              top_p=.4,
            #                              early_stopping=True,
            #                              pad_token_id=processor.tokenizer.pad_token_id,
            #                              eos_token_id=processor.tokenizer.eos_token_id,
            #                             )
            # val_output = processor.batch_decode(predictions, skip_special_tokens=True)
            # val_outputs.extend(val_output)
            # true_outputs.extend(processor.batch_decode(batch['labels'], skip_special_tokens=True))
            # val_score = score(true_outputs, val_outputs)
            # print(f'score: {val_score}')
            
            if scheduler is not None: scheduler.step()

        del batch
        del outputs
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        try:
            del val_outputs, true_outputs, predictions, val_output
        except:
            pass


    avg_loss = avg_loss / len(train_loader)
    print(f"Average training loss: {avg_loss:.4f}")
    #wandb_log(train_loss=avg_loss)
    return avg_loss

def valid_one_epoch(model, processor, valid_loader):
    """
    Validates the model on all batches (in val set) for one epoch
    """
    model.eval()
    avg_loss = 0
    prog_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for idx, batch in prog_bar:
        labels = batch.pop("labels").to(Config['DEVICE'])
        flattened_patches = batch.pop("flattened_patches").to(Config['DEVICE'])
        attention_mask = batch.pop("attention_mask").to(Config['DEVICE'])
        
        outputs = model(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        prog_bar.set_description(f"loss: {loss.item():.4f}")
        #wandb_log(val_step_loss=loss.item())
        avg_loss += loss.item()

        del batch
        del outputs
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    avg_loss = avg_loss / len(valid_loader)
    print(f"Average validation loss: {avg_loss:.4f}")
    #wandb_log(val_loss=avg_loss)
    return avg_loss

def fit(model, processor, train_loader, valid_loader, optimizer, scaler, scheduler):
    """
    A nice function that binds it all together and reminds me of Keras days from 2018 :)
    """
    best_val_loss = int(1e+5)
    print('-----', Config['NB_EPOCHS'])

    for epoch in range(Config['NB_EPOCHS']):
        print(f"{'='*20} Epoch: {epoch+1} / {Config['NB_EPOCHS']} {'='*20}")
        _ = train_one_epoch(model, processor, train_loader, optimizer, scaler, scheduler)
        # val_avg_loss = valid_one_epoch(model, processor, valid_loader)
        n_epoch = epoch
        path = f'/home/scratch/yifuc/checkpoints/benetech/model/matcha_last_e{n_epoch}'
        model.save_pretrained(path)
        processor.save_pretrained(path)
    print(f"Best model with val_loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    with open("/home/scratch/yifuc/checkpoints/benetech/data/data.json", "r") as fl:
        dataset = json.load(fl)['data']
    
    # Shuffle the dataset and select however samples you want for training
    shuffle(dataset)
    dataset = dataset[:Config['ALL_SAMPLES']]

    for i in range(len(dataset)):
        name = dataset[i]['image'].split('/')[-1]
        dataset[i]['image'] = '/home/scratch/yifuc/checkpoints/benetech/data/train/images/' + name

    #adding sub dataset########
    # with open("/home/scratch/yifuc/checkpoints/benetech/data/data_sub_subset.json", "r") as fl:
    #     dataset_sub = json.load(fl)
    
    # shuffle(dataset_sub)

    # for i in range(len(dataset_sub)):
    #     name = dataset_sub[i]['image']
    #     dataset_sub[i]['image'] = '/home/scratch/yifuc/checkpoints/benetech/data/' + name
    #################

    # We are splitting the data naively for now
    split = 1.0
    train_samples = int(len(dataset) * split)
    train_ds = dataset[:train_samples+1]
    valid_ds = dataset[train_samples:]

    # train_ds = train_ds + dataset_sub
    # shuffle(train_ds)
    print('combine dataset successful')

    # Yeah all that
    processor, model = get_model()
    print('sucessfully obtained model')
    model.to(Config['DEVICE'])
    #wandb.watch(model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=Config['LR'])

    # Load the data into Datasets and then make DataLoaders for training
    train_dataset = BeneTechDataset(train_ds, processor, augments=augments())
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=Config['TRAIN_BS'], collate_fn=collator)

    valid_dataset = BeneTechDataset(valid_ds, processor, augments=augments())
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=Config['VALID_BS'], collate_fn=collator)

    nb_train_steps = int(train_samples / Config['TRAIN_BS'] * Config['NB_EPOCHS'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=3e-7, verbose=True)

    # Print out the data sizes we are training on
    print(f"Training on {len(train_ds)} samples, Validating on {len(valid_ds)} samples")

    # Train the model now
    fit(
        model=model,
        processor=processor,
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        optimizer=optimizer,
        scaler=GradScaler(),
        scheduler=scheduler
    )