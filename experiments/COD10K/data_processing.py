"""
    Code for processing the segmentation dataset into a better format. 
"""
import os
import h5py
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from nltk.corpus import wordnet as wn

background_concepts = ["sky", "grass", "water", "background"]

def prompt_concepts_generator(file_name):
    parts = file_name.split('-')
    object = parts[-2]
    category = parts[3]
    is_camo = True if parts[1] == "CAM" else False

    if is_camo:
        prompt = f"An image of a camouflaged {category} {object}"
    else:
        prompt = f"An image of a non-camouflaged {category} animal"
        object = "animal"

    return object, background_concepts, prompt



def process_dataset(directory: str="data/COD10K-v3/",):
    # Make the files
    if not os.path.exists(f"{directory}"):
        os.makedirs(f"{directory}")
    if not os.path.exists(f"{directory}/Image"):
        os.makedirs(f"{directory}/Image")
    if not os.path.exists(f"{directory}/GT_Object"):
        os.makedirs(f"{directory}/GT_Object")    
    # Make a pandas dataframe
    df = pd.DataFrame(
        columns=["image_path", "segmentation_mask_path", "object_name", "background_concepts", "prompt"]
    )


    # Iterate through the data
    image_directory = "./data/COD10K-v3/Train/Image"
    target_directory = "./data/COD10K-v3/Train/GT_Object"

    for file_name in os.listdir(image_directory):        

        # Load the image
        concepts, prompt = prompt_concepts_generator(file_name)
        img = Image.open(image_directory + "/" + file_name)
        img = np.array(img).transpose((2,1,0))

        target_img = Image.open(target_directory + "/" + file_name)

        # Load the target segmentation
        target = np.array(target_img).transpose((1, 0))

        # Get the simplified name
        object_name, background_concepts, prompt = prompt_concepts_generator(file_name)
        # Save the image
        img_path = f"{image_directory}/{file_name}.png"
        Image.fromarray(img).save(img_path)
        # Save the target segmentation
        target_path = f"{target_directory}/{file_name}.png"
        Image.fromarray(target).save(target_path)
        # Add the row to the pandas dataframe
        df = pd.concat([
            df,
            pd.DataFrame(
                {
                    "image_path": [img_path],
                    "segmentation_mask_path": [target_path],
                    "object_name": [object_name],
                    "background_concepts" : background_concepts,
                    "prompt" : prompt
                },
                index=[file_name]
            )
        ])
        # Save the pandas data frame 
        df.to_csv(f"{directory}/data.csv")

class Cod10K_Segmentation(data.Dataset):
    CLASSES = 2

    def __init__(
        self,
        directory: str="data/COD10K-v3",
        transform=None,
        target_transform=None
    ):
        self.directory = directory
        self.image_directory = "./data/COD10K-v3/Train/Image"
        self.target_directory = "./data/COD10K-v3/Train/GT_Object"
        
        if not os.path.exists(f"{self.directory}/data.csv"):
            process_dataset(directory=self.directory)
        # Load the csv as a dataframe
        self.df = pd.read_csv(f"{self.directory}/data.csv")
        self.data_length = len(self.df)

    def __getitem__(self, index):
        # Load the image file
        img = Image.open(f"{self.image_directory}/{index}.png").convert("RGB")
        # Load the target segmentation file
        target = Image.open(f"{self.target_directory}/{index}.png")
        object_name = self.df.iloc[index]["object_name"]
        background_concepts = self.df.iloc[index]["background_concepts"]
        prompt = self.df.iloc[index]["prompt"]

        return img, target, object_name, background_concepts, prompt

    def __len__(self):
        return self.data_length

if __name__ == "__main__":
    process_dataset()