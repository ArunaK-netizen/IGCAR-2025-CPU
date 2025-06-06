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
    try:
        object = parts[-2]
    except:
        print(parts)
    category = parts[3]
    is_camo = True if parts[1] == "CAM" else False

    if is_camo:
        prompt = f"An image of a camouflaged {category} {object}"
    else:
        prompt = f"An image of a non-camouflaged {category} animal"
        object = "animal"

    return object, background_concepts, prompt



def process_dataset(directory: str="/kaggle/input/cod10k/COD10K-v2",):
    # Make the files
    # if not os.path.exists(f"{directory}"):
    #     os.makedirs(f"{directory}")
    # if not os.path.exists(f"{directory}/Image"):
    #     os.makedirs(f"{directory}/Image")
    # if not os.path.exists(f"{directory}/GT_Object"):
    #     os.makedirs(f"{directory}/GT_Object")    
    # Make a pandas dataframe
    working_directory = f"/kaggle/working"
    if not os.path.exists(f"{working_directory}/data.csv"):

        df = pd.DataFrame(
            columns=["image_path", "segmentation_mask_path", "object_name", "background_concepts", "prompt"]
        )


        # Iterate through the data
        image_directory = f"{directory}/Train/Images/Image"
        target_directory = f"{directory}/Train/GT_Objects/GT_Object"

        for file_name in os.listdir(image_directory):        

            # Load the image
            img = Image.open(image_directory + "/" + file_name)
            img = np.array(img).transpose((2,1,0))

            target_img = Image.open(target_directory + "/" + file_name[:-3:] + "png")
            # Load the target segmentation
            target = np.array(target_img).transpose((1, 0))

            # Get the simplified name
            object_name, background_concepts, prompt = prompt_concepts_generator(file_name)
            # Save the image
            img_path = f"{image_directory}/{file_name}"
            # Save the target segmentation
            target_path = f"{target_directory}/{file_name[:-4]}.png"

            # Add the row to the pandas dataframe
            df = pd.concat([
                df,
                pd.DataFrame(
                    {
                        "image_path": [img_path],
                        "segmentation_mask_path": [target_path],
                        "object_name": [object_name],
                        "background_concepts" : [background_concepts],
                        "prompt" : [prompt]
                    },
                    index=[file_name]
                )
            ])
            # Save the pandas data frame 
            if not os.path.exists(f"{working_directory}/data.csv"):
                df.to_csv(f"{working_directory}/data.csv")

class Cod10K_Segmentation(data.Dataset):
    CLASSES = 2

    def __init__(
        self,
        directory: str="/kaggle/input/cod10k/COD10K-v2",
        transform=None,
        target_transform=None
    ):
        self.directory = directory
        self.working_directory = "/kaggle/working"
        self.image_directory = f"{directory}/Train/Images/Image"
        self.target_directory = f"{directory}/Train/GT_Objects/Object"
        
        if not os.path.exists(f"{self.directory}/data.csv"):
            process_dataset(directory=self.directory)
        # Load the csv as a dataframe
        self.df = pd.read_csv(f"{self.working_directory}/data.csv")
        self.data_length = len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        img_path = os.path.join(self.directory, row["image_path"])
        target_path = os.path.join(self.directory, row["segmentation_mask_path"])

        img = Image.open(img_path).convert("RGB")
        target = Image.open(target_path)

        object_name = row["object_name"]
        background_concepts = row["background_concepts"]
        prompt = row["prompt"]

        return img, target, object_name, background_concepts, prompt


    def __len__(self):
        return self.data_length

if __name__ == "__main__":
    process_dataset()