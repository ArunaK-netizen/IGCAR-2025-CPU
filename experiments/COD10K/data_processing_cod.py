import os

default_concepts = ["sky", "grass", "water", "background"]

def prompt_concepts_generator(file_name):
    parts = file_name.split('-')
    object = parts[-2]
    category = parts[3]
    is_camo = True if parts[1] == "CAM" else False

    concepts = [object] + default_concepts
    if is_camo:
        prompt = f"An image of a camouflaged {category} {object}"
    else:
        prompt = f"An image of a non-camouflaged {category} {object}"

    return concepts, prompt


def process_dataset():
    directory = "./data/COD10K-v3/Train/Image"
    
    for file_name in os.listdir(directory):
        print(file_name)
        concepts, prompt = prompt_concepts_generator(file_name)
        print(concepts, prompt)


if __name__ == "__main__":
    process_dataset()