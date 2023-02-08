# Append different text files into one giant text file 
# Keep only second line 
# Keep in JSON format 

import glob 
import re 
import os 
import json 
from tqdm import tqdm 

root = "/home/munfai98/Documents/NSC/part1/data/channel1/"
label_files = glob.glob(f"{root}/script/*.txt", recursive= True) 

for file in tqdm(label_files): 
    dict_for_each_file = {} 

    # Convert to JSON 
    with open(file, "r",  encoding='utf-8-sig') as f: 
        for line in f: 
            line = line.lstrip("\t")
            if "\t" in line: 
                parts = line.split("\t")
                id = parts[0]
                # Use top sentence for label 
                label = parts[1]
                label=label.lstrip("\t").rstrip("\n")

                # Use bottom sentence for label 
                # label = next(f)

                dict_for_each_file[id] = label

    # Write to new text file - do NOT overwrite 
    new_file_name = file.replace("script", "labels")
    # print(file, new_file_name)
    with open(f'{new_file_name}', 'w') as f:
        json_string = json.dumps(dict_for_each_file)
        f.write(json_string)

     

