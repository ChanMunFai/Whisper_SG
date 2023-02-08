"""
This script is used to move audio and label files from the train folder into the test folder
"""
import glob 
import random
import shutil 
import os 
from pathlib import Path 

random.seed(10)

current_folder = "/home/munfai98/Documents/NationalSpeechCorpus/part1/data/train/channel0"
target_folder = "/home/munfai98/Documents/NationalSpeechCorpus/part1/data/test/channel0"

folders = glob.glob(f"{current_folder}/wave/**", recursive= False) 
labels = glob.glob(f"{current_folder}/labels/**/*.TXT", recursive= True) 


def get_test_indices(num_items, percentage_test): 
    """
    Arguments: 
        num_items: Total number of examples
        percentage_test: Percentage of examples to put into test split 
    """
    test_indices = random.sample(range(num_items), int(percentage_test * num_items))
    return test_indices


def get_test_waves(test_indices, folders): 
    """ Extract folder (Speaker) names of audio waves. 
    We split by speakers to reduce data leakage. 

    Arguments: 
        test_indices: List of indices to be used for test split 
        folders: List of all folders (speakers) - consist of both train and test examples
    """
    test_waves = [folders[i] for i in test_indices]
    return test_waves


def find_corresponding_label(test_waves, labels): 

    speaker_list = [i[-4:] for i in test_waves]
    assert len(speaker_list) == len(test_waves)

    test_labels = []

    for i in labels: 
        speaker = i[-9:-5]
        if speaker in speaker_list: 
            test_labels.append(i)

    assert len(test_labels) >= len(test_waves)
    assert len(test_labels) <= 2 * len(test_waves)

    return test_labels

def move_audio_label(test_waves, test_labels, current_folder, target_folder): 
    """
    Arguments: 
        test_waves: List of file names for audio waves 
        test labels: List of file names for labels 
        current folder
        target folder
    """

    def move_file(source_folder, target_folder): 
        # Scenario 1: Move folder
        if not os.path.isfile(source_folder): 
            for src_file in Path(source_folder).rglob('*.*'):
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                shutil.move(src_file, target_folder)
            
            print(f"Moved folder {source_folder} to {target_folder}")

        # Scenario 2: Move files
        else: 
            parent_target_folder = target_folder.rsplit("/", 1)[0]
            if not os.path.exists(parent_target_folder):
                os.makedirs(parent_target_folder)
            shutil.move(source_folder, target_folder)

   
    permission = input(f"\n Are you sure you want to proceed moving {len(test_waves)+len(test_labels)} files from \n {current_folder} to \n {target_folder}. \n Enter YES if you are sure. \n")
    
    if permission == "YES": 
        print("Copying files...")
        pass 

    else: 
        print("Mission aborted")
        return

    for i in test_waves: 
        filename = i[len(current_folder):]
        target_filename = target_folder + filename
        move_file(i, target_filename)

    for i in test_labels: 
        filename = i[len(current_folder):]
        target_filename = target_folder + filename
        move_file(i, target_filename)
        

print(f"Number of speaker is {len(folders)}. Number of label files is {len(labels)}.")

test_indices = get_test_indices(len(folders), 0.2)
test_waves = get_test_waves(test_indices, folders)

test_labels = find_corresponding_label(test_waves, labels)

print(f"Number of audio folders to be used for test split is {len(test_waves)}. ")
print(f"Number of label text files to be used for test split is {len(test_labels)}. ")

move_audio_label(test_waves, test_labels, current_folder, target_folder)



