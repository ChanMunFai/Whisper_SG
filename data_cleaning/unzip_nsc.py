import os 
import glob 
import zipfile
from tqdm import tqdm 

class Unzip():
    def __init__(self, directory):
        self.directory = directory 
        self.filenames = glob.glob(f"{directory}/**")
        self.zipfilenames = glob.glob(f"{directory}/**.zip")
        self.extracted_files = [f for f in self.filenames if f not in self.zipfilenames]
        # print(self.zipfilenames)

    def unzip(self): 
        for f in tqdm(self.filenames): 
            with zipfile.ZipFile(f , 'r') as zip_ref:
                zip_ref.extractall(self.directory)

        print("Finished unzipping all files!")
        return 

    def delete(self): 
        for f in tqdm(self.filenames): 
            os.remove(f)

        return   

    def unzip_and_delete(self): 
        # Delete folder if it exists 
        for f in tqdm(self.zipfilenames): 
            

            if f in self.extracted_files: 
                os.remove(f) 

            else: 
                try: 
                    with zipfile.ZipFile(f , 'r') as zip_ref:
                        zip_ref.extractall(self.directory)
                    os.remove(f)
                except: 
                    pass 

        print("Finished extracting all files")
            
  

if __name__ == "__main__": 
    unzipper = Unzip("/home/munfai98/Documents/NationalSpeechCorpus/part1/data/train/channel0/wave")
    unzipper.unzip_and_delete()