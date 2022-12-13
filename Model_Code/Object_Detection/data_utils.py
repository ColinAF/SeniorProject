
### Data Utilities ###
# A tool meant to merge files from different databases created in the Test_Server
# !!! WARNING: this code is almost totally untested !!!

### External Imports ###
import os
import shutil
import tarfile
### External Imports ###

# Paths relative to main senior project directory #
DATASET_NAME = "fruit_test"
DATASETS_PATH = "assets/datasets/" + DATASET_NAME + "/images/" 

ARCHIVE_NAME = ""
ARCHIVE_PATH = "assets/archives/" + ARCHIVE_NAME # 
# Paths relative to main senior project directory #


'''
Files in the dataset are of form classnameXXX.jpg 
Where classname is one of the object classifications
and 'XXX' is a its ID with padded zeros.
'''
FILL_SIZE = 3 # Zero Padding for filenames 
FILE_EXTENSION_LEN = len(".jpg")
FILE_IDENTIFIER_LEN = FILL_SIZE + FILE_EXTENSION_LEN

# Counts files in the DATASETS_PATH directory 
# This is actually silly it can all be done in one line 
def countFiles():
        files = os.listdir(DATASETS_PATH)

        numFiles = 0 

        # Loop through every file in DATASETS_PATH
        for file in files:
            if os.path.isfile(os.path.join(DATASETS_PATH, file)):
                # This is a funky way to chop of the file extension and number 
                class_name = file[:-(FILE_IDENTIFIER_LEN)]
                numFiles += 1 

        print(numFiles)

# Counts the number of files of each class in a dataset at the specified path
# Returns a dict holding this info in the form classname : classcount 
def countFilesPerClass(pathToDataset, dict, rename):
    
    files = os.listdir(pathToDataset)

    # Loop through every file in DATASETS_PATH
    for file in files:
        if os.path.isfile(os.path.join(pathToDataset, file)):
            # This is a funky way to chop of the file extension and number 
            class_name = file[:-(FILE_IDENTIFIER_LEN)]
            
            updateClass(class_name, dict)

            if rename is not False:
                reName(pathToDataset, class_name, file, dict)
    
    # Guess I won't need the returns 
    return dict

# Update class count 
def updateClass(class_name, dict):

    class_value = dict.get(class_name)

    if class_value == None:
        print("Adding Class: " + class_name) 
        dict.update({class_name: 1})  
    else:
        print("Updating Class: " + class_name) 
        class_value += 1
        dict.update({class_name: class_value})

# Rename a given file 
def reName(pathToDataset, class_name, cur_name, dict):
    class_rename = str(dict.get(class_name)).zfill(FILL_SIZE) + ".jpg"
    os.rename(pathToDataset + cur_name, pathToDataset + class_rename)

# Move files in pathTwo to pathOne
def moveDataset(pathTwo, pathOne):
    
    assert (pathTwo != pathOne), "Destination path is the same as source path!"

    files = os.listdir(pathTwo)
        
    for file in files:
        shutil.move(os.path.join(pathTwo, file), pathOne)
        print()

# Merge Two Datasets 
# Result ends up at pathOne
def mergeDatasets(pathOne, pathTwo):

    assert (pathTwo != pathOne), "Destination path is the same as source path!"

     # A dict that holds class names and the total images in each
    class_dict = {} 

    class_dict = countFilesPerClass(pathOne, class_dict, rename=True)
    class_dict = countFilesPerClass(pathTwo, class_dict, rename=True)

    moveDataset(pathTwo, pathOne)


## Potentially change these functions so compression type can be found on the fly ##
# Export the Dataset at DATASETS_PATH in a tar archive with lzma compression
def exportDataset(destPath):
    files = os.listdir(DATASETS_PATH)
    with tarfile.open(ARCHIVE_PATH, "w:xz") as tar:
        for file in files:
            tar.add(file)

# Import a Dataset in a tar archive with lzma compression
def importDataset(archivePath):
    with tarfile.open(ARCHIVE_PATH, 'r:xz') as archive:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(archive, DATASETS_PATH)
## Potentially change these functions so compression type can be found on the fly ##


mergeDatasets("/home/colin/SeniorProject/assets/datasets/produce_train01/","/home/colin/Desktop/fruit_test01/")
### Data Utilities ###