import os

#Cargamos las clases 
def load_class_names(data_dir):
    return sorted([
        folder for folder in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, folder))
    ])
