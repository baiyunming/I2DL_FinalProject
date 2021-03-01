# I2DL_FinalProject

## Dataset
link of dataset: https://drive.google.com/drive/folders/1bhlj_fgV89Ta3LwKM1trVWoq_h5vPNgo?usp=sharing  
In addition to Kaggle African Wildlife animal dataset https://www.kaggle.com/biancaferreira/african-wildlife, 100 more images are collected for each class. Images beginning with an underscore are all self collected samples.  

## Generate txt and csv file 
Run Generate_txt_csv.ipynb to generate txt files for training and test dataset. The txt files contains path to the images and annotation file. The correponding cell may needs to run several times until there are 1595 lines in train.txt and 221 lines in test.txt.
Then the txt file in converted to csv file, which is need for creating an instance of defined AnimalDataset class in animal_dataset.py.  
