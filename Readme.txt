The code file "randomforest" in the zip file contains 10 files.

1)dataset - contains jpg image of flowers of 5 species in different folders.

2)Create_Dataset.py or .ipynb - this file creates the dataset that will be used by the model. This file needs path to the folder where
'dataset' and 'modelu_functions' are being stored. Please store these two files in the same folder with 'Create_Dataset' for easy
implementation.

3)feature_vectors.csv - this a sample dataset file generated using the "Create_Dataset.ipynb" file that can be used directly by the model
without requiring to run the upper two files.

4)flower_classification_src.py or ipynb - this is the main file that shows the output and accuracy of the model.

5)flower_images.npy - it contains 3-dimensional array of all the images present in the "dataset" file.

6)module_functions.py or ipynb - this function is another important file that contains all the functions required by the model. This 
function is required and imported by "Create_Dataset" and "flower_classification_src", so please keep it in the same folder with the other two.

7)forest.npy - this a sample random forest pregenerated and stored to check the working of the model. This is imported by
"flower_classification_src" to immediately check the results. Otherwise user defined random forest can be generated using the same code.

Please make sure all the python-libraries used by the model are installed in your system.