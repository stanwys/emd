# EMD
Exploration of Massive Data course project - classification problem of rating mobile apps.

# Run program and test classification model 
##Test model: 

python main.py 4 path_to_csv_file path_to_model

For example:
python main.py 4 path_to_csv_file models/mlp

##Train final models(MLP and RF):

python main.py 3 path_to_csv_file output_folder

##Test majority rule:
python main.py 5 path_to_csv_file 

##Test random classification:
python main.py 6 path_to_csv_file num_repeats