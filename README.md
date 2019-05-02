# cyberbullying-detection: Cyberbullying Detection using Deep Learning

- PREREQUISITES

	- please load GloVe vectors in a folder called 'GloVe' as they have not been uploaded due to large file size. Visit https://nlp.stanford.edu/projects/glove/
	- Python 3.6
	- Scikit-learn
	- Tensorflow
	- Keras
	- NLTK

This implementation has been submitted as a part of the L4 Advanced Project Module towards completion of the M.Eng. (Hons) Computer Science Degree at Durham University, UK.

Our project comprises several different parts. Below is a guide to files wrt our research questions (RQ):

- RQ1: Comparative analysis

	- nn_models.py: Our GRU binary classifier implementation
	- awekar_models.py: LSTM and BLSTM classifiers by (Agrawal and Awekar, 2018)
	- biclass.py: runs all 3 bi-classifiers for comparative analysis wrt RQ1
	- multi_models.py: Our binary classifiers modified for multi-classification
	- multiclass.py: Runs all 3 multi-classifiers for comparative analysis wrt RQ1


- RQ2: Staggered Classification

	- staggered.py: Simple staggered classification for all 3 models
	- sc1.py: Staggered consensus algo 1
	- sc2.py: Staggered consensus algo 2
	- sc3.py: Staggered consensus algo 3


- RQ3: Virtual Adversarial Training

	- adversarial.py: Virtual Adversarial Training Model
	- vadv.py: Script to run virtual adversarial training

- Others

	- data_preprocess.py: Data Cleaning Methods
	- embeddings.py: Text to Embeddings
	- segment.py: for word segmentation - used to split up hashtags
	- one-grams.txt: used for word segmentation