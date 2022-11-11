runtime_name = "huggingface_bert-base-uncased"
device = 'cpu'
base_folder = "/home/ravikumar/Desktop/NLP/Learning-NLP-with-PyTorch/"
base_dir = "Chapter - 1 : Coding a basic project/"
raw_dataset_file = "dataset/imdb_dataset.csv"
preprocessed_dataset_file = "dataset/preprocessed_dataset.csv"
mapping = {"negative":0,"positive" : 1}
vocab_file_name = "dataset/vocab.json"
checkpoint = "bert-base-uncased"
train_test_data_file =base_folder +  base_dir + "dataset/train_test_vocabed.pkl"
emb_vec_file = "dataset/prep_emb_vec.pkl"

MIN_COUNT = 4
TRIM_SIZE = 300 

weight_decay=1e-5
target_columns = ["sentiment"]
input_column = ["trimmed_review"]

max_seq_len = 300
BATCH_SIZE = 4
LEARNING_RATE = 0.00001
EPOCHS = 100

EMBED_SIZE = 32
HIDDEN_SIZE = 64
OUT_DIM = 16

n_labels = 1
patience = 3

debug_mode = True

save_checkpoint_dir =  base_dir + "checkpoints/"
checkpoints_file = f"{save_checkpoint_dir}{runtime_name}_seq2seq_hidden_{HIDDEN_SIZE}_embed_{EMBED_SIZE}.pt"