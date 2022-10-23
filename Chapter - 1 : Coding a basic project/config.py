runtime_name = "bidi_lstm_max_mean_pool_concat_"
device = 'cuda'

base_dir = "Chapter - 1 : Coding a basic project/"
raw_dataset_file = "dataset/imdb_dataset.csv"
preprocessed_dataset_file = "dataset/preprocessed_dataset.csv"
mapping = {"negative":0,"positive" : 1}
vocab_file_name = "dataset/vocab.json"

save_checkpoint_dir = base_dir + "trained_models/"
train_test_data_file = "dataset/train_test_vocabed.pkl"
# word2vec_file = "dataset/prep_word2vectors_32.model"#"dataset/word2vectors_32.model" #  .wordvectors word2vec.wordvectors" dataset/word2vec.wordvectors.vectors.npy
# emb_vec_file = "dataset/prep_emb_vec.pkl"#"dataset/emb_vec.pkl"

MIN_COUNT = 4
TRIM_SIZE = 300 

weight_decay=1e-5
target_columns = ["sentiment"]
input_column = ["trimmed_review"]

max_seq_len = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100

EMBED_SIZE = 32
HIDDEN_SIZE = 64
OUT_DIM = 16

n_labels = 1
patience = 5

debug_mode = False


#Chapter\ -\ 1\ \:\ Coding\ a\ basic\ project/