RUNTIME_NAME = "huggingface_bert-base-uncased_multilabel-classification"
BASE_DIR = "/home/Ravikumar/Developer/Learning-NLP-with-PyTorch/"
WORKING_DIR = "Chapter - 6 : Exploring transformer variants/"
TRAIN_DIR = BASE_DIR +WORKING_DIR + "dataset/prep_train_data.csv"
TEST_DIR = BASE_DIR +WORKING_DIR + "dataset/test.csv"
TEST_PRE_DIR = BASE_DIR +WORKING_DIR + "dataset/prep_test_data.csv"
DATASET_DIR = BASE_DIR +WORKING_DIR + "dataset/"
LABELS = ['Computer Science', 'Physics', 'Mathematics','Statistics', 'Quantitative Biology', 'Quantitative Finance']
NUM_LABELS = 6
MAX_LEN = 450
DEVICE = 'cuda'
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.00001
PATIENCE = 3
CHECKPOINT_DIR = BASE_DIR + WORKING_DIR +"/dataset/checkpoints/"
CHECKPOINT_NAME = CHECKPOINT_DIR + f"{RUNTIME_NAME}_{BATCH_SIZE}_{LEARNING_RATE}.pt"