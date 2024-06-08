from load_data import X_train, y_train

from load_dataset import x_train_cc, x_test_cc, y_train_cc, y_test_cc


TOTAL_NEURONS = 500
INPUT_NEURONS = X_train.shape[1]
HIDDEN_NEURONS = 100
OUTPUT_NEURONS = len(y_train[0])
TOTAL_EPOCHES = 500
PATIENCE = 50