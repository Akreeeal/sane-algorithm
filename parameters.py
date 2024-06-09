from load_data import X_train, y_train




TOTAL_NEURONS = 500
INPUT_NEURONS = X_train.shape[1]
HIDDEN_NEURONS = 100
OUTPUT_NEURONS = len(y_train[0])
TOTAL_EPOCHES = 500
PATIENCE = 50