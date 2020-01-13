import numpy as np
import random as rn
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_samples, silhouette_score, normalized_mutual_info_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import rbf_kernel

from typing import Tuple

def prepare_datasets(X_first: pd.DataFrame, X_second:pd.DataFrame, y:pd.DataFrame, test_size:float = 0.2, swap_noise:float = 0.15, split_ratio:float = 0.5):
    """Dataset Preprocessing function that splits the datasets into train and test sets, Selects the 25% features with more variance, 
        generates a noisy training dataset, normalizes all datasets, and returns all the preprocessed datasets needed to train the models. 
       Parameters: X_first: First layer dataset.
                   X_second: Second layer dataset.
                   y: original dataset labels.
                   test_size: float indicating the ratio of test to training samples.
                   swap_noise: float indicating the percentage of samples that will be swaped to add noise to the data.
                   split_ratio: float indicating the number of samples that will contain the first dataset, when split to work with multi-input models.
       Returns: the training datasets reduced and normalized,
                the training datasets reduced, noisy and normalized,
                the test datasets reduced and normalized,
                the training datasets reduced, normalized and concatenated, 
                the training datasets reduced, noisy, normalized and concatenated,
                the test dataset reduced and normalized and concatenated,
                the train labels,
                the test labels, 
                the train labels one-hot encoded,
                the test labels one-hot encoded
       the split in two training dataset, the split in two noisy dataset, and the split in two test dataset
    """
    # Split into train and test sets
    X_train_first, X_test_first, X_train_second, X_test_second, y_train, y_test = train_test_split(X_first, X_second, y, test_size=test_size, random_state=1) 
    
    # SELECT 25% OF FEATURES WITH HIGHER VARIANCE FIRST DATASET 
    X_std = np.std(X_train_first)
    X_threshold = np.percentile(X_std, 75)
    X_select = X_std > X_threshold
    X_train_first = X_train_first.loc[:,X_select]
    X_test_first = X_test_first.loc[:,X_select]
    
    # SELECT 25% OF FEATURES WITH HIGHER VARIANCE SECOND DATASET
    X_std = np.std(X_train_second)
    X_threshold = np.percentile(X_std, 75)
    X_select = X_std > X_threshold
    X_train_second = X_train_second.loc[:,X_select]
    X_test_second = X_test_second.loc[:,X_select]
    
    # Add swap noise to training dataset
    X_swapped_first = X_train_first
    X_swapped_second = X_train_second
    num_swaps = round(X_train_first.shape[0]*swap_noise)
    print(f"swapping: {num_swaps} rows.")

    for col in range(X_train_first.shape[1]):
        to_swap_rows = np.random.randint(X_train_first.shape[0], size=num_swaps)
        sample_rows = np.random.randint(X_train_first.shape[0], size=num_swaps)

        X_swapped_first.iloc[to_swap_rows,col] = X_train_first.iloc[sample_rows,col].values
    
    for col in range(X_train_second.shape[1]):
        to_swap_rows = np.random.randint(X_train_second.shape[0], size=num_swaps)
        sample_rows = np.random.randint(X_train_second.shape[0], size=num_swaps)

        X_swapped_second.iloc[to_swap_rows,col] = X_train_second.iloc[sample_rows,col].values
        
    # Normalization of data sets
    # Data Scaling MinMax
    scaler = MinMaxScaler()
    X_first_norm = X_train_first
    X_second_norm = X_train_second
    X_swapped_first_norm = X_swapped_first
    X_swapped_second_norm = X_swapped_second
    X_test_first_norm = X_test_first
    X_test_second_norm = X_test_second

    X_first_norm = pd.DataFrame(scaler.fit_transform(X_first_norm))
    X_test_first_norm = pd.DataFrame(scaler.transform(X_test_first_norm))
    X_second_norm = pd.DataFrame(scaler.fit_transform(X_second_norm))
    X_test_second_norm = pd.DataFrame(scaler.transform(X_test_second_norm))
    X_swapped_first_norm = pd.DataFrame(scaler.fit_transform(X_swapped_first_norm))
    X_swapped_second_norm = pd.DataFrame(scaler.fit_transform(X_swapped_second_norm))
     
    # One hot encode labels
    OH_encoder = LabelEncoder()
    OH_y_train = pd.DataFrame(OH_encoder.fit_transform(np.ravel(y_train)))
    OH_y_test = pd.DataFrame(OH_encoder.transform(np.ravel(y_test)))
    y_train_oh = keras.utils.to_categorical(OH_y_train)
    y_test_oh = keras.utils.to_categorical(OH_y_test)
    
    ## CONCAT DATASETS    
    X_train_concat = pd.concat([X_first_norm, X_second_norm],axis=1)
    X_swapped_concat = pd.concat([X_swapped_first, X_swapped_second],axis=1)
    X_test_concat = pd.concat([X_test_first_norm, X_test_second_norm],axis=1)
        
    return X_first_norm, X_second_norm, X_swapped_first_norm, X_swapped_second_norm, X_test_first_norm, X_test_second_norm, X_train_concat, X_swapped_concat, X_test_concat, y_train, y_test, y_train_oh, y_test_oh


def perform_PCA(X_train, y_train, X_test=None, y_test=None, n_components:int = 10):
    ## Perform PCA
    pca = PCA(n_components=n_components, random_state=1)

    X_train_pca = pca.fit_transform(X_train)
    X_train_pca_labeled = np.c_[X_train_pca , y_train]
    
    ## Plot components ordered by higher variance
    ax1 = plt.subplot(1,1,1)
    ax1.figure.set_size_inches((8, 4))
    sns.barplot(np.arange(np.shape(pca.explained_variance_ratio_)[0]),pca.explained_variance_ratio_)
    plt.xlabel("Eigen values")
    plt.ylabel("Explained variance")
    plt.show()
    print(f"PCA on single-modal explained variance ratio: {pca.explained_variance_ratio_.sum()}")
    
    pc1_explained_variance = pca.explained_variance_ratio_[0]
    pc2_explained_variance = pca.explained_variance_ratio_[1]
    pc1_ratio = pc1_explained_variance / (pc1_explained_variance + pc2_explained_variance)
    
    num_labels = y_train.nunique()[0]

    # Plot First 2 Components training set
    ax = plt.subplot(1,1,1)
    plot_principal_components(X_train_pca_labeled[:,0], X_train_pca_labeled[:,1] ,X_train_pca_labeled[:,-1], pc1_ratio , num_labels, ax)
    
    X_test_pca = []
    
    if X_test != None :
        X_test_pca = pca.transform(X_test)
        X_test_pca_labeled = np.c_[X_test_pca , y_test]

        # Plot First 2 Components test set  
        ax = plt.subplot(1,1,1)
        plot_principal_components(X_test_pca_labeled[:,0], X_test_pca_labeled[:,1] ,X_test_pca_labeled[:,-1] , pc1_ratio, num_labels, ax)
    
    return X_train_pca, X_test_pca

def perform_KPCA(X_train, y_train, X_test=None, y_test=None, n_components=20, kernel="rbf", gamma=0.008, variance_threshold=0.025):
    ## Feature selection
    X_train_select = X_train
    selector = VarianceThreshold(variance_threshold)
    X_train_select = selector.fit_transform(X_train_select)
    X_test_select = selector.transform(X_test)
    
    ## Perform KPCA
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
    X_kpca = kpca.fit_transform(X_train_select)
    X_test_kpca = kpca.transform(X_test_select)

    X_kpca_var = np.var(X_kpca,0)
    X_kpca_var_ratio = X_kpca_var / sum(X_kpca_var)

    X_kpca_train_labeled = np.c_[X_kpca , y_train]
    X_kpca_test_labeled = np.c_[X_test_kpca , y_test]

    num_labels = y_train.nunique()[0]
        
    ax1 = plt.subplot(1,1,1)
    ax1.figure.set_size_inches((8, 4))
    sns.barplot(np.arange(np.shape(X_kpca_var_ratio)[0]),X_kpca_var_ratio)
    plt.xlabel("Eigen values")
    plt.ylabel("Explained variance")
    plt.show()

    print(X_kpca_var_ratio[:6].sum())

    pc1_explained_variance = X_kpca_var_ratio[0]
    pc2_explained_variance = X_kpca_var_ratio[1]
    pc1_ratio = pc1_explained_variance / (pc1_explained_variance + pc2_explained_variance)
    
    # Plot First 2 Components training set
    ax = plt.subplot(1,1,1)
    plot_principal_components(X_kpca_train_labeled[:,0], X_kpca_train_labeled[:,1] , X_kpca_train_labeled[:,-1], pc1_ratio, num_labels, ax)
    
    # Plot First 2 Components test set  
    ax = plt.subplot(1,1,1)
    plot_principal_components(X_kpca_test_labeled[:,0], X_kpca_test_labeled[:,1] ,X_kpca_test_labeled[:,-1], pc1_ratio, num_labels, ax)

    return X_kpca, X_test_kpca

def perform_multi_KPCA(X_first, X_second, y, kernel="rbf", gamma=0.008, mu=0.5):
        num_labels = y.nunique()[0]
        # Apply rbf kernel to divided datasets
        K1 = rbf_kernel(X=X_first, gamma=gamma)
        K2 = rbf_kernel(X=X_second, gamma=gamma)

        Ktot = mu*K1 + (1-mu)*K2
        
        # Use Ktot to perform KPCA 
        kpca = KernelPCA(kernel="precomputed")
        X_kpca = kpca.fit_transform(Ktot)
        X_kpca_var = np.var(X_kpca,0)
        X_kpca_var_ratio = X_kpca_var / sum(X_kpca_var)
        X_kpca_train_labeled = np.c_[X_kpca , y]
        
        ax1 = plt.subplot(1,1,1)
        ax1.figure.set_size_inches((8, 4))
        sns.barplot(np.arange(np.shape(X_kpca_var_ratio[:20])[0]),X_kpca_var_ratio[:20])
        plt.xlabel("Eigen values")
        plt.ylabel("Explained variance")
        plt.show()
        
        print(X_kpca_var_ratio[:6].sum())

        pc1_explained_variance = X_kpca_var_ratio[0]
        pc2_explained_variance = X_kpca_var_ratio[1]
        pc1_ratio = pc1_explained_variance / (pc1_explained_variance + pc2_explained_variance)
        
        # Plot first 2 principal components
        ax = plt.subplot(1,1,1)
        plot_principal_components(X_kpca_train_labeled[:,0], X_kpca_train_labeled[:,1] ,X_kpca_train_labeled[:,-1], pc1_ratio, num_labels, ax)
    
        return X_kpca, kpca
    
def build_and_train_autoencoder(X_train_input, X_train_reconstruct, validation_data, encoding_dim=20, regularizer=tf.keras.regularizers.l1_l2(0.0001,0), dropout=0.5, epochs=100):
    """Single Input Autoencoder building and training function
       Parameters: X_train: training dataset.
                   X_test: test dataset.
                   y_train: training labels.
                   y_test: test labels.
                   encoding_dim: Size of the latent space (bottleneck layer size).
                   regularizer: keras regularizer object
                   dropout: float indicating dropout probability
       Returns the 3 trained models: full autoencoder, the encoder part and the decoder part. We will use the encoder to get the latent representation.
    """
    # Set Optimizer: Adam with learning rate=0.001
    optimizer = tf.keras.optimizers.Adam(0.001)
    ## Call autoencoder build function and get the AE, the encoder and the decoder.
    autoencoder, encoder, decoder = build_autoencoder(encoding_dim=encoding_dim, number_features=X_train_input.shape[1], regularizer=regularizer, dropout=dropout)
    # Compile the autoencoder using Mean Square Error loss function.
    autoencoder.compile(optimizer=optimizer,
                            loss="mse",
                            metrics=['mse'])
    
    # Set Early Stop Callback And Reduce LR on Plateu
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=10,  mode='auto', baseline=None, restore_best_weights=False, verbose=1)
    rlrop = keras.callbacks.ReduceLROnPlateau(monitor='loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
    
    ## TRAINING
    # Fit the training data into the autoencoder.
    history = autoencoder.fit(X_train_input,
                              X_train_reconstruct,
                              validation_data=validation_data,
                              epochs=epochs,
                              verbose=0,
                              callbacks=[early_stop, rlrop])
    # Plot training vs validation losses
    plt.plot(history.history["loss"], c = 'b', label = "Training")
    plt.plot(history.history["val_loss"], c = (0.5, 0.3, 0.2), label = "Validation")
    plt.title("Autoencoder Loss during training epochs")
    plt.legend()
    plt.show()
    loss = history.history["loss"][-1] 
    return autoencoder, encoder, decoder, loss


def build_and_train_multi_autoencoder(X_train_input, X_train_reconstruct, encoding_dim=20, regularizer=tf.keras.regularizers.l1_l2(0.0001,0), dropout=0.5, epochs=100, mu=0.5):
    """Single Input Autoencoder building and training function
       Parameters: X_train: training dataset.
                   X_test: test dataset.
                   y_train: training labels.
                   y_test: test labels.
                   encoding_dim: Size of the latent space (bottleneck layer size).
                   regularizer: keras regularizer object
                   dropout: float indicating dropout probability
                   epochs: number of epochs to train the model
                   mu: weight balance for each layer (i.e. mu=0.5 -> each layer loss function has equal weight)
       Returns the 3 trained models: full autoencoder, the encoder part and the decoder part. We will use the encoder to get the latent representation.
    """
    # Set Optimizer: Adam with learning rate=0.001
    optimizer = tf.keras.optimizers.Adam(0.001)
    ## Call autoencoder build function and get the AE, the encoder and the decoder.
    autoencoder_multi, encoder_multi, decoder_multi = build_multi_autoencoder(encoding_dim=encoding_dim, number_features=(X_train_input[0].shape[1],X_train_input[1].shape[1]), regularizer=regularizer, dropout=dropout)
    # Compile the autoencoder using Mean Square Error loss function.
    autoencoder_multi.compile(optimizer=optimizer,
                            loss=["mse","mse"],
                            loss_weights=[mu, 1-mu],
                            metrics=['mse'])
    
    # Set Early Stop Callback And Reduce LR on Plateu
    early_stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00005, patience=10,  mode='auto', baseline=None, restore_best_weights=False, verbose=1)
    rlrop = keras.callbacks.ReduceLROnPlateau(monitor='loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
    
    # Fit the training data into the multi-input DAE.
    history = autoencoder_multi.fit(X_train_input,X_train_reconstruct,
                                      epochs=epochs,
                                      verbose=0,
                                      callbacks=[early_stop, rlrop])
    # Plot training vs validation losses
    plt.plot(history.history["loss"], c = 'b', label = "Training")
    plt.title("Autoencoder (Multi) Loss during training epochs")
    plt.legend()
    plt.show()
    loss = history.history["loss"][-1] 
    print(loss)

    return autoencoder_multi, encoder_multi, decoder_multi, loss

class CustomEarlyStopping(keras.callbacks.EarlyStopping):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch % 50 == 0):
            print(f"epoch {epoch}")
            print(self.get_monitor_value(logs))
            print(self.min_delta)

# Define the model using the keras functional API
def build_autoencoder(encoding_dim: int, number_features: int, regularizer: tf.keras.regularizers.Regularizer, dropout: float):
    """Two-input autoencoder build function
       Parameters: encoding_dim: Size of the latent space (bottleneck layer size).
                   number_features: Tuple with the sizes of the two inputs.
                   regularizer: keras regularizer object
                   dropout: float indicating dropout probability
       Returns the 3 models: full autoencoder, the encoder part and the decoder part
    """
    if dropout > 1:
        dropout = 1
    elif dropout < 0:
        dropout = 0
    # this is the reduction of our encoded representations, in times.
    print(f"Compression: {number_features/encoding_dim}")

    first_layer_size = 400

    ## input->400->encoding_dim->Dropout->BatchNorm->400->input
    
    ## ENCODER
    # encoder first input placeholder.
    first_input = layers.Input(shape=(number_features))
    # encoder first Hidden Layer - H1
    H1 = layers.Dense(first_layer_size, activation='relu', kernel_regularizer=regularizer)(first_input)
   
    ## BOTTLENECK 
    bottleneck = layers.Dense(encoding_dim, activation='relu', kernel_regularizer=regularizer)(H1)

    # this model maps an input to its encoded representation
    encoder = keras.models.Model(first_input, bottleneck, name='encoder')

    ## DECODER
    # Decoder Input Layer - Encoding dimension
    encoded_input = layers.Input(shape=(encoding_dim,))
    # decoder first Dropout Layer - D3
    D3 = layers.Dropout(dropout)(encoded_input)
    # decoder first Batch Normalization Layer - BN3 
    BN3 = layers.BatchNormalization()(D3)
    # decoder first Hidden Layer - H3
    H3 = layers.Dense(first_layer_size, activation='relu', kernel_regularizer=regularizer)(BN3)
    # decoder reconstruction layer - O1
    O1 = layers.Dense(number_features, activation='sigmoid')(H3)

    # create the decoder model
    decoder = keras.models.Model(encoded_input, O1)

    # create the full autoencoder
    encoder_model = encoder(first_input)
    decoder_model = decoder(encoder_model)

    autoencoder = keras.models.Model(first_input, decoder_model, name="autoencoder")
    
    return autoencoder, encoder, decoder

def build_multi_autoencoder(encoding_dim: int, number_features: Tuple, regularizer: tf.keras.regularizers.Regularizer, dropout: float):
    """Two-input autoencoder build function
       Parameters: encoding_dim: Size of the latent space (bottleneck layer size).
                   number_features: Tuple with the sizes of the two inputs.
                   regularizer: keras regularizer object
       Returns the 3 models: full autoencoder, the encoder part and the decoder part
    """
    if dropout > 1:
        dropout = 1
    elif dropout < 0:
        dropout = 0
    # this is the reduction of our encoded representations, in times.
    print(f"Compression: {sum(number_features)/encoding_dim}")

    first_layer_size = 200
    
    ## input_first -> 400 -\                                          /-> 400 -> input_first
    ##                      |-> encoding_dim -> dropout -> batchNorm |
    ## input_second-> 400 -/                                          \-> 400 -> input_second
    
    ## First Dataset input path
    # encoder first input placeholder.
    first_input = layers.Input(shape=(number_features[0]))
    # encoder first path first Hidden Layer - H11
    H11 = layers.Dense(first_layer_size, activation='relu', kernel_regularizer=regularizer)(first_input)
   
    ## Second Dataset input path
    # encoder second input placeholder
    second_input = layers.Input(shape=(number_features[1]))
    # encoder second path first Hidden Layer - H21
    H21 = layers.Dense(first_layer_size, activation='relu', kernel_regularizer=regularizer)(second_input)
    
    ## Concatenate paths - Bottleneck 
    concatenated = layers.concatenate([H11, H21], axis=-1)
    bottleneck = layers.Dense(encoding_dim, activation='relu', kernel_regularizer=regularizer)(concatenated)

    # this model maps an input to its encoded representation
    encoder = keras.models.Model([first_input,second_input], bottleneck, name='encoder')

    ## Decoder Outputs
    # Decoder Input Layer - Encoding dimension - D1
    encoded_input = layers.Input(shape=(encoding_dim,))
    # decoder bottleneck Dropout Layer - Db
    Db = layers.Dropout(dropout)(encoded_input)
    # decoder bottleneck Batch Normalization Layer - BNb 
    BNb = layers.BatchNormalization()(Db)
    
    ## Paths Split
    ## First Dataset output path
    # decoder first path first Hidden Layer - H12
    H12 = layers.Dense(first_layer_size, activation='relu', kernel_regularizer=regularizer)(BNb)
    # decoder first path reconstruction layer - O1
    O1 = layers.Dense(number_features[0], activation='sigmoid')(H12)
    
    ## Second path output hidden
    # decoder second path first Hidden Layer - H22
    H22 = layers.Dense(first_layer_size, activation='relu', kernel_regularizer=regularizer)(BNb)
    # decoder second path reconstruction layer - O2
    O2 = layers.Dense(number_features[1], activation='sigmoid')(H22)

    # create the decoder model
    decoder = keras.models.Model(encoded_input, [O1, O2])

    # create the full autoencoder
    encoder_model = encoder([first_input, second_input])
    decoder_model = decoder(encoder_model)

    autoencoder = keras.models.Model([first_input,second_input], decoder_model, name="autoencoder")
    
    return autoencoder, encoder, decoder

def encode_dataset(X, encoder):
    # Encode datasets using the trained encoder.
    X_encoded = encoder.predict(X)
    # Renormalize data
    scaler = MinMaxScaler()
    X_encoded = pd.DataFrame(scaler.fit_transform(X_encoded))
    return X_encoded


def classify(X, X_test, y, y_test, model_type="Model"):
    """Classification function. Classifies the X dataset using the 3 models: LR, SVM y RF
       Parameters: X: training dataset.
                   X_test: test dataset.
                   y: training labels.
                   y_test: test labels.
                   model_type: name of the model, for displaying
       Returns the test accuracy for the 3 models.
    """
    print(f"Results for {model_type}: \n")
    lr_accuracy, lr_auc = logistic_regression(X, X_test, y, y_test)
    svm_accuracy, svm_auc = support_vector_machine(X, X_test, y, y_test)
    rf_accuracy, rf_auc = random_forest(X, X_test, y, y_test)
    return [lr_accuracy, svm_accuracy, rf_accuracy, lr_auc, svm_auc, rf_auc]

def classify_with_cv(X, X_test, y, y_test, model_type="Model"):
    """Classification function. Classifies the X dataset using the 3 models: LR, SVM y RF
       Parameters: X: training dataset.
                   X_test: test dataset.
                   y: training labels.
                   y_test: test labels.
                   model_type: name of the model, for displaying
       Returns the test accuracy for the 3 models.
    """
    print(f"Results for {model_type}: \n")
    params_grid = [{'C': [0.0001,0.0005,0.001,0.005,0.01,0.05,0.075,0.1, 0.5]}]
    lr_accuracy, lr_auc = logistic_regression(X, X_test, y, y_test, params_grid)
    
    params_grid = [{'gamma': [1e-3,5e-3,1e-2,5e-2,1e-1,5e-1],
                         'C': [0.001,0.005,0.01,0.05,0.1,0.5]}]
    svm_accuracy, svm_auc = support_vector_machine(X, X_test, y, y_test, params_grid)
    
    params_grid = [{'n_estimators': [80,90,100,110,115,120], 'max_depth':[8,9,10,11,12]}]
    rf_accuracy, rf_auc = random_forest(X, X_test, y, y_test, params_grid)
    
    return [lr_accuracy, svm_accuracy, rf_accuracy, lr_auc, svm_auc, rf_auc]
    
def logistic_regression(X_train, X_test, y_train, y_test, params_grid=[]):
    """Logistic Regression classifier function, with parameters tuned in Singleomic_Classifiers notebook. Prints the confusion matrix and classification report.
       Parameters: X: training dataset.
                   X_test: test dataset.
                   y: training labels.
                   y_test: test labels.
        Returns: The model test accuracy
    """
    clf = LogisticRegression(random_state=0, solver='lbfgs', dual=False, penalty='l2')

    if len(params_grid) == 0:
        params_grid = [{'C':[0.08]}]
        
    # Perform Cross Validation Grid Search to find best hyperparamters
    clf_grid = GridSearchCV(estimator=clf, param_grid=params_grid, cv=5)
    clf_grid.fit(X_train, y_train)

    # Print training scores
    print('Best score for training data:', clf_grid.best_score_,"\n") 
    print('Best C:',clf_grid.best_estimator_.C,"\n") 

    # Select the estimator with the best hyperparameters
    clf = clf_grid.best_estimator_
    
    # Predict classification with final model
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    
    roc_auc = calc_roc_curve(y_test, y_score[:,1])
    cm = confusion_matrix(y_test,y_pred)

    print(cm)
    print("\n")
    print(classification_report(y_test,y_pred))
    test_score = clf.score(X_test  , y_test)
    print("Training set score for Logistic Regression: %f" % clf.score(X_train, y_train))
    print("Testing  set score for Logistic Regression: %f" % test_score)
    return test_score, roc_auc

def support_vector_machine(X_train, X_test, y_train, y_test, params_grid=[]):
    """Support Vector Machine classifier function, with parameters tuned in Singleomic_Classifiers notebook. Prints the confusion matrix and classification report.
       Parameters: X: training dataset.
                   X_test: test dataset.
                   y: training labels.
                   y_test: test labels.
        Returns: The model test accuracy
    """
    if len(params_grid) == 0:
        params_grid = [{'C':[0.1]},{"gamma":[0.1]}]

    svm = SVC(kernel = 'rbf', random_state = 0, probability=True)
    # Perform CV to tune parameters for best SVM fit

    # Perform Cross Validation Grid Search to find best hyperparamters
    svm_grid = GridSearchCV(estimator=svm, param_grid=params_grid, cv=5)
    svm_grid.fit(X_train, y_train)

    # Print training scores
    print('Best score for training data:', svm_grid.best_score_,"\n") 
    print('Best C:',svm_grid.best_estimator_.C,"\n") 
    print('Best Gamma:',svm_grid.best_estimator_.gamma,"\n")

    # Select the estimator with the best hyperparameters
    svm = svm_grid.best_estimator_

    # Predict classification with final model
    y_pred = svm.predict(X_test)
    y_score = svm.predict_proba(X_test)
    
    roc_auc = calc_roc_curve(y_test, y_score[:,1])
    cm = confusion_matrix(y_test,y_pred)

    print(cm)
    print("\n")
    print(classification_report(y_test,y_pred))
    test_score = svm.score(X_test  , y_test)
    print("Training set score for SVM: %f" % svm.score(X_train, y_train))
    print("Testing  set score for SVM: %f" % test_score)
    return test_score, roc_auc

def random_forest(X_train, X_test, y_train, y_test, params_grid=[]):
    """Random Forest classifier function, with parameters tuned in Singleomic_Classifiers notebook. Prints the confusion matrix and classification report.
       Parameters: X: training dataset.
                   X_test: test dataset.
                   y: training labels.
                   y_test: test labels.
        Returns: The model test accuracy
    """
    if len(params_grid) == 0:
        params_grid = [{'n_estimators':[140]},{"max_depth":[12]}]
    
    rfc = RandomForestClassifier(random_state=0, class_weight="balanced_subsample") # try class_weights "balanced" and "balanced_subsample"
    
    # Perform Cross Validation Grid Search to find best hyperparamters
    rfc_grid = GridSearchCV(estimator=rfc, param_grid=params_grid, cv=5)
    rfc_grid.fit(X_train, y_train)

    # Print training scores
    print('Best score for training data:', rfc_grid.best_score_,"\n") 
    print('Best #estimators:',rfc_grid.best_estimator_.n_estimators,"\n") 
    print('Best max depth:',rfc_grid.best_estimator_.max_depth,"\n") 

    # Select the estimator with the best hyperparameters
    rfc = rfc_grid.best_estimator_
    # Predict classification with final model
    y_pred = rfc.predict(X_test)
    y_score = rfc.predict_proba(X_test)
    
    roc_auc = calc_roc_curve(y_test, y_score[:,1])
    cm = confusion_matrix(y_test,y_pred)

    print(cm)
    print("\n")
    print(classification_report(y_test,y_pred))
    test_score = rfc.score(X_test  , y_test)
    print("Training set score for RFC: %f" % rfc.score(X_train, y_train))
    print("Testing  set score for RFC: %f" % test_score)
    return test_score, roc_auc
 
    
def calc_roc_curve(y_test, y_pred):
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr,tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc
    
def cluster(X, y, model_type="Model"):
    """Clustering function. Performs clustering algorithms (KMeans, Hierarchical Clustering and Spectral Clustering) on the input dataset.
       Parameters: X: training dataset.
                   y: training labels.
                   model_type: the name of the model that was used to compute the input dataset.
        Returns: the resulting silhouette score and mutual information score
    """
    n_clusters = [2,3,4,5,6]
    silhouette_kmeans, mutual_info_kmeans = k_means(X, y, n_clusters, model_type)
    silhouette_spectral, mutual_info_spectral = spectral_cluster(X, y, n_clusters, model_type)
    silhouette_hierarchical, mutual_info_hierarchical = hierarchical_cluster(X, y, n_clusters, model_type)
    return [silhouette_kmeans, silhouette_spectral, silhouette_hierarchical ,mutual_info_kmeans, mutual_info_spectral, mutual_info_hierarchical]

def k_means(X:pd.DataFrame, y:pd.DataFrame, n_clusters:int, model_type:str ="Model"):

    silhouette_scores = []
    mutual_info_scores = []
    mutual_info = 0
    for n in n_clusters:
        kmeans = KMeans(n_clusters=n, random_state=0)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        if (n==2):
            mutual_info = normalized_mutual_info_score(np.ravel(y), cluster_labels,average_method='arithmetic')
            print(f"mutual information: {mutual_info}")
            ## PCA to visualize new labels
            pca = PCA(n_components=n_components, random_state=1)
            X_train_pca = pca.fit_transform(X)
            X_train_pca_labeled = np.c_[X , y]
            X_train_pca_cluster_labeled = np.c_[X , cluster_labels]
        
        print(f"{model_type} {n} clusters -  silhoutte score: {silhouette_avg}")
        silhouette_scores.append(silhouette_avg)
        
    plt.plot(n_clusters,silhouette_scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()
    
    return silhouette_scores[0], mutual_info

def spectral_cluster(X, y, n_clusters, model_type="Model"):
    silhouette_scores = []
    mutual_info_scores = []
    mutual_info = 0
    for n in n_clusters:
        spectral = SpectralClustering(n_clusters=n, random_state=0)
        cluster_labels = spectral.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        if (n==2):
            mutual_info = normalized_mutual_info_score(y, cluster_labels,average_method='arithmetic')
            print(f"mutual information: {mutual_info}")
        
        print(f"{model_type} {n} clusters -  silhoutte score: {silhouette_avg}")
        silhouette_scores.append(silhouette_avg)
        
    plt.plot(n_clusters,silhouette_scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()
    return silhouette_scores[0], mutual_info

def hierarchical_cluster(X, y, n_clusters, model_type="Model"):
    silhouette_scores = []
    mutual_info_scores = []
    mutual_info = 0
    for n in n_clusters:
        hierarchical = AgglomerativeClustering(n_clusters=n)
        cluster_labels = hierarchical.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        if (n==2):
            mutual_info = normalized_mutual_info_score(y, cluster_labels,average_method='arithmetic')
            print(f"mutual information: {mutual_info}")
        
        print(f"{model_type} {n} clusters -  silhoutte score: {silhouette_avg}")
        silhouette_scores.append(silhouette_avg)
        
    plt.plot(n_clusters,silhouette_scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()
    return silhouette_scores[0], mutual_info
    
## PLOT FUNCTIONS
def plot_principal_components(pc1, pc2, y, pc1_ratio, num_labels, ax):
    
    size_multiplier = (pc1_ratio/(1 - pc1_ratio))
    if (pc1_ratio == 1):
        size_multiplier = 1
    
    sns.scatterplot(x=pc1, 
                    y=pc2, 
                    alpha = 0.8, 
                    s= 50, legend='full', 
                    hue=y,
                    palette=sns.color_palette("hls")[:(-num_labels-1):-1])
    ax.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.,framealpha=1, frameon=True, fontsize="x-small")
    ax.set_xlabel(f"PC 1 {pc1_ratio:.2f}")
    ax.set_ylabel(f"PC 2 {1-pc1_ratio:.2f}")
    ax.figure.set_size_inches((4*size_multiplier) , 4)
    ax.set_title("PCA")
    plt.yticks(rotation=45) 
    plt.show()

def plot_hyperparam_tune(hyperparam, results, legends):

    
    
    ax.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.,framealpha=1, frameon=True, fontsize="x-small")
    ax.set_xlabel(f"PC 1 {pc1_ratio:.2f}")
    ax.set_ylabel(f"PC 2 {1-pc1_ratio:.2f}")
    ax.figure.set_size_inches((4*(pc1_ratio/(1 - pc1_ratio)) , 4))
    ax.set_title("PCA")
    plt.yticks(rotation=45) 
    plt.show()
    
