import numpy as np
import random as rn
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

from typing import Tuple

def prepare_datasets(X: pd.DataFrame, y:pd.DataFrame, test_size: float):
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=1) # Drop the Donor ID column from both datasets
    
    # Add swap noise to training dataset
    # Swap Noise 15% - 1700*0.15 = 255
    X_swapped = X_train
    swap_noise = 0.15
    num_swaps = round(X_train.shape[0]*swap_noise)
    print(f"swapping: {num_swaps} rows.")

    for col in range(X_train.shape[1]):
        to_swap_rows = np.random.randint(X_train.shape[0], size=num_swaps)
        sample_rows = np.random.randint(X_train.shape[0], size=num_swaps)

        X_swapped.iloc[to_swap_rows,col] = X_train.iloc[sample_rows,col].values

    # Normalization of data sets
    # Data Scaling MinMax
    scaler = MinMaxScaler()
    X_train_norm = X_train
    X_train_swapped = X_swapped
    X_test_norm = X_test

    X_train_norm = pd.DataFrame(scaler.fit_transform(X_train_norm))
    X_train_swapped = pd.DataFrame(scaler.fit_transform(X_train_swapped))
    X_test_norm = pd.DataFrame(scaler.transform(X_test_norm))
     
    # One hot encode labels
    OH_encoder = LabelEncoder()
    OH_y_train = pd.DataFrame(OH_encoder.fit_transform(y_train))
    OH_y_test = pd.DataFrame(OH_encoder.transform(y_test))
    y_train_oh = keras.utils.to_categorical(OH_y_train)
    y_test_oh = keras.utils.to_categorical(OH_y_test)
    
    return X_train_norm, X_train_swapped, X_test_norm, y_train, y_test, y_train_oh, y_test_oh

def perform_PCA(X_train, X_test, y_train, y_test, n_components: int):
    ## Perform PCA
    pca = PCA(n_components=n_components, random_state=1)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    X_train_pca_labeled = np.c_[X_train_pca , y_train]
    X_test_pca_labeled = np.c_[X_test_pca , y_test]

    num_labels = y_train.nunique()
    
    sns.set_style("white")
    sns.set_context("talk")
    sns.set_style("ticks")
    
    ## Plot components ordered by higher variance
    ax1 = plt.subplot(1,1,1)
    ax1.figure.set_size_inches((8, 4))
    sns.barplot(np.arange(np.shape(pca.explained_variance_ratio_)[0]),pca.explained_variance_ratio_)
    plt.xlabel("Eigen values")
    plt.ylabel("Explained variance")
    plt.show()
    print(f"PCA on single-modal explained variance ratio: {pca.explained_variance_ratio_.sum()}")
    
    # Plot First 2 Components training set
    ax = plt.subplot(1,2,1)
    plot_principal_components(X_train_pca_labeled[:,0], X_train_pca_labeled[:,1] ,X_train_pca_labeled[:,-1] , num_labels, ax)
    
    # Plot First 2 Components test set  
    ax = plt.subplot(1,2,2)
    plot_principal_components(X_test_pca_labeled[:,0], X_test_pca_labeled[:,1] ,X_test_pca_labeled[:,-1] , num_labels, ax)
    
    plt.show()
    return X_train_pca, X_test_pca
    
def perform_KPCA(X_train, X_test, y_train, y_test, n_components=20, kernel="rbf", gamma=0.008, variance_threshold=0.025):
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

    num_labels = y_train.nunique()
        
    ax1 = plt.subplot(1,1,1)
    ax1.figure.set_size_inches((8, 4))
    sns.barplot(np.arange(np.shape(X_kpca_var_ratio)[0]),X_kpca_var_ratio)
    plt.xlabel("Eigen values")
    plt.ylabel("Explained variance")
    plt.show()

    print(X_kpca_var_ratio[:6].sum())

    sns.set_style("white")
    sns.set_context("talk")
    sns.set_style("ticks")
  
    # Plot First 2 Components training set
    ax = plt.subplot(1,2,1)
    plot_principal_components(X_kpca_train_labeled[:,0], X_kpca_train_labeled[:,1] , X_kpca_train_labeled[:,-1] , num_labels, ax)
    
    # Plot First 2 Components test set  
    ax = plt.subplot(1,2,2)
    plot_principal_components(X_kpca_test_labeled[:,0], X_kpca_test_labeled[:,1] ,X_kpca_test_labeled[:,-1] , num_labels, ax)
    
    plt.show()

    return X_kpca, X_test_kpca

def build_and_train_autoencoder(X_train_input, X_train_reconstruct, X_test, y_train, y_test, encoding_dim=20, regularizer=tf.keras.regularizers.l1_l2(0.0001,0), dropout=0.5, epochs=100):
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
    
    ## TRAINING
    # Fit the training data into the autoencoder.
    history = autoencoder.fit(X_train_input,
                              X_train_reconstruct,
                              validation_data=(X_test,X_test),
                              epochs=epochs,
                              verbose=1,
                              callbacks=[])
    # Plot training vs validation losses
    plt.plot(history.history["loss"], c = 'b', label = "Training")
    plt.plot(history.history["val_loss"], c = 'r', label = "Validation")
    plt.title("Autoencoder Loss during training epochs")
    plt.legend()
    plt.show()
    print(history.history["loss"][-1])
    return autoencoder, encoder, decoder

def encode_dataset(X, encoder):
    # Encode datasets using the trained encoder.
    X_train_encoded = encoder.predict(X)
    # Renormalize data
    scaler = MinMaxScaler()
    X_train_encoded = pd.DataFrame(scaler.fit_transform(X_train_encoded))
    return X_train_encoded


def classify(X, X_test, y, y_test, model_type="Model"):
    """Single Input Autoencoder building and training function
       Parameters: X: training dataset.
                   X_test: test dataset.
                   y: training labels.
                   y_test: test labels.
                   encoding_dim: Size of the latent space (bottleneck layer size).
                   regularizer: keras regularizer object
                   dropout: float indicating dropout probability
       Returns the 3 trained models: full autoencoder, the encoder part and the decoder part. We will use the encoder to get the latent representation.
    """
    # Set Early Stop Callback
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10,  mode='auto', baseline=None, restore_best_weights=False, verbose=1)

    # Fit best model with dimensionality reduction data
    classifier = Models.build_best_classifier(input_shape=(X.shape[1],) ,dropout=0.5, l1=0.0001, l2=0.0001)
    history = classifier.fit(X, y, epochs=150,
                        validation_split = 0.1, verbose=0, callbacks=[early_stop], shuffle=False)
    hist = pd.DataFrame(history.history)

    test_loss, test_acc = classifier.evaluate(X_test, y_test)
    print(f"Results for {model_type}: Loss: {test_loss} - Accuracy: {test_acc}")


def build_best_classifier(input_shape: Tuple, dropout: int, l1: int, l2: int):
    model = keras.Sequential([
        layers.Dense(1000, activation=tf.nn.relu ,kernel_regularizer=keras.regularizers.l1_l2(l1,l2), input_shape=input_shape),
        layers.Dropout(dropout),
        layers.BatchNormalization(),  
        layers.Dense(20,activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l1_l2(l1,l2)),
        layers.Dropout(dropout),
        layers.BatchNormalization(),
        layers.Dense(2,activation=tf.nn.softmax)
  ])

    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    return model


def cluster(X, n_clusters, model_type="Model"):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)

    print(f"{Model} silhoutte score: {silhouette_avg}")

    ### PLOT SILOHUETTE SCORE FOR CLUSTERS
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()


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

    first_layer_size = number_features/40
    second_layer_size = number_features/120
    
    ## ENCODER
    # encoder first input placeholder.
    first_input = layers.Input(shape=(number_features))
    # encoder first Hidden Layer - H1
    H1 = layers.Dense(first_layer_size, activation='relu', kernel_regularizer=regularizer)(first_input)
    # encoder first Dropout Layer - D1
    D1 = layers.Dropout(dropout)(H1)
    # encoder first Batch Normalization Layer - BN1
    BN1 = layers.BatchNormalization()(D1)
    # encoder second Hidden Layer - H2
    H2 = layers.Dense(second_layer_size, activation='relu', kernel_regularizer=regularizer)(BN1)
    # encoder second Dropout Layer - D2
    D2 = layers.Dropout(dropout)(H2)
    # encoder first path second Batch Normalization Layer - BN2
    BN2 = layers.BatchNormalization()(D2)

   
    ## BOTTLENECK 
    bottleneck = layers.Dense(encoding_dim, activation='relu', kernel_regularizer=regularizer)(BN2)

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
    H3 = layers.Dense(second_layer_size, activation='relu', kernel_regularizer=regularizer)(BN3)
    # decoder second Dropout Layer - D4
    D4 = layers.Dropout(dropout)(H3)
    # decoder second Batch Normalization Layer - BN4 
    BN4 = layers.BatchNormalization()(D4)
    # decoder reconstruction layer - O1
    O1 = layers.Dense(number_features, activation='sigmoid')(BN4)

    # create the decoder model
    decoder = keras.models.Model(encoded_input, O1)

    # create the full autoencoder
    encoder_model = encoder(first_input)
    decoder_model = decoder(encoder_model)

    autoencoder = keras.models.Model(first_input, decoder_model, name="autoencoder")
    
    return autoencoder, encoder, decoder


## PLOT FUNCTIONS
def plot_principal_components(pc1, pc2, y, num_labels, ax):
    sns.scatterplot(x=pc1, 
                    y=pc2, 
                    alpha = 0.8, 
                    s= 75, legend='full', 
                    hue=y,
                    palette=sns.color_palette("hls")[:(-num_labels-1):-1])
    ax.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.,framealpha=1, frameon=True, fontsize="x-small")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.figure.set_size_inches( (16,8) )
    ax.set_title("PCA")
    plt.yticks(rotation=45) 
    

