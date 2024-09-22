# The Free Spoken Digit Dataset (FSDD)
This project makes use of the Free Spoken Digit Dateset.
This dataset consists of recordings of spoken digits by a number of different speakers, all recorded at a sample rate of 8kHz. 
The recordings are trimmed so that they have near minimal silence at the beginnings and ends.

The dataset consists 2,500 recordings from 5 different speakers. 
Using the TensorFlow Datasets API, this dataset can be downloaded and stored directly in a Dataset object using the code below.

```
ds = tfds.load(
    'spoken_digit',
    split='train',
    # data_dir=os.path.join(os.getenv("HOME"), 'tensorflow_datasets', 'spoken_digit'),
    data_dir=os.path.join("data", "spoken_digit"),
    shuffle_files=False
)
```
# VAE 
> An encoder network which parameterises a posterior distribution $q(z|x)$ of discrete latent
random variables $z$ given the input date $x$, a prior distribution $p(z)$, and a decoder with a distribution $p(x|z)$ over input data.

> In this work we introduce the VQ-VAE where we use discrete latent variables with a new way of
training, inspired by vector quantisation (VQ). The posterior and prior distributions are categorical,
and the samples drawn from these distributions index an embedding table. These embeddings are
then used as input into the decoder network.

# VQ-VAE
We develop and train a vector-quantised variational autoencoder (VQ-VAE) model.
This is a variant of the VAE algorithm that makes use of a discrete latent space.
In particular, the VQ-VAE defines a codebook $\mathbf{e} \in \mathbb{R}^{K \times D}$ 
for the latent embedding space, consisting of $K$ latent embedding vectors $e_i$ ($i=1,\ldots,K$), 
each of dimension $D$. The algorithm involves training encoder and decoder networks as usual.
However, for a given input $x$, the encoder output $E(x) \in \mathbb{R}^D$ is quantised to the nearest latent embedding vector:

$$VQ(E(x)) = e_k,\quad\textrm{where }k = \underset{j}{\arg\min}||E(x) - e_j||_2$$

where the $||\cdot||$ norm above is the Euclidean norm in $\mathbb{R}^D$. 
This quantized latent vector is then passed through the decoder to output the likelihood $p_\theta(x \mid z)$ as usual.

The quantisation stage of the VQ-VAE means that it is not possible to compute gradients with respect to the encoder variables. The solution to this problem in the VQ-VAE is to use the _straight through estimator_, in which the gradients computed with respect to the quantised embeddings are simply passed unaltered to the encoder. This process means that the codebook embeddings $e_i$ do not receive any gradient updates (for details, refer to the implementation referenced in question 3, which makes this process explicit). The VQ-VAE objective therefore adds two additional terms to learn the codebook embeddings:

$$
L = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z) ] + ||\textrm{sg}[E(x)] - VQ(E(x))||^2_2 + \beta || E(x) - \textrm{sg}[VQ(E(x))]||_2^2
$$

In the above, $\textrm{sg}$ is the _stop gradient_ operator that is defined as identity in the forward pass but has zero partial derivates
(see [`tf.stop_gradient`](https://www.tensorflow.org/api_docs/python/tf/stop_gradient)). This means that when evaluating the loss function above, the stop gradient operator can be ignored, but when computing derivatives of the loss with respect to the model parameters, the partial derivatives of $\textrm{sg}[E(x)]$ and $\textrm{sg}[VQ(E(x))]$ with respect to the model parameters will be zero. The constant $\beta$ is usually taken set to $\beta=0.25$ by default, as in the original paper.

The first term in the objective $L$ above is the reconstruction loss, 
the second term is the _codebook loss_, 
and the third term is the _commitment loss_. 
The Kullback-Leibler divergence term in the ELBO is constant and so is ignored for training.

In this assessment, we will design, implement, train and evaluate a VQ-VAE for the spoken digit dataset, 
and use it to learn a generative model of the spoken audio.

# Part 1: Data analysis
We will carry out a basic exploration and analysis of the dataset;
computing, displaying and visualising any properties we deem to be relevant.

```
tf.config.list_physical_devices("GPU")

import tensorflow_io as tfio

from tensorflow.keras.layers import Layer, Conv2D, Conv1D, Conv1DTranspose, Conv2DTranspose, Dense, Masking
from tensorflow.keras.models import Sequential, Model
```

## Dataset EDA

```
# Audio sample rate

sr = 8000

# Compute lengths of audio and check the labels present and number of examples

lengths = []
labels = []
for example in ds:
    lengths.append(example['audio'].numpy().size)
    labels.append(example['label'].numpy())

print(f"{len(lengths)} examples in the dataset")
print(f"Maximum audio length: {np.max(lengths)} samples ({np.max(lengths) / sr} seconds)")
print(f"Dataset labels: {set(labels)}")

lengths_secs = np.array(lengths) / sr

plt.figure(figsize=(6, 3))
plt.hist(lengths_secs, histtype='bar', bins=40, edgecolor='black',)
plt.xlabel("Seconds")
plt.ylabel("Count")
plt.title("Unnormalized histogram of audio lengths (in seconds)")
plt.show()


# Check the dataset is balanced
​
unique_labels, counts = np.unique(labels, return_counts=True)
print("Number of examples per label in the dataset:")
pd.DataFrame([[l, c] for l, c in zip(unique_labels, counts)], columns=["Label", "Count"]).set_index('Label').transpose()

```

We can see in the table above that the dataset is balanced, with 250 examples per class. 
(According to the [TFDS webpage](https://www.tensorflow.org/datasets/catalog/spoken_digit),
there are also 5 speakers, with each speaker providing 50 examples of each digi.)

```
# Listen to some samples

for e in ds.shuffle(500).take(5):
    print(f"Label: {e['label']}")
    display(Audio(e['audio'], rate=sr))

# Inspect the scale/range of the digital samples

def compute_sample_range(dataset):
    minval = np.inf
    maxval = -np.inf
    abs_max = []

    for example in dataset:
        audio = example['audio'].numpy()
        minval = np.min([minval, audio.min()])
        maxval = np.max([maxval, audio.max()])
        abs_max.append(np.max([audio.max(), -audio.min()]))

    return minval, maxval, abs_max

minval, maxval, abs_max = compute_sample_range(ds)
print(f"Minimum sample value: {minval}\nMaximum sample value: {maxval}")


```
The audio is 16 bit ($2^{16} / 2 = 32768$). 
The following histogram shows the distribution of the maximum amplitude for each audio example.m


```
plt.figure(figsize=(6, 3))
plt.hist(abs_max, histtype='bar', bins=40, edgecolor='black',)
plt.xlabel("Maximum absolute sample value")
plt.ylabel("Count")
plt.title("Unnormalized histogram of maximum absolute sample value per audio file")
plt.show()
```
As we can see, there is a wide range of maximum amplitudes across the dataset, 
with many examples having a small amplitude.
As a result, we choose to normalise the amplitudes so that each audio example has maximum amplitude equal to one.


## Prepare training and validation Datasets

```
def normalise_samples(example):
    max_val = tf.reduce_max(tf.math.abs(example['audio']))
    example['audio'] = tf.cast(example['audio'] / max_val, tf.float32)
    return example

ds = ds.map(normalise_samples)
```
We will create a stratified 80/20 split of the Dataset, for training and validation.

```
# Sample dataset indices for the training and validation partitions

def compute_val_set_inx(dataset):
    # Recompute the labels here. NB the Dataset is not shuffled!
    labels = []
    for example in dataset:
        labels.append(example['label'].numpy())

    val_size = 0.2  # 50 examples per label
    val_inx = []

    for l in set(labels):
        label_inx = np.where(labels == l)[0]
        num_samples = int(val_size * label_inx.size)
        assert num_samples == 50, num_samples
        val_inx.extend(np.random.choice(label_inx, num_samples, replace=False).tolist())

    return val_inx
val_inx = compute_val_set_inx(ds)
```
```
def create_train_val_datasets(ds, val_inx):
    # Make a tensor of type tf.int64 to match the one by Dataset.enumerate().
    val_inx = tf.constant(val_inx, dtype=tf.int64)

    def index_is_in(index, rest):
        """Returns True is the index is in val_inx"""
        return tf.math.reduce_any(index == val_inx)

    def index_is_not_in(index, rest):
        """Returns True is the index is not `in val_inx"""
        return tf.math.logical_not(tf.math.reduce_any(index == val_inx))

    def drop_index(index, rest):
        return rest

    # Dataset.enumerate() is similar to Python's enumerate().
    # The method adds indices to each elements. Then, the elements are filtered
    # by using the specified indices. Finally unnecessary indices are dropped.
    train_ds = ds.enumerate().filter(index_is_not_in).map(drop_index)
    val_ds = ds.enumerate().filter(index_is_in).map(drop_index)

    return train_ds, val_ds

train_ds, val_ds = create_train_val_datasets(ds, val_inx)
del ds

# Extract only the audio. Add a dummy channel dimension

def extract_audio(example):
    audio = example['audio'][..., tf.newaxis]
    return audio

train_ds = train_ds.map(extract_audio)
val_ds = val_ds.map(extract_audio)


train_ds.element_spec


```

As we can see in the dataset exploration above, 
there is some variation in the lengths of the audio files.
If we create padded batches while training, it is likely that the batch will be filled with a lot of zeros,
and this could affect the quality of the training. 
We consider the following two options:
1) Sample a random, short slice (say 0.5s) from each audio, to minimise the amount of padding
2) Propagate a mask through the model, and mask any parts of the sequence that correspond to padding

We will choose to go with option 2) above. 
This is a more complicated option, to implement,
but will allow us to feed in each audio example in its entirety in every batch. 
Seeing as the audio files are very short, this is computationally feasible.

We cannot use zero values for padding as this is within the range of the sample values.
Hence we choose to pad with a value of -2.0
(this is arbitrary; any value outside the range of $[-1, 1]$ would do).

```
# Shuffle and batch the Datasets

PADDING_VALUE = -2.0
BATCH_SIZE = 64

train_ds = train_ds.shuffle(100).padded_batch(BATCH_SIZE, padding_values=PADDING_VALUE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.padded_batch(BATCH_SIZE, padding_values=PADDING_VALUE).prefetch(tf.data.AUTOTUNE)


```
