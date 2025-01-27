<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="https://i.ibb.co/zZNbRCq/Music-logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center"><b>Songs Recommender System</b></h3>
</div>

# Table of Contents

1. [About](#about)  

2. [Data](#data)  
   - [How was the dataset prepared](#how-was-the-dataset-prepared)  
   - [Song fragmentation](#song-fragmentation)  

3. [Vector Representation Using Neural Networks](#vector-representation-using-neural-networks)  
   - [Neural network architecture](#neural-network-architecture)  
   - [Evaluation of vector quality](#evaluation-of-vector-quality)  
     - [Grouped](#grouped)  
     - [Full breakdown](#full-breakdown)
   - [Training Infrastructure and Data Storage](#training-infrastructure-and-data-storage)  

4. [How Recommendations Are Made](#how-recommendations-are-made)  
   - [Way of choosing recommendations](#way-of-choosing-recommendations)  
   - [Explainability](#explainability)  
   - [Recommendation example](#recommendation-example)  

5. [Future Work and Improvements](#future-work-and-improvements)  


# About

This project focuses on developing a recommendation system for songs based on their audio tracks, leveraging melodic spectrograms.
The core idea behind the project was to go beyond metadata like artist names or genres, using audio data directly to uncover deeper musical connections.
We aim to represent song fragments as vectors, which will be obtained through the use of a neural network, enabling more accurate song recommendations based on the audio content itself.

# Data

To gather the necessary data for the recommendation system, we developed a scraper that automatically retrieved all the songs from NoCopyrightSounds
(you can find the website [here](https://ncs.io/music-search?q=&genre=&mood=&version=regular)), along with their associated tags.  
In total, the dataset contained ~1700 songs and 89 distinct tags, which helped in categorizing and better understanding the relationships between different songs.
### How was the dataset prepared

<p align="center">
  <img src="https://i.ibb.co/j3Srd5y/spectrogram-vs-mel-spectrogram-vertical.png" alt="spec vs mel spec" />
</p>
<p align="center"><i>Spectrogram vs Mel Spectrogram</i></p>

The dataset was prepared by first dividing each song into fragments, 
with the specifics of how the fragments were made outlined below. 
After fragmenting the songs, mel spectrograms were extracted instead of 
traditional spectrograms to better capture the perceptual characteristics of sound. 
Mel spectrograms map frequencies to the mel scale, which is more aligned 
with how humans perceive pitch and loudness, 
making them more suitable for tasks involving music and audio recognition.

### Song fragmentation
<p align="center">
  <img src="https://i.ibb.co/KKffMcZ/set-frag.jpg" alt="spec vs mel spec" />
</p>
<p align="center"><i>Song fragmentation</i></p>
<p align="center"><i>green - training set, purple - validation set</i></p>

The song fragmentation process divides each audio file
into equally-sized fragments of `n_seconds` duration. 
Fragments are randomly assigned to either the training or validation set
based on a provided `validation_probability`, ensuring that fragments from the training
and validation sets do not overlap. The `step` defines the interval at which
consecutive fragments start, allowing for either overlapping or non-overlapping fragments.
For this project, 5-second fragments with a 1-second step were used.

# Vector Representation Using Neural Networks

The neural network was structured in several blocks, with the final block being a classifier
composed of a single dense layer. This design choice was intentional, aiming to minimize
the classifier's contribution to the creation of the representation. 
The purpose of the classifier was to predict the tags associated with a song,
so the representation had to capture the essential features for this task.
_**To obtain the vector, we pass the mel spectrogram of a song segment through the network, bypassing the classifier entirely.**_

### Neural network architecture

<div style="text-align: center;">
  <h6>First Part - Convolutional Layers</h6>
  <p><i>padding_0</i>: ConstantPad2d(padding=(1, 0, 0, 0), value=0)</p>
  <p><i>conv_0</i>: Conv1d(80, 128, kernel_size=(2,), stride=(1,))</p>
  <p><i>activation_0</i>: ReLU()</p>
  <p><i>padding_1</i>: ConstantPad2d(padding=(1, 0, 0, 0), value=0)</p>
  <p><i>conv_1</i>: Conv1d(128, 128, kernel_size=(2,), stride=(1,))</p>
  <p><i>activation_1</i>: ReLU()</p>
  <p><i>padding_2</i>: ConstantPad2d(padding=(1, 0, 0, 0), value=0)</p>
  <p><i>conv_2</i>: Conv1d(128, 128, kernel_size=(2,), stride=(1,))</p>
  <p><i>activation_2</i>: ReLU()</p>
  <p><i>padding_3</i>: ConstantPad2d(padding=(1, 0, 0, 0), value=0)</p>
  <p><i>conv_3</i>: Conv1d(128, 128, kernel_size=(2,), stride=(1,))</p>
  <p><i>activation_3</i>: ReLU()</p>
  <p><i>max_pool_reduce</i>: MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)</p>
  <p><i>end_block_activation</i>: ReLU()</p>

  <h6>Second Part - LSTM Layers</h6>
  <p><i>LSTM</i>: LSTM(128, 256, num_layers=4, batch_first=True, dropout=0.4)</p>

  <h6>Final Part - Fully Connected Classifier</h6>
  <p><i>dense_classifier</i>: Linear(in_features=256, out_features=91, bias=True)</p>
  <p><i>batch_norm_classifier_end</i>: BatchNorm1d(91, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)</p>
  <p><i>classifier_end_activation</i>: Sigmoid()</p>
</div>

### Evaluation of vector quality

To evaluate the quality of the generated vectors, we use classification based on these vectors, 
as previously mentioned. The neural network includes a classifier at its final block, 
designed specifically for this purpose. To evaluate how well the vectors capture meaningful features of the songs, 
we pass the mel spectrogram of a song segment through the entire network, including the classifier.
By training the classifier to predict categories based on the extracted vector representations,
we can assess the effectiveness of the vectors. The accuracy and performance of the vectors are measured by evaluating the classifier's predictions.

The following metrics were used to evaluate the model's performance:
- **BCELoss**: 0.0437
- **Hamming Accuracy**: 0.9835
- **Exact Match**: 0.6223

Additionally we provide tables with more detailed breakdown:

#### Grouped
This table shows aggregated groups of incorrect tags, with the number of samples and their share of the total for each group.

| **number_of_incorrect_tags** | **number_of_samples** | **share_of_total** |
|------------------------------|-----------------------|--------------------|
| 0                            | 18,933                | 0.622305           |
| 1 to 6                        | 9,752                 | 0.320536           |
| 7 to 10                       | 1,590                 | 0.052261           |
| 11 to 14                      | 144                   | 0.004733           |
| 15 or more                   | 5                     | 0.000164           |

#### Full breakdown
This table provides the exact count of samples for each specific number of incorrect tags, along with their share of the total dataset.

| **number_of_incorrect_tags** | **number_of_samples** | **share_of_total** |
|------------------------------|-----------------------|--------------------|
| 0                            | 18,933                | 0.622305           |
| 1                            | 2,375                 | 0.078063           |
| 2                            | 1,777                 | 0.058408           |
| 3                            | 1,622                 | 0.053313           |
| 4                            | 1,503                 | 0.049402           |
| 5                            | 1,350                 | 0.044373           |
| 6                            | 1,125                 | 0.036977           |
| 7                            | 726                   | 0.023863           |
| 8                            | 456                   | 0.014988           |
| 9                            | 248                   | 0.008151           |
| 10                           | 160                   | 0.005259           |
| 11                           | 82                    | 0.002695           |
| 12                           | 32                    | 0.001052           |
| 13                           | 18                    | 0.000592           |
| 14                           | 12                    | 0.000394           |
| 15                           | 2                     | 0.000066           |
| 17                           | 1                     | 0.000033           |
| 18                           | 1                     | 0.000033           |
| 20                           | 1                     | 0.000033           |


### Training infrastructure and data storage

The neural network was trained using Google Cloud Platform's Vertex AI, 
which provided computational resources for training. 
Different versions of the data and models were stored and managed in Google Cloud Storage, 
ensuring easy access and version control throughout the training process. 
However, we had a budget of $300 and were limited to the resources available through the free trial, which restricted the scale and duration of the training.

# How Recommendations Are Made

With the vectors obtained as described above, we select a set of songs (the ones the user liked). 
We load all vectors representing the fragments of the songs liked by the user, and then randomly choose a subset of them. 
Based on this subset, we search in the vector space representing other songs. 
The found vectors will be the basis for the recommendations.

### Way of choosing recommendations

Each vector in the space is assigned a distance `d`, which is the sum of distances ($d = \sum_{i=1}^{n} \| x_i \|_{\ell_2}$) to the vectors 
representing fragments of songs liked by the user. 
Next, we reject vectors representing fragments that:

- Represent a fragment of a song liked by the user.  
- Have another vector representing a fragment of the same song with a smaller `d` (we want to avoid recommending the same song twice; one close fragment is enough to suggest the song).  

From the remaining vectors, we select a few fragments with the smallest `d`.

### Explainability

For each recommended song, we provide:  

- The fragment used as the basis for the recommendation (the moment in the song where the fragment starts).  
- Its tags.  
- The distance `d`.  

Additionally, we also list the tags of the songs liked by the user.

### Recommendation example
```
Liked songs:
Liked song -> 094_-_JOXION:
        Tags: ['Dark', 'Dreamy', 'Future House', 'Glamorous', 'Gloomy', 'Sexy', 'energetic']
Liked song -> 4_Love_(Fresh_Stuff_Remix)_-_Fresh_Stuff__Alltair__Wiguez:
        Tags: ['Euphoric', 'Future House', 'Hopeful', 'Mysterious', 'Quirky', 'Restless', 'energetic']

A random selection of fragments from songs liked by the user:
Selected fragment -> 094_-_JOXION:
        Start time: 5s | Idx: 5 | Tags: ['Dark', 'Dreamy', 'Future House', 'Glamorous', 'Gloomy', 'Sexy', 'energetic']  
Selected fragment -> 094_-_JOXION:
        Start time: 15s | Idx: 15 | Tags: ['Dark', 'Dreamy', 'Future House', 'Glamorous', 'Gloomy', 'Sexy', 'energetic']
Selected fragment -> 094_-_JOXION:
        Start time: 17s | Idx: 17 | Tags: ['Dark', 'Dreamy', 'Future House', 'Glamorous', 'Gloomy', 'Sexy', 'energetic']
Selected fragment -> 094_-_JOXION:
        Start time: 70s | Idx: 70 | Tags: ['Dark', 'Dreamy', 'Future House', 'Glamorous', 'Gloomy', 'Sexy', 'energetic']
Selected fragment -> 4_Love_(Fresh_Stuff_Remix)_-_Fresh_Stuff__Alltair__Wiguez:
        Start time: 46s | Idx: 1035 | Tags: ['Euphoric', 'Future House', 'Hopeful', 'Mysterious', 'Quirky', 'Restless', 'energetic']     
Selected fragment -> 4_Love_(Fresh_Stuff_Remix)_-_Fresh_Stuff__Alltair__Wiguez:
        Start time: 67s | Idx: 1056 | Tags: ['Euphoric', 'Future House', 'Hopeful', 'Mysterious', 'Quirky', 'Restless', 'energetic']     
Selected fragment -> 4_Love_(Fresh_Stuff_Remix)_-_Fresh_Stuff__Alltair__Wiguez:
        Start time: 101s | Idx: 1090 | Tags: ['Euphoric', 'Future House', 'Hopeful', 'Mysterious', 'Quirky', 'Restless', 'energetic']    
Selected fragment -> 4_Love_(Fresh_Stuff_Remix)_-_Fresh_Stuff__Alltair__Wiguez:
        Start time: 137s | Idx: 1126 | Tags: ['Euphoric', 'Future House', 'Hopeful', 'Mysterious', 'Quirky', 'Restless', 'energetic']    

Recommendations:
Reccomended fragment with d=19.765941619873047-> Pretty_woman_-_The_Saint__Magnus:
        Start time: 159s | Idx: 217837 | Tags: ['Epic', 'Euphoric', 'Happy', 'House', 'Restless']
Reccomended fragment with d=19.851806640625-> Ber_Zer_Ker_(Rob_Gasser_Remix)_-_Rob_Gasser__WATEVA:
        Start time: 7s | Idx: 21388 | Tags: ['Angry', 'Drum & Bass', 'Gloomy', 'Restless', 'energetic']
Reccomended fragment with d=19.854427337646484-> Double_D_-_Debris:
        Start time: 173s | Idx: 67871 | Tags: ['Angry', 'Dark', 'Future House', 'Gloomy', 'energetic']
Reccomended fragment with d=19.859947204589844-> Destiny_-_Tobu__DEAF_KEV__Anna_Yvette__Electro-Light__Jim_Yosef:
        Start time: 15s | Idx: 61049 | Tags: ['Epic', 'Euphoric', 'Happy', 'Hopeful', 'House']
Reccomended fragment with d=19.891176223754883-> Psycho_(feat__Nieko)_-_Zack_Merci__Nieko:
        Start time: 60s | Idx: 218915 | Tags: ['Future House', 'Gloomy', 'Mysterious', 'Quirky', 'Restless', 'Suspense', 'energetic']
```

# Future Work and Improvements

While the current system performs well, there are clear opportunities for improvement: 
- **Improving Neural Network**: Testing wider range of architectures could create better vectors.  
- **Refining Recommendation Methods**: Adopting more sophisticated techniques of making recommendations based on vectors.
- **Expanding the Dataset**: Including more songs across a wider range of genres and styles could improve the system's diversity and accuracy.
- **Introducing a Vector Database**: Using a dedicated vector database could improve how efficiently the system stores and retrieves song vectors, especially with larger datasets.  