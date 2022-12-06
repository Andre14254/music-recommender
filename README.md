# music-recommender
A Python Flask App which takes an input of YouTube music and returns recommendations based on a machine-learning model.

An audioset dataset of YouTube songs gets passed into an embedding generator, which converts the data into a vector, and passes it into a recommendation engine with a library that performs an Approximate Nearest Neighbor algorithm to find a list of the closest recommendations in the dataset.

Used Flask and HTML/CSS to create a web application that takes an input and a number of songs we want recommended and return a list of videos that most closely matches the inputted song.

# How To Use

```git clone https://github.com/Andre14254/music-recommender.git```

Install dependencies (there may be others)

```pip install Flask

pip install annoy

pip install ast

pip install numpy

pip install json

pip install os

pip install tensorflow```

In terminal, run ```python app.py```.

Enter an ID from a song below and choose the number of songs you want returned and a list of videos will appear.