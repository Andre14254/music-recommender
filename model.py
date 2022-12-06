#!pip3 install tensorflow

# %%
import numpy as np
import json
import tensorflow as tf
import os
import pandas as pd

# %%
directory = "audioset_v1_embeddings/eval"

dataset = []
for file_name in os.listdir(directory):
     if file_name.endswith(".tfrecord"):
            dataset.append(os.path.join(directory,file_name))

# %%
tf.compat.v1.enable_eager_execution()

# %%
raw_dataset = tf.data.TFRecordDataset(dataset)

# %%
class_labels = pd.read_csv('class_labels_indices.csv')
labels = class_labels['display_name'].tolist()

music_class = class_labels[class_labels['display_name'].str.contains('Music', case=False)]
music_labels = music_class['index'].tolist()

# %%
audios = []
counter = 0
NUM_SECONDS = 10

for raw_record in raw_dataset:
    example = tf.train.SequenceExample()
    example.ParseFromString(raw_record.numpy())
    
    audio_labels = example.context.feature['labels'].int64_list.value
    start_time = example.context.feature['start_time_seconds'].float_list.value
    end_time = example.context.feature['end_time_seconds'].float_list.value
    video_id = example.context.feature['video_id'].bytes_list.value
    
    if not (set(music_labels) & set(audio_labels)):
        continue

    feature_list = example.feature_lists.feature_list['audio_embedding'].feature
    final_features = [list(feature.bytes_list.value[0]) for feature in feature_list]
    audio_embedding = [item for sublist in final_features[:NUM_SECONDS] for item in sublist]
    
    if len(final_features) < NUM_SECONDS:
        continue
    
    audio = {
        'label': audio_labels,
        'video_id': video_id[0],
        'start_time': start_time[0],
        'end_time': end_time[0],
        'data': audio_embedding
    }
    
    audios.append(audio)
    counter += 1
    if counter % 100 == 0:
        print(f"Processing {counter}th file ...")

# %%
with open('music_set.json', 'w') as file:
    str_audio = repr(audios)
    json.dump(str_audio, file)

# %%
[audio['data'][:10] for audio in audios[:4]]

# inspired by CodeEmporium