from flask import Flask, request, render_template
#!pip3 install annoy
from annoy import AnnoyIndex
import json
from ast import literal_eval
import pandas as pd
app = Flask(__name__)

class_labels = pd.read_csv('class_labels_indices.csv')
music_dict = dict(zip(class_labels.index, class_labels.display_name))

with open('music_set.json', 'r') as file:
    file_read = json.loads(file.read())
    music_dataset = literal_eval(file_read)

audio_dim = 1280
annoy_index = AnnoyIndex(audio_dim, 'angular')  # Length of item vector that will be indexed
for index in range(len(music_dataset[:1000])):
    vector = music_dataset[index]['data']
    annoy_index.add_item(index, vector)

annoy_index.build(50) # 50 trees
annoy_index.save('nearest_neightbor_graph.ann')

annoy_index = AnnoyIndex(audio_dim, 'angular')
annoy_index.load('nearest_neightbor_graph.ann')


# @app.route('/')
# def form():
#     # if (0):
#     #     nns_index = annoy_index.get_nns_by_item(request.form['id'], 10)
#     #     for index in nns_index:
#     #         sample = music_dataset[index]
#     #         songs = [music_dict[idx] for idx in sample['label']]
#     return '''
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <title>{% block title %} {% endblock %} - FlaskApp</title>
#     <style>
#         .message {
#             padding: 10px;
#             margin: 5px;
#             background-color: #f3f3f3
#         }
#         nav a {
#             color: #d64161;
#             font-size: 3em;
#             margin-left: 50px;
#             text-decoration: none;
#         }

#         .alert {
#             padding: 20px;
#             margin: 5px;
#             color: #970020;
#             background-color: #ffd5de;
#         }

#     </style>
# </head>
# <h2 style='color: #ececec'> Please input a song ID</h2>
# 			<div class="break"></div>
# 				<div class="ml-container">
# 					<form action="{{ url_for('id') }}"  id="form" method="POST">
# 						<input class="input" type="text" name="URL">
# 						<div class="break"></div>
# 						<small id="emailHelp" class="form-text text-muted"> You can find the playlist link by following instructions <a href="https://wordpress.com/support/audio/spotify/" target="_blank"> here </a> </small>
# 						<div class="break"></div>
# 							<label style='color: #ececec' for="number-of-recs" >How many songs do you want?:</label>
# 							<select name="number-of-recs" id="number-of-recs" form="form1">
# 								<option value="5" selected> 5</option>
# 								<option value="10">10</option>
# 								<option value="15">15</option>
# 								<option value="20">20</option>
# 							</select>
# 							<div class="break"></div>
# 						<input class='button1' form='form' value='Get recommendations!' type="submit">
# 					</form>
# 				</div>
#                 <div class="results">
# 				<div class='results'>
#     <h2 style='color: #ececec'>Results</h2>
#         {% for song in songs %}
#         <p class='song' style="font-size: 11pt"> <a  href={{song[1]}} target="_blank"> {{ song[0]}} </a> </p>
#         {% endfor %}    
# </div>
# 			</div>
# </html>'''

@app.route('/index', methods=['GET','POST'])
def execute():
    songs = []
    recs=[]
    a = "grg"
    if request.method == 'POST':
        nns_index = annoy_index.get_nns_by_item(int(request.form['ID']), int(request.form['number-of-recs']))
        for index in nns_index:
            sample = music_dataset[index]
            #songs = [music_dict[idx] for idx in sample['label']]
            recs.append(sample['video_id'].decode('utf-8'))

        for index in range(len(music_dataset[:100])):
            sample = music_dataset[index]
            music_labels = [music_dict[idx] for idx in sample['label']]
            songs.append(sample['video_id'].decode('utf-8'))

    return render_template("index.html", recs = recs, songs=songs)

if __name__ == '__main__':
    app.run()