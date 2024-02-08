from flask import Flask, render_template, request
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load data and fit model
data = pd.read_excel(r'C:\Users\Admin\Desktop\Python Practice\Place Recommender\statesnew (1).xlsx')
text_data = data['Description'].values.astype('U')
vectorizer = TfidfVectorizer()
text_vectors = vectorizer.fit_transform(text_data)
dense_vectors = text_vectors.toarray()
dense_df = pd.DataFrame(dense_vectors, index=data.index)
result_df = pd.concat([data, dense_df], axis=1)
X = result_df[list(dense_df.columns)]
X = X.astype(str)
X.columns = X.columns.astype(str)
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)
result_df['cluster'] = kmeans.labels_

# Define function to get recommendations
def place_finder(description: str, state: str):
    tfvector = vectorizer.transform(np.array([description]))
    review_class = kmeans.predict(tfvector)[0]
    clean = result_df.loc[(result_df.cluster == review_class) & (result_df.State == state)]
    clean['Rating'] = clean['Rating'].astype(float)  # convert Rating column to float
    result = clean[['Name', 'Rating', 'State']].sort_values(by="Rating", ascending=False).head(5)
    return result.to_dict('records')


# Define route for homepage
@app.route('/')
def home():
    states = get_unique_states()
    return render_template('index.html', states=states)

# Define route for getting recommendations
@app.route('/recommendations', methods=['POST'])
def recommendations():
    description = request.form['description']
    state = request.form['state']
    recommendations = place_finder(description, state)
    return render_template('index.html', recommendations=recommendations, states=get_unique_states())

def get_unique_states():
    return data['State'].unique().tolist()

if __name__ == '__main__':
    app.run(debug=True)
