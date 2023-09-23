import pandas as pd
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn import datasets
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the Iris dataset from scikit-learn
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Create a K-means clustering model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model to the data
kmeans.fit(df)

# Get the labels assigned to each cluster center
cluster_labels = kmeans.predict(df)

@app.route('/')
def index():
    # Serve the index.html file
    return render_template('index.html')

@app.route('/predict_cluster', methods=['POST'])
def predict_cluster():
    try:
        input_data = request.get_json()
        if all(feature in input_data for feature in iris.feature_names):
            # Convert input data values to float
            for feature_name in iris.feature_names:
                input_data[feature_name] = float(input_data[feature_name])

            input_df = pd.DataFrame([input_data])
            cluster = int(kmeans.predict(input_df)[0])  # Convert to regular Python integer

            # Get the cluster label from the cluster_labels array
            cluster_label = iris.target_names[cluster]

            # Calculate distances to each cluster center
            distances = pairwise_distances_argmin_min(input_df, kmeans.cluster_centers_)[1].tolist()

            # Calculate differences from cluster centers for each feature
            differences = []
            for center in kmeans.cluster_centers_:
                diff = {}
                for feature_name in iris.feature_names:
                    input_feature_value = input_data[feature_name]
                    center_feature_value = center[iris.feature_names.index(feature_name)]
                    diff[feature_name] = input_feature_value - center_feature_value
                differences.append(diff)

            # Calculate Euclidean distance to each cluster center
            euclidean_distances = np.linalg.norm(input_df.values - kmeans.cluster_centers_, axis=1).tolist()

            return jsonify({'cluster': cluster_label, 'distances': distances, 'differences': differences, 'euclidean_distances': euclidean_distances, 'error': None})
        else:
            return jsonify({'cluster': None, 'distances': None, 'differences': None, 'euclidean_distances': None, 'error': 'Input data must contain all 4 features'})
    except Exception as e:
        return jsonify({'cluster': None, 'distances': None, 'differences': None, 'euclidean_distances': None, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False)
