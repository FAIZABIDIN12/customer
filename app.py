from flask import Flask, render_template, request
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

def perform_clustering(data):
    # Praproses data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    scaled_df = pd.DataFrame(scaled_data, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

    # Klasterisasi dengan 3 klaster
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(scaled_df)
    data['Cluster'] = kmeans.labels_

    # Konversi kolom ke tipe numerik untuk menghindari TypeError
    numeric_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Analisis karakteristik setiap klaster
    cluster_summary = data.groupby('Cluster')[numeric_columns].mean()

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    colors = ['red', 'green', 'blue']
    for i in range(3):
        cluster_data = data[data['Cluster'] == i]
        plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], c=colors[i], label=f'Cluster {i}')

    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Clustered Data')
    plt.legend()
    
    # Save plot to BytesIO object
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode('utf-8')
    
    return data, cluster_summary, img_data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            # Simpan file di folder 'uploads' (pastikan folder sudah ada)
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Baca data dari file CSV
            data = pd.read_csv(file_path)

            # Lakukan klasterisasi
            clusters, cluster_summary, img_data = perform_clustering(data)

            # Hapus file setelah digunakan
            os.remove(file_path)

            return render_template('result.html', clusters=clusters, cluster_summary=cluster_summary, img_data=img_data)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
