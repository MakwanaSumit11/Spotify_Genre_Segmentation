# Spotify Genre Segmentation Analysis: A K-Means Clustering App

This project is a Streamlit web application designed to perform **Exploratory Data Analysis (EDA)** and **K-Means Clustering** on music track data (like the Spotify Million Song Dataset). The goal is to segment songs into meaningful groups or "clusters" based on their audio features, which serves as the core of a content-based music recommendation system.

##  Features at a Glance

* **Interactive Data Upload:** Securely upload your own CSV dataset via the sidebar.
* **Comprehensive Data Overview:** Analyze dataset size, column types, and missing values.
* **In-depth EDA:** Explore feature distributions (`tempo`, `popularity`, audio features), and genre composition.
* **Correlation Analysis:** Visualize feature relationships using a detailed **Plotly Heatmap**.
* **K-Means Clustering:**
    * Segment tracks based on key audio features (`danceability`, `energy`, `valence`, `tempo`, `loudness`).
    * **Configurable Clusters:** Choose the optimal number of clusters (2 to 10).
    * **Cluster Profiling:** View detailed statistics, top genres, and average feature values for each resulting cluster.
    * **Interactive Visualization:** 2D and 3D scatter plots to visually inspect cluster separation.
* **Recommendation Framework:** A dedicated section outlining the architecture for a cluster-based music recommender.
* **Data Export:** Download the final clustered dataset and a cluster summary table.

---

##  Tech Stack

| Category | Tools / Libraries |
| :--- | :--- |
| **App Framework** | `streamlit` |
| **Data Analysis** | `pandas`, `numpy` |
| **Machine Learning** | `scikit-learn` (`KMeans`, `StandardScaler`) |
| **Visualization** | `plotly.express`, `plotly.graph_objects`, `matplotlib`, `seaborn` |

---

##  Expected Data Format

The application is designed to work best with music datasets that include common Spotify audio features and genre information.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `track_name` | object | Name of the song |
| `track_artist` | object | Name of the artist |
| `playlist_genre` | object | The genre label (used for validation/profiling) |
| `danceability` | float | How suitable a track is for dancing (0.0 to 1.0) |
| `energy` | float | Perceptual measure of intensity and activity (0.0 to 1.0) |
| `valence` | float | Musical positiveness conveyed by a track (0.0 to 1.0) |
| `tempo` | float | Estimated overall tempo in BPM |
| `loudness` | float | Overall loudness in decibels (dB) |
| *... other features* | *...* | `speechiness`, `acousticness`, `liveness`, `duration_ms`, etc. |

A sample of the expected data is displayed in the "Data Overview" tab upon starting the application without an uploaded file.

---

##  Getting Started

Follow these steps to set up and run the application locally.

### 1. Prerequisites

You'll need Python 3.8+ installed on your system.

### 2. Clone the Repository

```bash
git clone [YOUR_REPOSITORY_URL]
cd [YOUR_REPOSITORY_NAME]
```

### 3. Install Dependencies

Install all the necessary Python libraries using pip:

```bash
pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn
```

### 4. Run the App

Launch the Streamlit application from your terminal:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser, usually at the address http://localhost:8501.

### 5. Start Analysis

1. In the running application, use the file uploader in the sidebar labeled "Choose a CSV file" to upload your music dataset.

2. Navigate through the tabs (Data Overview, Clustering, Results) to perform the analysis, explore the visualizations, and generate your song segments!
