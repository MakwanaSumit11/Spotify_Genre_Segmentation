import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Spotify Genre Segmentation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f0fdf4 0%, #dbeafe 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f3f4f6;
        border-radius: 5px 5px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-bottom: 2px solid #3b82f6;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("""
    <div style='background: linear-gradient(90deg, #10b981 0%, #3b82f6 100%); 
                padding: 30px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0;'>🎵 Spotify Genre Segmentation Analysis</h1>
        <p style='color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 18px;'>
            Machine Learning-based Music Recommendation System
        </p>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.session_state.processed = True
            st.success(f"✅ Loaded {len(df)} records")
            
            st.markdown("---")
            st.subheader("📊 Dataset Info")
            st.write(f"**Rows:** {df.shape[0]}")
            st.write(f"**Columns:** {df.shape[1]}")
            st.write(f"**Memory:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# Main content
if st.session_state.data is None:
    st.info("👆 Please upload a CSV file from the sidebar to begin analysis")
    
    # Show sample data format
    st.markdown("### 📋 Expected Data Format")
    sample_data = {
        'track_name': ['Song 1', 'Song 2'],
        'track_artist': ['Artist 1', 'Artist 2'],
        'playlist_genre': ['pop', 'rock'],
        'danceability': [0.748, 0.650],
        'energy': [0.916, 0.833],
        'valence': [0.518, 0.725],
        'tempo': [122.036, 123.976]
    }
    st.dataframe(pd.DataFrame(sample_data))
    
else:
    df = st.session_state.data
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview", 
        "📈 Preprocessing", 
        "🎨 Visualizations", 
        "🔗 Correlation", 
        "🎯 Clustering", 
        "✅ Results"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='color: #3b82f6; margin: 0;'>Total Records</h4>
                    <h2 style='margin: 10px 0 0 0;'>{}</h2>
                </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='color: #10b981; margin: 0;'>Unique Genres</h4>
                    <h2 style='margin: 10px 0 0 0;'>{}</h2>
                </div>
            """.format(df['playlist_genre'].nunique()), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='color: #8b5cf6; margin: 0;'>Total Features</h4>
                    <h2 style='margin: 10px 0 0 0;'>{}</h2>
                </div>
            """.format(len(df.columns)), unsafe_allow_html=True)
        
        with col4:
            avg_pop = df['track_popularity'].mean() if 'track_popularity' in df.columns else 0
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='color: #f59e0b; margin: 0;'>Avg Popularity</h4>
                    <h2 style='margin: 10px 0 0 0;'>{:.1f}</h2>
                </div>
            """.format(avg_pop), unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Type': df.dtypes.values
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Missing Values Analysis")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
    
    # Tab 2: Preprocessing
    with tab2:
        st.header("Data Preprocessing & Statistics")
        
        # Clean data
        numeric_cols = ['track_popularity', 'danceability', 'energy', 'loudness', 
                       'speechiness', 'acousticness', 'instrumentalness', 
                       'liveness', 'valence', 'tempo', 'duration_ms']
        
        # Filter numeric columns that exist
        available_numeric = [col for col in numeric_cols if col in df.columns]
        
        st.subheader("Feature Statistics")
        stats_df = df[available_numeric].describe().T
        stats_df['variance'] = df[available_numeric].var()
        
        st.dataframe(stats_df.style.background_gradient(cmap='YlGnBu'), 
                    use_container_width=True)
        
        st.markdown("---")
        
        # Distribution of key features
        st.subheader("Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'track_popularity' in df.columns:
                fig = px.histogram(df, x='track_popularity', nbins=30,
                                 title='Track Popularity Distribution',
                                 color_discrete_sequence=['#3b82f6'])
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'tempo' in df.columns:
                fig = px.histogram(df, x='tempo', nbins=30,
                                 title='Tempo Distribution',
                                 color_discrete_sequence=['#10b981'])
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        # Genre distribution
        st.markdown("---")
        st.subheader("Genre Distribution")
        
        if 'playlist_genre' in df.columns:
            genre_counts = df['playlist_genre'].value_counts().head(15)
            
            fig = px.bar(x=genre_counts.values, y=genre_counts.index,
                        orientation='h',
                        title='Top 15 Genres by Track Count',
                        labels={'x': 'Number of Tracks', 'y': 'Genre'},
                        color=genre_counts.values,
                        color_continuous_scale='viridis')
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Visualizations
    with tab3:
        st.header("Data Visualizations")
        
        # Audio features distribution
        st.subheader("Audio Features Analysis")
        
        audio_features = ['danceability', 'energy', 'valence', 'acousticness']
        available_audio = [f for f in audio_features if f in df.columns]
        
        if len(available_audio) >= 4:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Audio Features Distribution', fontsize=16, fontweight='bold')
            
            for idx, feature in enumerate(available_audio[:4]):
                row = idx // 2
                col = idx % 2
                axes[row, col].hist(df[feature].dropna(), bins=30, 
                                   color=['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'][idx],
                                   alpha=0.7, edgecolor='black')
                axes[row, col].set_title(feature.capitalize(), fontsize=12, fontweight='bold')
                axes[row, col].set_xlabel('Value')
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Scatter plots
        st.subheader("Feature Relationships")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("Select X-axis feature", available_audio, index=0)
        with col2:
            y_feature = st.selectbox("Select Y-axis feature", available_audio, index=1)
        
        if x_feature and y_feature and 'playlist_genre' in df.columns:
            fig = px.scatter(df.sample(min(1000, len(df))), 
                           x=x_feature, y=y_feature,
                           color='playlist_genre',
                           title=f'{x_feature.capitalize()} vs {y_feature.capitalize()}',
                           opacity=0.6,
                           height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Box plots
        st.subheader("Feature Distribution by Genre")
        
        selected_feature = st.selectbox("Select feature for box plot", available_audio)
        
        if selected_feature and 'playlist_genre' in df.columns:
            top_genres = df['playlist_genre'].value_counts().head(8).index
            df_filtered = df[df['playlist_genre'].isin(top_genres)]
            
            fig = px.box(df_filtered, x='playlist_genre', y=selected_feature,
                        title=f'{selected_feature.capitalize()} Distribution by Top Genres',
                        color='playlist_genre',
                        height=500)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Correlation
    with tab4:
        st.header("Correlation Analysis")
        
        st.write("Correlation between audio features shows how different musical characteristics relate to each other.")
        
        corr_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                        'acousticness', 'valence', 'tempo']
        available_corr = [f for f in corr_features if f in df.columns]
        
        if len(available_corr) > 2:
            corr_matrix = df[available_corr].corr()
            
            # Plotly heatmap
            fig = px.imshow(corr_matrix,
                          labels=dict(color="Correlation"),
                          x=corr_matrix.columns,
                          y=corr_matrix.columns,
                          color_continuous_scale='RdBu_r',
                          aspect="auto",
                          title="Feature Correlation Matrix",
                          zmin=-1, zmax=1)
            
            fig.update_layout(height=600)
            
            # Add correlation values
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    fig.add_annotation(
                        x=j, y=i,
                        text=f"{corr_matrix.iloc[i, j]:.2f}",
                        showarrow=False,
                        font=dict(size=10, color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
                    )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Key insights
            st.subheader("Key Correlation Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔴 Strong Positive Correlations (> 0.5)**")
                strong_pos = []
                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        if corr_matrix.iloc[i, j] > 0.5:
                            strong_pos.append(f"• {corr_matrix.index[i]} ↔ {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")
                
                if strong_pos:
                    for item in strong_pos:
                        st.write(item)
                else:
                    st.write("None found")
            
            with col2:
                st.markdown("**🔵 Strong Negative Correlations (< -0.5)**")
                strong_neg = []
                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        if corr_matrix.iloc[i, j] < -0.5:
                            strong_neg.append(f"• {corr_matrix.index[i]} ↔ {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")
                
                if strong_neg:
                    for item in strong_neg:
                        st.write(item)
                else:
                    st.write("None found")
    
    # Tab 5: Clustering
    with tab5:
        st.header("K-Means Clustering Analysis")
        
        st.write("Songs are grouped into clusters based on their audio features for better music recommendations.")
        
        # Clustering parameters
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Cluster Configuration")
        with col2:
            n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=10, value=5)
        
        clustering_features = ['danceability', 'energy', 'valence', 'tempo', 'loudness']
        available_cluster_features = [f for f in clustering_features if f in df.columns]
        
        if len(available_cluster_features) >= 3:
            # Perform clustering
            df_cluster = df[available_cluster_features].dropna()
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_cluster)
            
            # K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_cluster['cluster'] = kmeans.fit_predict(X_scaled)
            
            # Merge back to original df
            df['cluster'] = pd.Series(df_cluster['cluster'], index=df_cluster.index)
            
            st.markdown("---")
            
            # Cluster statistics
            st.subheader("Cluster Statistics")
            
            cols = st.columns(n_clusters)
            for i in range(n_clusters):
                cluster_data = df[df['cluster'] == i]
                with cols[i]:
                    st.markdown(f"""
                        <div style='background: white; padding: 15px; border-radius: 10px; 
                                    border-left: 4px solid #{["ef4444", "3b82f6", "10b981", "f59e0b", "8b5cf6", "ec4899", "06b6d4", "84cc16", "f97316", "a855f7"][i]};
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='margin: 0 0 10px 0;'>Cluster {i+1}</h4>
                            <p style='font-size: 24px; font-weight: bold; margin: 0;'>{len(cluster_data)}</p>
                            <p style='color: #666; margin: 5px 0 0 0;'>songs</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Detailed cluster profiles
            st.subheader("Cluster Profiles")
            
            cluster_profiles = []
            for i in range(n_clusters):
                cluster_data = df[df['cluster'] == i]
                profile = {'Cluster': f'Cluster {i+1}', 'Size': len(cluster_data)}
                
                for feature in available_cluster_features:
                    if feature in cluster_data.columns:
                        profile[f'Avg {feature.capitalize()}'] = cluster_data[feature].mean()
                
                if 'playlist_genre' in cluster_data.columns:
                    top_genre = cluster_data['playlist_genre'].mode()
                    profile['Top Genre'] = top_genre.values[0] if len(top_genre) > 0 else 'N/A'
                
                cluster_profiles.append(profile)
            
            profile_df = pd.DataFrame(cluster_profiles)
            st.dataframe(profile_df.style.background_gradient(subset=[col for col in profile_df.columns if 'Avg' in col], cmap='YlOrRd'),
                        use_container_width=True)
            
            st.markdown("---")
            
            # Visualization
            st.subheader("Cluster Visualization")
            
            if len(available_cluster_features) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    viz_x = st.selectbox("X-axis", available_cluster_features, index=0, key='viz_x')
                with col2:
                    viz_y = st.selectbox("Y-axis", available_cluster_features, index=1, key='viz_y')
                
                sample_size = min(2000, len(df[df['cluster'].notna()]))
                df_sample = df[df['cluster'].notna()].sample(sample_size)
                
                fig = px.scatter(df_sample, x=viz_x, y=viz_y, color='cluster',
                               title=f'Cluster Distribution: {viz_x.capitalize()} vs {viz_y.capitalize()}',
                               color_continuous_scale='viridis',
                               opacity=0.6,
                               height=600)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # 3D visualization
            if len(available_cluster_features) >= 3:
                st.markdown("---")
                st.subheader("3D Cluster Visualization")
                
                sample_3d = df[df['cluster'].notna()].sample(min(1000, len(df[df['cluster'].notna()])))
                
                fig_3d = px.scatter_3d(sample_3d, 
                                      x=available_cluster_features[0],
                                      y=available_cluster_features[1],
                                      z=available_cluster_features[2],
                                      color='cluster',
                                      title='3D Cluster Distribution',
                                      opacity=0.7,
                                      height=700)
                
                st.plotly_chart(fig_3d, use_container_width=True)
    
    # Tab 6: Results
    with tab6:
        st.header("Model Results & Recommendation System")
        
        # Success message
        st.markdown("""
            <div style='background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                        padding: 30px; border-radius: 10px; border-left: 5px solid #10b981;
                        margin-bottom: 30px;'>
                <h2 style='color: #065f46; margin: 0 0 10px 0;'>✅ Model Successfully Built!</h2>
                <p style='color: #047857; margin: 0; font-size: 16px;'>
                    Successfully segmented {} songs into {} clusters based on audio features.
                </p>
            </div>
        """.format(len(df), n_clusters if 'cluster' in df.columns else 5), unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div style='background: white; padding: 20px; border-radius: 10px; text-align: center;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h1 style='color: #3b82f6; margin: 0;'>{}</h1>
                    <p style='color: #666; margin: 10px 0 0 0;'>Total Songs</p>
                </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div style='background: white; padding: 20px; border-radius: 10px; text-align: center;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h1 style='color: #10b981; margin: 0;'>{}</h1>
                    <p style='color: #666; margin: 10px 0 0 0;'>Clusters</p>
                </div>
            """.format(n_clusters if 'cluster' in df.columns else 5), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div style='background: white; padding: 20px; border-radius: 10px; text-align: center;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h1 style='color: #8b5cf6; margin: 0;'>{}</h1>
                    <p style='color: #666; margin: 10px 0 0 0;'>Key Features</p>
                </div>
            """.format(len(available_cluster_features) if 'available_cluster_features' in locals() else 7), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recommendation System Framework
        st.subheader("🎯 Recommendation System Framework")
        
        st.markdown("""
        <div style='background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h4 style='color: #1f2937; margin-bottom: 20px;'>How the Recommendation Engine Works:</h4>
            
            <div style='margin-bottom: 15px; padding: 15px; background: #eff6ff; border-radius: 8px;'>
                <h5 style='color: #1e40af; margin: 0 0 10px 0;'>📊 Step 1: Feature Extraction</h5>
                <p style='color: #1e3a8a; margin: 0;'>
                    Extract audio features (danceability, energy, valence, tempo, loudness) from the target song
                </p>
            </div>
            
            <div style='margin-bottom: 15px; padding: 15px; background: #f0fdf4; border-radius: 8px;'>
                <h5 style='color: #15803d; margin: 0 0 10px 0;'>🎯 Step 2: Cluster Assignment</h5>
                <p style='color: #166534; margin: 0;'>
                    Normalize features and assign the song to the nearest cluster using Euclidean distance
                </p>
            </div>
            
            <div style='margin-bottom: 15px; padding: 15px; background: #fef3c7; border-radius: 8px;'>
                <h5 style='color: #92400e; margin: 0 0 10px 0;'>🔍 Step 3: Similar Songs Selection</h5>
                <p style='color: #78350f; margin: 0;'>
                    Find songs within the same cluster with similar feature profiles using cosine similarity
                </p>
            </div>
            
            <div style='padding: 15px; background: #f3e8ff; border-radius: 8px;'>
                <h5 style='color: #6b21a8; margin: 0 0 10px 0;'>⭐ Step 4: Ranking & Filtering</h5>
                <p style='color: #581c87; margin: 0;'>
                    Rank recommendations by similarity score, popularity, and user preferences
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Cluster characteristics summary
        st.subheader("📋 Cluster Characteristics Summary")
        
        if 'cluster' in df.columns:
            for i in range(n_clusters):
                cluster_data = df[df['cluster'] == i]
                
                # Determine profile
                profile = "Balanced"
                if 'energy' in cluster_data.columns and 'danceability' in cluster_data.columns:
                    avg_energy = cluster_data['energy'].mean()
                    avg_dance = cluster_data['danceability'].mean()
                    avg_valence = cluster_data['valence'].mean() if 'valence' in cluster_data.columns else 0.5
                    
                    if avg_energy > 0.7 and avg_dance > 0.7:
                        profile = "High Energy Dance"
                    elif avg_valence > 0.6:
                        profile = "Positive/Upbeat"
                    elif avg_energy < 0.5:
                        profile = "Calm/Relaxed"
                
                with st.expander(f"🎵 Cluster {i+1} - {profile} ({len(cluster_data)} songs)"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if available_cluster_features:
                            feature_avgs = {f: cluster_data[f].mean() for f in available_cluster_features if f in cluster_data.columns}
                            
                            fig = go.Figure(data=[
                                go.Bar(x=list(feature_avgs.keys()), 
                                      y=list(feature_avgs.values()),
                                      marker_color='#3b82f6')
                            ])
                            fig.update_layout(title=f"Average Feature Values", height=300, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Top Genres:**")
                        if 'playlist_genre' in cluster_data.columns:
                            top_genres = cluster_data['playlist_genre'].value_counts().head(5)
                            for genre, count in top_genres.items():
                                st.write(f"• {genre}: {count} songs")
        
        st.markdown("---")
        
        # Export option
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Clustered Data as CSV", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Click to Download",
                    data=csv,
                    file_name="spotify_clustered_data.csv",
                    mime="text/csv"
                )
        
        with col2:
            if 'cluster' in df.columns and available_cluster_features:
                if st.button("📊 Download Cluster Summary", use_container_width=True):
                    cluster_summary = []
                    for i in range(n_clusters):
                        cluster_data = df[df['cluster'] == i]
                        summary = {'Cluster': i+1, 'Size': len(cluster_data)}
                        for feature in available_cluster_features:
                            if feature in cluster_data.columns:
                                summary[f'Avg_{feature}'] = cluster_data[feature].mean()
                        cluster_summary.append(summary)
                    
                    summary_df = pd.DataFrame(cluster_summary)
                    csv_summary = summary_df.to_csv(index=False)
                    st.download_button(
                        label="Click to Download Summary",
                        data=csv_summary,
                        file_name="cluster_summary.csv",
                        mime="text/csv"
                    )
        
        st.markdown("---")
        
        # Final recommendations
        st.success("✅ **Your recommendation system is ready!** You can now use these clusters to recommend similar songs based on audio feature similarity.")
        
        st.info("""
        **💡 Next Steps:**
        1. Use the cluster assignments to group similar songs
        2. For a given song, find its cluster and recommend other songs from the same cluster
        3. Rank recommendations by feature similarity within the cluster
        4. Consider adding user preferences and listening history for personalized recommendations
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🎵 Spotify Genre Segmentation Analysis | Built with Streamlit</p>
        <p style='font-size: 12px;'>Machine Learning-based Music Recommendation System</p>
    </div>
    """, unsafe_allow_html=True)