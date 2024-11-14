import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

import paths

word_list = [
    "GPT",
    "Transformers",
    "LLMs",
    "YOLO",
    "Object Detection",
    "Graph NN",
    "RAG",
    "Time Series",
    "XGBoost",
    "Attention",
    "Causal AI",
    "BERT",
    "Matrix Factorization",
    "AI Agents",
    "XAI",
    "Recommendation Systems",
    "Knowledge Graphs",
    "Anomaly Detection",
    "Collaborative Filtering",
    "Forecasting",
    "LightGBM",
    "CatBoost",
    "ResNet",
    "Regression",
    "Classification",
    "AutoML",
    "LLaMA",
    "Gradient Descent",
    "Backpropagation",
    "Clustering",
    "PCA",
    "Neural Networks",
    "CNN",
    "K Means",
    "GANs",
    "PyTorch",
    "TensorFlow",
    "SVM",
    "Image Segmentation",
    "Face Recognition",
    "ARIMA",
    "SVD",
    "Random Forest",
    "Seasonality",
    "Stationarity",
    "Prophet",
    "Model Bias",
    "Data Drift",
    "PEFT",
    "Agents",
    "Pose Detection",
    "RecSys",
]


words = " ".join(word_list)

dimensions = (1200, 700)

width, height = dimensions
gradient = np.linspace(30, 40, height).reshape((height, 1)).repeat(width, axis=1)
background = np.dstack((gradient, gradient, gradient)).astype(np.uint8)
background_image = Image.fromarray(background)

word_colors = [
    "rgb(102, 255, 255)",  # Softer Cyan
    "rgb(255, 102, 255)",  # Softer Magenta
    "rgb(255, 255, 102)",  # Softer Yellow
    "rgb(102, 255, 102)",  # Softer Lime
    "rgb(255, 178, 102)",  # Softer Orange
    "rgb(255, 102, 178)",  # Softer Pink
    "rgb(178, 102, 255)",  # Softer Purple
    "rgb(102, 178, 255)",  # Softer Blue
    "rgb(255, 153, 153)",  # Salmon
    "rgb(153, 255, 204)",  # Mint
    "rgb(204, 153, 255)",  # Lavender
    "rgb(255, 204, 153)",  # Peach
    "rgb(153, 204, 255)",  # Sky Blue
    "rgb(255, 178, 178)",  # Light Coral
    "rgb(178, 255, 178)",  # Light Green
]

word_color_map = {
    word: word_colors[i % len(word_colors)]
    for i, word in enumerate(word_list)
}


def color_func(
    word,
    font_size=None,
    position=None,
    orientation=None,
    font_path=None,
    random_state=None,
):
    return word_color_map[word]


# Create a dictionary of word frequencies without underscores
word_frequencies = Counter(word_list)

wordcloud = WordCloud(
    width=width,
    height=height,
    background_color=None,
    mode="RGBA",
    random_state=1,
    color_func=color_func,
    font_path="arial.ttf",
    prefer_horizontal=0.7,
    collocations=False,
    relative_scaling=0.5,
    margin=20,
    # scale=2,
    max_font_size=200,   # Set max font size
    min_font_size=20,    # Set min font size
    normalize_plurals=False,
).generate_from_frequencies(word_frequencies)

wordcloud_image = wordcloud.to_image()
combined_image = Image.alpha_composite(
    background_image.convert("RGBA"), wordcloud_image
)

combined_image.save(paths.WORD_CLOUD_FPATH)
plt.figure(figsize=(15, 8))
plt.imshow(combined_image, interpolation="bilinear")
plt.axis("off")
plt.show()
