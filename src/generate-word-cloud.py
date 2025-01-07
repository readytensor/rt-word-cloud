import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

import paths

word_list = [
    "Transformers",
    "RAG",
    "YOLO",
    "LLMs",
    "Object Detection",
    "Agentic AI",
    "Graph NN",
    "Recommendation Systems",
    "Forecasting",
    "ReAct",
    "Classification",
    "Causal AI",
    "LangChain",
    "Time Series",
    "XAI",
    "Clustering",
    "XGBoost",
    "Knowledge Graphs",
    "PEFT",
    "Crew AI",
    "Pose Detection",
    "Image Segmentation",
    "GPT",
    "Attention",
    "Matrix Factorization",
    "LLaMA",
    "BERT",
    "Anomaly Detection",
    "Collaborative Filtering",
    "LightGBM",
    "CatBoost",
    "ResNet",
    "AutoML",
    "Gradient Descent",
    "Backpropagation",
    "PCA",
    "Neural Networks",
    "CNN",
    "K Means",
    "GANs",
    "PyTorch",
    "TensorFlow",
    "SVM",
    "Face Recognition",
    "ARIMA",
    "SVD",
    "Random Forest",
    "Seasonality",
    "Stationarity",
    "Prophet",
    "Model Bias",
    "Data Drift",
]


width, height = 800, 800
padding_horizontal = 30  # 50px on each side
padding_vertical = 30    # 25px on top and bottom
width_with_padding = width + (2 * padding_horizontal)
height_with_padding = height + (2 * padding_vertical)

# Create gradient background with padding
gradient = np.linspace(30, 40, height_with_padding).reshape((height_with_padding, 1)).repeat(width_with_padding, axis=1)
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
   "rgb(178, 255, 178)"   # Light Green
]

word_color_map = {word: word_colors[i % len(word_colors)] 
                for i, word in enumerate(word_list)}

def color_func(word, font_size=None, position=None, orientation=None, 
             font_path=None, random_state=None):
   return word_color_map[word]

word_frequencies = Counter(word_list)

# Create word cloud with original width
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
   margin=25,
   max_font_size=200,
   min_font_size=20,
   normalize_plurals=False,
).generate_from_frequencies(word_frequencies)

# Center word cloud on padded background
wordcloud_image = wordcloud.to_image()
# Update offset variables 
x_offset = padding_horizontal
y_offset = padding_vertical

background_with_padding = background_image.convert("RGBA")
background_with_padding.paste(wordcloud_image, (x_offset, y_offset), wordcloud_image)

combined_image = background_with_padding

combined_image.save(paths.WORD_CLOUD_FPATH)
plt.figure(figsize=(15, 8))
plt.imshow(combined_image, interpolation="bilinear")
plt.axis("off")
plt.show()
