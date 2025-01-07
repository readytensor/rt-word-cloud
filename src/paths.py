import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
WORD_CLOUD_FPATH = os.path.join(OUTPUT_DIR, "wordcloud-square.png")