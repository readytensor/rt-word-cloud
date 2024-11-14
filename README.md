# Word Cloud Generator

Create visual word clouds from a list of AI/ML terms with customizable colors and styling.

## Requirements

- Python 3.x
- Packages: numpy, pillow, matplotlib, wordcloud (see `requirements.txt`)

## Usage

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

```bash
source venv/bin/activate
# `venv\Scripts\activate`  in windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Customize the word list, colors, and dimensions in the script `src/generate_word-cloud.py`.

- Edit word list in `word_list` variable
- Modify colors in `word_colors`
- Adjust dimensions with `dimensions` variable

5. Run:

```bash
python generate_wordcloud.py
```

Output will be saved as `wordcloud.png` under `output` directory.
