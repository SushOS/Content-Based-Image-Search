# Content-based-Image-Search

Content-Based Image Search (CBIS) is a technique for retrieving images from a database based on visual content features rather than textual descriptions. By indexing images using attributes like color, texture, and shape, CBIS systems deliver more accurate and relevant search results compared to traditional methods.

# Team Members

- Rushi Shah (B21AI032)
- Sushant Ravva (B21CS084)
- Jaysukh Makvana (B21CS043)

# Usage CLIP

To use the image retrieval system, follow these steps:

1. Clone or download this repository to your local machine.

2. To install the required dependencies, you can use pip with the provided requirements.txt file. Simply run the following command:

```
pip install -r requirements.txt
```

3. Navigate to the directory containing the `interface_clip.py` script.

4. Run the script using the following command:

```
streamlit run interface_clip.py
```

5. The Streamlit interface will launch in your default web browser.

6. Enter the path to the folder containing the images you want to search through in the provided text input field on the sidebar.

7. Input your search query or prompt in the text input field.

8. Click the "Search" button to perform the image search.

9. The top 5 images most relevant to your query will be displayed along with their paths.

## About Script

The `interface_clip.py` script performs content-based image search using CLIP. Here's a brief overview of its functionality:

- It loads the CLIP model and sets up necessary configurations.
- Provides a Streamlit interface for user interaction.
- Allows users to input the path to the image folder and their search prompts.
- Utilizes the CLIP model to retrieve the most relevant images based on the user's prompt.
- Displays the top 5 images along with their paths.

# Usage BLIP

To use the image search engine, follow these steps:

1. Clone or download this repository to your local machine.
2. To install the required dependencies, you can use pip with the provided requirements.txt file. Simply run the following command:

```
pip install -r requirements.txt
```

3. Navigate to the directory containing the `interface_blip.py` script.
4. Download the precomputed embeddings file `embeddings.pkl` and place it in the same directory.
5. Run the script using the following command:

```
streamlit run interface_blip.py
```

6. The Streamlit interface will launch in your default web browser.
7. Enter your search query in the provided text input field.
8. Click the "Search" button to perform the image search.
9. The top 5 images most relevant to your query will be displayed.

## About Script

The `interface_blip.py` script utilizes the BLIP model for generating embeddings of text queries and Faiss indexing for efficient similarity search. It loads precomputed embeddings and sets up a Faiss index to perform nearest neighbor search. The user can input a search query, and the script will return the top 5 images most similar to the query.

# Notes

- Ensure that the provided image folder contains images in a compatible format (e.g., JPEG, PNG).

- CLIP model loading might take some time depending on your hardware configuration.

# Acknowledgements

- CLIP script utilizes the CLIP model developed by OpenAI.
- BLIP script utilizes the Sentence Transformers library for text embedding.
