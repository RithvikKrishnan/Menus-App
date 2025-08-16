# Menus-App

This app takes in an image of a plate of food, identifies and segments the food in the plate, and outputs the nutritional information of the plate based on data taken from Purdue's dining services.

**(It would be awesome to add a screenshot or a GIF of the app working right here!)**

### How It Works: The Two-Model Approach

The item reading code employs two models to balance speed and accuracy:

1.  **First, a Quick Shortlist with CLIP:** The app uses OpenAI's CLIP model to shortlist the 1000+ food labels down to the 10 foods most likely to be in the image. This is extremely fast because the only expensive part—generating label embeddings—is already done and stored. While it's great for quickly narrowing things down, it can't tell you *where* the food is on the plate.

2.  **Then, Bounding Boxes with GroundingDINO:** With that shortlist of 10 labels, the much slower GroundingDINO model takes over. Its job is to find those items in the image, draw boxes around them, and provide the final annotation. This step takes significantly less time than having it search for all 1000+ original labels.

> **Note:** This second step usually takes 15-20 seconds, as even the lightest GroundingDINO model used here is computationally intensive.

### The Data

The nutritional data comes from a JSON file with over 1000 food items. The data collection code, which is also in this repository, makes 20 threads of "scrapers" to crawl through Purdue's dining API and store that data.

### How to Use It

As of right now, the best way to use this is to clone this repository and run it locally.

1.  Clone the repository and move into the directory:
    ```sh
    git clone https://github.com/RithvikKrishnan/Menus-App.git
    cd Menus-App
    ```

2.  Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3.  Run the app with Streamlit:
    ```sh
    streamlit run FoodItemReader.py
    ```
    > **Heads up:** The initial loading time will be 15-20 seconds because the CLIP and GroundingDINO models need to be loaded into memory.

### What's Next?

In the future, I intend to:
*   Deploy this and make it publicly accessible.
*   Find ways to increase accuracy without increasing computational requirements.
*   Periodically run my scraper and update the database (though this can be done locally if you have the file).
