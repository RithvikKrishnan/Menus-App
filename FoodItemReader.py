import torch
import torchvision
import clip
from PIL import Image
import time
import os
import numpy as np
from groundingdino.util import box_ops
from groundingdino.util.inference import load_model, load_image, predict, annotate

device = "cpu"

# Load the CLIP model and the necessary image preprocessor
clip_model, preprocess = clip.load("ViT-B/32", device=device)
#print(f"CLIP model loaded on device: {device}")


# Third-Party Libraries
import cv2
import supervision as sv
import torch

# Your project's specific libraries


#config_path = r"C:\Users\rithv\OneDrive\Desktop\PurdueDiningApp\venv\src\groundingdino\groundingdino\config/GroundingDINO_SwinT_OGC.py" #the r is to ensure the path is treated as a raw string
#weights_path = r"C:\Users\rithv\OneDrive\Desktop\PurdueDiningApp\groundingdino_swint_ogc.pth" 

config_path = r"GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
weights_path = r"groundingdino_swint_ogc.pth"
model = load_model(config_path, weights_path, device = device)
#print("Grounding DINO model loaded on device:", device)

EMBEDDING_CACHE_PATH = "./clip_text_embeddings.pt" # The file where embeddings will be stored


def generate_text_embeddings(labels: list):
  """
  Takes a list of text labels and returns their CLIP embeddings.
  Efficiently uses a file cache to avoid re-computing existing embeddings.

  Args:
    labels: A list of string labels.

  Returns:
    A tensor containing the normalized text features (embeddings), ordered
    according to the input list.
  """
  # 1. Load existing embeddings from the cache file
  if os.path.exists(EMBEDDING_CACHE_PATH):
      cached_embeddings = torch.load(EMBEDDING_CACHE_PATH, map_location=device)
  else:
      cached_embeddings = {}

  # 2. Identify which labels are new and need to be computed
  labels_to_compute = [label for label in labels if label not in cached_embeddings]

  # 3. If there are new labels, compute their embeddings
  if labels_to_compute:
      #print(f"Found {len(labels_to_compute)} new labels to encode. Computing embeddings...")
      with torch.no_grad():
          # Your original logic, but ONLY for the new labels
          text_tokens = clip.tokenize(labels_to_compute).to(device)
          new_features = clip_model.encode_text(text_tokens)
          new_features /= new_features.norm(dim=-1, keepdim=True)
      
      # 4. Add the new embeddings to our cache dictionary
      for i, label in enumerate(labels_to_compute):
          # Store each embedding tensor individually
          cached_embeddings[label] = new_features[i]
      
      # 5. Save the updated cache back to the disk
      #print(f"Saving updated cache with {len(cached_embeddings)} total embeddings...")
      torch.save(cached_embeddings, EMBEDDING_CACHE_PATH)
  else:
      print("All labels found in cache. No new embeddings needed.")

  # 6. Assemble the final tensor in the correct order from the cache
  # This ensures the output tensor matches the order of the input 'labels' list
  ordered_embeddings = [cached_embeddings[label] for label in labels]
  
  return torch.stack(ordered_embeddings)

def classify_image_with_embeddings(img_path: str, k: int, text_embeddings, labels: list):
    """
    Classifies an image against a set of pre-computed text embeddings.
    Only returns labels with confidence >= 0.35.

    Args:
        img_path: Path to the input image file.
        k: Number of top labels to consider.
        text_embeddings: Pre-computed tensor of embeddings for the labels.
        labels: Original list of string labels.

    Returns:
        A list of the top labels (strings) with confidence >= 0.35
    """
    try:
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    except Exception as e:
        return f"Error processing image: {e}"

    with torch.no_grad():
        # Encode and normalize image
        image_features = clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity and probabilities
        similarity = (100.0 * image_features @ text_embeddings.T)
        probs = similarity.softmax(dim=-1).cpu().numpy()[0]

    # Get top-k indices
    top_k_indices = probs.argsort()[-k:][::-1]

    # Filter and return labels only
    results = [labels[i] for i in top_k_indices]

    return results


def detect_and_draw(
    image_path: str,
    labels: list,
    box_threshold=0.35,
    text_threshold=0.35,
    iou_threshold=0.5
):
    """
    Takes an image path and labels, runs Grounding DINO, applies Non-Max Suppression
    to reduce overlapping boxes, and draws the filtered results.

    Args:
        image_path (str): The path to the input image.
        labels (list): A list of strings representing the object labels to detect.
        box_threshold (float, optional): The confidence threshold for a box to be considered.
                                          Defaults to 0.35.
        text_threshold (float, optional): The confidence threshold for a label to match a box.
                                           Defaults to 0.35.
        iou_threshold (float, optional): The Intersection over Union threshold for NMS.
                                         Boxes with IoU > threshold will be suppressed.
                                         Defaults to 0.5.

    Returns:
        a list of labels detected in the image, or None if the model is not loaded or image loading fails.
    """
    if model is None:
        #print("Model is not loaded. Cannot perform detection.")
        return None
        
    try:
        image_source_pil, image_tensor = load_image(image_path)
        # We need the original image dimensions for un-normalizing boxes
        image_source_np = np.array(image_source_pil)
        h, w, _ = image_source_np.shape
    except FileNotFoundError:
        #print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        #print(f"Error loading image: {e}")
        return None

    text_prompt = " . ".join(labels).strip()
    #print("text prompt: ", text_prompt)

    # 1. Run the initial model prediction
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device="cpu"
    )

    #print(f"Found {len(boxes)} initial detections before NMS.")
    #print("found phrases", phrases)

    # 2. Perform Non-Max Suppression (NMS)
    if len(boxes) > 0:
        # Un-normalize boxes to pixel coordinates for NMS
        boxes_pixel = boxes * torch.Tensor([w, h, w, h])
        
        # Convert from center-width-height to xyxy format
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_pixel)

        # Perform NMS
        # torchvision.ops.nms returns the indices of the boxes to keep
        nms_indices = torchvision.ops.nms(boxes_xyxy, logits, iou_threshold)
        #print(f"Keeping {len(nms_indices)} boxes after NMS.")

        # Filter the results based on NMS indices
        filtered_boxes = boxes[nms_indices]
        filtered_logits = logits[nms_indices]
        # Phrases is a list, so we need to index it differently
        filtered_phrases = [phrases[i] for i in nms_indices]
    else:
        # If no boxes were detected initially, just use the empty lists
        filtered_boxes, filtered_logits, filtered_phrases = boxes, logits, phrases
    
    final_phrases = []
    for phrase in filtered_phrases:
        best_match = ""
        # Find the longest original label that is a substring of the model's output
        for original_label in labels:
            original_label = original_label.lower()
            if original_label in phrase and len(original_label) > len(best_match):
                #print(original_label, "is a substring of", phrase)
                best_match = original_label
        # If a match was found, use it. Otherwise, fall back to the raw phrase.
        final_phrases.append(best_match)
        if not best_match:
            print(f"Warning: No matching label found for phrase '{phrase}'. Using raw phrase.")
    print("Final phrases after matching:", final_phrases)

    # 3. Annotate the image with the filtered results
    annotated_image_np = annotate(
        image_source=image_source_np,
        boxes=filtered_boxes,
        logits=filtered_logits,
        phrases=final_phrases
    )
    
    # Display the final image
    cv2.imshow("Annotated Image (Non-Overlapping)", annotated_image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    return final_phrases
import json

def get_food_names_from_file(filename: str) -> list[str]:
    """
    Reads a master food database JSON file and extracts all food names.

    The function expects the JSON file to be a dictionary where the keys are
    item IDs and the values are objects, each containing details for one
    food item. The food's name is expected to be under the "Name" key
    in each of these value objects.

    Args:
        filename: The path to the JSON database file.

    Returns:
        A list of strings, where each string is the name of a food item.
        Returns an empty list if the file is not found, is empty, or
        cannot be parsed.
    """
    food_names = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # The data is a dictionary of items, we need to iterate through its values
            for item_details in data.values():
                # Safely get the 'Name' key to avoid errors if it's missing
                name = item_details.get("Name")
                if name:
                    food_names.append(name)

    except FileNotFoundError:
        #print(f"Error: The file '{filename}' was not found.")
        return []
    except json.JSONDecodeError:
        #print(f"Error: The file '{filename}' is not a valid JSON file.")
        return []
        
    return food_names
def get_nutrition_info(food_name: str, db_filepath: str = "master_food_database.json") -> str:
    """
    Searches a food database JSON file for an item by name and returns its
    nutritional information in a human-readable format.

    Args:
        food_name (str): The name of the food to search for (case-insensitive).
        db_filepath (str): The path to the master food database JSON file.

    Returns:
        str: A formatted string with the nutritional information, or a
             message if the food is not found or an error occurs.
    """
    # --- 1. Load the Database File ---
    try:
        with open(db_filepath, 'r', encoding='utf-8') as f:
            food_db = json.load(f)
    except FileNotFoundError:
        return f"Error: Database file not found at '{db_filepath}'"
    except json.JSONDecodeError:
        return f"Error: Could not parse the JSON file. It might be corrupted or empty."

    # --- 2. Find the Food Item by Name ---
    # The database is a dict of {ID: data}, so we iterate through the values.
    found_item = None
    for item_data in food_db.values():
        # Using .get() is safer in case an item is missing a "Name" key
        # Using .lower() makes the search case-insensitive
        if item_data.get("Name", "").lower() == food_name.lower():
            found_item = item_data
            break  # Stop searching once we find the first match

    if not found_item:
        #return f"Could not find a food item named '{food_name}' in the database."
        return ""

    # --- 3. Format the Output String ---
    # We will build a list of lines and join them at the end.
    output_lines = []

    # --- Basic Info ---
    output_lines.append(f"--- Nutrition for: {found_item['Name']} ---")
    output_lines.append(f"ID: {found_item.get('ID', 'N/A')}")
    output_lines.append(f"Ingredients: {found_item.get('Ingredients', 'Not available')}")
    output_lines.append("=" * 40) # Separator

    # --- Nutrition Details ---
    output_lines.append("Nutrition Facts:")
    if not found_item.get("NutritionReady") or not found_item.get("Nutrition"):
        output_lines.append("  - Nutrition information is not available for this item.")
    else:
        for nutrient in found_item["Nutrition"]:
            name = nutrient.get("Name", "Unknown")
            # Use LabelValue for a clean display (e.g., "21g" instead of 21.0)
            value = nutrient.get("LabelValue", "N/A")
            daily_value = nutrient.get("DailyValue")

            line = f"  - {name:<20} {value}" # Left-align name for clean columns
            if daily_value:
                line += f" ({daily_value} DV)" # Add daily value if it exists
            output_lines.append(line)

    output_lines.append("=" * 40) # Separator

    # --- Allergens and Dietary Info ---
    output_lines.append("Allergens & Dietary Tags:")
    # Filter the list to only include allergens/tags that are 'true'
    active_allergens = [
        allergen["Name"] for allergen in found_item.get("Allergens", []) if allergen.get("Value")
    ]

    if active_allergens:
        output_lines.append(f"  - Contains: {', '.join(active_allergens)}")
    else:
        output_lines.append("  - No specific allergens or tags listed.")


    return "\n".join(output_lines)
all_labels = get_food_names_from_file("master_food_database.json")
for i in range(len(all_labels)):
    label = all_labels[i]
    label = label.strip().strip(".")
    all_labels[i] = label

#print("Total food items loaded:", len(all_labels))
import time
start_time = time.perf_counter()
embeddings = generate_text_embeddings(all_labels)
end_time = time.perf_counter()
#print(end_time - start_time)
shortlisted_labels = classify_image_with_embeddings(img_path = r"C:\Users\rithv\Downloads\chicken-fajita-marinade-4.jpg", k = 10,
                                                    text_embeddings = embeddings, labels = all_labels)
#print(shortlisted_labels)

phrases = detect_and_draw(image_path = r"C:\Users\rithv\Downloads\chicken-fajita-marinade-4.jpg", labels = shortlisted_labels,
                box_threshold = 0.35, text_threshold = 0.25)
print(phrases)
for phrase in phrases:
    nutrition_info = get_nutrition_info(phrase)
    print(nutrition_info)
    #print(f"Detected phrase: {phrase}")
    #print(f"Nutrition info: {nutrition_info}")
