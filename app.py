from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
import os
from PIL import Image
import numpy as np
import json
import math
import re
import random
from collections import defaultdict
from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for
from transformers import CLIPProcessor, CLIPModel  # For image annotation
# Add these imports at the top of the file, after the existing imports
import numpy as np
import math
from collections import defaultdict
app = Flask(__name__)
# ---------------------------
# Load CLIP Model for Image Annotation
# ---------------------------
print("Loading CLIP model for image annotation...")
try:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP model loaded successfully")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    print("Will continue without CLIP annotation")
    model = None
    processor = None

# ---------------------------
# Utility Functions for Crawling and Downloading
# ---------------------------
def is_valid_image(img_path):
    """Check if image is valid and not all black"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False
        # Check if image is all black or nearly black
        mean_value = np.mean(img)
        return mean_value > 5  # Threshold for considering an image as valid
    except Exception as e:
        print(f"Error validating image {img_path}: {e}")
        return False

def download_image(url, img_path, headers, timeout=10):
    """Download image and verify it's valid"""
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=timeout)
        if response.status_code != 200:
            print(f"Failed to download image (status {response.status_code}): {url}")
            return False
            
        with open(img_path, 'wb') as f:
            f.write(response.content)
            
        # Verify the image is valid
        if is_valid_image(img_path):
            return True
        else:
            os.remove(img_path)
            print(f"Removed invalid or black image: {img_path}")
            return False
            
    except Exception as e:
        print(f"Download error for {url}: {e}")
        if os.path.exists(img_path):
            os.remove(img_path)
        return False

from PIL import Image

def enhance_annotation(img_path, original_alt):
    """Enhance image annotation using CLIP model"""
    if model is None or processor is None:
        return original_alt

    try:
        # Open the image using PIL and convert to RGB (CLIP expects RGB)
        image = Image.open(img_path).convert("RGB")

        # Provide a set of candidate annotations based on the query context
        candidate_texts = [
            "a photo of nature", 
            "a mountain landscape", 
            "a forest scene",
            "a beach scene",
            "a desert landscape",
            "a waterfall",
            "a lake view",
            "a sunset over mountains",
            "a river through forest",
            "a field of flowers",
            "a snowy mountain peak",
            "an autumn forest",
            "a tropical island"
        ]

        inputs = processor(text=candidate_texts, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image.softmax(dim=1)
        best_idx = logits_per_image.argmax().item()
        best_match = candidate_texts[best_idx]

        # Combine original alt text with CLIP annotation
        enhanced = f"{original_alt} {best_match}" if original_alt else best_match
        print(f"Enhanced annotation: '{original_alt}' -> '{enhanced}'")
        return enhanced

    except Exception as e:
        print(f"Error enhancing annotation: {e}")
        return original_alt


def setup_chrome_driver():
    """Setup and return a configured Chrome WebDriver"""
    options = Options()
    # Comment out headless mode for debugging - sometimes Google blocks headless browsers
    # options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')  # Hide automation
    
    # Use a realistic user agent
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59'
    ]
    options.add_argument(f'--user-agent={random.choice(user_agents)}')
    
    # Add language preference
    options.add_argument('--lang=en-US,en;q=0.9')
    
    # Set window size to a normal desktop size
    options.add_argument('--window-size=1920,1080')
    
    # Disable extensions
    options.add_argument('--disable-extensions')
    
    # Disable GPU acceleration
    options.add_argument('--disable-gpu')
    
    # Disable automation info bar
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    # Execute CDP commands to hide automation
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": random.choice(user_agents)
    })
    
    # Set navigator.webdriver to undefined
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def crawl_google_images(query="nature scenery", limit=1500, images_dir="images"):
    """Crawl Google Images and download images matching the query"""
    print(f"Starting to crawl Google Images for '{query}' (limit: {limit})")
    
    # Create images directory if it doesn't exist
    os.makedirs(images_dir, exist_ok=True)
    
    driver = setup_chrome_driver()
    image_data = []
    
    try:
        # Navigate to Google Images
        driver.get(f"https://www.google.com/search?q={query}&tbm=isch")
        
        # Wait for page to load
        time.sleep(3)
        
        # Try to accept cookies if the dialog appears
        try:
            cookie_buttons = driver.find_elements(By.XPATH, "//button[contains(., 'Accept all') or contains(., 'I agree') or contains(., 'Agree')]")
            if cookie_buttons:
                cookie_buttons[0].click()
                print("Accepted cookies")
                time.sleep(1)
        except Exception as e:
            print(f"Cookie handling error: {e}")
        
        # Keep track of already processed image URLs to avoid duplicates
        processed_urls = set()
        
        # Try different methods to find images
        def find_images():
            # Try multiple selectors to find images
            selectors = [
                "img.rg_i", 
                "img.Q4LuWd", 
                "img[data-src]",
                "img[src^='http']",
                "img.YQ4gaf"
            ]
            
            all_images = []
            for selector in selectors:
                try:
                    images = driver.find_elements(By.CSS_SELECTOR, selector)
                    if images:
                        all_images.extend(images)
                        print(f"Found {len(images)} images with selector: {selector}")
                except:
                    pass
            
            # Remove duplicates (same element found by different selectors)
            unique_images = []
            seen_elements = set()
            for img in all_images:
                element_id = img.id
                if element_id not in seen_elements:
                    seen_elements.add(element_id)
                    unique_images.append(img)
            
            return unique_images
        
        # Alternative method: try to get image URLs directly from page source
        def extract_image_urls_from_page():
            page_source = driver.page_source
            # Look for image URLs in the page source
            url_pattern = r'https?://[^"\']+\.(?:jpg|jpeg|png|gif|webp)'
            urls = re.findall(url_pattern, page_source)
            print(f"Found {len(urls)} image URLs in page source")
            return urls
        
        # Keep scrolling until we have enough images or can't load more
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        max_scroll_attempts = 50  # Increased for more attempts to get 1500 images
        
        while len(image_data) < limit and scroll_attempts < max_scroll_attempts:
            scroll_attempts += 1
            print(f"Scroll attempt {scroll_attempts}/{max_scroll_attempts} - Images collected: {len(image_data)}/{limit}")
            
            # Find images using element selectors
            img_elements = find_images()
            print(f"Found {len(img_elements)} total image elements")
            
            # Process images found by selectors
            new_images_found = 0
            for img in img_elements:
                try:
                    # Try different attributes where the actual image URL might be stored
                    src = (img.get_attribute('src') or 
                           img.get_attribute('data-src') or 
                           img.get_attribute('data-iurl'))
                    
                    if not src or not src.startswith('http') or 'base64' in src or src in processed_urls:
                        continue
                    
                    processed_urls.add(src)
                    
                    # Get alt text
                    alt_text = img.get_attribute('alt') or query
                    
                    # Download the image
                    img_name = f"{images_dir}/{len(image_data) + 1}.jpg"
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
                        'Referer': 'https://www.google.com/'
                    }
                    
                    if download_image(src, img_name, headers):
                        # Enhance annotation with CLIP
                        enhanced_alt = enhance_annotation(img_name, alt_text)
                        
                        image_data.append({
                            "path": img_name,
                            "text": enhanced_alt,
                            "url": src
                        })
                        new_images_found += 1
                        print(f"Downloaded image {len(image_data)}/{limit}: {img_name}")
                        
                        if len(image_data) >= limit:
                            break
                except Exception as e:
                    print(f"Error processing image element: {e}")
            
            # If no images found by selectors, try extracting from page source
            if new_images_found == 0:
                image_urls = extract_image_urls_from_page()
                for url in image_urls:
                    if url in processed_urls:
                        continue
                    
                    processed_urls.add(url)
                    
                    # Download the image
                    img_name = f"{images_dir}/{len(image_data) + 1}.jpg"
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
                        'Referer': 'https://www.google.com/'
                    }
                    
                    if download_image(url, img_name, headers):
                        # Enhance annotation with CLIP
                        enhanced_alt = enhance_annotation(img_name, query)
                        
                        image_data.append({
                            "path": img_name,
                            "text": enhanced_alt,
                            "url": url
                        })
                        new_images_found += 1
                        print(f"Downloaded image {len(image_data)}/{limit} from page source: {img_name}")
                        
                        if len(image_data) >= limit:
                            break
            
            if len(image_data) >= limit:
                break
                
            # If no new images were found, try to click "Show more results" button or break
            if new_images_found == 0:
                try:
                    # Try different button selectors
                    button_selectors = [
                        "//input[@type='button' and @value='Show more results']",
                        "//button[contains(text(), 'Show more')]",
                        "//button[contains(text(), 'Load more')]",
                        "//span[contains(text(), 'Show more results')]/.."
                    ]
                    
                    button_found = False
                    for selector in button_selectors:
                        try:
                            buttons = driver.find_elements(By.XPATH, selector)
                            if buttons:
                                buttons[0].click()
                                print(f"Clicked button with selector: {selector}")
                                button_found = True
                                time.sleep(2)
                                break
                        except:
                            pass
                    
                    if not button_found:
                        print("No 'Show more results' button found")
                        # If we've scrolled multiple times with no results, try a different approach
                        if scroll_attempts % 5 == 0:  # Every 5 attempts
                            # Try to refresh the page or modify the search query slightly
                            if scroll_attempts % 10 == 0:  # Every 10 attempts
                                # Modify the search query slightly
                                modified_query = f"{query} {random.choice(['beautiful', 'amazing', 'stunning', 'high resolution'])}"
                                print(f"Modifying search query to: {modified_query}")
                                driver.get(f"https://www.google.com/search?q={modified_query}&tbm=isch")
                                time.sleep(3)
                            else:
                                # Refresh the page
                                print("Refreshing the page")
                                driver.refresh()
                                time.sleep(3)
                except Exception as e:
                    print(f"Error clicking 'Show more' button: {e}")
            
            # Scroll down to load more images
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for new images to load
            
            # Check if we've reached the bottom of the page
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("Reached end of page, no more scrolling possible")
                # Try one more time with a different approach
                driver.execute_script("window.scrollBy(0, 1000);")  # Scroll a fixed amount
                time.sleep(1)
                
                # If we've been stuck for a while, try a different approach
                if scroll_attempts % 5 == 0:  # Every 5 attempts when stuck
                    # Try clicking on a thumbnail to open the larger view, then go back
                    try:
                        thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.rg_i, img.Q4LuWd")
                        if thumbnails:
                            random_index = random.randint(0, min(10, len(thumbnails)-1))
                            print(f"Clicking on thumbnail {random_index} to trigger more image loading")
                            thumbnails[random_index].click()
                            time.sleep(2)
                            driver.back()
                            time.sleep(2)
                    except Exception as e:
                        print(f"Error clicking thumbnail: {e}")
            
            last_height = new_height
        
        print(f"Successfully downloaded {len(image_data)} images")
        
        # If we didn't find any images, create a few dummy entries for testing
        if not image_data:
            print("No images found. Creating dummy data for testing...")
            for i in range(5):
                image_data.append({
                    "path": f"dummy_{i+1}.jpg",
                    "text": f"Dummy image {i+1} for {query}",
                    "url": "https://example.com/dummy.jpg"
                })
        
        # Save image data to JSON file for search functionality
        with open('image_data.json', 'w') as f:
            json.dump(image_data, f, indent=2)
            
        return image_data
        
    except Exception as e:
        print(f"Error during crawling: {e}")
        return image_data
    finally:
        driver.quit()

# ---------------------------
# Text Processing and Search Functions
# ---------------------------
def preprocess_text(text):
    """Preprocess text for search"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text.strip()

def compute_bm25(query, documents, k1=1.5, b=0.75):
    """Compute BM25 scores for documents given a query"""
    # Handle empty document list
    if not documents:
        print("Warning: No documents to search")
        return []
        
    query_terms = preprocess_text(query).split()
    if not query_terms:
        return []
        
    # Calculate average document length
    avg_doc_len = sum(len(preprocess_text(doc["text"]).split()) for doc in documents) / max(len(documents), 1)
    
    results = []
    for doc in documents:
        doc_id = doc["path"]
        doc_text = preprocess_text(doc["text"])
        doc_terms = doc_text.split()
        doc_len = len(doc_terms)
        
        score = 0
        for term in query_terms:
            # Count term frequency in document
            term_freq = doc_terms.count(term)
            if term_freq == 0:
                continue
                
            # Count documents containing the term
            docs_with_term = sum(1 for d in documents if term in preprocess_text(d["text"]).split())
            
            # Calculate IDF (Inverse Document Frequency)
            idf = math.log((len(documents) - docs_with_term + 0.5) / (docs_with_term + 0.5) + 1)
            
            # BM25 formula
            term_score = idf * (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * (doc_len / avg_doc_len)))
            score += term_score
        
        if score > 0:
            results.append({"path": doc_id, "text": doc["text"], "score": score})
    
    # Sort by score in descending order
    return sorted(results, key=lambda x: x["score"], reverse=True)

# Add the evaluation metrics functions after the compute_bm25 function
def compute_precision_at_k(relevant_docs, retrieved_docs, k=5):
    """
    Compute Precision@k: the proportion of relevant documents in the top-k results
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs in ranked order
        k: Number of top results to consider
        
    Returns:
        Precision@k score (float)
    """
    if not retrieved_docs or k <= 0:
        return 0.0
    
    # Consider only top k documents
    top_k_docs = retrieved_docs[:k]
    
    # Count relevant documents in top k
    relevant_in_top_k = sum(1 for doc in top_k_docs if doc in relevant_docs)
    
    # Calculate precision
    return relevant_in_top_k / k

def compute_average_precision(relevant_docs, retrieved_docs):
    """
    Compute Average Precision: the average of precision values at each relevant document
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs in ranked order
        
    Returns:
        Average Precision score (float)
    """
    if not relevant_docs or not retrieved_docs:
        return 0.0
    
    relevant_count = 0
    sum_precision = 0.0
    
    for i, doc in enumerate(retrieved_docs):
        # If the current document is relevant
        if doc in relevant_docs:
            relevant_count += 1
            # Precision at current position (i+1 because i is 0-indexed)
            precision_at_i = relevant_count / (i + 1)
            sum_precision += precision_at_i
    
    # If no relevant documents were retrieved
    if relevant_count == 0:
        return 0.0
    
    # Average precision
    return sum_precision / len(relevant_docs)

def compute_mean_average_precision(all_queries_relevant_docs, all_queries_retrieved_docs):
    """
    Compute Mean Average Precision (MAP): the mean of average precision scores over all queries
    
    Args:
        all_queries_relevant_docs: Dictionary mapping query IDs to lists of relevant document IDs
        all_queries_retrieved_docs: Dictionary mapping query IDs to lists of retrieved document IDs
        
    Returns:
        MAP score (float)
    """
    if not all_queries_relevant_docs:
        return 0.0
    
    sum_ap = 0.0
    query_count = 0
    
    for query_id in all_queries_relevant_docs:
        if query_id in all_queries_retrieved_docs:
            relevant_docs = all_queries_relevant_docs[query_id]
            retrieved_docs = all_queries_retrieved_docs[query_id]
            
            ap = compute_average_precision(relevant_docs, retrieved_docs)
            sum_ap += ap
            query_count += 1
    
    # If no queries were evaluated
    if query_count == 0:
        return 0.0
    
    # Mean average precision
    return sum_ap / query_count

def compute_ndcg_at_k(relevant_docs, retrieved_docs, relevance_scores=None, k=5):
    """
    Compute Normalized Discounted Cumulative Gain at k (NDCG@k)
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs in ranked order
        relevance_scores: Dictionary mapping document IDs to relevance scores (if None, binary relevance is assumed)
        k: Number of top results to consider
        
    Returns:
        NDCG@k score (float)
    """
    if not retrieved_docs or k <= 0:
        return 0.0
    
    # Consider only top k documents
    top_k_docs = retrieved_docs[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, doc in enumerate(top_k_docs):
        # Position in the ranking (1-indexed)
        pos = i + 1
        
        # Get relevance score (binary if not provided)
        if relevance_scores and doc in relevance_scores:
            rel = relevance_scores[doc]
        else:
            rel = 1.0 if doc in relevant_docs else 0.0
        
        # DCG formula: rel_i / log_2(i+1)
        dcg += rel / math.log2(pos + 1)
    
    # Calculate ideal DCG (IDCG)
    # For ideal ranking, sort relevant documents by their relevance scores
    if relevance_scores:
        # Get relevance scores for relevant documents
        relevant_with_scores = [(doc, relevance_scores.get(doc, 0.0)) for doc in relevant_docs]
        # Sort by relevance score in descending order
        sorted_relevant = [doc for doc, score in sorted(relevant_with_scores, key=lambda x: x[1], reverse=True)]
    else:
        # With binary relevance, any ordering of relevant docs is ideal
        sorted_relevant = relevant_docs
    
    # Calculate IDCG
    idcg = 0.0
    for i, doc in enumerate(sorted_relevant[:k]):
        pos = i + 1
        if relevance_scores and doc in relevance_scores:
            rel = relevance_scores[doc]
        else:
            rel = 1.0
        
        idcg += rel / math.log2(pos + 1)
    
    # If there are no relevant documents, IDCG is 0, and NDCG is defined as 0
    if idcg == 0:
        return 0.0
    
    # NDCG = DCG / IDCG
    return dcg / idcg

def evaluate_search_system(queries, ground_truth, search_function):
    """
    Evaluate a search system using multiple metrics
    
    Args:
        queries: List of query strings
        ground_truth: Dictionary mapping query strings to lists of relevant document IDs
        search_function: Function that takes a query and returns a list of document IDs
        
    Returns:
        Dictionary with evaluation metrics
    """
    all_queries_retrieved_docs = {}
    all_queries_relevant_docs = {}
    
    p_at_5_sum = 0.0
    ndcg_at_5_sum = 0.0
    
    for i, query in enumerate(queries):
        query_id = f"q{i}"
        
        # Get relevant documents for this query
        if query in ground_truth:
            relevant_docs = ground_truth[query]
        else:
            relevant_docs = []
        
        # Get retrieved documents using the search function
        retrieved_docs = [doc["path"] for doc in search_function(query)]
        
        # Store for MAP calculation
        all_queries_retrieved_docs[query_id] = retrieved_docs
        all_queries_relevant_docs[query_id] = relevant_docs
        
        # Calculate P@5
        p_at_5 = compute_precision_at_k(relevant_docs, retrieved_docs, k=5)
        p_at_5_sum += p_at_5
        
        # Calculate NDCG@5
        ndcg_at_5 = compute_ndcg_at_k(relevant_docs, retrieved_docs, k=5)
        ndcg_at_5_sum += ndcg_at_5
    
    # Calculate MAP
    map_score = compute_mean_average_precision(all_queries_relevant_docs, all_queries_retrieved_docs)
    
    # Calculate average P@5 and NDCG@5
    avg_p_at_5 = p_at_5_sum / len(queries) if queries else 0.0
    avg_ndcg_at_5 = ndcg_at_5_sum / len(queries) if queries else 0.0
    
    return {
        "MAP": map_score,
        "P@5": avg_p_at_5,
        "NDCG@5": avg_ndcg_at_5
    }

# ---------------------------
# Flask Web Interface for Image Search
# ---------------------------
app = Flask(__name__)

# Basic HTML template for search
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Search Engine with CLIP Annotations</title>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-dark: #4338ca;
            --secondary: #10b981;
            --dark: #1f2937;
            --light: #f9fafb;
            --gray: #6b7280;
            --light-gray: #e5e7eb;
            --danger: #ef4444;
            --card-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            --transition: all 0.3s ease;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f3f4f6;
            color: var(--dark);
            line-height: 1.6;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--light-gray);
        }
        
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: var(--gray);
            font-weight: 300;
            font-size: 1.1rem;
        }
        
        .search-form {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: var(--card-shadow);
            margin-bottom: 2rem;
            transition: var(--transition);
        }
        
        .search-form:hover {
            transform: translateY(-5px);
        }
        
        .search-input-group {
            display: flex;
            gap: 0.5rem;
        }
        
        .search-input {
            flex: 1;
            padding: 0.75rem 1rem;
            border: 2px solid var(--light-gray);
            border-radius: 8px;
            font-size: 1rem;
            font-family: inherit;
            transition: var(--transition);
        }
        
        .search-input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2);
        }
        
        .search-button {
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: var(--transition);
        }
        
        .search-button:hover {
            background-color: var(--primary-dark);
            transform: translateY(-2px);
        }
        
        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }
        
        .results-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--dark);
        }
        
        .results-count {
            background-color: var(--primary);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1.5rem;
        }
        
        .result-card {
            background-color: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: var(--card-shadow);
            transition: var(--transition);
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        
        .result-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
        
        .image-container {
            height: 200px;
            overflow: hidden;
            position: relative;
            background-color: #f3f4f6;
        }
        
        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: var(--transition);
        }
        
        .result-card:hover .image-container img {
            transform: scale(1.05);
        }
        
        .card-content {
            padding: 1rem;
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .card-text {
            font-size: 0.875rem;
            margin-bottom: 0.5rem;
            flex: 1;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
        }
        
        .card-score {
            display: inline-block;
            background-color: #e0f2fe;
            color: #0369a1;
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        
        .no-results, .error-message {
            background-color: white;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            box-shadow: var(--card-shadow);
        }
        
        .no-results-icon, .error-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .no-results-icon {
            color: var(--gray);
        }
        
        .error-icon {
            color: var(--danger);
        }
        
        .no-results-text {
            font-size: 1.25rem;
            color: var(--gray);
            margin-bottom: 1rem;
        }
        
        .error-text {
            font-size: 1.25rem;
            color: var(--danger);
            margin-bottom: 1rem;
        }
        
        .error-details {
            color: var(--gray);
            font-size: 0.875rem;
        }
        
        .dummy-image {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            color: var(--gray);
            font-size: 0.875rem;
        }
        
        /* Loading animation */
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        
        .loading {
            animation: pulse 1.5s infinite;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            .results-grid {
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 1rem;
            }
            
            .search-input-group {
                flex-direction: column;
            }
            
            .search-button {
                width: 100%;
            }
        }
        
        /* Footer */
        footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--light-gray);
            color: var(--gray);
            font-size: 0.875rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Image Search Engine</h1>
            <p class="subtitle">Powered by CLIP Annotations and BM25 Ranking</p>
        </header>
        
        <div class="search-form">
            <form method="GET" action="/search">
                <div class="search-input-group">
                    <input 
                        type="text" 
                        name="query" 
                        class="search-input" 
                        placeholder="Search for images..." 
                        value="{{ query }}"
                        autofocus
                    >
                    <button type="submit" class="search-button">Search</button>
                </div>
            </form>
        </div>
        
        {% if error %}
            <div class="error-message">
                <div class="error-icon">⚠️</div>
                <div class="error-text">Something went wrong</div>
                <div class="error-details">{{ error }}</div>
            </div>
        {% endif %}
        
        {% if results %}
            <div class="results-header">
                <h2 class="results-title">Results for "{{ query }}"</h2>
                <span class="results-count">{{ results|length }} found</span>
            </div>
            
            <div class="results-grid">
                {% for result in results %}
                    <div class="result-card">
                        <div class="image-container">
                            {% if result.path.startswith('dummy_') %}
                                <div class="dummy-image">Dummy Image</div>
                            {% else %}
                                <img 
                                    src="/{{ result.path }}" 
                                    alt="{{ result.text }}" 
                                    onerror="this.onerror=null;this.src='data:image/svg+xml;charset=UTF-8,%3Csvg width=\\'200\\' height=\\'150\\' xmlns=\\'http://www.w3.org/2000/svg\\'%3E%3Crect width=\\'200\\' height=\\'150\\' fill=\\'%23EEEEEE\\'/%3E%3Ctext x=\\'100\\' y=\\'75\\' font-size=\\'12\\' text-anchor=\\'middle\\' alignment-baseline=\\'middle\\' fill=\\'%23999999\\'%3EImage not found%3C/text%3E%3C/svg%3E';"
                                >
                            {% endif %}
                        </div>
                        <div class="card-content">
                            <div class="card-text">{{ result.text }}</div>
                            <div class="card-score">Score: {{ '{:.3f}'.format(result.score) }}</div>
                        </div>
                    </div>
                {% endfor %}
            </div>
        {% elif query %}
            <div class="no-results">
                <div class="no-results-icon">🔍</div>
                <div class="no-results-text">No results found for "{{ query }}"</div>
                <p>Try a different search term or browse our image collection.</p>
            </div>
        {% endif %}
        
        <footer>
            <p id="copyright">© Image Search Engine with CLIP Annotations | Built with Python, Flask, and CLIP</p>
        </footer>
    </div>
    
    <script>
        // Add current year to footer
        document.addEventListener('DOMContentLoaded', function() {
            const year = new Date().getFullYear();
            const copyright = document.getElementById('copyright');
            copyright.textContent = '© ' + year + ' Image Search Engine with CLIP Annotations | Built with Python, Flask, and CLIP';
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

# Modify the Flask route for search to include evaluation metrics
@app.route('/search')
def search():
    query = request.args.get('query', '')
    results = []
    error = None
    metrics = None
    
    if query:
        # Load image data from JSON file
        try:
            if os.path.exists('image_data.json'):
                with open('image_data.json', 'r') as f:
                    image_data = json.load(f)
                
                # Perform search
                results = compute_bm25(query, image_data)[:20]  # Limit to top 20 results
                
                # For demonstration purposes, we'll use a mock ground truth
                # In a real system, you would have actual ground truth data
                mock_ground_truth = {
                    query: [results[i]["path"] for i in range(min(3, len(results)))]  # Assume top 3 are relevant
                }
                
                # Define a search function that uses our BM25 implementation
                def search_function(q):
                    return compute_bm25(q, image_data)
                
                # Calculate evaluation metrics
                metrics = evaluate_search_system([query], mock_ground_truth, search_function)
                
            else:
                error = "No image data found. Please run the crawler first."
        except Exception as e:
            error = f"Error during search: {str(e)}"
            print(error)
    
    # Update the HTML template to include metrics
    html_with_metrics = HTML_TEMPLATE.replace(
        '<span class="results-count">{{ results|length }} found</span>',
        '''<span class="results-count">{{ results|length }} found</span>
            {% if metrics %}
                <div class="metrics-container">
                    <div class="metric">MAP: {{ "{:.4f}".format(metrics.MAP) }}</div>
                    <div class="metric">P@5: {{ "{:.4f}".format(metrics.P_at_5) }}</div>
                    <div class="metric">NDCG@5: {{ "{:.4f}".format(metrics.NDCG_at_5) }}</div>
                </div>
            {% endif %}'''
    )
    
    # Add CSS for metrics
    html_with_metrics = html_with_metrics.replace(
        '.results-count {',
        '''.metrics-container {
            display: flex;
            gap: 1rem;
            margin-top: 0.5rem;
        }
        .metric {
            background-color: var(--secondary);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        .results-count {'''
    )
    
    return render_template_string(
        html_with_metrics, 
        results=results, 
        query=query, 
        error=error,
        metrics={
            'MAP': metrics['MAP'] if metrics else 0,
            'P_at_5': metrics['P@5'] if metrics else 0,
            'NDCG_at_5': metrics['NDCG@5'] if metrics else 0
        } if metrics else None
    )

# Add a new route for evaluation
@app.route('/evaluate')
def evaluate():
    # This would be a more comprehensive evaluation page
    # For now, we'll just redirect to the search page
    return redirect(url_for('home'))

# Serve images
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

# Update the main function to include an evaluation example
def main():
    # Check if image data already exists
    if os.path.exists('image_data.json'):
        try:
            with open('image_data.json', 'r') as f:
                image_data = json.load(f)
                print(f"Loaded {len(image_data)} images from existing data")
        except Exception as e:
            print(f"Error loading existing image data: {e}")
            image_data = []
    else:
        image_data = []
    
    # If no existing data or very few images, crawl for more
    if len(image_data) < 1500:  # Increased threshold to match the target
        # Crawl images (adjust query and limit as needed)
        query = "nature scenery"
        limit = 1500  # Increased to 1500 as requested
        
        image_data = crawl_google_images(query=query, limit=limit)
        print(f"Collected {len(image_data)} images")
    
    # Example search and evaluation
    if image_data:
        test_queries = ["mountain", "beach", "forest"]
        print("\nEvaluating search system with test queries:")
        
        # Mock ground truth for demonstration
        mock_ground_truth = {}
        
        # For each query, perform a search and use top 3 results as mock ground truth
        for query in test_queries:
            results = compute_bm25(query, image_data)
            mock_ground_truth[query] = [results[i]["path"] for i in range(min(3, len(results)))]
            
            print(f"\nQuery: '{query}'")
            for i, result in enumerate(results[:5]):  # Show top 5 results
                print(f"{i+1}. {result['path']} (Score: {result['score']:.3f})")
                print(f"   Text: {result['text']}")
        
        # Define search function for evaluation
        def search_function(query):
            return compute_bm25(query, image_data)
        
        # Evaluate the search system
        metrics = evaluate_search_system(test_queries, mock_ground_truth, search_function)
        
        print("\nSearch System Evaluation:")
        print(f"MAP: {metrics['MAP']:.4f}")
        print(f"P@5: {metrics['P@5']:.4f}")
        print(f"NDCG@5: {metrics['NDCG@5']:.4f}")
    else:
        print("No images to search")
    
    # Start the Flask app
    print("\nStarting web server. Access the search interface at http://127.0.0.1:5000")
    app.run(debug=True)

if __name__ == "__main__":
    main()
