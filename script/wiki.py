import requests
import json
import itertools
import urllib.parse
import pandas as pd
import time
import multiprocessing
from functools import partial
import logging
from collections import deque
import concurrent.futures

from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


session = requests.Session()
base_url = "https://my.wikipedia.org/w/api.php"

def get_page_by_url(url):
    """
    Get a Wikipedia page by its URL

    Args:
        url (str): Full Wikipedia page URL

    Returns:
        dict: Page content information or None if not found
    """
    # Extract the page title from the URL

    # Parse the URL
    parsed_url = urllib.parse.urlparse(url)

    # Get the path component and remove the '/wiki/' prefix
    path = parsed_url.path

    if path.startswith('/wiki/'):
        # Extract the title and decode it
        page_title = urllib.parse.unquote(path.replace("/wiki/", ""))
    else:
        raise f"Link has issue: {url}"

    # Now use the API to get the page by title
    params = {
        'action': 'query',
        'titles': page_title,
        'prop': 'extracts|categories|info',
        'explaintext': True,
        'exsectionformat': 'plain',
        'inprop': 'url',
        'format': 'json',
        'redirects': True
    }

    response = requests.get(url=base_url, params=params).json()

    if 'query' in response and 'pages' in response['query']:
        # The API returns a dict with page IDs as keys
        # We don't know the page ID in advance, so we get the first (and only) page
        pages = response['query']['pages']

        # Check if the page exists (page ID -1 means it doesn't exist)
        if '-1' in pages:
            raise f"ID has issue: {url}"

        page_id = next(iter(pages))
        page_data = pages[page_id]

        return {
            'title': page_data.get('title', ''),
            'content': page_data.get('extract', ''),
            'categories': [cat['title'] for cat in page_data.get('categories', [])],
            'url': page_data.get('fullurl', ''),
            'last_modified': page_data.get('touched', ''),
        }
    else:
        raise f"Response has issue: {url}"


def get_page_redirect_title(title):
    # Now use the API to get the page by title
    params = {
        'action': 'query',
        'titles': title,
        'prop': 'extracts|categories|info',
        'explaintext': True,
        'exsectionformat': 'plain',
        'inprop': 'url',
        'format': 'json',
        'redirects': True
    }

    max_retries = 100

    for attempt in range(max_retries):
        try:
            response = requests.get(url=base_url, params=params).json()

            if 'query' in response and 'pages' in response['query']:
                # The API returns a dict with page IDs as keys
                # We don't know the page ID in advance, so we get the first (and only) page
                pages = response['query']['pages']

                page_id = next(iter(pages))
                page_data = pages[page_id]

                return page_data.get('title', '')
        except Exception as e:
            wait_time = (attempt + 1) * 5  # Increasing backoff
            logger.warning(f"Error fetching category {category}. Retrying in {wait_time}s... ({attempt+1}/{max_retries})")
            time.sleep(wait_time)


def get_category_members(category):
    # Add Category: prefix if not present
    if not category.startswith(('Category:', 'ကဏ္ဍ:')):
        category = f"Category:{category}"

    pages = []
    subcategories = []

    params = {
        'action': 'query',
        'list': 'categorymembers',
        'cmtitle': category,
        'cmtype': 'page|subcat',
        'cmlimit': 500,
        'format': 'json'
    }

    continuation = True



    max_retries = 100
    response = None
    for attempt in range(max_retries):
        if not continuation:
            return pages, subcategories

        try:
            while continuation:
                response = session.get(url=base_url, params=params).json()

                if 'error' in response:
                    print(f"Error: {response['error']['info']}")
                    break

                if response is None:
                    print(f"Timeout: {category}")
                    break

                if 'query' not in response:
                    print(f"No results found for {category}")
                    break

                members = response['query']['categorymembers']

                for member in members:
                    ns = member['ns']
                    title = member['title']

                    if ns == 14:  # Namespace 14 is for categories
                        subcategories.append(title)
                    else:
                        pages.append({
                            'title': title,
                            'pageid': member['pageid']
                        })

                if 'continue' in response:
                    params['cmcontinue'] = response['continue']['cmcontinue']
                else:
                    continuation = False

        except (requests.exceptions.RequestException, ValueError) as e:
            wait_time = (attempt + 1) * 5  # Increasing backoff
            logger.warning(f"Error fetching category {category}. Retrying in {wait_time}s... ({attempt+1}/{max_retries})")
            time.sleep(wait_time)


def get_page_content(page_id, session=None):
    # Create a new session if none is provided
    local_session = session or requests.Session()

    params = {
        'action': 'query',
        'prop': 'extracts|categories|info',
        'pageids': page_id,
        'explaintext': True,
        'exsectionformat': 'plain',
        'inprop': 'url',
        'format': 'json'
    }

    max_retries = 100
    for attempt in range(max_retries):
        try:
            response = local_session.get(url=base_url, params=params).json()
            if 'query' in response and 'pages' in response['query']:
                page_data = response['query']['pages'][str(page_id)]
                return {
                    'title': page_data.get('title', ''),
                    'content': page_data.get('extract', ''),
                    'categories': [cat['title'] for cat in page_data.get('categories', [])],
                    'url': page_data.get('fullurl', ''),
                    'last_modified': page_data.get('touched', '')
                }
            else:
                return None
        except (requests.exceptions.RequestException, ValueError) as e:
            wait_time = (attempt + 1) * 5  # Increasing backoff
            logger.warning(f"Error fetching page {page_id}: {e}. Retrying in {wait_time}s... ({attempt+1}/{max_retries})")
            time.sleep(wait_time)

def process_page(page, category, session=None):
    page_data = get_page_content(page['pageid'], session)

    if page_data and page_data['content']:
        page_data['source_category'] = category
        return page_data
    return None

def process_batch(batch, category):
    """Process a batch of pages"""
    results = []
    for page in batch:
        result = process_page(page, category)
        if result:
            results.append(result)
    return results

def get_category_info(category):
    """
    Get details about a category including its normalized name and parent categories
    Returns a tuple of (normalized_name, parent_categories)
    """
    # Add Category: prefix if not present
    if not category.startswith(('Category:', 'ကဏ္ဍ:')):
        category = f"Category:{category}"

    params = {
        'action': 'query',
        'titles': category,
        'prop': 'categories',
        'cllimit': 500,
        'format': 'json'
    }

    try:
        response = session.get(url=base_url, params=params).json()

        # Extract the normalized "to" name
        normalized_name = None
        if 'normalized' in response.get('query', {}):
            for norm in response['query']['normalized']:
                if norm.get('from') == category:
                    normalized_name = norm.get('to')
                    break

        # If no normalization happened, use the original category
        if normalized_name is None:
            normalized_name = category

        # Extract parent categories
        parent_categories = []
        if 'pages' in response.get('query', {}):
            pages = response['query']['pages']

            for page_id, page_info in pages.items():
                if 'categories' in page_info:
                    for cat in page_info['categories']:
                        parent_categories.append(cat['title'])

        return normalized_name, parent_categories

    except Exception as e:
        logger.error(f"Error getting category info: {str(e)}")
        return category, []


def fetch_pages_and_subcategories(category):
    _, categories = get_category_info(category[-1])
    pages, subcategories = get_category_members(category[-1])

    if len(pages) == 0:
        return pd.DataFrame(columns=['title', 'content', 'categories', 'url', 'last_modified', 'source_category']), subcategories + categories

    num_processes = 10
    batch_size = min(num_processes, len(pages))

    # num_cores = multiprocessing.cpu_count()


    logger.info(f"Category: {category[-1]}, Pages: {len(pages)}, Subcategories: {len(subcategories)}, Categories: {len(categories)}")

    all_results = []
    total_batches = (len(pages) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(pages), batch_size), total=total_batches, desc="Processing batches"):
        batch = pages[i:i+batch_size]

        # Create a pool for each batch to ensure fresh connections
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Create a partial function with fixed arguments
            process_page_partial = partial(process_page, category=category, session=session)

            # Process the batch
            results = pool.map(process_page_partial, batch)
            all_results.extend(results)

        # Filter out None results
    all_pages = [result for result in all_results if result is not None]

    # Convert to DataFrame
    if all_pages:
        return pd.DataFrame(all_pages), subcategories + categories
    else:
        return pd.DataFrame(columns=['title', 'content', 'categories', 'url', 'last_modified', 'source_category']), subcategories + categories


if __name__ == "__main__":

    with open("../data/myanmar_wiki_entry_links.json") as file:
        myanmar_wiki_links = json.load(file)

    categories = myanmar_wiki_links['category_links']
    pages = myanmar_wiki_links['page_links']
    categories = [category['text'] for category in categories]

    # Use multiprocessing Pool to process categories in parallel
    with multiprocessing.Pool(processes=10) as pool:
        # Use tqdm to show a progress bar
        categories = list(tqdm(
            pool.imap(get_page_redirect_title, categories),
            total=len(categories),
            desc="Processing categories"
        ))

    categories = set(categories)
    categories = [[category] for category in categories]

    pd.DataFrame(columns=['title', 'content', 'categories', 'url', 'last_modified', 'source_category']).to_csv('../data/myanmar_wikipedia_dataset.csv', index=False)

    for page in pages:
        page_url = page['url']
        page_data = get_page_by_url(page_url)
        page_data['source_category'] = [page['header']]
        df = pd.DataFrame([page_data])
        df.to_csv('../data/myanmar_wikipedia_dataset.csv', mode='a', header=False, index=False)

    # Initialize with your starting categories
    queue = deque(list(categories))
    processed_categories = set()  # To track what we've already processed
    result_categories = list(categories)

    # Create a pool of workers
    while queue:

        category = queue.popleft()  # For breadth-first (use pop() for depth-first)

        # Convert category to tuple for hashability (if it's a list)
        category_key = category[-1]

        # Skip if already processed
        if category_key in processed_categories:
            continue

        processed_categories.add(category_key)

        # Get subcategories
        df, subcategories = fetch_pages_and_subcategories(category)

        # Use multiprocessing Pool to process categories in parallel
        with multiprocessing.Pool(processes=10) as pool:
            # Use tqdm to show a progress bar
            subcategories = list(tqdm(
                pool.imap(get_page_redirect_title, subcategories),
                total=len(subcategories),
                desc="Processing categories"
            ))

        # Create new pairs and filter out existing ones
        for subcat in subcategories:
            new_category = category + [subcat]
            new_category_key = new_category[-1]

            if new_category_key not in processed_categories:
                queue.append(new_category)  # Add to queue for processing
                result_categories.append(new_category)  # Add to results
        df.to_csv('../data/myanmar_wikipedia_dataset.csv', mode='a', header=False, index=False)

    print(f"Processed {len(result_categories)} category paths in total")

    with open("../data/myanmar_wikipedia_categories.json", "w") as file:
        json.dump(result_categories, file, indent=4)
