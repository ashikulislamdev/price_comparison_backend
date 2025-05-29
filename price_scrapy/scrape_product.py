import requests
import json

RAINFOREST_API_KEY = "06ED87269C5A47B292440F35DD02DE4A"  # Your actual API key


def search_amazon_products(search_term: str, num_results: int = 5) -> list:
    """
    Args:
        search_term (str): Product name to search.
        num_results (int): Number of results to fetch.

    Returns: List of product dictionaries with title, price, rating, etc.
    """
    params = {
        'api_key': RAINFOREST_API_KEY,
        'type': 'search',
        'amazon_domain': 'amazon.com',
        'search_term': search_term,
        'output': 'json',
        'num_results': num_results,
    }

    print(f"Calling Rainforest API for search term: '{search_term}' on amazon.com")
    print(f"Request URL: https://api.rainforestapi.com/request?{requests.compat.urlencode(params)}")

    try:
        response = requests.get('https://api.rainforestapi.com/request', params=params)
        print(f"HTTP Status Code: {response.status_code}")
        response.raise_for_status()
        data = response.json()

        if not data.get('search_results'):
            print("No search results returned.")
            return []

        found_products = []

        for product in data['search_results']:
            raw_price = None
            currency_symbol = None

            # Primary: product['price']
            price_obj = product.get("price")
            if price_obj and isinstance(price_obj, dict):
                raw_price = price_obj.get("raw")
                currency_symbol = price_obj.get("symbol")

            # Fallback: product['buybox_winner']['price']
            if not raw_price:
                buybox_winner = product.get("buybox_winner")
                if buybox_winner and isinstance(buybox_winner, dict):
                    buybox_price = buybox_winner.get("price")
                    if buybox_price and isinstance(buybox_price, dict):
                        raw_price = buybox_price.get("raw")
                        currency_symbol = buybox_price.get("symbol")

            # Last fallback: Check 'prices' array
            if not raw_price:
                prices_list = product.get("prices", [])
                if isinstance(prices_list, list) and len(prices_list) > 0:
                    raw_price = prices_list[0].get("raw")
                    currency_symbol = prices_list[0].get("symbol")

            found_products.append({
                "store": "Amazon",
                "product_name": product.get("title"),
                "product_url": product.get("link"),
                "price": raw_price,
                "currency_symbol": currency_symbol,
                "image_url": product.get("image"),
                "rating": product.get("rating"),
                "reviews_count": product.get("ratings_total"),
                "delivery": product.get("delivery"),
                "is_prime": product.get("is_prime"),
                "has_variants": product.get("has_variants"),
            })

        return found_products

    except requests.exceptions.RequestException as e:
        print(f"Error making Rainforest API request: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return []


def get_product_prices_from_stores(product_name: str) -> list:
    print(f"Searching Amazon for: '{product_name}'...")
    amazon_listings = search_amazon_products(product_name, num_results=5)
    if amazon_listings:
        print(f"Found {len(amazon_listings)} Amazon listings for '{product_name}'.")
        return amazon_listings
    else:
        print(f"No Amazon listings found for '{product_name}'.")
        return []


if __name__ == "__main__":
    detected_product_name = "Sony PlayStation 5 Console"
    print(f"Starting price comparison for: '{detected_product_name}'\n")

    found_products = get_product_prices_from_stores(detected_product_name)

    if found_products:
        print("\n--- Consolidated Product Listings ---")

        # Sort by numeric price (handle missing price gracefully)
        found_products_sorted = sorted(
            found_products,
            key=lambda x: float(
                (str(x.get('price') or '')
                 .replace('$', '')
                 .replace(',', '')
                 .strip()) or '0'
            )
        )

        for i, product in enumerate(found_products_sorted[:5]):
            print(f"\nListing {i+1}:")
            print(f"  Store: {product.get('store', 'N/A')}")
            print(f"  Product: {product.get('product_name', 'N/A')}")
            print(f"  Price: {product.get('price', 'N/A')}")
            print(f"  Rating: {product.get('rating', 'N/A')} ({product.get('reviews_count', 'N/A')} reviews)")
            print(f"  URL: {product.get('product_url', 'N/A')}")
            print(f"  Image: {product.get('image_url', 'N/A')}")
    else:
        print("No product listings found.")
