from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import json
import pathlib

app = FastAPI()

# Enable CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# More robust way to define DATA_DIR relative to this script file
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
# If your data directory is truly one level up from where main.py is, you might use:
# DATA_DIR = SCRIPT_DIR.parent / "data"
# Or, if data is in the same dir as main.py:
# DATA_DIR = SCRIPT_DIR / "data" # this is what you have effectively with "./data" if main.py is in the root

@app.get("/products")
async def get_products(store: str = Query(...), class_name: str = Query(...)):
    # print(f"--- Request for store: {store}, class: {class_name} ---")
    # print(f"FastAPI script directory: {SCRIPT_DIR}")
    # print(f"Resolved DATA_DIR: {DATA_DIR.resolve()}") # Shows absolute path

    filename_str = f"{store.lower()}_products.json"
    filepath = DATA_DIR / filename_str # Use pathlib for path joining

    # print(f"Attempting to access filepath: {filepath.resolve()}") # Shows absolute path

    if not filepath.exists(): # Use .exists() for pathlib.Path
        print(f"!!! File NOT FOUND: {filepath.resolve()}")
        return {"error": f"Store '{store}' not found at path."} 

    # print(f"--- File FOUND: {filepath.resolve()} ---")
    with open(filepath, "r", encoding="utf-8") as f:
        store_data = json.load(f)

    # Extract class name 
    class_key = class_name.split(" (")[0].strip()

    products = store_data.get(class_key, [])
    if not products:
        return {"error": f"No products found for class '{class_key}' in store '{store}'."}
    return {"products": products}

