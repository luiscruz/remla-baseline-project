import json
import os

key = os.environ["API_KEY_SECRET"]
key_json = json.loads(key, strict=False)
with open(os.environ["KEY_FILE"], "w+") as f:
    json.dump(key_json, f)
