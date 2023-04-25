import sys
sys.path.append('..')
from flask import Flask, request
import scrubber as s

app = Flask(__name__)

@app.route("/scrub", methods=["POST"])
def scrub():
    content_type = request.headers.get("Content-Type")
    if content_type == "application/json":
        data = request.json
        return s.scrub_text(data['text'], data['options'])
    else:
        return 'Content-Type not supported!'
