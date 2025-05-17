from flask import Flask, render_template, request, jsonify
from trt_interface import estimate_trt
from simplifier_interface import simplify_word

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    input_text = request.form.get("input_text","")
    results = []
    if input_text:
        results = estimate_trt(input_text)
    min_trt = min([result['trt'] for result in results]) if results else 0
    max_trt = max([result['trt'] for result in results]) if results else 1000
    return render_template("index.html", input_text=input_text, results=results, min_trt=float(min_trt), max_trt=float(max_trt))

@app.route("/simplify", methods=["POST"])
def simplify():
    data = request.json
    sentence = data["sentence"]
    idx = data["idx"]
    replacement, replacement_trt = simplify_word(idx, sentence)
    return jsonify({"replacement": replacement, "new_trt": float(replacement_trt)})

if __name__ == "__main__":
    app.run(debug=True)
