# webapp/app.py

from flask import Flask, request, render_template
from trt_interface import estimate_trt

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    input_text = ""
    if request.method == "POST":
        input_text = request.form.get("input_text", "")
        results = estimate_trt(input_text)
    return render_template("index.html", input_text=input_text, results=results)

if __name__ == "__main__":
    app.run(debug=True)
