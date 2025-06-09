from flask import Flask, request, render_template, redirect, url_for
from app import index_documents, query_documents, generate_response
import os
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "./news_articles"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Extract topic list from filenames
def extract_topics(directory="./news_articles"):
    topics = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            clean = file.split("-", 3)[-1].replace(".txt", "").replace("-", " ").capitalize()
            topics.append(clean)
    return sorted(topics)

@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    source_chunks = []
    selected_topic = ""
    question = ""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    topics = extract_topics()
    files = [f for f in os.listdir(app.config["UPLOAD_FOLDER"]) if f.endswith(".txt")]

    if request.method == "POST":
        selected_topic = request.form.get("topic")
        question = request.form.get("question") or selected_topic
        if question:
            chunks = query_documents(question)
            source_chunks = chunks
            answer = generate_response(question, chunks)

    return render_template(
        "index.html",
        answer=answer,
        topics=topics,
        selected=selected_topic,
        sources=source_chunks,
        question=question,
        timestamp=timestamp,
        files=files
    )

@app.route("/reindex", methods=["POST"])
def reindex():
    index_documents()
    return redirect(url_for("home"))

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    if file and file.filename.endswith(".txt"):
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], file.filename))
        index_documents()
        return redirect(url_for("home"))
    return "Only .txt files are allowed!", 400

@app.route("/delete", methods=["POST"])
def delete_file():
    fname = request.form.get("file_to_delete")
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    if os.path.exists(fpath):
        os.remove(fpath)
        index_documents()
    return redirect(url_for("home"))

@app.route("/files")
def list_files():
    folder = app.config["UPLOAD_FOLDER"]
    files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    file_previews = []
    for fname in files:
        with open(os.path.join(folder, fname), encoding="utf-8") as f:
            content = f.read(300)
            file_previews.append((fname, content))
    return render_template("files.html", file_previews=file_previews)

if __name__ == "__main__":
    app.run(debug=True)
