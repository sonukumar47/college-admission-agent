"""
Flask Web Application
College Admission RAG Agent — Web Interface
"""

import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

from src.agent import create_agent

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")
CORS(app)

# Initialize the RAG agent once at startup
agent = None


def get_agent():
    global agent
    if agent is None:
        logger.info("Initializing College Admission Agent...")
        agent = create_agent()
        logger.info("Agent ready.")
    return agent


@app.route("/")
def index():
    """Serve the main chat UI."""
    college_name = os.getenv("COLLEGE_NAME", "Greenfield University")
    return render_template("index.html", college_name=college_name)


@app.route("/api/ask", methods=["POST"])
def ask():
    """Handle a student question via JSON API."""
    data = request.get_json(silent=True)
    if not data or not data.get("question"):
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"].strip()
    if len(question) < 3:
        return jsonify({"error": "Question is too short"}), 400
    if len(question) > 1000:
        return jsonify({"error": "Question is too long (max 1000 characters)"}), 400

    try:
        rag = get_agent()
        response = rag.query(question)

        sources = [
            {
                "id": s.id,
                "title": s.title,
                "category": s.category,
            }
            for s in response.sources
        ]

        return jsonify({
            "answer": response.answer,
            "sources": sources,
            "query": response.query,
        })

    except ValueError as e:
        # Configuration errors (missing API key etc.)
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        return jsonify({"error": "An error occurred. Please try again."}), 500


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "College Admission RAG Agent"})


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    logger.info(f"Starting College Admission Agent at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
