from flask import Blueprint, render_template

bp = Blueprint(
    "index",
    __name__,
)

@bp.route("/", methods=["GET"])
def index():
    """
    Home page / navigation entry point.
    """
    return render_template("index.html")
