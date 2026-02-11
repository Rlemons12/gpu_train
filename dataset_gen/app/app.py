from flask import Flask

# Force env normalization immediately
from dataset_gen.dataset_qanda_generator.configuration import env_adapter  # noqa

# --------------------------------------------------
# Blueprint imports (top-level, explicit)
# --------------------------------------------------
from dataset_gen.app.routes.index import bp as index_bp
from dataset_gen.app.routes.documents import bp as documents_bp
from dataset_gen.app.routes.document_metadata import bp as document_metadata_bp
from dataset_gen.app.routes.pipeline import bp as pipeline_bp
from dataset_gen.app.routes.dataset_generator import dataset_generator_bp


def create_app():
    app = Flask(__name__)

    # --------------------------------------------------
    # Register Blueprints
    # --------------------------------------------------
    app.register_blueprint(index_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(document_metadata_bp)
    app.register_blueprint(pipeline_bp)
    app.register_blueprint(dataset_generator_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5001)
