"""
    Diurnal model package.

    The package contains models designed to predict RNA secondary
    structure from their primary structure. All models are classes that
    implement the following interface:

    - constructor
      - Create and initialize the model.
    - `train(primary_structure, secondary_structure) -> None`
      - Train a model with primary and secondary structures.
    - `predict(primary_structure) -> secondary_structure`
      - Predict a secondary structure from a primary structure.
"""

__all__ = ["baseline"]
