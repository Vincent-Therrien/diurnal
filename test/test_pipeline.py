"""
    System tests for the data processing pipeline to ensure that diurnal
    modules can interact correctly for non-NN-related tasks.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: June 2023
    License: MIT
"""

import pytest

from diurnal import database, structure, train, evaluate
import diurnal.models.baseline as baseline
from utils.fileio import TMP_PATH, STRUCTURE_PATH


DIM = 512


@pytest.mark.parametrize(
    "model, f_range",
    [
        (baseline.Random(), [0.2, 0.5]),
        (baseline.Uniform([1, 0, 0]), [0.2, 0.5]),
        (baseline.Uniform([0, 1, 0]), [0.2, 0.5]),
        (baseline.Uniform([0, 0, 1]), [0.2, 0.5]),
        (baseline.Uniform([0, 0, 0]), [0.0, 0.0])
    ]
)
def test_pipeline_dryrun(tmp_rna_structure_files, model, f_range):
    """Test a complete diurnal pipeline sans the neural network part.

    The test case:
    - Reads `.ct` files and convert the RNA data into vectors.
    - Simulates secondary structure prediction (NN part).
    - Cleans up the results.
    - Evaluates the results.

    Since models are baselines, they will, on average, correctly predict
    one pairing out of three - except if the model is designed to make
    incorrect predictions.
    """
    # Preprocessing
    database.format(STRUCTURE_PATH, TMP_PATH, DIM)
    data = train.load_data(TMP_PATH, False)
    # Simulated training
    model.train(data["structures"]["primary"], data["structures"]["secondary"])
    prediction = model.predict(data["structures"]["primary"][0])
    # Clean up
    bases, true, pred = train.clean_vectors(
        data["structures"]["primary"][0],
        data["structures"]["secondary"][0],
        prediction
    )
    true = structure.Secondary.to_bracket(true)
    pred = structure.Secondary.to_bracket(pred)
    # Evaluation
    f1 = evaluate.Vector.get_f1(true, pred)
    assert f_range[0] <= f1 <= f_range[1], f"Abnormal F1-score: {f1}."
