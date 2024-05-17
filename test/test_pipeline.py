"""
    System tests for the data processing pipeline to ensure that diurnal
    modules can interact correctly for non-NN-related tasks.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2023
    - License: MIT
"""

import pytest

from diurnal import database, structure, train, evaluate
import diurnal.models.baseline as baseline
from utils.fileio import TMP_PATH, STRUCTURE_PATH


DIM = 512


@pytest.mark.parametrize(
    "model, f_range",
    [
        (baseline.Random(), [0.1, 0.5]),
        (baseline.Uniform([1, 0, 0]), [0.1, 0.5]),
        (baseline.Uniform([0, 1, 0]), [0.1, 0.5]),
        (baseline.Uniform([0, 0, 1]), [0.1, 0.5]),
        (baseline.Majority(), [0.1, 0.5])
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

    Args:
        model: Baseline model to test.
        f_range (List[float]): Range of acceptable f1-score for a model.
    """
    # Preprocessing
    database.format_basic(STRUCTURE_PATH, TMP_PATH, DIM)
    data = train.load_data(TMP_PATH, False)
    train_set, test_set, validate_set = train.split_data(data, [1/3, 1/3, 1/3])
    # Simulated training
    model.train(train_set)
    prediction = model.predict(test_set["input"][0])
    # Clean up
    bases, true, pred = train.clean_vectors(
        test_set["input"][0][0],
        test_set["output"][0],
        prediction
    )
    true = structure.Secondary.to_bracket(true)
    pred = structure.Secondary.to_bracket(pred)
    # Evaluation
    f1 = evaluate.Bracket.micro_f1(true, pred)
    assert f_range[0] <= f1 <= f_range[1], f"Abnormal F1-score: {f1}."
