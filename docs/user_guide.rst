.. _user-guide:

User Guide
==========


The Data Processing Pipeline
----------------------------


Model Training and Evaluation
-----------------------------


Save and Load Models
--------------------

.. code-block:: python

   # Write the model into the `saved_model` directory.
   model.save("saved_model")

   # Erase the model to clear space.
   del model

   # Load the model stored in the `saved_model` directory.
   loaded_model = diurnal.models.NN(
       cnn.Pairings_1,
       SIZE,
       3,
       torch.optim.Adam,
       torch.nn.MSELoss,
       {"eps": 1e-4},
       None,
       verbosity=1)
   loaded_model.load("saved_model")

   # Test the model. Should obtain the same result as before.
   f = loaded_model.test(test_set)
   print(f"Average F1-score of the saved model: {sum(f)/len(f):.4}")

   # Visualize an example of a prediction.
   print(f"\nSample prediction from the test set (`{test_set['names'][0]}`).")
   p = test_set["primary_structures"][0]
   s = test_set["secondary_structures"][0]
   visualize.prediction(p, s, loaded_model.predict(p))
