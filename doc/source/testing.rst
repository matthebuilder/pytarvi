Testing
========

This section describes how to run and write tests for the PyTorch Architecture Visualizer.

Running Tests
-------------
The project uses ``pytest`` for unit and integration testing. To run all tests, execute the following command from the project root:

.. code-block:: bash

   PYTHONPATH=. pytest tests/

Running Specific Tests
----------------------
To run a specific test file:

.. code-block:: bash

   PYTHONPATH=. pytest tests/test_extractor.py

To run a specific test function:

.. code-block:: bash

   PYTHONPATH=. pytest tests/test_extractor.py::test_linear_flow

Writing Tests
--------------
Tests are located in the ``tests/`` directory. When adding new functionality:
1. Create a corresponding test case in the relevant test file.
2. Use fixtures or temporary files for model code analysis tests.
3. Ensure that all tests pass before committing changes.
