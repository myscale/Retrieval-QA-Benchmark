.. _write-your-own:

Write your own pipeline
=======================

Add a dataset
-------------


1. Inherit from :class:`retrieval_qa_benchmark.schema.BaseDataset`
2. Implement ``build`` function that parse your data to list of :class:`retrieval_qa_benchmark.schema.QARecord`.
3. Register your dataset with :class:`retrieval_qa_benchmark.utils.registry.REGISTRY.register_dataset` like this:

.. code-block:: python

    @retrieval_qa_benchmark.utils.registry.REGISTRY.register_dataset("type name you want")
    class YourDataset(retrieval_qa_benchmark.schema.BaseDataset):
        pass

4. Create a PR on `github <https://github.com/myscale/Retrieval-QA-Benchmark>`_.


Add a transform
---------------

1. Inherit from :class:`retrieval_qa_benchmark.schema.BaseTransform`.
2. Implement any of 
    - :attr:`retrieval_qa_benchmark.schema.BaseTransform.transform_question`
    - :attr:`retrieval_qa_benchmark.schema.BaseTransform.transform_choices`
    - :attr:`retrieval_qa_benchmark.schema.BaseTransform.transform_context`
    - and other fields you would like to change in ``QARecord``
3. Register your transform with :class:`retrieval_qa_benchmark.utils.registry.REGISTRY.register_transform` like this:

.. code-block:: python

    @retrieval_qa_benchmark.utils.registry.REGISTRY.register_transform("type name you want")
    class YourTransform(retrieval_qa_benchmark.schema.BaseTransform):
        pass

4. Create a PR on `github <https://github.com/myscale/Retrieval-QA-Benchmark>`_


Add a LLM
---------

1. Inherit from :class:`retrieval_qa_benchmark.schema.BaseLLM`
2. Implement all of below
    - :attr:`retrieval_qa_benchmark.schema.BaseLLM.build`
    - :attr:`retrieval_qa_benchmark.schema.BaseLLM.generate`
3. Register your language model with :class:`retrieval_qa_benchmark.utils.registry.REGISTRY.register_model` like this:

.. code-block:: python

    @retrieval_qa_benchmark.utils.registry.REGISTRY.register_model("type name you want")
    class YourLLM(retrieval_qa_benchmark.schema.BaseLLM):
        pass

4. Create a PR on `github <https://github.com/myscale/Retrieval-QA-Benchmark>`_



Add a evaluator
---------------

1. Inherit from :class:`retrieval_qa_benchmark.schema.BaseEvaluator`
2. Change the ``matcher`` function of :class:`retrieval_qa_benchmark.schema.BaseEvaluator` like this
3. Register your dataset with :class:`retrieval_qa_benchmark.utils.registry.REGISTRY.register_evaluator` like this:

.. code-block:: python

    @retrieval_qa_benchmark.utils.registry.REGISTRY.register_evaluator("type name you want")
    class YourLLM(retrieval_qa_benchmark.schema.BaseLLM):
        pass

4. Create a PR on `github <https://github.com/myscale/Retrieval-QA-Benchmark>`_