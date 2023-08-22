.. retrieval_qa_benchmark documentation master file, created by
   sphinx-quickstart on Fri Aug 11 08:08:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Retrieval QA Benchmark's documentation!
==================================================

Retreival QA Benchmark (**RQABench** in short) is an open-sourced, end-to-end test workbench for *Retrieval Augmented Generation* (RAG) systems. 
We intend to build an open benchmark for all developers and researchers to reproduce and design new RAG systems.

The overall data flow will look like this:

.. graphviz::

  digraph test {
    rankdir=LR;
    D [shape=polygon,sides=4, label="Dataset"];
    T [shape=polygon,sides=4, label="TransformGraph"];
    L [shape=polygon,sides=4, label="Language Model"];
    D -> T [label="QARecord"];
    T -> L [label="QARecord"];
  }

There are 3 major modules in a :class:`retrieval_qa_benchmark.evaluators.base.BaseEvaluator`.

1. :class:`retrieval_qa_benchmark.schema.BaseDataset`
2. :class:`retrieval_qa_benchmark.schema.TransformGraph`
3. :class:`retrieval_qa_benchmark.schema.BaseLLM`

All data flows over modules are :class:`retrieval_qa_benchmark.schema.QARecord`. So the data schema is constrained instead of the modules.
The dataset outputs formatted ``QARecord`` to the ``TransformGraph``. Graph can be defined using our YAML configuration.
Here is where you can design your retrieval system. You can modify the ``context`` field in ``QARecord`` objects with in nodes in ``TransformGraph``.
Then ``LLM`` accepts QARecord and format ``QARecord`` using template defined in YAML. 
Finally, ``LLM`` throw a :class:`retrieval_qa_benchmark.schema.datatypes.QAPrediction` to the ``Evaluator``.


Here are some major feature of this benchmark:

- **Flexibility**: We maximize the flexibility when design your retrieval system, as long as you accept ``QARecord`` as input and ``QARecord`` as output.
- **Reproducibility**: We gather all settings in the evaluation process into a single YAML configuration. It helps you to track and reproduce experiements.
- **Traceability**: We collect more than the accuracy and scores. We also focus on running times and the tokens used in the whole RAG system.

Table of Content
================

.. toctree::
  :maxdepth: 2

  Home <self>
  getting-started
  write-your-own
  api




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
