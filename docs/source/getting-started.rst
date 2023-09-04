.. _getting_started:

Getting Started
================================================

Currently we only support `install from source`_.


.. _install from source:

Install from source
-------------------

1. Clone the repository from Github


.. code-block:: bash
    
    git clone https://github.com/myscale/Retrieval-QA-Benchmark


2. Install using ``pip``

.. code-block:: bash

    cd Retrieval-QA-Benchmark && python3 -m pip3 install -e .

.. note::
    For users who does not have any GPU, you need a CPU version of pytorch installed by
    
    .. code-block:: bash

        pip install torch --index-url https://download.pytorch.org/whl/cpu --upgrade

.. _YAML configuration:

Editing YAML configuration
--------------------------

Basics on configurations
^^^^^^^^^^^^^^^^^^^^^^^^

Configuration defines an end-to-end experiemnt. It will help you to manage, track and accumulate experiements.
A YAML configuration will contain these sections:

- ``evaluator``: Root of this configuration
    - ``outfile``: File that contains all evaluated ``QAPrediction`` with results.
    - ``dataset``: can be a dataset config or list of datasets. A list of dataset looks like this:
        .. code-block:: yaml
            
            dataset:
                - type: mmlu
                  args:
                    subset: astronomy
                - type: mmlu
                  args:
                    subset: clinical_knowledge
    - ``model``: language model configuration
    - ``transform``: can be a list of transforms or a dictionary of transforms. See how to `construct transform graph`_

All config for modules has ``type``, ``args``, and ``run_args``. ``type`` is the name registered in 
:class:`retrieval_qa_benchmark.utils.register.REGISTRY`. ``args`` is the arguments used for module construction, 
and ``run_args`` is used when calling those modules. You can get a feeling when `write a yaml config from scratch`_.

.. _include pre-defined configuration:

Include pre-defined configuration for modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: 
    You should use :class:`retrieval_qa_benchmark.utils.config.load` 
    to load these yaml file with pre-defined modular configuration.

We provide a several pre-defined module settings under ``config/`` directory.
If you want to use them, you can include those pre-defined modular configuration in your YAML like this:

.. code-block:: yaml

    evaluator: 
        type: mcsa
        outfile: mmlu-llama2-remote-retrieval-base.jsonl
        dataset: !include datasets/mmlu.yaml
        model: !include models/remote-llama2-13b.yaml
        transform:
            nodes:
               - !include transforms/faiss-pre.yaml
               - !include transforms/hybrid-rerank.yaml
  
The ``node`` section under ``transform`` is a sequence of transforms. The execution order will like this:


.. graphviz:: 
    
    digraph t1 {
        rankdir=LR;
        A [label="faiss-pre"];
        B [label="hybrid-rerank"];
        A -> B;
    }


''''''

.. _write a yaml config from scratch:

Write a YAML configuration from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to write your own configuration without pre-defined, you can start from here:

.. code-block:: yaml
    
    evaluator:
        type: mcsa
        out_file: None
        dataset: 
            type: "mmlu"
            args:
                subset: "astronomy"
        transform:
            nodes:
               - type: Faiss
                 args: 
                    num_selected: 100
                    index_path: "path-to-index"
               - type: RRFHybrid
                 args: 
                    num_selected: 5
                    num_filtered: 100
                    rank_dict:
                        mpnet: 30
                        bm25: 40
        model:
            type: "chatgpt35"
            args:
                model_name: "gpt-3.5-turbo"
                api_key: "sk-some-super-secret-key-you-will-never-know"
                system_prompt: ""
            run_args:
                temperature: 0.0
                top_p: 1.0
                max_tokens: 30
                stop: "\n\n"

This is identical to the previous configuration in `include pre-defined configuration`_.

'''''''

.. _construct transform graph:

Construct a transform graph in YAML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning:: 
    For a graph transform, you should specify the ``entry_id`` in your YAML configuration under ``transform`` section.
    For sequential transforms, ``entry_id`` must **NOT** be set.

You can construct a transform graph in your configuration. We take these codes as example:


Suppose we have three transforms like ``dummy1``, ``dummy2`` and ``dummy3``.

.. code-block:: python

    from typing import Any, Dict

    from retrieval_qa_benchmark.schema import BaseTransform, QARecord
    from retrieval_qa_benchmark.utils.factory import TransformGraphFactory
    from retrieval_qa_benchmark.utils.registry import REGISTRY


    @REGISTRY.register_transform("dummy1")
    class Dummy_1(BaseTransform):
        def check_status(self, current: Dict[str, Any]) -> int:
            return int(len(current["question"]) > 100)

        def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
            return "dummy1" + data["question"]


    @REGISTRY.register_transform("dummy2")
    class Dummy_2(BaseTransform):
        def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
            return "dummy2" + data["question"]


    @REGISTRY.register_transform("dummy3")
    class Dummy_3(BaseTransform):
        def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
            return "dummy3" + data["question"]

``dummy2`` and ``dummy3`` is just inserting string to question while ``dummy1`` insert and check the status 
that determines what is the next transform. We can build up a graph using those components:

.. graphviz:: 

    digraph t2 {
        rankdir=LR;
        A [label="dummy1"];
        B [label="dummy2"];
        C [label="dummy3"];
        S [shape=polygon, sides=4, skew=.4, label="start"];
        E [shape=polygon, sides=4, skew=.4, label="end"];
        S -> A;
        A -> B [label="len(current['question']) <= 100"];
        A -> C [label="len(current['question']) > 100"];
        B -> A;
        C -> E;
    }

This can be constructed with yaml file below

.. code-block:: yaml

    evaluator:
        # .. some other configurations
        transform:
            entry_id: t1
            nodes:
                t1:
                    type: dummy1
                    next: 
                      - t2
                      - t3
                t2:
                    type: dummy2
                    next: 
                      - null
                      - t1
                t3:
                    type: dummy3
                    next: 
                      - null
                      - null


''''''

Evaluation examples
-------------------

End-to-end evaluation
^^^^^^^^^^^^^^^^^^^^^

This is a sample code for end-to-end evaluation

.. code-block:: python

    from retrieval_qa_benchmark.models import *
    from retrieval_qa_benchmark.datasets import *
    from retrieval_qa_benchmark.transforms import *
    from retrieval_qa_benchmark.evaluators import *
    from retrieval_qa_benchmark.utils.profiler import PROFILER
    # This is for loading our special yaml configuration with `!include` keyword
    from retrieval_qa_benchmark.utils.config import load
    # This is where you can contruct evaluator from config
    from retrieval_qa_benchmark.utils.factory import EvaluatorFactory

    # This will print all loaded modules. You can also use it as reference to edit your configuration
    print(str(REGISTRY))

    # Choose a configuration to evaluate
    config = load(open("config/mmlu-myscale.yaml"))
    evaluator = EvaluatorFactory.from_config(config).build()
    
    # evaluator will return accuracy in float and list of `QAPrediction`
    acc, result = evaluator()

    # you can set out_file to generate a JSONL file or write it as your own.
    with open("some-file-name-to-store-result.jsonl", "w") as f:
        f.write("\n".join([r.model_dump_json() for r in result]))


``TransformGraph``-only evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a sample code for retrieval system benchmark

.. code-block:: python

    from tqdm import tqdm
    from retrieval_qa_benchmark.models import *
    from retrieval_qa_benchmark.datasets import *
    from retrieval_qa_benchmark.transforms import *
    from retrieval_qa_benchmark.evaluators import *
    from retrieval_qa_benchmark.utils.profiler import PROFILER
    from retrieval_qa_benchmark.utils.config import load
    from retrieval_qa_benchmark.utils.factory import EvaluatorFactory

    config = load(open("config/mmlu-myscale.yaml"))
    evaluator = EvaluatorFactory.from_config(config).build()

    # externally clear the profiler's counter
    PROFILER.clear()

    for r in tqdm(map(evaluator.transform, evaluator.dataset.iterator()), 
                  total=len(evaluator.dataset)):
        # transform every element in dataset, and get `QARecord`
        data.append(r)

    print(str(PROFILER))

    # you can dump all QARecord as JSONL as well
    with open("some-file-name-to-store-result.jsonl", "w") as f:
        f.write("\n".join([r.model_dump_json() for r in result]))


``LLM``-only evaluation
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from retrieval_qa_benchmark.models import *
    from retrieval_qa_benchmark.datasets import *
    from retrieval_qa_benchmark.transforms import *
    from retrieval_qa_benchmark.evaluators import *
    from retrieval_qa_benchmark.utils.profiler import PROFILER
    from retrieval_qa_benchmark.utils.config import load
    from retrieval_qa_benchmark.utils.factory import EvaluatorFactory

    config = load(open("config/mmlu.yaml"))
    evaluator = EvaluatorFactory.from_config(config).build()

    # purge out all transforms
    evaluator.transform.nodes = []
    acc, result = evaluator()