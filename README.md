# RQABench: Retrieval QA Benchmark

[![Documentation Status](https://readthedocs.org/projects/retrieval-qa-benchmark/badge/?version=latest)](https://retrieval-qa-benchmark.readthedocs.io/en/latest/?badge=latest)

Retreival QA Benchmark (RQABench in short) is an open-sourced, end-to-end test workbench for Retrieval Augmented Generation (RAG) systems. We intend to build an open benchmark for all developers and researchers to reproduce and design new RAG systems. We also want to create a platform for everyone to share their lego blocks to help others to build up their own retrieval + LLM system.

Here are some major feature of this benchmark:

- **Flexibility**: We maximize the flexibility when design your retrieval system, as long as your transform accept `QARecord` as input and `QARecord` as output.
- **Reproducibility**: We gather all settings in the evaluation process into a single YAML configuration. It helps you to track and reproduce experiements.
- **Traceability**: We collect more than the accuracy and scores. We also focus on running times on any function you want to watch and the tokens used in the whole RAG system.

## Getting started

### Clone and install

```bash
# Clone to your local machine
git clone https://github.com/myscale/Retrieval-QA-Benchmark
# install it as editable package
cd Retrieval-QA-Benchmark && python3 -m pip3 install -e .
```

### Run it

```python
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

# Choose a configuration to evaluatoe
config = load(open("config/mmlu.yaml"))
evaluator = EvaluatorFactory.from_config(config).build()

# evaluator will return accuracy in float and list of `QAPrediction`
acc, result = evaluator()

# you can set out_file to generate a JSONL file or write it as your own.
with open("some-file-name-to-store-result.jsonl", "w") as f:
    f.write("\n".join([r.model_dump_json() for r in result]))
```

## Replicate our FAISS / MyScale Benchmark

1. RAG with FAISS

- Download the index for wikipedia (around 26G). (index is still uploading)
- Download dataset from huggingface with our code (around 140G). It will automatically download the dataset for the first time.
- Set the index path to the download index. 


2. RAG with MyScale

- Download the data for wikipedia in parquet. (data is still uploading)
- Insert the data and create vector index


## Result with Simple RAG pipeline

### with MyScale
<table>
 <thead>
 <tr>
 <th colspan=2>Setup</th>
 <th colspan=5>Dataset</th>
 <th rowspan=2>Average</th>
 </tr>
 <tr>
 <th>LLM</th>
 <th>Contexts</th>
 <th>mmlu-astronomy</th>
 <th>mmlu-prehistory</th>
 <th>mmlu-global-facts</th>
 <th>mmlu-college-medicine</th>
 <th>mmlu-clinical-knowledge</th>
 </tr>
 </thead>
 <tbody>
 <tr>
 <td rowspan=5>gpt-3.5-turbo</td>
 <td>❌</td>
 <td>71.71%</td>
 <td>70.37%</td>
 <td>38.00%</td>
 <td>67.63%</td>
 <td>74.72%</td>
 <td>68.05%</td>
 </tr>
 <tr>
 <td>✅<br>(Top-1)</td>
 <td>75.66%<br>(+3.95%)</td>
 <td>78.40%<br>(+8.03%)</td>
 <td>46.00%<br>(+8.00%)</td>
 <td>67.05%<br>(-0.58%)</td>
 <td>73.21%<br>(-1.51%)</td>
 <td>71.50%<br>(+3.45%)</td>
 </tr>
 <tr>
 <td>✅<br>(Top-3)</td>
 <td>76.97%<br>(+5.26%)</td>
 <td>81.79%<br>(+11.42%)</td>
 <td>48.00%<br>(+10.00%)</td>
 <td>65.90%<br>(-1.73%)</td>
 <td>73.96%<br>(-0.76%)</td>
 <td>72.98%<br>(+4.93%)</td>
 </tr>
 <tr>
 <td>✅<br>(Top-5)</td>
 <td>78.29%<br>(+6.58%)</td>
 <td>79.63%<br>(+9.26%)</td>
 <td>42.00%<br>(+4.00%)</td>
 <td>68.21%<br>(+0.58%)</td>
 <td>74.34%<br>(-0.38%)</td>
 <td>72.39%<br>(+4.34%)</td>
 </tr>
 <tr>
 <td>✅<br>(Top-10)</td>
 <td>78.29%<br>(+6.58%)</td>
 <td>79.32%<br>(+8.95%)</td>
 <td>44.00%<br>(+6.00%)</td>
 <td>71.10%<br>(+3.47%)</td>
 <td>75.47%<br>(+0.75%)</td>
 <td>73.27%<br>(+5.22%)</td>
 </tr>
 <tr>
 <td rowspan=5>llama2-13b-chat-q6_0</td>
 <td>❌</td>
 <td>53.29%</td>
 <td>57.41%</td>
 <td>33.00%</td>
 <td>44.51%</td>
 <td>50.19%</td>
 <td>50.30%</td>
 </tr>
 <tr>
 <td>✅<br>(Top-1)</td>
 <td>58.55%<br>(+5.26%)</td>
 <td>61.73%<br>(+4.32%)</td>
 <td>45.00%<br>(+12.00%)</td>
 <td>46.24%<br>(+1.73%)</td>
 <td>54.72%<br>(+4.53%)</td>
 <td>55.13%<br>(+4.83%)</td>
 </tr>
 <tr>
 <td>✅<br>(Top-3)</td>
 <td>63.16%<br>(+9.87%)</td>
 <td>63.27%<br>(+5.86%)</td>
 <td>49.00%<br>(+16.00%)</td>
 <td>46.82%<br>(+2.31%)</td>
 <td>55.85%<br>(+5.66%)</td>
 <td>57.10%<br>(+6.80%)</td>
 </tr>
 <tr>
 <td>✅<br>(Top-5)</td>
 <td>63.82%<br>(+10.53%)</td>
 <td>65.43%<br>(+8.02%)</td>
 <td>51.00%<br>(+18.00%)</td>
 <td>51.45%<br>(+6.94%)</td>
 <td>57.74%<br>(+7.55%)</td>
 <td>59.37%<br>(+9.07%)</td>
 </tr>
 <tr>
 <td>✅<br>(Top-10)</td>
 <td>65.13%<br>(+11.84%)</td>
 <td>66.67%<br>(+9.26%)</td>
 <td>46.00%<br>(+13.00%)</td>
 <td>49.71%<br>(+5.20%)</td>
 <td>57.36%<br>(+7.17%)</td>
 <td>59.07%<br>(+8.77%)</td>
 </tr>
 </tbody>
 <tfoot>
 <tr>
 <td colspan=8>
 <i>* The benchmark uses MyScale MSTG as vector index</i><br>
 <i>* This benchmark can be reproduced with our github repository <a href="https://github.com/myscale/Retrieval-QA-Benchmark/">retrieval-qa-benchmark</a></i>
 </td>
 </tr>
 </tfoot>
</table>

------------------

### with FAISS
<table>
 <thead>
 <tr>
 <th colspan=2>Setup</th>
 <th colspan=5>Dataset</th>
 <th rowspan=2>Average</th>
 </tr>
 <tr>
 <th>LLM</th>
 <th>Contexts</th>
 <th>mmlu-astronomy</th>
 <th>mmlu-prehistory</th>
 <th>mmlu-global-facts</th>
 <th>mmlu-college-medicine</th>
 <th>mmlu-clinical-knowledge</th>
 </tr>
 </thead>
 <tbody>
 <tr>
 <td rowspan=5>gpt-3.5-turbo</td>
 <td>❌</td>
 <td>71.71%</td>
 <td>70.37%</td>
 <td>38.00%</td>
 <td>67.63%</td>
 <td>74.72%</td>
 <td>68.05%</td>
 </tr>
 <tr>
 <td>✅<br>(Top-1)</td>
 <td>75.00%<br>(+3.29%)</td>
 <td>77.16%<br>(+6.79%)</td>
 <td>44.00%<br>(+6.00%)</td>
 <td>66.47%<br>(-1.16%)</td>
 <td>73.58%<br>(-1.14%)</td>
 <td>70.81%<br>(+2.76%)</td>
 </tr>
 <tr>
 <td>✅<br>(Top-3)</td>
 <td>75.66%<br>(+3.95%)</td>
 <td>80.25%<br>(+9.88%)</td>
 <td>44.00%<br>(+6.00%)</td>
 <td>65.90%<br>(-1.73%)</td>
 <td>73.21%<br>(-1.51%)</td>
 <td>71.70%<br>(+3.65%)</td>
 </tr>
 <tr>
 <td>✅<br>(Top-5)</td>
 <td>78.29%<br>(+6.58%)</td>
 <td>79.32%<br>(+8.95%)</td>
 <td>46.00%<br>(+8.00%)</td>
 <td>65.90%<br>(-1.73%)</td>
 <td>73.58%<br>(-1.14%)</td>
 <td>72.09%<br>(+4.04%)</td>
 </tr>
 <tr>
 <td>✅<br>(Top-10)</td>
 <td>78.29%<br>(+6.58%)</td>
 <td>80.86%<br>(+10.49%)</td>
 <td>49.00%<br>(+11.00%)</td>
 <td>69.94%<br>(+2.31%)</td>
 <td>75.85%<br>(+1.13%)</td>
 <td>74.16%<br>(+6.11%)</td>
 </tr>
 <tr>
 <td rowspan=5>llama2-13b-chat-q6_0</td>
 <td>❌</td>
 <td>53.29%</td>
 <td>57.41%</td>
 <td>33.00%</td>
 <td>44.51%</td>
 <td>50.19%</td>
 <td>50.30%</td>
 </tr>
 <tr>
 <td>✅<br>(Top-1)</td>
 <td>57.89%<br>(+4.60%)</td>
 <td>61.42%<br>(+4.01%)</td>
 <td>48.00%<br>(+15.00%)</td>
 <td>45.66%<br>(+1.15%)</td>
 <td>55.09%<br>(+4.90%)</td>
 <td>55.22%<br>(+4.92%)</td>
 </tr>
 <tr>
 <td>✅<br>(Top-3)</td>
 <td>59.21%<br>(+5.92%)</td>
 <td>65.74%<br>(+8.33%)</td>
 <td>50.00%<br>(+17.00%)</td>
 <td>50.29%<br>(+5.78%)</td>
 <td>56.98%<br>(+6.79%)</td>
 <td>58.28%<br>(+7.98%)</td>
 </tr>
 <tr>
 <td>✅<br>(Top-5)</td>
 <td>65.79%<br>(+12.50%)</td>
 <td>64.51%<br>(+7.10%)</td>
 <td>48.00%<br>(+15.00%)</td>
 <td>50.29%<br>(+5.78%)</td>
 <td>58.11%<br>(+7.92%)</td>
 <td>58.97%<br>(+8.67%)</td>
 </tr>
 <tr>
 <td>✅<br>(Top-10)</td>
 <td>65.13%<br>(+11.84%)</td>
 <td>66.05%<br>(+8.64%)</td>
 <td>48.00%<br>(+15.00%)</td>
 <td>47.40%<br>(+2.89%)</td>
 <td>56.23%<br>(+6.04%)</td>
 <td>58.38%<br>(+8.08%)</td>
 </tr>
 </tbody>
 <tfoot>
 <tr>
 <td colspan=8>
 <i>* The benchmark uses FAISS IVFSQ (nprobes=128) as vector index</i><br>
 <i>* This benchmark can be reproduced with our github repository <a href="https://github.com/myscale/Retrieval-QA-Benchmark/">retrieval-qa-benchmark</a></i>
 </td>
 </tr>
 </tfoot>
</table>