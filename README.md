# Zero-Shot Question Answering

This repository provides a simple code to implement zero-shot learning approach for Question Answering task. For each question `q` with available answer options `a`, `b`, and `c`, it computes each option's score as the negative log-likelihood under the language model conditioned on the question. More formally, score(a) = P<sub>m</sub>(a|q); score(b) = P<sub>m</sub>(b|q); and score(c) = P<sub>m</sub>(c|q). It then returns the option with the highest score as the most probable answer to the question `q`.

## Setup Environment Requirements

- `PyTorch 1.11`
- `numpy 1.22.3`
- `transformers 4.16.2`

## Usage

The Question Answering model (***QAModel***) is defined inside `Zero_Shot_QA_Model.py` file. It loads the tokenizer and the language model in the initializer method. The `get_answer()` method goes over all available answer options for a given question and computes ***log-likelihood*** as the score for each option. It returns the option with the highest score. In addition to this, `Inference.py` file is also made available with a few examples of how this implementation is being used for Question Answering task in zero-shot manner.

## Results

```
- Question: Where is capital of France?
  Available Options: London, Berlin, Paris, Lyon
  Predicted Answer: Paris

- Question: Who is best known for developing the theory of relativity?
  Available Options: Albert Einstein, Isaac Newton, Stephen Hawking, Max Planck
  Predicted Answer: Albert Einstein
  
- Question: Who is CEO of Tesla?
  Available Options: Bill Gates, Elon Musk, Steve Jobs, Tim cook
  Predicted Answer: Elon Musk
```
