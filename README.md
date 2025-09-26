# LLM Scenarios
_What influence does framing a prompt have on performance and safety?_

## Motivation
This repository contains code for running LLM performance benchmarks (using [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness)) and a safety benchmark ([SORRY-Bench](https://sorry-bench.github.io/)) for answering the question of the influence of prompt _framing_, i.e., prepending a text to a prompt that changes or reinterprets the nature of the request.
For instance, we could frame our request as being urgent, e.g.,
```
"I need help with this quickly, time is of the essence!"
```
Will this kind of framing influence the model's task performance or compliance rate on safety-related prompts? 

## Setup
Clone this repository:
```commandline
git clone git@github.com:kmerkelbach/llm-request-tone.git
```
Also, you need to clone my fork of SORRY-Bench which has been modified to work with OpenRouter (https://openrouter.ai/):
```commandline
git clone git@github.com:kmerkelbach/sorry-bench.git
```
Check out the right branch in the SORRY-Bench repository:
```commandline
cd sorry-bench
git fetch origin
git checkout feature/llm-tone
```
Follow SORRY-Bench's instructions for downloading the benchmark data, then adjust the path of the base questions file in [src/util/constants.py]() (note that this is within the `llm-request-tone` repository).

Finally, install [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) using its instructions. No modifications to it are necessary.

### OpenRouter
[OpenRouter](https://openrouter.ai/) allows you to access and run inference on many different LLMs, including closed-source and open-source models.
If you want to use OpenRouter, create an API key after signing up. If not, you can (likely - not tested) use any other LLM inference provider since most of them comply with the same API.

In your `~/.bashrc` or `~/.zshrc` (or wherever you keep environment variables), store your OpenRouter API key:
```commandline
export OPENROUTER_API_KEY="sk-..."
```

## Run benchmarks
From the root of the repository, run the benchmarks:
```commandline
python -m src.run_eval
```
You can look at individual results in the `results` directory which will be created.

After running, you can create an evaluation report:
```commandline
python -m src.report_results
```
This creates evaluation tables.

_Note that reports generation is currently in an unfinished state._

## Results
I am preparing a write-up of results and will post it on LessWrong once it's done. Watch this space for the link.
