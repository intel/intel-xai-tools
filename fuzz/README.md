## Fuzz Testing in Intel Explainable AI Tools
Fuzz testing is an automated software testing technique that involves providing invalid, unexpected, or random data as inputs to a computer program. The program is then monitored for exceptions such as crashes, failing built-in code assertions, or potential memory leaks. This README details the use of Google's Atheris, a coverage-guided Python fuzzing engine, to conduct fuzz testing in our project. 
Inside this fuzz folder holds all fuzz testing programs.

### Requirements
* Python: Version 3.8 or newer
* Atheris: Google's fuzzing engine for Python
* Coverage: Code coverage measurement for Python

## Setup
To prepare your environment for fuzz testing with Atheris, follow these steps:

# Install Dependencies
```
pip install -r requirements.txt
```
## Running Fuzz Tests
```
python3 -m coverage run fuzz_test.py -atheris_runs=0 ../model_card_gen/intel_ai_safety/model_card_gen/docs/examples/json/
```
# Interpreting Results
When running fuzz tests with Atheris it is important to understand the output to idenfity potential issues effectively. 

### Crashes and Exceptions
Atheris reports when the fuzzed input causes the program to crash or raise unhandled exceptions. These input are crucial for identifying vulnerabilities. 

~~~
ERROR: atheris detected an error in fuzz_test.py.
CRASH: Test input caused an unhandled IndexError exception.
~~~

In this example, the fuzzer has discovered an input that causes an IndexError in fuzz_test.py. This indicates that the code does not properly handle cases where list or array access is out of bounds. The developer should examine the stack trace provided by Atheris, identify the problematic code, and implement proper bounds checking or error handling to resolve the issue.

### Coverage Metrics
Atheris provides information about code coverage, which helps in understanding which parts of your code were exercised by the fuzz tests. Low coverage might indicate that additional fuzzing targets or more diverse inputs are needed. 

To generate the coverage report, run the following command inside the fuzz folder:

`python3 -m coverage report`

The output will be:

| Name                                                        | Stmts | Miss | Cover |
|-------------------------------------------------------------|-------|------|-------|
| fuzz_test.py                                                | 25    | 6    | 76%   |
| intel_ai_safety/model_card_gen/__init__.py                  | 0     | 0    | 100%  |
| intel_ai_safety/model_card_gen/analyze/__init__.py          | 4     | 0    | 100%  |
| intel_ai_safety/model_card_gen/analyze/analyzer.py          | 26    | 15   | 42%   |
| ...                                                         | ...   | ...  | ...   |
| intel_ai_safety/model_card_gen/validation.py                | 26    | 12   | 54%   |
|-------------------------------------------------------------|-------|------|-------|
| TOTAL                                                       | 835   | 416  | 50%   |

### Leak Detection

If you're using a Python extension module that interfaces with C code, you might encounter memory leaks due to improper memory management in the C layer. Here's an example of how a memory leak might be reported:

```plaintext
Leak detected: an object of type 'MyCExtension.Object' with a size of 1024 bytes was not freed.
Call stack of the allocation:
  File "my_c_extension.py", line 58, in create_object
    obj = MyCExtension.Object()
```

Developers should review the create_object function to ensure proper memory management.

### Reproducing Issues 
For every failure, detected, Atheris outputs a test case that can reproduce the issues. These test cases can help debug and fix the vulnerabilities in your code. When Atheris encounters an issue such as an unhandled exception, it can provide a serialized input that caused the problem. This allows you to reproduce the issue for debugging purposes. Here's an example of the output you might see:

```plaintext
EXCEPTION: Test input caused a KeyError in your Python code.
Reproducing input written to: exception-abcdef1234567890.pickle
To reproduce, run: python3 -m atheris reproduce exception-abcdef1234567890.pickle
```

In this example, the fuzzer has discovered an input that causes a KeyError in the Python code. The input has been saved to a file named exception-abcdef1234567890.pickle. To reproduce the issue, the developer can run the provided command, which will execute the fuzzer with the exact same input that caused the exception, allowing for consistent reproduction and easier debugging.