# RethinkMCTS
Repository for [RethinkMCTS: Refining Erroneous Thought in Monte Carlo Tree Search For Code Generation].
## Run:
Download the raw data, change the path in function "get_raw_data_path()" in "utils/util.py" to your raw data path.


First set your own OpenAI API key in run.py.

Then, to test RethinkMCTS, run run.py

	python run.py --rollout 16

The parameter settings are also in run.py. Common modifications include parameters such as `rollout`, `dataset`, `model`, `arch`.
Set the `experiment_idx` before each run. And the experiment result will be saved in "results/{dataset}/Experiment_{experiment_idx}/".

If you want to see the result, run "ExpPro.py", change the dataset and playlist contain the "experiment_idx" you want to see.
