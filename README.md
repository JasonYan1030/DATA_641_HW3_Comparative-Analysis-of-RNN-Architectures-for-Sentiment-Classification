## Setup the environment

First download all the file in order and put into a folder. Then, at that folder, create a virtual enviroment. For example:

```bash
python -m venv hw_3_env
```

Then, activate the virtual environment. Here, since my desktop is Window system so activate by this:

```bash
.\hw_3_env\Scripts\Activate
```

Once activated, your prompt should have the (hw_3_env) in the beginning such as:

```bash
(hw_3_env) PS E:\学校内容\DATA 641\DATA 641 HW3>
```

After you activate the virtual environment, upgrade pip and install all requirement libraries. It takes a few minutes to install all of them:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Setup GPU acceleration

Here, I highly suggest that you setup a GPU acceleration. My models are constructed using pytorch and it takes around 2 hours to run once with GPU acceletation. The estimated time of running with CPU only based on the task would be 10-12 hours to run once. I'm not sure why it is so time consuming but given that we have 162 different combination of parameters, it is expected to spend a long time. 

Here, in your activated virtual environment, run the following code to check that if your installed pytorch support GPU acceleration:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

It is expected to show something like:

```bash
CUDA available: True
Device name: (Your GPU name)
```

If it shown something like this:

```bash
CUDA available: False
Device name: CPU only
```

This means that your installed pytorch does not support the GPU acceleration. Use the following code to reinstall the pytorch:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

Then check again whether this time the pytorch is compatible with your GPU using following code:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Here, I have to notify you that it still takes a long time to run even with GPU acceleration (mine is 2 hours).

## Run the script.

Here, I integrate the functions of "train.py" and "utils.py" to other files so these 2 files are empty. The "preprocess.py" file contains the function that do the data preprocessing (train-test split, tokenization, etc.). The "models.py" file contains the 3 models (RNN, LSTM, and Bidirectional LSTM) structures and their training process in it. The "evaluate.py" file contains everything include the code that run the evaluation.

It is a little bit messy to see individual file, so I organize everything in a "main.py" file. If you need to run the code, just run that "main.py" would be enough. On the main folder terminal, activate the virtual environment first. Then run the code:

```bash
python main.py
```

Then it will run the code. Here, if you plan to run the code, please delete the "evaluation_summary" table in the result/ folder and all the plots in the result/plots/ folder before.
