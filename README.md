## Setup the environment

First download all the file in order and put into a folder. Then, at that folder, create a virtual enviroment. For example:

```bash
python -m venv hw_3_env
```

Then, activate the virtual environment. Here, since my desktop is Window system so activate by this:

```bash
.\hw_3_env\Scripts\Activate
```

If you are in macOS then run:

```bash
source hw_3_env/bin/activate
```

Once activated, your prompt should have the (hw_3_env) in the beginning such as:

```bash
(hw_3_env) PS E:\学校内容\DATA 641\DATA 641 HW3>
```

After you activate the virtual environment, upgrade pip and install all requirement libraries:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

When you finish running, use the following code to deactivate the virtual environment and quit:

```bash
deactivate
```
