# preprocessing-of-speech
VAD + resampling



# Usage
1. pip install -r requirements.txt
2. Move `main.py` to where the `.wav` files are located.
3. **Run** `main.py`
4. The folder will be created and the files will be downloaded to that folder.

# Arguments

```
python3 main.py [--opt OPT] [--path PATH]
```
```
Preprocessing of Speech

optional arguments:
 --opt OPT    preprecessing mode : vad=1, resampling=2, vad+resampling=3 (default: 3)
 --path PATH  wav file location (default: current directory)
```
