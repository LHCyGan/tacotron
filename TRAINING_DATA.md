# Training Data

This repo supports the following speech datasets:

* [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) (Public Domain)
* [Blizzard 2012](http://www.cstr.ed.ac.uk/projects/blizzard/2012/phase_one) (Creative Commons Attribution Share-Alike)
* [标贝科技中文标准女声音库](http://www.data-baker.com/open_source.html) (Data Baker Chinese female voice, non-commercial use only)

You can use any other dataset if you write a preprocessor for it.

### Writing a Preprocessor

Each training example consists of:
  1. The text that was spoken
  2. A mel-scale spectrogram of the audio
  3. A linear-scale spectrogram of the audio

The preprocessor is responsible for generating these. See [ljspeech.py](datasets/ljspeech.py) for a
commented example.

For each training example, a preprocessor should:

  1. Load the audio file:
     ```python
     wav = audio.load_wav(wav_path)
     ```

  2. Compute linear-scale and mel-scale spectrograms (float32 numpy arrays):
     ```python
     spectrogram = audio.spectrogram(wav).astype(np.float32)
     mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
     ```

  3. Save the spectrograms to disk:
     ```python
     np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
     np.save(os.path.join(out_dir, mel_spectrogram_filename), mel_spectrogram.T,  allow_pickle=False)
     ```
     Note that the transpose of the matrix returned by `audio.spectrogram` is saved so that it's
     in time-major format.

  4. Generate a tuple `(spectrogram_filename, mel_spectrogram_filename, n_frames, text)` to
     write to train.txt. n_frames is just the length of the time axis of the spectrogram.


After you've written your preprocessor, you can add it to [preprocess.py](preprocess.py) by
following the example of the other preprocessors in that file.


### Non-English Data

If your training data is in a language other than English, you will probably want to change the
text cleaners by setting the `cleaners` hyperparameter.

  * If your text is in a Latin script or can be transliterated to ASCII using the
    [Unidecode](https://pypi.python.org/pypi/Unidecode) library, you can use the transliteration
    cleaners by setting the hyperparameter `cleaners=transliteration_cleaners`.

  * If you don't want to transliterate, you can define a custom character set.
    This allows you to train directly on the character set used in your data.

    To do so, edit [symbols.py](text/symbols.py) and change the `_characters` variable to be a
    string containing the UTF-8 characters in your data. Then set the hyperparameter `cleaners=basic_cleaners`.

  * If you're not sure which option to use, you can evaluate the transliteration cleaners like this:

    ```python
    from text import cleaners
    cleaners.transliteration_cleaners('Здравствуйте')   # Replace with the text you want to try
    ```

### Data Baker Data

Download and uncompress Data Baker waves to `databaker` folder. The structure will be like `databaker/Waves`. To be convenient, use the Pinyin labels from `ProsodyLabeling.rar` for easier transliteration. Set the hyperparameter `cleaners=basic_cleaners` with digits 0-9 in `_characters` (`symbols.py` is already updated with the change). The pre-preprocessed `databaker/metadata.csv` will be like:

```text
000001|ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1
000002|jia2 yu3 cun1 yan2 bie2 zai4 yong1 bao4 wo3
000003|bao2 ma3 pei4 gua4 bo3 luo2 an1 diao1 chan2 yuan4 zhen3 dong3 weng1 ta4
000004|deng4 xiao3 ping2 yu3 sa4 qie4 er3 hui4 wu4
000005|lao2 hu3 you4 zai3 yu2 chong3 wu4 quan3 wan2 shua3
```
