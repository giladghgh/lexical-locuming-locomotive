# Thesaurus Engine

#### This is my attempt at a (nonsensical) synonym replacement program — or a *lexical locuming locomotive*, as it likes to call itself.

<img src="poster.png" width="250" height="250"/>

---

### Table of Contents

- [Overview](#overview)
- [Briefing](#how-to-use)
- [License](#license)

---

## Overview

A WordNet synonym engine, which seemingly takes any text input and gives it an 1850's vernacular.

---

## Briefing

The main program is `ThesaurusEngine.py`. My best efforts to package this into an executable were in vain. I've tried countless methods but `lemminflect` seems to disagree with all of them. 

Note that this program will **not** fix your spelling, punctuation, and/or grammar. In fact, these will only be made worse if entered wrong.

#### Dependencies

You will need to run the following commands as part of the installation (these commands are present but commented in the `ipynb` file):

- `pip install nltk`
- `pip install numpy`
- `pip install lemminflect`

#### Interface

The user interface will prompt you for a direct input, or for the command to read input from `input.txt`. Using the text pre-loaded into `input.txt`, this is an example of one possible console output:

```shell
Enter grammatically correct text below so I can synonymise it, or enter "file" (no quotes) to read from input.txt:
file

ORIGINAL:
'Better to remain silent and be thought a fool than to speak and remove all doubt' -- Mark Twain.


FINAL:
'Better to persist silent and be celebrated a chump than to talk and polish off all doubtfulness' -- Mark Twain.



Press enter to exit.
```
---

## License

Everything you see is wholly [unlincensed](LICENSE).

---

[Back Up Top](#Thesaurus-Engine)
