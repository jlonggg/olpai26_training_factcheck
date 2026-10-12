"""Regenerate solution.ipynb from pipeline.py (cells are separated by `# %%` markers)."""
import json, re
src = open("pipeline.py", encoding="utf8").read(); cells = []
for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
    body = chunk.split("\n", 1)[1].strip("\n")
    if chunk.startswith(" [markdown]"):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body})
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("solution.ipynb", "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(len(cells), "cells")
