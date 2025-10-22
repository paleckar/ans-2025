# Aplikace neuronových sítí 2025


## Přednášky

| #  | datum      | přednáška                                          | obsah                                         |
|----|------------|----------------------------------------------------|-----------------------------------------------|
| 0. | 17.9.2025  | [Úvod](slides/ans-00-intro.pdf)                    | podmínky předmětu, úvod                       |
| 1. | 17.9.2025  | [Lineární klasifikace](slides/ans-01-linear.pdf)   | lineární klasifikace, softmax, svm            |
| 2. | 24.9.2025  | [Optimalizace](slides/ans-02-optimization.pdf)     | optimalizace, gradient, SGD, momentum, Adam   |
| 3. | 8.10.2025  | [Zpětná propagace](slides/ans-03-backprop.pdf)     | zpětná propagace, reverzní aut. derivování    |
| 4. | 15.10.2025 | [Trénování sítí v praxi I](slides/ans-04-mlp.pdf)  | vícevrstvý perceptron, aktivace, inicializace |
| 5. | 22.10.2025 | [Trénování sítí v praxi II](slides/ans-04-mlp.pdf) | regularizace, preprocessing, trénování        |


## Úlohy

- Bodování úloh a termíny odevzdání jsou specifikovány v konfiguračním souboru [assignments.yaml](assignments.yaml).
- Odevzdání úlohy po termínu bude penalizováno odečtením 50 % bodů.
- Další podmínky a známkování jsou v přednášce [Úvod](slides/ans-00-intro.pdf).

| #  | úloha                                                           | body    | deadline   |
|----|-----------------------------------------------------------------|---------|------------|
| 1. | [Lineární klasifikace](assignments/linear_classification.ipynb) | 20 (32) | 21.10.2025 |
| 2. | [Dvouvrstvý perceptron](assignments/two_layer_perceptron.ipynb) | 14 (17) | 28.10.2025 |
| 3. | [Autograd](assignments/autograd.ipynb)                          | 33 (33) | 18.11.2025 |
| 4. | [Perceptron autogradem](assignments/perceptron_autograd.ipynb)  | 29 (43) | 2.12.2025  |


## Nastavení

### 1. Balíčkovací manažer
- Předmět využívá [uv](https://docs.astral.sh/uv/) jako manažer balíčků.
- Lze jej nainstalovat příkazy  
  ```
  # Windows:
  winget install --id=astral-sh.uv  -e
  # Ubuntu:
  sudo snap install astral-uv
  # macOS brew:
  brew install uv
  # Other Linux distros and macOS:
  wget -qO- https://astral.sh/uv/install.sh | sh
  ```

### 2. Virtuální prostředí a instalace potřebných balíků
- Potřebné balíčky jsou uvedeny v souborech `pyproject.toml` a `uv.lock`.
- Pokud jsou uvedené soubory v aktuálním adresáři, všechny balíčky lze nainstalovat pomocí příkazu  
  ```
  uv sync
  ```
- Příkaz vytvoří sub-adresář `./.venv` a v něm nové virtuální prostředí se vším potřebným.

### 3. Obsah předmětu
- Úlohy jsou ve formě [Jupyter notebooks](https://jupyter.org/).
- Doporučený nástroj na práci s kódy je [Visual Studio Code](https://code.visualstudio.com/) s rozšířeními [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) a [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).
