# Aplikace neuronových sítí 2025

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
- Všechny je lze nainstalovat pomocí příkazu  
  ```
  uv sync
  ```
- Pro príkaz je nutné, aby soubory `pyproject.toml` a `uv.lock` byly v aktuálním adresáři, ze kterého je příkaz spouštěn.

### 3. Obsah předmětu
- Úlohy jsou ve formě [Jupyter notebooks](https://jupyter.org/).
- Doporučený nástroj na práci s kódy je [Visual Studio Code](https://code.visualstudio.com/) s rozšířeními [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) a [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).

## Přednášky

| #   | datum      | přednáška                                           | obsah                                         |
|-----|------------|-----------------------------------------------------|-----------------------------------------------|
| 0.  | 17.9.2025  | [Úvod](slides/ans-00-intro.pdf)                     | podmínky předmětu, úvod                       |
| 1.  | 17.9.2025  | [Lineární klasifikace](slides/ans-01-linear.pdf)    | lineární klasifikace, softmax, svm            |
| 2.  | 24.9.2024  | [Optimalizace](slides/ans-02-optimization.pdf)      | optimalizace, gradient, SGD, momentum, Adam   |
