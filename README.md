# Instruções

Para rodar os trabalhos, primeiro instale o gerenciador de dependências:

```bash
pip install poetry
```

Em seguida, instale as dependências:

```bash
poetry install
```

Ative o ambiente virtual com:

```bash
poetry shell
```

# Projeto

Vídeo: <https://youtu.be/aSjzR3vxKYA>

Uma vez que o ambiente virtual está ativo, inicie um interpretador:

```bash
cd proj
python
```

No interpretador, importe o arquivo do algoritmo:

```python
import src.X
```

Em que X é um dos algoritmos: `adaboost`, `random_forest` ou `xgboost`. Certifique-se que os dados lhe foram fornecidos e estão na pasta `data`.

Os arquivos `util.py` e `preprocessing.py` contém funções auxiliares. O arquivo `playground.py` é usado apenas para exploração.

Para gerar alguns gráficos, pode ser necessário instalar no seu sistema o pacote `graphviz`.

# TPs

Para executar os TPs, execute os programas como módulos em Python:

```bash
python -m tpX.src
```

Substitua `X` pelo número do TP em questão.

**Atenção**: o *backend* do `matplotlib` foi configurado para rodar apenas com Linux.
