# Intruções para rodar o EP3

O repositório foi organizado com a seguinte estrututra:

```
7971751_ep3/
├── ep3.py  <- Pasta com o fluxo principal de treinamento
├── code_utils  <- biblioteca interna com a principal lógica 
│    └── preprocessing <- módulo que isola a lógica de pré processamento textual
│     │     └── data.py <- código com a lógica de computação de métricas
│     └── metrics <- módulo que isola a lógica de computação de métricas
│            └── comput_metrics.py <- código com a lógica de computação de métricas
│────└── models <- módulo que isola a lógica de cada um dos modelos 
│          └── bilstm <- módulo com apenas a lógica do modelo lstm bidirecional 
│                 └── inference.py <- código que contém a lógica necessária para inferência  
│                 └── training.py <- código que contém a lógica necessária para treinamento 
├── data  <- Pasta com input de dados
│    └── bw2-10k.csv <- Dataset com reviews da B2W baixado do github
│    └── cbow_s50.txt <- Embeddings do NILC baixados do site e extraidos aqui
├── notebooks  <- Pasta notebooks de estudo
│    └── estudo-bilsm.ipynb <- notebook com estudos que levaram as tomadas de decisão do Encoder Decoder Bidirecional
├── requirements.txt     <- arquivos `requirements.txt` com as versões de bibliotecas utilizadas
├── nUSP7971751_ep3.pdf  <- Relatório final do EP
```

**Atenção**: o código, para fins de CLI, usou a biblioteca [click](https://github.com/pallets/click). Garanta que essa biblioteca foi instalada antes de executar o código!

***Este EP foi rodado no servidor do IME `brucutuvi`. Para acessar, execute os comandos
```
ssh  <seu.usuario>@shell.ime.usp.br -N -f -L 8888:localhost:8888
ssh  <seu.usuario>@shell.ime.usp.br
ssh  <seu.usuario>@brucutuvi.ime.usp.br -N -f -L 8888:localhost:8888
ssh  <seu.usuario>@brucutuvi.ime.usp.br
```
***Uma vez dentro da máquina, as dependências do projeto foram instaladas com o poetry
```
HOME=/var/fasttemp/<some-folder>
pip3 install --user --upgrade poetry
python3 -m poetry config virtualenvs.create true
python3 -m poetry install
python3 -m poetry run ipython kernel install --user --name=ep3
python3 -m jupyter notebook --no-browser
```
*A lista de comandos acima só irá funcionar se você estiver dentro da máquina do IME-USP*


Para executar o EP, rode o comando na pasta raiz do repositório:
```
python ep3.py
```

Ao rodar, a linha de comando aparecerá com as opções do experimento que você quer definir (bilstm ou bert). Caso prefira, você pode passar o comando direto, como por exemplo:

```
python ep3.py --model_definition bilstm
```


