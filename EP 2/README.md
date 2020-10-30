# Intruções para rodar o EP2

O repositório foi organizado com a seguinte estrututra:

```
7971751_ep2/
├── src  <--------- Pasta com o código do EP
│   └── ep2.py <--------- Toda lógica do EP está dentro dessa pasta
├── data  <--------- Pasta com input de dados
│   └── B2W-Reviews01.csv <--------- Dataset com reviews da B2W baixado do github
    └── cbow_s50.txt <--------- Embeddings do NILC baixados do site e extraidos aqui
├── nUSP7971751_ep2.pdf                      <--------- Relatório final do EP
```

**Atenção**: o código, para fins de CLI, usou a biblioteca [click](https://github.com/pallets/click). Garanta que essa biblioteca foi instalada antes de executar o código!

Para executar o EP, rode o comando na pasta raiz do repositório:
```
python scr/ep2.py
```

Ao rodar, a linha de comando aparecerá com as opções do experimento que você quer definir (usar a rede bidirecional e a taxa de dropout). Caso prefira, você pode passar o comando direto, como por exemplo:

```
python src/ep2.py --usar_bidirecional True --taxa_de_dropout 0.25
```


