# Tradutor de Libras

## Objetivo
O Tradutor de Libras é um projeto do PET Computação que visa ajudar os membros a aprenderem e aplicarem modelos de previsão, bem como ajudar de incentivo ao aprendizado de Libras.

## Metodologia
O projeto foca na previsão de sinais do alfabeto de Libras que não exigem movimento (ou seja, todas as letras, excluindo H, J, K, X e Z). <br><br>
O dataset foi coletado pelos próprios membros do PET e voluntários. <br><br>
Esse projeto foi fortemente inspirado nesse repositório: https://github.com/computervisioneng/sign-language-detector-python

## Instalando bibliotecas necessárias
```console
sudo apt install python3-pip
```

```console
pip3 install -U scikit-learn
```

```console
sudo apt-get install python3-opencv
```

```console
pip install mediapipe
```

## Como usar
Caso queira apenas usar a aplicação diretamente, siga esses passos:

### Criando dataset base
Os dados do projeto estão separados por pessoa, e o primeiro passo é juntá-los em apenas um arquivo.
Execute o código "merge_data.py":
```console
python3 datasets/merge_data.py
```
<br>
Os dados serão salvos no arquivo "datasets/base_dataset.pickle".

### Separando dados de treino e teste
Para ser possível treinar o modelo de posteriormente testá-lo, é necessário fazer o split dos dados
Execute o código "merge_data.py":
```console
python3 models/create_split.py
```
<br>
Os dados serão salvos na pasta "models/TrainTestData".

### Treinando o modelo
O melhor modelo dentro os analisados deve ser criado para ser utilizado na aplicação.
Execute o código "create_best_model.py":
```console
python3 models/create_best_model.py
```
<br>
O modelo está salvo no arquivo "models/best_model.sav".

### Criando a aplicação
Cria a aplicação que irá acessar a câmera e analisar os sinais feitos e tentar prevê-los.
Execute o código "application.py":
```console
python3 ./application.py
```

## Algumas dicas
Dependendo da máquina, os comando acimas podem usar "python" ou "python3".

<br><br>

Nos arquivos que usam câmera, pode ser necessário mudar o argumento do cv2.VideoCapture. Para a maioria das máquinas, usar 0 como argumento deve funcionar, mas dependendo da situação pode ser necessário usar outros valores (por exemplo, se tiver mais de uma câmera ligada ao sistema). Nesses casos, 1 ou 2 como argumento devem funcionar.

## Dificuldades com Python -> Vale ressaltar a existência do material do curso básico da linguagem dentro do repositório do PET e gravado nos arquivos do projeto a aula com o material.
