# Tradutor de Libras

## Objetivo
O Tradutor de Libras é um projeto do PET Computação que visa ajudar os membros a aprenderem e aplicarem modelos de previsão, bem como ajudar de incentivo ao aprendizado de Libras.

## Metodologia
O projeto foca na previsão de sinais do alfabeto de Libras que não exigem movimento (ou seja, todas as letras excluindo H, J, K, X e Z). <br><br>
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
Caso queira apenas usar a aplicação diretamente, execute o último passo apenas.
### Coleta de Dados
Para escolher os símbolos a serem coletados, edite o arquivo "symbols" <br><br>
Para coletar as imagens, execute o arquivo "collect_imgs.py":
```console
python3 ./collect_images.py
```
<br>
O terminal irá avisar qual imagem está sendo coletada. Para que a imagem seja salva, aperte a tecla ENTER <br><br>
O código consegue reconhecer quais símbolos faltam e quantas imagens, então você pode parar a coleta no meio e continuar depois.

### Criando dataset parcial
Execute o código "create_dataset.py":
```console
python3 datasets/create_dataset.py
```
<br>
O terminal irá requisitar o nome do arquivo. De preferência, nomeie "partial_identificador", onde identificador é seu nome, por exemplo. Dessa forma, no passo seguinte esses dados serão adicionados ao modelo principal.

### Criando dataset base
Execute o código "merge_data.py":
```console
python3 datasets/merge_data.py
```
<br>
Os dados serão salvos no arquivo "datasets/base_dataset.pickle".

### Treinando o modelo
Há vários modelos no projeto, pois eles estão sendo testados para escolher o melhor.
Execute o código "forest_minimum_classifier.py":
```console
python3 models/forest_minimum_classifier.py
```

### Criando a aplicação
Execute o código "application.py":
```console
python3 ./application.py
```

## Algumas dicas
Dependendo da máquina, os comando acimas podem usar "python" ou "python3".<br><br>
Nos arquivos que usam camera, pode sernecessário mudar o argumento do cv2.VideoCapture. Para a maioria das máquinas, usar 0 como argumento deve funcionar, mas dependendo da situação pode ser necessário usar outros valores (por exemplo, se tiver mais de uma câmera ligada ao sistema). Nesses casos, 1 ou 2 como argumento devem funcionar. <br><br>
O padrão de coleta de imagens é 10 por classe. Isso pode ser alterado no "collect_images.py", na variável DATASET_SIZE. Os símbolos a serem capturados estão no arquivo "symbols" (eles devem ser símbolos de um caracter obrigatoriamente").

