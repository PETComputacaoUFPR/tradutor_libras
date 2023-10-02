# Tradutor de Libras

## Objetivo
O Tradutor de Libras é um projeto do PET Computação que visa ajudar os membros a aprenderem e aplicarem modelos de previsão, bem como ajudar de incentivo ao aprendizado de Libras.

## Metodologia
O projeto foca na previsão de sinais do alfabeto de Libras que não exigem movimento (ou seja, todas as letras excluindo H, J, K, X e Z). <br><br>
O dataset foi coletado pelos próprios membros do PET e voluntários. <br><br>
Esse projeto foi fortemente inspirado nesse repositório: https://github.com/computervisioneng/sign-language-detector-python

## Requerimentos
opencv-python==4.7.0.68 <br>
mediapipe==0.9.0.1 <br>
scikit-learn==1.2.0 <br>

## Como usar
### Coleta de Dados
Para escolher os símbolos a serem coletados, edite o arquivo "symbols" <br><br>
Para coletar as imagens, execute o arquivo "collect_imgs.py" <br>
```console
python collect_images.py
```
<br>
O terminal irá avisar qual imagem está sendo coletada. Para que a imagem seja salva, aperte a tecla ENTER <br><br>
O código consegue reconhecer quais símbolos faltam e quantas imagens, então você pode parar a coleta no meio e continuar depois.

### Criando dataset parcial
Execute o código "create_dataset.py"
```console
python ./datasets/create_dataset.py
```
<br>
O terminal irá requisitar o nome do arquivo. De preferência, nomeie "partial_identificador", onde identificador é seu nome, por exemplo. Dessa forma, no passo seguinte esses dados serão adicionados ao modelo principal.

### Criando dataset base
Execute o código "merge_data.py"
```console
python ./datasets/merge_data.py
```
<br>
Os dados serão salvos no arquivo "datasets/base_dataset.pickle".

### Treinando o modelo
Execute o códgo "train_classifier.py"
```console
python ./models/train_classifier.py
```

### Criando a aplicação
Execute o código "application.py" 
```console
python application.py
```

