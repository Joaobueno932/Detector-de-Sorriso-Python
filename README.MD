# Detector de Sorriso – OpenCV e CVZone

## Visão Geral

Este projeto usa **OpenCV** e **CVZone** para capturar vídeo em tempo real pela webcam e detectar sorrisos.
Quando um sorriso é identificado, o sistema tira uma foto e a salva no diretório atual com um nome baseado no timestamp.

## Funcionalidades

* **Captura de vídeo** pela webcam
* **Detecção de sorrisos** usando funcionalidades do **CVZone**
* **Captura automática de foto** ao detectar um sorriso
* **Salvamento de imagem** com nome único baseado no horário

## Pré-requisitos

* Python 3.12 ou superior
* Bibliotecas necessárias:

  ```bash
  pip install opencv-python cvzone
  ```

## Como Executar

1. Clone ou baixe o repositório.
2. Execute o script principal:

   ```bash
   python nome_do_arquivo.py
   ```
3. Sorria para a câmera! Quando um sorriso for detectado, uma foto será salva automaticamente.

## Notas

* A detecção pode ser afetada por iluminação ruim ou baixa resolução da câmera.
* As imagens são salvas no diretório de execução com nomes no formato de timestamp.
* É possível ajustar parâmetros de detecção no código para maior sensibilidade ou precisão.

## Habilidades demonstradas

* Captura e manipulação de vídeo em tempo real com **OpenCV**
* Uso de **CVZone** para detecção facial e de expressões
* Automação de captura e salvamento de imagens
* Integração de visão computacional com lógica condicional em Python
