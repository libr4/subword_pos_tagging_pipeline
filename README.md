# Projeto de Marcação POS com Segmentação de Subpalavras

Este projeto implementa um pipeline de Processamento de Linguagem Natural (PLN) para marcação de Part-of-Speech (POS) utilizando uma rede neural BiLSTM-CRF. O diferencial do projeto reside na experimentação com diferentes estratégias de segmentação de subpalavras (Baseline, BPE, Morfessor, FlatCat) para analisar seu impacto na performance do marcador POS.

## Configuração do Ambiente

Para configurar e rodar o projeto, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd your_project_name
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

## Preparação dos Dados

Certifique-se de que seus corpora no formato CoNLL-U estejam na pasta `data/corpora/<LANGUAGE>/`. Por exemplo, para o idioma `nheengatu`, você teria:

```
data/corpora/nheengatu/
├── dev_corpus.conllu
└── train_corpus.conllu
```


## Executando o Pipeline

O pipeline completo é orquestrado pelo script `run_pipeline.py`. Você pode controlar o idioma e o tipo de segmentação usando variáveis de ambiente.

### Variáveis de Ambiente:

* **`LANGUAGE`**: Define o idioma do corpus a ser processado (ex: `nheengatu`, `bororo5k`). O script `extract_tokens_tags.py` tem um tratamento especial para `bororo5k`.
* **`SEGMENTER`**: Define a estratégia de segmentação de palavras a ser utilizada:
    * `baseline`: Não aplica segmentação de subpalavras (usa os tokens originais).
    * `bpe`: Utiliza Byte-Pair Encoding para segmentação.
    * `morfessor`: Utiliza o Morfessor para segmentação morfológica.
    * `flatcat`: Utiliza o FlatCat para segmentação morfológica.

### Execução:

Após configurar o arquivo `.env`, execute o pipeline a partir da raiz do projeto:

```bash
python run_pipeline.py
```
O script `run_pipeline.py` irá:

    Extrair tokens e tags dos arquivos CoNLL-U (para LANGUAGE != "bororo5k").

    Realizar a segmentação de subpalavras, se SEGMENTER não for "baseline".

    Propagar as tags POS originais para os tokens segmentados.

    Treinar o modelo BiLSTM-CRF com os dados preparados.