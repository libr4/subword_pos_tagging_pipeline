# POS Tagging com Segmentação de Subpalavras

Este projeto implementa um pipeline de PLN para POS Tagging, utilizando BiLSTM-CRF e avalia seu desempenho após treinamento com métodos de segmentação em subpalavras.

## Configuração do Ambiente

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/libr4/subword_pos_tagging_pipeline.git
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

## Preparação dos Dados

Certifique-se de que seus corpora no formato CoNLL-U estejam na pasta `data/corpora/<LANGUAGE>/` (O projeto já inclui alguns corpora). Por exemplo, para o idioma `nheengatu`, você teria:

```
data/corpora/nheengatu/
├── dev_corpus.conllu
└── train_corpus.conllu
```


## Executando o Pipeline

O pipeline completo é executado pelo script `run_pipeline.py`. Você pode controlar o idioma e o tipo de segmentação usando variáveis de ambiente no arquivo .env.

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
