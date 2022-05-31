This repositiory includes code used in experiments for the master thesis "Multilingual information retrieval with word embeddings".

## Remarks
Creating an index is specific for the [CLEF dataset](https://catalogue.elra.info/en-us/repository/browse/ELRA-E0008/).

Ranking with neural models happens at the second stage and requires a list of query-document pairs to estimate their relevane and rank them based on the computed score.

For the aligned mBERT model see [[1]](#1).

The trained classification model for reranking based on mBERT and aligned mBERT is available [here](https://drive.google.com/drive/folders/1VZ4DNvlwMSOvZqM-g2KJP3P0dxvuqMiu?usp=sharing).

## References
<a id="1">[1]</a> 
Zhao, Wei et al. “Inducing Language-Agnostic Multilingual Representations.” STARSEM (2021).