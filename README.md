# Pipeline Oficial CEMIG (Atual)

Este repositório segue oficialmente o fluxo mais recente baseado em:
- snapshot novo (40k)
- dataset v2 (Top-25)
- ciclo 3 com reranker fine-tuned

## Versão oficial

- Encoder oficial: e5_large_ft_v2 em experiments/exp_v2_40k
- Reranker oficial: reranker_ft em experiments/exp_v2_40k_reranker_ft
- Dataset oficial de treino do encoder: datasets/train_pos_v2.jsonl

## Organização de pastas

- Datasets ativos: datasets/
- Datasets legados: removidos
- Experimentos legados: removidos
- Scripts legados: removidos

Datasets atuais esperados em datasets/:
- snapshot_v1.jsonl
- snapshot_v1_enriquecido.jsonl
- snapshot_v1_filtrado.jsonl
- train_pos_v1.jsonl
- train_pos_v2.jsonl
- train_cross_encoder.jsonl

## Ordem oficial dos scripts

1. python3 01_validate_snapshot_v1.py
2. python3 02_enrich_snapshot_v1.py
3. python3 03_filter_snapshot_v1.py
4. python3 04_build_train_pairs_v1.py
5. python3 05_split_dataset_v1.py
6. python3 06_eval_baseline_e5_v1.py
7. python3 07_train_e5_encoder_v1.py
8. python3 08_eval_e5_ft_v1.py
9. python3 09_rerank_bge_m3_v1.py
10. python3 10_filter_top25_v1.py
11. python3 11_split_v2.py
12. python3 12_train_e5_encoder_v2.py
13. python3 13_eval_e5_ft_v2.py
14. python3 14_rerank_bge_m3_v2.py
15. python3 15_results_summary_cycle2.py
16. python3 16_build_cross_encoder_dataset_v2.py
17. python3 17_train_reranker_ft_v2.py
18. python3 18_eval_reranker_ft_v2.py

## Artefatos finais de referência

- experiments/exp_v2_40k/results_summary_cycle2.md
- experiments/exp_v2_40k_reranker_ft/results_summary_cycle3.md

## Notas

- Scripts antigos foram removidos apenas quando havia substituto mais novo.
- A documentação antiga do fluxo MiniLM foi descontinuada.
- Acervo legado foi removido para manter apenas o fluxo oficial.
