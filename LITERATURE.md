# Современная литература по LoRA и оптимизации предпочтений

Ниже приведены первичные публикации, непосредственно связанные с низкоранговой адаптацией, reward modeling и preference optimization. Для статьи SUMMA используется формат IEEE; этот список дополнительно поясняет роль каждого источника.

## Низкоранговая адаптация

1. Hu E. J. et al. LoRA: Low-Rank Adaptation of Large Language Models // International Conference on Learning Representations. 2022. URL: <https://openreview.net/forum?id=nZeVKeeFYf9>. Базовая параметризация обновления весов произведением двух матриц малого ранга.
2. Dettmers T. et al. QLoRA: Efficient Finetuning of Quantized LLMs // Advances in Neural Information Processing Systems. Vol. 36. 2023. P. 10088–10115. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html>. Сочетание 4-битной квантизации и LoRA.
3. Hayou S., Ghosh N., Yu B. LoRA+: Efficient Low Rank Adaptation of Large Models // Proceedings of the 41st International Conference on Machine Learning. PMLR 235. 2024. P. 17783–17806. URL: <https://proceedings.mlr.press/v235/hayou24a.html>. Раздельные скорости обучения для двух матриц адаптера.
4. Liu S.-Y. et al. DoRA: Weight-Decomposed Low-Rank Adaptation // Proceedings of the 41st International Conference on Machine Learning. PMLR 235. 2024. P. 32100–32121. URL: <https://proceedings.mlr.press/v235/liu24bn.html>. Разложение веса на величину и направление.
5. Meng F., Wang Z., Zhang M. PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models // Advances in Neural Information Processing Systems. Vol. 37. 2024. URL: <https://proceedings.neurips.cc/paper_files/paper/2024/hash/db36f4d603cc9e3a2a5e10b93e6428f2-Abstract-Conference.html>. Инициализация адаптера ведущими сингулярными компонентами исходного веса.

## Оптимизация предпочтений и reward-модели

6. Rafailov R. et al. Direct Preference Optimization: Your Language Model Is Secretly a Reward Model // Advances in Neural Information Processing Systems. Vol. 36. 2023. P. 53728–53741. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html>.
7. Ethayarajh K. et al. Model Alignment as Prospect Theoretic Optimization // Proceedings of the 41st International Conference on Machine Learning. PMLR 235. 2024. P. 12634–12651. URL: <https://proceedings.mlr.press/v235/ethayarajh24a.html>. KTO показывает зависимость подходящей функции потерь от предположений о человеческой полезности.
8. Meng Y., Xia M., Chen D. SimPO: Simple Preference Optimization with a Reference-Free Reward // Advances in Neural Information Processing Systems. Vol. 37. 2024. DOI: 10.52202/079017-3946. URL: <https://proceedings.neurips.cc/paper_files/paper/2024/hash/e099c1c9699814af0be873a175361713-Abstract-Conference.html>.
9. Hong J., Lee N., Thorne J. ORPO: Monolithic Preference Optimization without Reference Model // Proceedings of EMNLP. 2024. P. 11170–11189. DOI: 10.18653/v1/2024.emnlp-main.626. URL: <https://aclanthology.org/2024.emnlp-main.626/>.
10. Lambert N. et al. RewardBench: Evaluating Reward Models for Language Modeling // Findings of NAACL. 2025. P. 1755–1797. DOI: 10.18653/v1/2025.findings-naacl.96. URL: <https://aclanthology.org/2025.findings-naacl.96/>. Современный набор проверок reward-моделей на диалогах, рассуждениях и безопасности.
11. Yang K. et al. Selective Preference Optimization via Token-Level Reward Function Estimation // Proceedings of EMNLP. 2025. P. 7032–7056. DOI: 10.18653/v1/2025.emnlp-main.359. URL: <https://aclanthology.org/2025.emnlp-main.359/>. Оптимизация только информативных токенов с уменьшением вычислительных затрат.

Для расширения текущего эксперимента наиболее важны LoRA+, PiSSA и RewardBench: первые два дают проверяемые альтернативы параметризации адаптера, а RewardBench — внешнюю оценку переносимости reward-модели за пределами HH-RLHF.
