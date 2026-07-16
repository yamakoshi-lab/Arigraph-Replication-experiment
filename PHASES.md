# フェーズ

このリポジトリでの作業を、大きく2つのフェーズに分けて記録する。
境界はGitタグ`repro-phase-complete`。

## Phase 1: 再現実験(〜タグ`repro-phase-complete`まで)

AriGraph論文(arXiv:2407.04363v3)の再現。

- **MuSiQue**: 6要因分析(埋め込み方式・QA回答指示・記憶の件数・抽出の指示文・
  ランダム性・重要語抽出の指示)で性能変化の内訳を特定(dec-033)。
- **TextWorld**: 論文バックボーン(Llama-3-70B, gpt-4-0125-preview)は実地調査の
  結果いずれも入手不可能と確定。代替バックボーン(Llama-3.1-70B)でアーキテクチャの
  妥当性(dec-037)と性能(dec-041/042)を確認(dec-043)。

詳細: `lab/decisions.md`(dec-001〜dec-043)、発表資料
`lab/presentations/20260716_combined_compressed_report.pptx`。

## Phase 2: 失敗分析＋改良(タグ`repro-phase-complete`から、進行中)

Mindful-RAG(arXiv:2407.12216、査読あり:IEEE FLLM 2024)型を模倣し、
失敗分析とその知見に基づく改良を1つ実装することを目指す(dec-044)。

MuSiQueの分析を2部に分離:

1. **AriGraph検索部分の分析**(LLM不要、優先着手): エピソード記憶の選択が、
   回答に必要な段落をどれだけ拾えているかを測定。
2. **オラクルQAの分析**(LLM使用、後回し): 検索を無視し、正解に必要な段落のみを
   LLMに与えた場合の点数を測定。検索の失敗とQA推論の失敗を切り分ける。

作業場所: `experiments/phase2_failure_analysis/`(コード)、進捗は`lab/STATUS.md`・
`lab/decisions.md`(dec-044〜)を参照。
