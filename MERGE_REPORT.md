# マージ報告: TextWorldバックボーン対応の`project/`への移植

**日付**: 2026-07-16
**ブランチ**: `merge-textworld-backbone-support`(`main`からの派生、`main`へは未マージ)
**担当**: Claude Code(本来Antigravity担当の実装作業だが、今回はユーザーの明示的許可により
Claude Codeが直接実施。指示書: `lab/prompts/claude/20260716-1909_dec-038_merge-textworld-backbone-support_v1.md`)

## 変更ファイル一覧

- `agents/parent_agent.py`(`GPTagent`): `__init__`に`base_url=None, price_per_1m=None`を追加。
  `OpenAI`クライアントへ`base_url`を渡し、コスト計算を`price_per_1m`指定時のみ切替(未指定時は
  既存のデフォルト計算式のまま)。
- `graphs/parent_graph.py`(`TripletGraph`): 同様に`base_url`/`price_per_1m`対応を追加。
  ついでに、追加箇所のすぐ下にあった無害な重複行(`cost = ...`が2回書かれていたバグ)を、
  同じ箇所を触るタイミングで解消。
- `graphs/contriever_graph.py`(`ContrieverGraph`): `__init__`に`base_url`/`price_per_1m`を追加し、
  親クラス(`TripletGraph`)へそのまま渡すだけの変更。
- `pipeline_arigraph.py`: `.env`読み込み用の`load_dotenv()`ヘルパーを追加(ファイルが
  存在しない場合は何もしない)。「Changeable part」に`base_url = None`・`price_per_1m = None`を
  テンプレートとして追加(具体的なDeepInfra URLやLlama-3.1-70Bの料金はハードコードしていない)。
  各`GPTagent`/`ContrieverGraph`のインスタンス化箇所に`base_url`/`price_per_1m`を渡すよう変更。
- `resume_run.py`(新規): `sandbox-textworld`で一度もコミットされていなかったdec-040の
  実行再開スクリプトを、変更なしでそのままコピー・新規追加。
- `.gitignore`: `.env`を追加(APIキーの誤コミット防止)。

## 意図的に持ち込まなかったもの

- `sandbox-textworld`のコミット`370d7f7`・`bcfa893`(共有ファイルを著者オリジナルへ完全復元)は
  一切反映していない。
- `pipeline_arigraph.py`の「Changeable part」にあった、TextWorld実行時の具体的な設定値
  (`env_name = "hunt"`、`model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"`、
  DeepInfraの実URL・料金の数値)は持ち込んでいない。既存のデフォルト値
  (`env_name = "hunt_hard"`, `model = "gpt-4o"`など)を維持している。

## 後方互換性の確認

- `musique_test_big.py`は今回一切変更していない。同ファイル95行目の
  `agent.generate(prompt, t=0.0)[0]`(temperature固定)は変更前のまま。
- `graphs/contriever_graph.py`のretrieval実装(`RETRIEVAL_MODE`環境変数、デフォルト`dense`)と
  `RETRIEVAL_THRESHOLD`(デフォルト`0.6`)は、`__init__`のみを変更したため無傷であることを
  該当箇所を直接読んで確認済み。
- 変更した5ファイル+新規1ファイルは`python -m py_compile`で構文エラーがないことを確認済み。
- `resume_run.py`のimport先(`utils.utils`, `utils.win_cond`, `utils.textworld_adapter`,
  `graphs.contriever_graph`, `pipeline_arigraph`)は全て`project/`に実在することを確認済み。

## 未実施

- `main`へのマージ・コミット・pushは行っていない。人間・Claude Codeのレビュー待ち。
