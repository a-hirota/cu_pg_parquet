gpupgparserにテストプランを追加したい。
## 目的
1. Postgresバイナリデータからarrowファイルの作成について、cpu版と比較して行数が全件一致していることを確認したい。
## 実行方法 
python cu_pg_parquet.py --table lineorder  --parallel 16 --chunks 8 にtestオプションをつけること。 python cu_pg_parquet.py --test --table lineorder --parallel 16 --chunks 8
- テスト内容：
  1. 行数比較
     1. 正解値の行数：psqlより算出。select count(*) from lineorder;
     2. GPU版の行数：parquetファイルをチャンク分読み込み行数出力
  2. 行値確認
     1. 正解値：バイナリデータダンプのlo_orderkey>lo_linenumberより該当する行を表示.GPU版の行数で検知したlo_orderkey>lo_linenumberを選択する。
     2. GPU版の行数
        1. ：サンプルスレッドのfield_offsets, field_lenghsおよび、それに対応したPostgresバイナリデータと前後20B.
        2. validate_and_extract_fields_liteの戻り値
        3. parse_rows_and_fields_liteの戻り値
        4. サンプルスレッドは、grid境界スレッドとする。
## tool
- pytest
- cuDFテストに準拠したい。
https://docs.rapids.ai/api/cudf/stable/developer_guide/testing/

## 作業場所
- フォルダはテストフォルダを利用すること。./test
- psql -U postgres -h localhostでpsqlに接続。lineorderテーブルがない場合は中止。