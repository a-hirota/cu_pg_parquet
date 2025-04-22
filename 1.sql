BEGIN;

-- 既存インデックスは型が合わなくなるので事前に削除
DROP INDEX CONCURRENTLY IF EXISTS lineorder_lo_ok_ln_idx;

-- 型変更（全列を numeric へキャスト）
ALTER TABLE lineorder
  ALTER COLUMN lo_orderkey
  TYPE numeric                                 -- 必要に応じて numeric(20,0) など精度指定
  USING lo_orderkey::numeric;

-- インデックスを作り直す（CONCURRENTLY 推奨）
CREATE INDEX CONCURRENTLY lineorder_lo_ok_ln_idx
  ON lineorder (lo_orderkey, lo_linenumber);

-- 統計情報を更新
ANALYZE lineorder;

COMMIT;

