-- チャンク26のページ範囲でデータが存在するか確認
SELECT COUNT(*) as chunk_26_rows
FROM lineorder 
WHERE (ctid::text::point)[0]::int >= 1542008 
  AND (ctid::text::point)[0]::int < 1601316;

-- チャンク25のページ範囲
SELECT COUNT(*) as chunk_25_rows
FROM lineorder 
WHERE (ctid::text::point)[0]::int >= 1482700 
  AND (ctid::text::point)[0]::int < 1542008;

-- チャンク31（最後）のページ範囲
SELECT COUNT(*) as chunk_31_rows
FROM lineorder 
WHERE (ctid::text::point)[0]::int >= 1838548 
  AND (ctid::text::point)[0]::int < 1897885;

-- 実際の最大ページ番号を確認
SELECT MAX((ctid::text::point)[0]::int) as actual_max_page
FROM lineorder;