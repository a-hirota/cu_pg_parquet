-- ctid範囲分割で行が欠落する理由を実証するSQL

-- 1. まず全体の行数を確認
SELECT COUNT(*) as total_rows FROM customer;

-- 2. ctidの範囲を確認
SELECT 
    MIN(ctid) as min_ctid,
    MAX(ctid) as max_ctid,
    pg_relation_size('customer'::regclass) / 8192 as total_pages
FROM customer;

-- 3. 最初の10ページの行数分布を確認
SELECT 
    (ctid::text::point)[0]::int as page_num,
    COUNT(*) as rows_in_page,
    MIN((ctid::text::point)[1]::int) as min_tuple,
    MAX((ctid::text::point)[1]::int) as max_tuple
FROM customer
WHERE (ctid::text::point)[0]::int < 10
GROUP BY page_num
ORDER BY page_num;

-- 4. ページ境界での問題を確認
-- ページ0の最後の行とページ1の最初の行
SELECT 
    ctid,
    c_custkey,
    (ctid::text::point)[0]::int as page,
    (ctid::text::point)[1]::int as tuple_id
FROM customer
WHERE ctid IN (
    (SELECT MAX(ctid) FROM customer WHERE (ctid::text::point)[0]::int = 0),
    (SELECT MIN(ctid) FROM customer WHERE (ctid::text::point)[0]::int = 1)
);

-- 5. チャンク分割シミュレーション（2チャンク）
WITH chunk_info AS (
    SELECT 
        (pg_relation_size('customer'::regclass) / 8192) / 2 as pages_per_chunk
)
SELECT 
    'Chunk 0' as chunk,
    COUNT(*) as rows_chunk0
FROM customer, chunk_info
WHERE ctid >= '(0,1)'::tid 
  AND ctid < '(' || pages_per_chunk || ',1)'::tid

UNION ALL

SELECT 
    'Chunk 1' as chunk,
    COUNT(*) as rows_chunk1
FROM customer, chunk_info
WHERE ctid >= '(' || pages_per_chunk || ',1)'::tid
  AND ctid < '(' || (pages_per_chunk * 2) || ',1)'::tid;

-- 6. 問題の核心：タプルIDが1から始まらないページを探す
WITH page_first_tuple AS (
    SELECT 
        (ctid::text::point)[0]::int as page_num,
        MIN((ctid::text::point)[1]::int) as first_tuple_id
    FROM customer
    WHERE (ctid::text::point)[0]::int BETWEEN 100 AND 200
    GROUP BY page_num
)
SELECT 
    page_num,
    first_tuple_id
FROM page_first_tuple
WHERE first_tuple_id > 1
ORDER BY page_num
LIMIT 10;

-- 7. 欠落の実証：ページ境界でのctid範囲クエリ
-- 例：ページ100の境界
SELECT 
    'Page 99 rows' as description,
    COUNT(*) as count
FROM customer
WHERE ctid >= '(99,1)'::tid AND ctid < '(100,1)'::tid

UNION ALL

SELECT 
    'Page 99 actual rows' as description,
    COUNT(*) as count
FROM customer
WHERE (ctid::text::point)[0]::int = 99;

-- 8. なぜ欠落するか：ページ100のタプル1が存在しない場合
SELECT 
    ctid,
    c_custkey
FROM customer
WHERE (ctid::text::point)[0]::int = 100
  AND (ctid::text::point)[1]::int <= 5
ORDER BY ctid;

-- 9. 正しい方法：ページ番号で分割
WITH chunk_ranges AS (
    SELECT 
        0 as chunk_id,
        0 as start_page,
        (pg_relation_size('customer'::regclass) / 8192) / 2 as end_page
    UNION ALL
    SELECT 
        1 as chunk_id,
        (pg_relation_size('customer'::regclass) / 8192) / 2 as start_page,
        (pg_relation_size('customer'::regclass) / 8192) as end_page
)
SELECT 
    chunk_id,
    COUNT(*) as rows_in_chunk
FROM customer, chunk_ranges
WHERE (ctid::text::point)[0]::int >= start_page
  AND (ctid::text::point)[0]::int < end_page
GROUP BY chunk_id
ORDER BY chunk_id;