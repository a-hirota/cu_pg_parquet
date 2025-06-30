-- テーブルの統計情報を確認
SELECT 
    schemaname,
    tablename,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples,
    n_dead_tup::float / NULLIF(n_live_tup, 0) as dead_ratio,
    last_vacuum,
    last_autovacuum
FROM pg_stat_user_tables
WHERE tablename = 'lineorder';

-- テーブルのサイズ情報
SELECT 
    pg_size_pretty(pg_relation_size('lineorder')) as table_size,
    pg_size_pretty(pg_total_relation_size('lineorder')) as total_size,
    (pg_relation_size('lineorder')::float / 8192) as total_pages,
    (pg_relation_size('lineorder')::float / 8192 / 246012324 * 129.6) as pages_per_row_expected;

-- 実際のページ使用状況を確認（サンプル）
WITH page_sample AS (
    SELECT 
        (ctid::text::point)[0]::int as page_num,
        COUNT(*) as rows_in_page
    FROM lineorder
    WHERE (ctid::text::point)[0]::int < 1000  -- 最初の1000ページのみサンプル
    GROUP BY page_num
)
SELECT 
    COUNT(*) as used_pages,
    AVG(rows_in_page) as avg_rows_per_page,
    MIN(rows_in_page) as min_rows,
    MAX(rows_in_page) as max_rows
FROM page_sample;