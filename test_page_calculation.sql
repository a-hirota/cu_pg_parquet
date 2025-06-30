-- lineorderテーブルの実際のページ数を確認
SELECT 
    pg_relation_size('lineorder'::regclass) as table_size_bytes,
    current_setting('block_size')::int as block_size,
    (pg_relation_size('lineorder'::regclass) / current_setting('block_size')::int)::int as total_pages,
    (pg_relation_size('lineorder'::regclass) / current_setting('block_size')::int)::int / 16 as pages_per_chunk_16
;