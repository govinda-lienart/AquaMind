-- see list of all active views
SHOW FULL TABLES WHERE Table_type = 'VIEW';

-- session_summary: shows annotation sessions grouped by video
-- useful for selecting which sessions to combine for training
CREATE OR REPLACE VIEW session_summary AS
SELECT 
    a.session_id,
    COUNT(DISTINCT a.frame_id) AS frames,
    COUNT(*) AS total_annotations,
    MIN(a.created_at) AS run_at,
    v.file_path
FROM annotations a
JOIN frames f ON a.frame_id = f.id
JOIN videos v ON f.video_id = v.id
GROUP BY a.session_id, v.file_path
ORDER BY a.session_id;

-- using the view
SELECT * FROM session_summary;
