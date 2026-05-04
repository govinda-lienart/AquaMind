-- schema.sql
CREATE TABLE frames (
    id INT AUTO_INCREMENT PRIMARY KEY,
    video_path VARCHAR(255),
    frame_path VARCHAR(255),
    frame_number INT,
    timestamp FLOAT,
    extracted_at TIMESTAMP
);