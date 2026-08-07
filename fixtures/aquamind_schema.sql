-- AquaMind test schema — mirrors production schema exactly

SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS annotations;
DROP TABLE IF EXISTS annotation_sets;
DROP TABLE IF EXISTS frames;
DROP TABLE IF EXISTS videos;
SET FOREIGN_KEY_CHECKS = 1;

CREATE TABLE videos (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    file_path      VARCHAR(500) NOT NULL,
    fps            INT,
    resolution     ENUM('720p', '1080p', '4K'),
    session_type   ENUM('behaviour', 'tagging', 'close_up', 'mixed'),
    obstacles      TINYINT(1),
    fish_count     INT,
    notes          TEXT,
    filmed_at      DATETIME,
    species        VARCHAR(100),
    morph          VARCHAR(100),
    tank_width_cm  FLOAT,
    tank_height_cm FLOAT,
    tank_depth_cm  FLOAT,
    UNIQUE KEY (file_path)
);

CREATE TABLE frames (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    video_id     INT NOT NULL,
    frame_path   VARCHAR(255),
    frame_number INT,
    timestamp    FLOAT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_video_frame (video_id, frame_number),
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE TABLE annotation_sets (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    video_id         INT NOT NULL,
    frame_source     ENUM('regular', 'crossing_event', 'ghosting_event') NOT NULL,
    notes            TEXT,
    frames_extracted INT,
    iou_threshold    FLOAT,
    dedup_window     INT,
    sample_rate      INT,
    start_seconds    FLOAT,
    end_seconds      FLOAT,
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE TABLE annotations (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    frame_id          INT NOT NULL,
    annotation_set_id INT,
    class_id          INT,
    label             VARCHAR(255),
    x_center          FLOAT,
    y_center          FLOAT,
    width             FLOAT,
    height            FLOAT,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (frame_id) REFERENCES frames(id),
    FOREIGN KEY (annotation_set_id) REFERENCES annotation_sets(id)
);
