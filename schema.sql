-- AquaMind database schema (structure only, no data)
-- MySQL 8.4, InnoDB, utf8mb4
--
-- Reproduce locally:
--   docker run --name cont-aquamind-sql \
--     -e MYSQL_ROOT_PASSWORD=aquamind -e MYSQL_DATABASE=aquamind \
--     -p 3306:3306 -d mysql:8.4
--   mysql -h 127.0.0.1 -P 3306 -u root -paquamind aquamind < schema.sql
--
-- 4 tables, created in dependency order (each FK target exists before
-- the table that references it):
--   videos -> frames -> annotation_sets -> annotations

-- One row per registered video. Written by sync_videos.py, which reads
-- video_metadata.xlsx (one row per video) and inserts each row here. The
-- spreadsheet exists because filling a spreadsheet row by hand is far
-- easier than writing manual SQL INSERTs for experimental metadata that
-- isn't recoverable from the video file itself (species, morph, tank
-- dimensions, filming context). fps/resolution are the exception: those are
-- read from the video file with OpenCV rather than trusted from the
-- spreadsheet, so they can't drift out of sync with the actual video.
CREATE TABLE `videos` (
  `id` int NOT NULL AUTO_INCREMENT,
  `file_path` varchar(500) NOT NULL,
  `fps` int DEFAULT NULL,
  `resolution` enum('720p','1080p','4K') DEFAULT NULL,
  `activity` varchar(100) DEFAULT NULL,
  `plants` int DEFAULT NULL,
  `fish_count` int DEFAULT NULL,
  `notes` text,
  `filmed_at` datetime DEFAULT NULL,
  `species` varchar(100) DEFAULT NULL,
  `morph` varchar(100) DEFAULT NULL,
  `tank_width_cm` float DEFAULT NULL,
  `tank_height_cm` float DEFAULT NULL,
  `tank_depth_cm` float DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `file_path` (`file_path`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- One row per extracted frame. Written by extract_frames.py at 1 frame/sec.
-- unique_video_frame makes it impossible to record the same frame index
-- twice for the same video, so re-running extraction is safe.
CREATE TABLE `frames` (
  `id` int NOT NULL AUTO_INCREMENT,
  `video_id` int NOT NULL,
  `frame_path` varchar(255) DEFAULT NULL,
  `frame_number` int DEFAULT NULL,
  `timestamp` float DEFAULT NULL,
  `extracted_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `unique_video_frame` (`video_id`,`frame_number`),
  CONSTRAINT `frames_ibfk_1` FOREIGN KEY (`video_id`) REFERENCES `videos` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- One row per LabelStudio export batch downloaded into MySQL, grouping the
-- `annotations` rows it produced. frame_source distinguishes a regular
-- 1 FPS sample from a targeted re-labeling pass (crossing_event /
-- ghosting_event) triggered by the tracker flagging hard frames.
CREATE TABLE `annotation_sets` (
  `id` int NOT NULL AUTO_INCREMENT,
  `video_id` int NOT NULL,
  `frame_source` enum('regular','crossing_event','ghosting_event') DEFAULT NULL,
  `notes` text,
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  `frames_extracted` int DEFAULT NULL,
  `iou_threshold` float DEFAULT NULL,
  `dedup_window` int DEFAULT NULL,
  `sample_rate` int DEFAULT NULL,
  `start_seconds` float DEFAULT NULL,
  `end_seconds` float DEFAULT NULL,
  `ls_project_name` varchar(200) DEFAULT NULL,
  `ls_project_id` int DEFAULT NULL,
  `ls_downloaded_at` datetime DEFAULT NULL,
  `ls_min_task_id` int DEFAULT NULL,
  `ls_max_task_id` int DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `video_id` (`video_id`),
  CONSTRAINT `annotation_sets_ibfk_1` FOREIGN KEY (`video_id`) REFERENCES `videos` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- One row per labeled bounding box. Written by store_annotations.py, parsed
-- from a LabelStudio YOLO export. class_id/label: 0 = danio_rerio,
-- 1 = reflection (labeled separately so the model learns to ignore tank
-- reflections rather than mistaking them for fish).
CREATE TABLE `annotations` (
  `id` int NOT NULL AUTO_INCREMENT,
  `frame_id` int NOT NULL,
  `annotation_set_id` int DEFAULT NULL,
  `class_id` int DEFAULT NULL,
  `label` varchar(255) DEFAULT NULL,
  `x_center` float DEFAULT NULL,
  `y_center` float DEFAULT NULL,
  `width` float DEFAULT NULL,
  `height` float DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `frame_id` (`frame_id`),
  KEY `annotation_set_id` (`annotation_set_id`),
  CONSTRAINT `annotations_ibfk_1` FOREIGN KEY (`frame_id`) REFERENCES `frames` (`id`),
  CONSTRAINT `annotations_ibfk_2` FOREIGN KEY (`annotation_set_id`) REFERENCES `annotation_sets` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
