INSERT INTO videos (id, file_path, fps, resolution, session_type, obstacles, fish_count, species)
VALUES (19, 'videos/IMG_0350.MOV', 60, '1080p', 'behaviour', 1, 5, 'danio_rerio');

INSERT INTO frames (video_id, frame_path, frame_number, timestamp)
VALUES (19, 'frames/frames_IMG_0350_20260101_2000/frame_360_IMG_0350.png', 360, 6.0);

INSERT INTO annotation_sets (id, video_id, frame_source, sample_rate, notes)
VALUES (1, 19, '1fps', 1, 'test annotation set');

INSERT INTO annotations (frame_id, annotation_set_id, class_id, label, x_center, y_center, width, height)
VALUES
  (LAST_INSERT_ID(), 1, 0, 'danio_rerio', 0.5, 0.4, 0.1, 0.08),
  (LAST_INSERT_ID(), 1, 1, 'reflection',  0.2, 0.3, 0.05, 0.04);
