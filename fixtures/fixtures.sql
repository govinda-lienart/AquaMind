INSERT INTO videos (file_path, fps, resolution, session_type, obstacles, fish_count, species)
VALUES ('videos/test.MOV', 60, '1080p', 'behaviour', 1, 5, 'danio_rerio');

INSERT INTO frames (video_id, frame_path, frame_number, timestamp)
VALUES (LAST_INSERT_ID(), 'frames/frames_test/frame_360_test.png', 360, 6.0);
