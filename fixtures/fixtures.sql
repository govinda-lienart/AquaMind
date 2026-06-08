INSERT INTO videos (file_path, fps, resolution, session_type, obstacles, fish_count, species)
VALUES ('videos/IMG_0350.MOV', 60, '1080p', 'behaviour', 1, 5, 'danio_rerio');

INSERT INTO frames (video_id, frame_path, frame_number, timestamp)
VALUES (LAST_INSERT_ID(), 'frames/frames_IMG_0350/frame_360.png', 360, 6.0);
