# IMPORTS

import os
import datetime

# ── LOAD CONFIG ──────────────────────────────────────────────
import yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

# ── DB CONNECTION ──────────────────────────────────────────────
from db import get_connection

with get_connection() as conn:
    reading_cursor = conn.cursor()
    insert_cursor  = conn.cursor()

    # ── GENERATE SESSION ID ────────────────────────────────────────
    session_id = 'annot_' + datetime.datetime.now().strftime('%Y%m%d_%H%M')
    print(f"Session ID: {session_id}")

    # ── ENSURE KEYPOINTS TABLE EXISTS ─────────────────────────────
    insert_cursor.execute("""
        CREATE TABLE IF NOT EXISTS keypoints (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            annotation_id INT NOT NULL,
            name          VARCHAR(50) NOT NULL,
            x             FLOAT NOT NULL,
            y             FLOAT NOT NULL,
            visible       INT NOT NULL DEFAULT 2,
            created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (annotation_id) REFERENCES annotations(id)
        )
    """)

    # ── LOOP THROUGH LABEL FILES ───────────────────────────────────
    labels_path   = cfg['store_annotations']['labels_path']
    frames_folder = cfg['store_annotations']['frames_folder']

    listing_labeltxt = os.listdir(labels_path)   # ['e6d83681-frame_360.txt', ...]

    for x in listing_labeltxt:

        # ── PARSE FILENAME → FRAME NUMBER ─────────────────────────
        extract_frame_numb   = x.split("-")[1]                                        # frame_360.txt
        extracted_label_numb = int(os.path.splitext(extract_frame_numb)[0].split("_")[1])  # 360

        # ── QUERY DB → GET FRAME ID ────────────────────────────────
        frame_path_pattern = f"{frames_folder}/frame_{extracted_label_numb}%.png"
        reading_cursor.execute(
            "SELECT id FROM frames WHERE frame_path LIKE %s", (frame_path_pattern,)
        )
        frame_id = reading_cursor.fetchone()[0]
        reading_cursor.fetchall()
        print()
        print(f"frame number: {extracted_label_numb} → frame_id: {frame_id}")

        # ── READ ANNOTATION FILE ───────────────────────────────────
        annotation_txt = os.path.join(labels_path, x)
        with open(annotation_txt, 'r') as f:
            lines = f.readlines()
        print(lines)

        # ── PARSE EACH LINE ────────────────────────────────────────
        label_map = {0: "danio_rerio", 1: "reflection"}

        for value in lines:
            values = value.split()  # splits on whitespace, strips \n

            if len(values) == 5:
                # bbox only: class_id x y w h
                class_id = int(values[0])
                x_center = float(values[1])
                y_center = float(values[2])
                width    = float(values[3])
                height   = float(values[4])
                has_keypoint = False

            elif len(values) == 8:
                # bbox + eye keypoint: class_id x y w h kp_x kp_y visible
                class_id     = int(values[0])
                x_center     = float(values[1])
                y_center     = float(values[2])
                width        = float(values[3])
                height       = float(values[4])
                kp_x         = float(values[5])
                kp_y         = float(values[6])
                kp_visible   = int(values[7])
                has_keypoint = True

            else:
                print(f"WARNING: {x} has {len(values)} values — expected 5 or 8. Skipping.")
                continue

            label      = label_map[class_id]
            created_at = datetime.datetime.now()

            # ── INSERT BBOX INTO ANNOTATIONS ──────────────────────
            insert_cursor.execute(
                "INSERT INTO annotations (frame_id, class_id, label, x_center, y_center, width, height, created_at, session_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (frame_id, class_id, label, x_center, y_center, width, height, created_at, session_id)
            )
            annotation_id = insert_cursor.lastrowid

            # ── INSERT KEYPOINT IF PRESENT ────────────────────────
            if has_keypoint and label == 'danio_rerio':
                insert_cursor.execute(
                    "INSERT INTO keypoints (annotation_id, name, x, y, visible, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
                    (annotation_id, 'eye', kp_x, kp_y, kp_visible, created_at)
                )
                print(f"  keypoint eye → ({kp_x:.4f}, {kp_y:.4f}) visible={kp_visible}")

            print(values)

    # ── COMMIT ─────────────────────────────────────────────────
    conn.commit()
    print("\nAnnotations stored successfully.")
