# Program to extract faces from videos using dlib or mediapipe


# dlib is used when processing videos from DFDC Preview dataset

# input_folder is the folder where the video is stored
# cropped_faces_folder is the folder where cropped faces will be stored
# original_frames_folder is the folder where individual video frames will be saved (optional)



import cv2
import dlib
import os
import mediapipe as mp



def dlib_face(input_folder, cropped_faces_folder, original_frames_folder):

    os.makedirs(cropped_faces_folder, exist_ok=True)
    os.makedirs(original_frames_folder, exist_ok=True)

    detector = dlib.get_frontal_face_detector()

    for video_file in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_file)

        if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Processing video: {video_file}")

            base_name = os.path.splitext(video_file)[0]

            video_cropped_folder = os.path.join(cropped_faces_folder, base_name)
            video_real_frame_folder = os.path.join(original_frames_folder, base_name)
            os.makedirs(video_cropped_folder, exist_ok=True)
            os.makedirs(video_real_frame_folder, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector(rgb_frame)

                if len(faces) > 0:
                    real_frame_filename = f"{base_name}_frame_{frame_count:06d}.jpg"
                    real_frame_path = os.path.join(video_real_frame_folder, real_frame_filename)
                    cv2.imwrite(real_frame_path, frame)

                    for face in faces:
                        x1 = face.left()
                        y1 = face.top()
                        x2 = face.right()
                        y2 = face.bottom()

                        cropped_face = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

                        if cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
                            cropped_filename = f"{base_name}_frame_{frame_count:06d}_face.jpg"
                            cropped_path = os.path.join(video_cropped_folder, cropped_filename)
                            cv2.imwrite(cropped_path, cropped_face)

            cap.release()
            print(f"Finished processing: {video_file}")
            print(f"→ Cropped faces: {video_cropped_folder}")
            print(f"→ Real frames with faces: {video_real_frame_folder}")

    cv2.destroyAllWindows()


def mediapipe_face(input_folder, cropped_faces_folder, original_frames_folder):
    os.makedirs(cropped_faces_folder, exist_ok=True)
    os.makedirs(original_frames_folder, exist_ok=True)

    mp_face_detection = mp.solutions.face_detection
    DETECTION_CONFIDENCE = 0.5

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=DETECTION_CONFIDENCE) as face_detector:
        for video_file in os.listdir(input_folder):
            video_path = os.path.join(input_folder, video_file)

            if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"Processing video: {video_file}")
                base_name = os.path.splitext(video_file)[0]

                video_cropped_folder = os.path.join(cropped_faces_folder, base_name)
                video_real_frame_folder = os.path.join(original_frames_folder, base_name)
                os.makedirs(video_cropped_folder, exist_ok=True)
                os.makedirs(video_real_frame_folder, exist_ok=True)

                cap = cv2.VideoCapture(video_path)
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detector.process(rgb_frame)

                    if results.detections:
                        real_frame_filename = f"{base_name}_frame_{frame_count:06d}.jpg"
                        real_frame_path = os.path.join(video_real_frame_folder, real_frame_filename)
                        cv2.imwrite(real_frame_path, frame)

                        for i, detection in enumerate(results.detections):
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, _ = frame.shape
                            x1 = int(bboxC.xmin * iw)
                            y1 = int(bboxC.ymin * ih)
                            w = int(bboxC.width * iw)
                            h = int(bboxC.height * ih)
                            x2 = x1 + w
                            y2 = y1 + h

                            cropped_face = frame[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)]

                            if cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
                                cropped_filename = f"{base_name}_face_{frame_count:06d}.jpg"
                                cropped_path = os.path.join(video_cropped_folder, cropped_filename)
                                cv2.imwrite(cropped_path, cropped_face)

                cap.release()
                print(f"Finished processing: {video_file}")
                print(f"→ Real frames saved to: {video_real_frame_folder}")
                print(f"→ Cropped faces saved to: {video_cropped_folder}")

    cv2.destroyAllWindows()
