import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceCropper:
    def __init__(
        self,
        model_path="face_detector.task",
        min_detection_confidence=0.6,
        bbox_margin=0.25,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.bbox_margin = bbox_margin

        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=self.min_detection_confidence,
        )

        self.detector = vision.FaceDetector.create_from_options(options)

    def _expand_bbox(self, x, y, w, h, img_w, img_h):
        margin_w = int(w * self.bbox_margin)
        margin_h = int(h * self.bbox_margin)

        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(img_w, x + w + margin_w)
        y2 = min(img_h, y + h + margin_h)

        return x1, y1, x2, y2

    def get_largest_face(self, frame_bgr):
        img_h, img_w, _ = frame_bgr.shape

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        detection_result = self.detector.detect(mp_image)

        if not detection_result.detections:
            return None, None

        best_area = -1
        best_bbox = None

        for detection in detection_result.detections:
            bbox = detection.bounding_box

            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            w = int(bbox.width)
            h = int(bbox.height)

            x1, y1, x2, y2 = self._expand_bbox(x, y, w, h, img_w, img_h)

            area = (x2 - x1) * (y2 - y1)

            if area > best_area:
                best_area = area
                best_bbox = (x1, y1, x2, y2)

        if best_bbox is None:
            return None, None

        x1, y1, x2, y2 = best_bbox
        face_crop = frame_bgr[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None, None

        return face_crop, best_bbox

    def close(self):
        self.detector.close()