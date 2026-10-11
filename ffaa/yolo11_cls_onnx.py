import numpy as np
import onnxruntime as ort
import cv2


class Yolo11ClsONNX:
    def __init__(
        self,
        onnx_path: str,
        imgsz: int = 224,
        class_names=None,
        providers=None,
    ):
        """
        YOLO11 classification ONNX inference wrapper.

        Args:
            onnx_path: path to .onnx model
            imgsz: input image size (must match training)
            class_names: list of class names in correct order
            providers: onnxruntime providers (CUDA / CPU)
        """
        self.onnx_path = onnx_path
        self.imgsz = imgsz
        self.class_names = class_names or ["0", "180", "270", "90"]

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            self.onnx_path,
            providers=providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("ONNX input shape :", self.session.get_inputs()[0].shape)
        print("ONNX output shape:", self.session.get_outputs()[0].shape)

    def _preprocess(self, img) -> np.ndarray:
        # img = cv2.imread(image_path)
        # if img is None:
        #     raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.resize(img, (self.imgsz, self.imgsz))

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))          # HWC → CHW
        img = np.expand_dims(img, axis=0)           # batch dim

        return img

    # -----------------------------
    # Inference
    # -----------------------------
    def predict(self, img: str):
        """
        Runs inference and returns (angle, confidence).
        """
        input_tensor = self._preprocess(img)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )

        probs = outputs[0][0]  # shape: (num_classes,)

        top1_id = int(np.argmax(probs))
        confidence = float(probs[top1_id])
        angle = int(self.class_names[top1_id])

        return angle, confidence
