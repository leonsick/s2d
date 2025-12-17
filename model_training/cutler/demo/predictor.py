# Copyright (c) Meta Platforms, Inc. and affiliates.
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import numpy as np
import torch

from detectron2.data import MetadataCatalog
import sys
import pycocotools.mask as mask_util

from detectron2.utils.colormap import random_color
from detectron2.utils import colormap

sys.path.append('./')
from engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer, _create_text_labels, GenericMask
import matplotlib.colors as mplc
import matplotlib as mpl


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, davis_mode=False):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        raw = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        if davis_mode:
            visualizer = CustomVisualizer(np.zeros_like(image), self.metadata, instance_mode=self.instance_mode)
        else:
            visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                if davis_mode:
                    vis_output, raw = visualizer.draw_instance_predictions_on_black(instances)
                else:
                    vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output, raw

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5


class CustomVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE, font_size_scale=1.0):
        super(CustomVisualizer, self).__init__(img_rgb, metadata, scale, instance_mode, font_size_scale)

        # Get a list of 20 colors, this list shall remained fixed for all the masks
        self.colors = [colormap._COLORS[i+1] for i in range(50)]

    def draw_instance_predictions_on_black(self, predictions, jittering: bool = True):
        """
        Draw instance-level prediction results on a black background.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
            jittering: if True, in color mode SEGMENTATION, randomly jitter the colors per class
                to distinguish instances from the same class

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = (
                [self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes]
                if jittering
                else [
                    tuple(mplc.to_rgb([x / 255 for x in self.metadata.thing_colors[c]]))
                    for c in classes
                ]
            )
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        # Create a black background
        black_background = np.zeros_like(self.img)
        self.output.reset_image(black_background)
        alpha = 1.0

        self.overlay_masks_for_davis(
            masks=masks,
            assigned_colors=colors,
            alpha=alpha,
        )

        raw = self.overlay_masks_numpy(
            masks=masks,
            colors=colors,
            base_img=black_background,
        )

        return self.output, raw

    def overlay_masks_numpy(self, masks, colors, base_img):
        num_instances = 0
        if masks is not None:
            masks = self._convert_masks(masks)
            num_instances = len(masks)

        if num_instances == 0:
            print("Detected 0 instances.")
            return base_img

        # Display in largest to smallest order to reduce occlusion.
        areas = np.asarray([x.area() for x in masks])
        sorted_idxs = np.argsort(-areas).tolist()
        masks = [masks[idx] for idx in sorted_idxs]
        assigned_colors = [self.colors[idx] for idx in sorted_idxs]

        new_annotation = np.zeros_like(base_img)

        for i in range(num_instances):
            color = np.array(assigned_colors[i]) * 255  # Color is a scalar float value
            mask_obj = masks[i]

            # If mask_obj has a 'polygons' attribute, treat it as a PolygonMasks object.
            binary_mask = np.zeros((new_annotation.shape[0], new_annotation.shape[1]), dtype=np.uint8)
            if hasattr(mask_obj, "polygons"):
                for segment in mask_obj.polygons:
                    polygon = segment.reshape(-1, 2)
                    pts = np.round(polygon).astype(np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(binary_mask, [pts], 1)

            # If the mask object has a 'mask' attribute (e.g., GenericMask or BitMasks), use it directly.
            elif hasattr(mask_obj, "mask"):
                binary_mask = mask_obj.mask.astype(np.uint8)  # Ensure it's a binary mask

            # Otherwise, assume the mask is already a binary NumPy array.
            else:
                binary_mask = mask_obj.astype(np.uint8)

            # Assign mask color directly to the new_annotation image
            for c in range(3):
                new_annotation[:, :, c] = np.where(binary_mask == 1, color[c], new_annotation[:, :, c])

        return new_annotation

    def overlay_masks_for_davis(
            self,
            *,
            masks=None,
            assigned_colors=None,
            alpha=0.5,
    ):
        """
        Args:
            masks (masks-like object): Supported types are:

                * :class:`detectron2.structures.PolygonMasks`,
                  :class:`detectron2.structures.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                  The first level of the list corresponds to individual instances. The second
                  level to all the polygon that compose the instance, and the third level
                  to the polygon coordinates. The third level should have the format of
                  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.
        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = 0
        if masks is not None:
            masks = self._convert_masks(masks)
            num_instances = len(masks)
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]



        if num_instances == 0:
            return self.output

        # Display in largest to smallest order to reduce occlusion.
        areas = np.asarray([x.area() for x in masks])
        sorted_idxs = np.argsort(-areas).tolist()
        masks = [masks[idx] for idx in sorted_idxs]
        assigned_colors = [self.colors[idx] for idx in sorted_idxs]

        for i in range(num_instances):
            color = assigned_colors[i]
            mask_obj = masks[i]

            # If mask_obj has a 'polygons' attribute, treat it as a PolygonMasks object.
            if hasattr(mask_obj, "polygons"):
                for i_s, segment in enumerate(mask_obj.polygons):
                    # Ensure the segment is in (N, 2) format
                    polygon = segment.reshape(-1, 2)

                    # Create an empty binary mask.
                    binary_mask = np.zeros((self.output.height, self.output.width), dtype=np.uint8)

                    # Convert polygon coordinates to integer format.
                    pts = np.round(polygon).astype(np.int32)
                    pts = pts.reshape((-1, 1, 2))

                    # Fill the polygon on the binary mask with the value 1.
                    cv2.fillPoly(binary_mask, [pts], 1)

                    # Overlay the binary mask on the image.
                    self.draw_binary_mask(binary_mask, color, alpha=alpha)

            # If the mask object has a 'mask' attribute (e.g., GenericMask or BitMasks), use it directly.
            elif hasattr(mask_obj, "mask"):
                binary_mask = mask_obj.mask  # Expected to be a NumPy array.
                self.draw_binary_mask(binary_mask, color, alpha=alpha)
            # Otherwise, assume the mask is already a binary NumPy array.
            else:
                self.draw_binary_mask(mask_obj, color, alpha=alpha)

        return self.output

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        """
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted. If not provided, a darker shade
                of the polygon color will be used instead.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with polygon drawn.
        """
        if edge_color is None:
            # make edge color darker than the polygon color
            edge_color = color

        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=mplc.to_rgb(color) + (alpha,),
            linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
        )

        self.output.ax.add_patch(polygon)
        return self.output
