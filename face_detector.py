import cv2
import argparse
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import chainer
from chainer import cuda, serializers, functions as F
from entity import params
from models.FaceNet import FaceNet

chainer.using_config('enable_backprop', False)

class FaceDetector(object):
    def __init__(self, arch=None, weights_file=None, model=None, device=-1):
        print('Loading FaceNet...')
        self.model = params['archs'][arch]()
        serializers.load_npz(weights_file, self.model)

        self.device = device
        if self.device >= 0:
            cuda.get_device_from_id(device).use()
            self.model.to_gpu()

            # create gaussian filter
            ksize = params['ksize']
            kernel = cuda.to_gpu(self.create_gaussian_kernel(sigma=params['gaussian_sigma'], ksize=ksize))
            self.gaussian_kernel = kernel

    def __call__(self, face_img, fast_mode=False):
        face_img_h, face_img_w, _ = face_img.shape

        resized_image = cv2.resize(face_img, (params["face_inference_img_size"], params["face_inference_img_size"]))
        x_data = np.array(resized_image[np.newaxis], dtype=np.float32).transpose(0, 3, 1, 2) / 256 - 0.5

        if self.device >= 0:
            x_data = cuda.to_gpu(x_data)

        hs = self.model(x_data)
        heatmaps = F.resize_images(hs[-1], (face_img_h, face_img_w)).data[0]
        keypoints = self.compute_peaks_from_heatmaps(heatmaps)

        return keypoints

    # compute gaussian filter
    def create_gaussian_kernel(self, sigma=1, ksize=5):
        center = int(ksize / 2)
        kernel = np.zeros((1, 1, ksize, ksize), dtype=np.float32)
        for y in range(ksize):
            distance_y = abs(y-center)
            for x in range(ksize):
                distance_x = abs(x-center)
                kernel[0][0][y][x] = 1/(sigma**2 * 2 * np.pi) * np.exp(-(distance_x**2 + distance_y**2)/(2 * sigma**2))
        return kernel

    def compute_peaks_from_heatmaps(self, heatmaps):
        keypoints = []
        xp = cuda.get_array_module(heatmaps)

        if xp == np:
            for i in range(heatmaps.shape[0] - 1):
                heatmap = gaussian_filter(heatmaps[i], sigma=params['gaussian_sigma'])
                max_value = heatmap.max()
                if max_value > params['face_heatmap_peak_thresh']:
                    coords = np.array(np.where(heatmap==max_value)).flatten().tolist()
                    keypoints.append([coords[1], coords[0], max_value]) # x, y, conf
                else:
                    keypoints.append(None)
        else:
            heatmaps = F.convolution_2d(heatmaps[:, None], self.gaussian_kernel, stride=1, pad=int(params['ksize']/2)).data.squeeze().get()
            for heatmap in heatmaps[:-1]:
                max_value = heatmap.max()
                if max_value > params['face_heatmap_peak_thresh']:
                    coords = np.array(np.where(heatmap==max_value)).flatten().tolist()
                    keypoints.append([coords[1], coords[0], max_value]) # x, y, conf
                else:
                    keypoints.append(None)

        return keypoints

def draw_face_keypoints(orig_img, face_keypoints, left_top):
    img = orig_img.copy()
    left, top = left_top

    for keypoint in face_keypoints:
        if keypoint:
            x, y, conf = keypoint
            cv2.circle(img, (x + left, y + top), 2, (255, 255, 0), -1)

    for face_line_index in params["face_line_indices"]:
        keypoint_from = face_keypoints[face_line_index[0]]
        keypoint_to = face_keypoints[face_line_index[1]]

        if keypoint_from and keypoint_to:
            keypoint_from_x, keypoint_from_y, _ = keypoint_from
            keypoint_to_x, keypoint_to_y, _ = keypoint_to
            cv2.line(img, (keypoint_from_x + left, keypoint_from_y + top), (keypoint_to_x + left, keypoint_to_y + top), (255, 255, 0), 1)

    return img

def crop_face(img, rect):
    orig_img_h, orig_img_w, _ = img.shape
    crop_center_x = rect[0] + rect[2] / 2
    crop_center_y = rect[1] + rect[3] / 2
    crop_width = rect[2] * params['face_crop_scale']
    crop_height = rect[3] * params['face_crop_scale']
    crop_left = max(0, int(crop_center_x - crop_width / 2))
    crop_top = max(0, int(crop_center_y - crop_height / 2))
    crop_right = min(orig_img_w-1, int(crop_center_x + crop_width / 2))
    crop_bottom = min(orig_img_h-1, int(crop_center_y + crop_height / 2))
    cropped_face = img[crop_top:crop_bottom, crop_left:crop_right]
    max_edge_len = np.max(cropped_face.shape[:-1])
    padded_face = np.zeros((max_edge_len, max_edge_len, cropped_face.shape[-1]), dtype=np.uint8)
    padded_face[0:cropped_face.shape[0], 0:cropped_face.shape[1]] = cropped_face

    return padded_face, (crop_left, crop_top)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face detector')
    parser.add_argument('arch', choices=params['archs'].keys(), default='facenet', help='Model architecture')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--img', help='image file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # load model
    face_detector = FaceDetector(args.arch, args.weights, device=args.gpu)

    # read image
    img = cv2.imread(args.img)

    # inference
    face_keypoints = face_detector(img)

    # draw and save image
    img = draw_face_keypoints(img, face_keypoints, (0, 0))
    print('Saving result into result.png...')
    cv2.imwrite('result.png', img)
