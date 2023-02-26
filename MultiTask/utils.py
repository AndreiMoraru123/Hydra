import os
from tqdm import tqdm
import torch
from torch.autograd.variable import Variable
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as co
import cv2

IMG_SCALE = 1. / 255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))


def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD


def pipeline(model, img, CMAP, NUM_CLASSES):
    with torch.no_grad():
        img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None]), requires_grad=False).float()
        if torch.cuda.is_available():
            img_var = img_var.cuda()
        depth, segm = model(img_var)
        segm = cv2.resize(segm[0, :NUM_CLASSES].cpu().data.numpy().transpose(1, 2, 0),
                          img.shape[:2][::-1],
                          interpolation=cv2.INTER_LANCZOS4)
        depth = cv2.resize(depth[0, 0].cpu().data.numpy(),
                           img.shape[:2][::-1],
                           interpolation=cv2.INTER_LANCZOS4)
        segm = CMAP[segm.argmax(axis=2)].astype(np.uint8)
        depth = np.abs(depth)
        return depth, segm


def depth_to_rgb(depth):
    normalizer = co.Normalize(vmin=0, vmax=80)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im


def predict_video(model, model_name, input_video_path, output_dir,
                  target_width, target_height, CMAP, NUM_CLASSES):
    file_name = input_video_path.split(os.sep)[-1].split('.')[0]
    output_filename = f'{file_name}_{model_name}_output.avi'
    output_video_path = os.path.join(output_dir, *[output_filename])

    # handles for input output videos
    input_handle = cv2.VideoCapture(input_video_path)
    output_handle = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'),
                                    30, (target_width, target_height))

    # create progress bar
    num_frames = int(input_handle.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=num_frames, position=0, leave=True)

    while input_handle.isOpened():
        ret, frame = input_handle.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # # create torch tensor to give as input to model
            # pt_image = preprocess(frame)
            # pt_image = pt_image.to(device)
            #
            # # get model prediction and convert to corresponding color
            # y_pred = torch.argmax(model(pt_image.unsqueeze(0)), dim=1).squeeze(0)
            # predicted_labels = y_pred.cpu().detach().numpy()
            # cm_labels = (train_id_to_color[predicted_labels]).astype(np.uint8)
            #
            # # overlay prediction over input frame
            # overlay_image = cv2.addWeighted(frame, 1, cm_labels, 0.25, 0)
            # overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)

            image = np.array(frame)

            depth, segm = pipeline(model, image, CMAP, NUM_CLASSES)

            # write output result and update progress
            output_handle.write(cv2.cvtColor(cv2.hconcat([image, segm, depth_to_rgb(depth)]), cv2.COLOR_RGB2BGR))
            pbar.update(1)

        else:
            break

    output_handle.release()
    input_handle.release()
