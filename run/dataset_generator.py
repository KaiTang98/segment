import os
import sys
import warnings
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

sys.path.append('/home/kaitang/workspace/sewing2d/segment/segment_anything')

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel


'''
    onnx and onnxruntime are required for exporting and running the model
    opencv and matplotlib are required for visualization.
'''

def show_mask(mask, ax):
    color = np.array([255/255, 1/255, 1/255, 0.1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_image
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

def set_up_env():
    
    # load checkpoint
    checkpoint = "/home/kaitang/workspace/sewing2d/segment/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    # Select onnx model, if None set to use an already exported model.
    # Run the following code to export an ONNX model.
    # onnx_model_path = None  
    onnx_model_path = "sam_onnx_example.onnx"

    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )    

    # onnx_model_quantized_path = "sam_onnx_quantized_example.onnx"
    # quantize_dynamic(
    #     model_input=onnx_model_path,
    #     model_output=onnx_model_quantized_path,
    #     optimize_model=True,
    #     per_channel=False,
    #     reduce_range=False,
    #     weight_type=QuantType.QUInt8,
    # )
    # onnx_model_path = onnx_model_quantized_path
    return sam, onnx_model_path

def loader_func(path):
    return Image.open(path)

def main(mode, image_path, image_folder, save_folder):

    # Environment setup
    sam, onnx_model_path =  set_up_env()
    
    if mode == "1" :
        
        # Load image
        image = cv2.imread(image_path)
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Here as an example, we use `onnxruntime` in python on CPU to execute the ONNX model. 
        # However, any platform that supports an ONNX runtime could be used in principle. Launch the runtime session below:
        ort_session = onnxruntime.InferenceSession(onnx_model_path)

        # To use the ONNX model, the image must first be pre-processed using the SAM image encoder. 
        # This is a heavier weight process best performed on GPU. SamPredictor can be used as normal, 
        # then `.get_image_embedding()` will retreive the intermediate features.
        sam.to(device='cuda')
        predictor = SamPredictor(sam)
        predictor.set_image(image_color)

        image_embedding = predictor.get_image_embedding().cpu().numpy()
        # print('The image embedding is ', image_embedding.shape)

        # define the output class and the corresponding color
        input_point = np.array([[70, 500]])
        input_label = np.array([1])

        # Add a batch index, concatenate a padding point, and transform.
        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

        onnx_coord = predictor.transform.apply_coords(onnx_coord, image_color.shape[:2]).astype(np.float32)

        # create empty mask
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        # Package the inputs to run in the onnx model
        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(image_color.shape[:2], dtype=np.float32)
        }

        # Run the model and obtain the mask
        masks, _, low_res_logits = ort_session.run(None, ort_inputs)
        masks = masks > predictor.model.mask_threshold
        # print('The mask shape is: ', masks.shape)

        # Visualize the results
        # plt.figure(figsize=(10,10))
        # plt.imshow(image_color)
        # mask_image = show_mask(masks, plt.gca())
        # show_points(input_point, input_label, plt.gca())
        # plt.axis('off')
        # plt.show() 

        input_point = np.array([[70, 500], [85, 375], [130, 375], [25, 280], [130, 280]]) # normal
        input_label = np.array([1, 1, 1, 1, 1])
        # input_point = np.array([[70, 500], [85, 375], [130, 375], [25, 370]]) # initial
        # input_label = np.array([1, 1, 1, 1])

        # Use the mask output from the previous run. It is already in the correct form for input to the ONNX model.
        onnx_mask_input = low_res_logits

        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

        onnx_coord = predictor.transform.apply_coords(onnx_coord, image_color.shape[:2]).astype(np.float32)

        onnx_has_mask_input = np.ones(1, dtype=np.float32)

        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(image_color.shape[:2], dtype=np.float32)
        }

        masks, _, _ = ort_session.run(None, ort_inputs)
        masks = masks > predictor.model.mask_threshold

        # Visualize the results
        plt.figure(figsize=(10,10))
        plt.imshow(image_color)
        mask_image = show_mask(masks, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.axis('off')
        plt.show() 
        print("success!!!")

    elif mode == "2" :

        for filename in os.listdir(image_folder):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".bmp"):
        
                # Load image
                image = cv2.imread(os.path.join(image_folder, filename))
                image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Here as an example, we use `onnxruntime` in python on CPU to execute the ONNX model. 
                # However, any platform that supports an ONNX runtime could be used in principle. Launch the runtime session below:
                ort_session = onnxruntime.InferenceSession(onnx_model_path)

                # To use the ONNX model, the image must first be pre-processed using the SAM image encoder. 
                # This is a heavier weight process best performed on GPU. SamPredictor can be used as normal, 
                # then `.get_image_embedding()` will retreive the intermediate features.
                sam.to(device='cuda')
                predictor = SamPredictor(sam)
                predictor.set_image(image_color)

                image_embedding = predictor.get_image_embedding().cpu().numpy()
                # print('The image embedding is ', image_embedding.shape)

                # define the output class and the corresponding color
                input_point = np.array([[70, 500]])
                input_label = np.array([1])

                # Add a batch index, concatenate a padding point, and transform.
                onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
                onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

                onnx_coord = predictor.transform.apply_coords(onnx_coord, image_color.shape[:2]).astype(np.float32)

                # create empty mask
                onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
                onnx_has_mask_input = np.zeros(1, dtype=np.float32)

                # Package the inputs to run in the onnx model
                ort_inputs = {
                    "image_embeddings": image_embedding,
                    "point_coords": onnx_coord,
                    "point_labels": onnx_label,
                    "mask_input": onnx_mask_input,
                    "has_mask_input": onnx_has_mask_input,
                    "orig_im_size": np.array(image_color.shape[:2], dtype=np.float32)
                }

                # Run the model and obtain the mask
                masks, _, low_res_logits = ort_session.run(None, ort_inputs)
                masks = masks > predictor.model.mask_threshold
                # print('The mask shape is: ', masks.shape)

                # Visualize the results
                # plt.figure(figsize=(10,10))
                # plt.imshow(image_color)
                # mask_image = show_mask(masks, plt.gca())
                # show_points(input_point, input_label, plt.gca())
                # plt.axis('off')
                # plt.show() 

                input_point = np.array([[70, 500], [85, 375], [130, 375], [25, 280], [130, 280]]) # normal
                input_label = np.array([1, 1, 1, 1, 1])
                # input_point = np.array([[70, 500], [85, 375], [130, 375], [25, 370]]) # initial
                # input_label = np.array([1, 1, 1, 1])

                # Use the mask output from the previous run. It is already in the correct form for input to the ONNX model.
                onnx_mask_input = low_res_logits

                onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
                onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

                onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

                onnx_has_mask_input = np.ones(1, dtype=np.float32)

                ort_inputs = {
                    "image_embeddings": image_embedding,
                    "point_coords": onnx_coord,
                    "point_labels": onnx_label,
                    "mask_input": onnx_mask_input,
                    "has_mask_input": onnx_has_mask_input,
                    "orig_im_size": np.array(image_color.shape[:2], dtype=np.float32)
                }

                masks, _, _ = ort_session.run(None, ort_inputs)
                masks = masks > predictor.model.mask_threshold

                # Visualize the results
                plt.figure(figsize=(10,10))
                plt.imshow(image_color)
                mask_image = show_mask(masks, plt.gca())
                show_points(input_point, input_label, plt.gca())
                plt.axis('off')
                # plt.show() 
            
                # save the label
                # Set the path to the rotated image file
                resized_array = np.squeeze(masks).astype(int)
                cv2.imwrite(os.path.join(save_folder, os.path.splitext(filename)[0] + '.bmp'), image)
                cv2.imwrite(os.path.join(save_folder, os.path.splitext(filename)[0] + '_show.png'), resized_array*200)
                cv2.imwrite(os.path.join(save_folder, os.path.splitext(filename)[0] + '.png'), resized_array)
                plt.savefig(os.path.join(save_folder, os.path.splitext(filename)[0] + '_mask.jpg'))

                # clsoe plot
                plt.close()
                print(filename, "success!!!")

                

        print("All images rotated and saved successfully!")
    
    else :
        
        print("Error: Invalid mode!")




if __name__ == "__main__":
    
     # Set prediction mode (1: single image, 2: folder)
    mode = "1" 

    # Set the path to the image file
    # image_path = "C:/Users/ktang/workspace/sewing2d_database/ufld/train/origin_combine_normal/img_1.jpg"
    image_path = "/home/kaitang/workspace/sewing2d_database/source/240424/combined/img_1.bmp"

    # Set the path to the folder containing the images
    # image_folder = "C:/Users/ktang/workspace/sewing2d_database/ufld/train/origin_combine_normal"
    image_folder = "./sewing2d_database/source/240424/combined"

    # Set the path to the folder where the labels will be saved
    # label_save_folder = "C:/Users/ktang/workspace/sewing2d_database/ufld/train/label_combine_normal"
    save_folder = "./sewing2d_database/source/240424/combined"

    # make folde
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    main(mode, image_path, image_folder, save_folder)











