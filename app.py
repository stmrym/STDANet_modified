import streamlit as st
import os
import glob
import cv2
import numpy as np
from time import time
from datetime import datetime
from PIL import Image
import tempfile
from models.Stack import Stack
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torchvision
from mmflow.datasets import visualize_flow
from typing import List, Tuple
from utils import util
from losses.multi_loss import *

ss = st.session_state

edge_dict = {   'Motion Edge':
                {   'dirname':'MotionEdge',
                    'output_type':'weighted',
                    'function_name':'motion_weighted_edge_extraction'
                },
            
                'Orthogonal Edge':
                {   'dirname':'OrthoEdge',
                    'output_type':'abs_weight',
                    'function_name':'orthogonal_edge_extraction'
                },
                
                'Sobel Edge':
                {   'dirname':'SobelEdge',
                    'output_type':'edge',
                    'function_name':'orthogonal_edge_extraction'}
}

def init_page() -> None:
    # Show initial page
    st.set_page_config(page_title = 'STDAN modified demo')
    st.title('Modified STDANet demo')
    st.sidebar.title('Options')


def select_weight() -> None:
    # Select ckpt file widget
    model_options = [file for file in os.listdir('./weights') if '.pth.tar' in file]
    selected_weight = st.sidebar.selectbox(
                            label = 'Choose your ckpt file. (weights/*.pth.tar)',
                            options = model_options,
                            disabled = ss.disabled
                            )
    if selected_weight is not None:
        ss.weight = selected_weight

def select_additional_save_imge() -> None:
    # Select which edge images save
    selected_add_image_type = st.sidebar.selectbox(
                            label = 'Choose additional save image type if you want.',
                            options = ['No save'] + list(edge_dict.keys()),
                            disabled = ss.disabled
                            )
    if selected_add_image_type is not None:
        ss.selected_add_image_type = selected_add_image_type


def select_network() -> None:
    # Select models
    selected_network = st.sidebar.radio(
                            label = 'Choose network. (Must be consistent with training settings)', 
                            options = tuple([os.path.splitext(f)[0].split('/')[-1] for f in sorted(glob.glob('models/model/*STDAN*.py'))]),
                            disabled = ss.disabled
                            )
    
    if selected_network is not None:
        ss.network = selected_network


def select_input_type() -> None:
    # Select 'video' or 'images'
    input_type = st.radio(
                            label = 'Choose your input format.',
                            options = ('video', 'images'),
                            disabled = ss.disabled
                            )
    
    if 'input_type' not in ss or ss.input_type != input_type:
        ss.input_type = input_type
        ss.uploaded_file = None


def upload_wiget() -> None:
    # Upload wiget of 'video' or 'images'
    if ss.input_type == 'video':
        uploaded_file = st.file_uploader(
                            label = 'Upload your video file. (.mp4)',
                            type = 'mp4',
                            disabled = ss.disabled
                            )
        if (uploaded_file is not None):
            ss.uploaded_file = uploaded_file
    
    if ss.input_type == 'images':
        uploaded_files = st.file_uploader(
                            label = 'Upload your image files. (.png, .jpg, .jpeg) (more than 5 images.)',
                            accept_multiple_files = True,
                            type = ['png', 'jpg', 'jpeg'],
                            disabled = ss.disabled
                            )
        if (uploaded_files != []):
            ss.uploaded_file = uploaded_files



def read_frames(upload_file) -> Tuple[np.ndarray, str, List[str]]:

    # Load uploaded file and read frame to convert tensor
    # Load input frames from a video
    if ss.input_type == 'video':

        # write upload_file to temp_file and read video
        with tempfile.NamedTemporaryFile() as tfile:
            tfile.write(upload_file.read())
            cap = cv2.VideoCapture(tfile.name)
            ss.video_fps = cap.get(cv2.CAP_PROP_FPS)

            inputs_list = []
            while True:
                ret, frame = cap.read()          
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inputs_list.append(frame)

        output_dir_name = datetime.now().strftime('%Y%m%dT%H%M%S') + '_' + ss.weight.split('.')[0] + '_' + os.path.splitext(upload_file.name)[0]    # e.g., 'ckpt-epoch-0390_000'
        output_frame_names = [f'{i:05}.png'  for i in range(0, len(inputs_list))]  # e.g., ['002.png', '003.png', '004.png',...]

    # Load input frames from images
    elif ss.input_type == 'images':
        
        upload_file = sorted(upload_file, key=lambda x: x.name)
        inputs_list = []
        for file in upload_file:
            image = Image.open(file)
            # (h, w, c)
            image = np.array(image)
            inputs_list.append(image)

        output_dir_name = datetime.now().strftime('%Y%m%dT%H%M%S') + '_' + ss.weight.split('.')[0] + '_' + os.path.splitext(upload_file[0].name)[0]  # e.g., 'ckpt-epoch-0390_000'
        output_frame_names = [file.name for file in upload_file]
        output_frame_names = output_frame_names[2:-2]   # e.g., ['002.png', '003.png', '004.png',...]
        ss.video_fps = 20

    return inputs_list, output_dir_name, output_frame_names
    

def convert_input_tesnor_list(input_list: List[np.ndarray]) -> List[torch.Tensor]:
    # Convert np.array list to input tensor list

    # [np.arrray(h, w, c), ...] -> torch.Tensor(n, h, w, c)
    inputs = np.stack(input_list)
    inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2).cuda()
    inputs = inputs.float() / 255
    # (1, n, c, h, w)
    input_tensor = torch.unsqueeze(inputs, dim = 0)

    num_frame = input_tensor.shape[1]
    input_tensor_list = []
    for i in range(0, num_frame - 4):
        input_tensor_list.append(input_tensor[:, i:i+5, :, :, :])

    return input_tensor_list

def load_model(network:str, weight:str):
    # Load model from 'network' and 'weight'
    # deblurnet = Stack(
    #     network_arch = network, 
    #     use_stack = cfg.NETWORK.USE_STACK, 
    #     n_sequence = cfg.NETWORK.INPUT_LENGTH, 
    #     in_channels = cfg.NETWORK.INPUT_CHANNEL,
    #     n_feat = cfg.NETWORK.NUM_FEAT,
    #     out_channels = cfg.NETWORK.OUTPUT_CHANNEL,
    #     n_resblock = cfg.NETWORK.NUM_RESBLOCK,
    #     kernel_size = cfg.NETWORK.KERNEL_SIZE,
    #     sobel_out_channels = cfg.NETWORK.SOBEL_OUT_CHANNEL,
    #     device = device
    # )
    deblurnet = Stack(
        network_arch = network, 
        use_stack = True, 
        n_sequence = 5, 
        in_channels = 3,
        n_feat = 32,
        out_channels = 3,
        n_resblock = 3,
        kernel_size = 5,
        sobel_out_channels = 2,
        device = 'cuda:0'
    )

    checkpoint = torch.load(os.path.join('weights', weight), map_location='cpu')    
    deblurnet.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['deblurnet_state_dict'].items()})
    deblurnet = torch.nn.DataParallel(deblurnet).cuda()

    return deblurnet


def video_inference(weight:str, network:str, upload_file) -> None:

    # Load model
    deblurnet = load_model(
        network = network,
        weight = weight
    )
    st.text(f'Weights: {weight}, Network: {network} loaded.')

    # Get input tensor and output file name
    inputs_list, output_dir_name, output_frame_names = read_frames(upload_file)
    

    # Saving settings
    os.makedirs(f'demo_output/{output_dir_name}/Input', exist_ok = True)
    os.makedirs(f'demo_output/{output_dir_name}/Output', exist_ok = True)
    os.makedirs(f'demo_output/{output_dir_name}/Flow', exist_ok = True)
    
    if ss.selected_add_image_type != 'No save':
        os.makedirs(f'demo_output/{output_dir_name}/{edge_dict[ss.selected_add_image_type]["dirname"]}', exist_ok = True)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fourcc2 = cv2.VideoWriter_fourcc(*'avc1')
    h, w, _ = inputs_list[0].shape
    output_video = cv2.VideoWriter(f'demo_output/{output_dir_name}/Output.mp4',fourcc, ss.video_fps, (w, h))
    flow_video = cv2.VideoWriter(f'demo_output/{output_dir_name}/Flow.mp4',fourcc2, ss.video_fps, (w, h))
    
    # Saving input images
    for input_image, frame_name in zip(inputs_list[2:-2], output_frame_names):
        cv2.imwrite(f'demo_output/{output_dir_name}/Input/{frame_name}', cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))

    # Convert ndarray to input tensor list
    input_tensor_list = convert_input_tesnor_list(inputs_list)

    # Start Inference
    text = st.empty()    
    progress_bar = st.progress(0)
    col1, col2 = st.columns(2)
    deblurnet.eval()

    # Calculating inference time
    torch.cuda.synchronize()
    process_start_time = time()

    with torch.no_grad():
        for i, (input_tensor, output_frame_name) in enumerate(zip(input_tensor_list, output_frame_names)):

            torch.cuda.synchronize()
            process_time = time() - process_start_time

            text.text(f'Processing frame : {i + 1} / {len(input_tensor_list)}  Inference time : {process_time} [s]')
            progress_bar.progress((i + 1)/(len(input_tensor_list)))

            # Inference
            output_dict = deblurnet(input_tensor)

            # Transform tensor to numpy
            output_image = output_dict['out']['final'].cpu().detach()*255
            output_image = output_image[0].permute(1,2,0).numpy().copy()
            output_image = np.clip(output_image, 0, 255).astype(np.uint8)
            
            output_flow = (output_dict['flow_forwards']['final'])[0][1].permute(1,2,0).cpu().detach().numpy()   
            flow_map = visualize_flow(output_flow, None)
            flow_map = cv2.resize(flow_map, (w, h), interpolation = cv2.INTER_NEAREST)
            # Show output and flow
            with col1:
                st.write(output_frame_name)
                st.image(output_image)
            with col2:
                st.write(output_frame_name)
                st.image(flow_map)

            # Saving images and video
            cv2.imwrite(f'demo_output/{output_dir_name}/Output/{output_frame_name}', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'demo_output/{output_dir_name}/Flow/{output_frame_name}', cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR))   
            
            # Saving each edge image
            if ss.selected_add_image_type != 'No save':
                # util.save_edge(
                #     savename = os.path.join('demo_output', output_dir_name, edge_dict[ss.selected_add_image_type]['dirname'], output_frame_name),
                #     out_image = output_dict['out']['final'],
                #     flow_tensor = output_dict['flow_forwards']['final'][:,1,:,:,:],
                #     key = edge_dict[ss.selected_add_image_type]['output_type'],
                #     edge_extraction_func = eval(edge_dict[ss.selected_add_image_type]['function_name']))
    
                util.save_edge(
                    savename = os.path.join('demo_output', output_dir_name, edge_dict[ss.selected_add_image_type]['dirname'], output_frame_name),
                    out_image = input_tensor[:,1],
                    flow_tensor = output_dict['flow_forwards']['final'][:,1,:,:,:],
                    key = edge_dict[ss.selected_add_image_type]['output_type'],
                    edge_extraction_func = eval(edge_dict[ss.selected_add_image_type]['function_name']))

            output_video.write(cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            flow_video.write(cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR))  


    output_video.release()
    flow_video.release()
    del deblurnet
    ss.output_dir_name = output_dir_name
    ss.output_frame_names = output_frame_names
    

def show_video_or_images() -> None:
    # Show video or images depending on input type
    if ss.input_type == 'video':
        st.write(ss.uploaded_file.name)
        st.video(ss.uploaded_file)

    elif ss.input_type == 'images':
        ss.uploaded_file = sorted(ss.uploaded_file, key=lambda x: x.name)

        for i, image in enumerate(ss.uploaded_file):
            st.write(ss.uploaded_file[i].name)
            st.image(image, use_column_width=True)


def show_results() -> None:
    # Show video and images tabs
    video_tab, image_tab = st.tabs(['🎥 Video', '📷 Images'])

    # Show video
    with video_tab:
        if ss.input_type == 'video':
            st.header('Input')
            st.video(ss.uploaded_file)
        st.header('Output')
        st.video(f'demo_output/{ss.output_dir_name}/Output.mp4')
        st.header('Flow')
        st.video(f'demo_output/{ss.output_dir_name}/Flow.mp4')
    
    # Show image column
    with image_tab:
        path = os.path.join('demo_output', ss.output_dir_name)
        result_options = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        type_list = st.multiselect(
                label = 'Type of image to display.',
                options = result_options,
                default = result_options
        )

        num_column = len(type_list)
        if num_column != 0:
            column_list = st.columns(num_column)
            # Read and display each image
            for col, image_type in zip(column_list, type_list):
                img_list = sorted(glob.glob(os.path.join('demo_output', ss.output_dir_name, image_type, '*')))
                # Read frames
                frames = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]
                frame_names = ss.output_frame_names
                # Display each type of images
                with col:
                    st.header(image_type)
                    for image_name, image in zip(frame_names, frames):
                        st.write(image_name)
                        st.image(image, use_column_width=True)
            


def set_session_start() -> None:
    # Jump to processing page
    if (ss.input_type == 'video') or (ss.input_type == 'images' and len(ss.uploaded_file) >= 5):
        ss.disabled = True
        ss.page_control = 'processing'
        ss.show_result_disable = True

def set_session_exit() -> None:
    # Jump to start page
    ss.disabled = False
    ss.page_control = 'start'

def set_session_finish() -> None:
    # Jump to finish page
    ss.page_control = 'finished'



def start_page() -> None:
    # Show input image
    if (ss.uploaded_file is not None) and (ss.uploaded_file != []):
        st.button('Start Deblurring', type = 'primary', on_click = set_session_start)
        show_video_or_images()
        
        

def processing_page() -> None:
    # Processing deblurring page
    col1, col2 = st.columns(2)
    with col1:
        st.button('Stop', on_click = set_session_exit)
    with col2:
        result = st.empty()

    video_inference(weight = ss.weight, network = ss.network, upload_file = ss.uploaded_file)
    
    with col2:
        result.button('View results', type = 'primary', on_click = set_session_finish)


def finished_page() -> None:
    # Show result images
    st.button('Return to start', on_click = set_session_exit)
    st.write(f'Results saved to "demo_output/{ss.output_dir_name}".')

    show_results()



def main() -> None:

    # Initialize
    if 'disabled' not in ss:
        ss.disabled = False
    
    # Initialize
    if 'page_control' not in ss:
        ss.page_control = 'start'
    
    init_page()
    select_weight()
    select_additional_save_imge()
    select_network()
    select_input_type()
    upload_wiget()

    if ss.page_control == 'start':
        start_page()
    elif ss.page_control == 'processing':
        processing_page()
    elif ss.page_control == 'finished':
        finished_page()



if __name__ == '__main__':
    main()