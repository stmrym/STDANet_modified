import streamlit as st
import os
import cv2
import numpy as np
from time import time
from PIL import Image
import importlib
import torch
import torchvision
from mmflow.datasets import visualize_flow

ss = st.session_state

def init_page():
    st.set_page_config(page_title = 'STDAN modified demo')
    st.title('Modified STDANet demo')
    st.sidebar.title('Options')


def select_weight():
    # select ckpt file widget
    model_options = [file for file in os.listdir('./weights') if '.pth.tar' in file]
    selected_weight = st.sidebar.selectbox(
                            label = 'Choose your ckpt file. (weights/*.pth.tar)',
                            options = model_options,
                            disabled = ss.disabled
                            )
    if selected_weight is not None:
        ss.weight = selected_weight


def select_network():
    # select 'STDANet' or 'STDANet_RAFT_Stack'
    selected_network = st.sidebar.radio(
                            label = 'Choose network. (Must be consistent with training settings)', 
                            options = ('STDAN_Stack', 'STDAN_RAFT_Stack'),
                            disabled = ss.disabled
                            )
    
    if selected_network is not None:
        ss.network = selected_network


def select_input_type():
    # select 'video' or 'images'
    input_type = st.sidebar.radio(
                            label = 'Choose your input format.',
                            options = ('video', 'images'),
                            disabled = ss.disabled
                            )
    
    if 'input_type' not in ss or ss.input_type != input_type:
        ss.input_type = input_type
        ss.uploaded_file = None



def load_model(network, weight):
    # Load model from 'network' and 'weight'
    module = importlib.import_module('models.' + network)
    deblurnet = module.__dict__[network]()

    checkpoint = torch.load(os.path.join('weights', weight), map_location='cpu')    
    deblurnet.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['deblurnet_state_dict'].items()})
    deblurnet = torch.nn.DataParallel(deblurnet).cuda()

    
    return deblurnet


def video_inference(weight, network, upload_file):

    # Load model
    deblurnet = load_model(
        network = network,
        weight = weight
    )
    st.text(f'Wights: {weight}, Model: {network} loaded.')

    # Load input frames from a video
    if ss.input_type == 'video':
        video_tensor, _, fps_dict = torchvision.io.read_video(os.path.join('demo_input', upload_file.name))
        input_tensor = (video_tensor.float() / 255).cuda()
        # (n, h, w, c) -> (n, c, h, w)
        input_tensor = input_tensor.permute(0, 3, 1, 2)
        ss.video_fps = int(fps_dict['video_fps'])

        output_dir_name = os.path.splitext(upload_file.name)[0]
        output_frame_names = [f'{i:05}.png'  for i in range(0, (input_tensor.shape)[0])]

    # Load input frames from images
    elif ss.input_type == 'images':
        
        upload_file = sorted(upload_file, key=lambda x: x.name)
        inputs = []
        for file in upload_file:
            image = Image.open(file)
            image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()
            inputs.append(image_tensor)

        input_tensor = torch.stack(inputs, dim = 0)

        output_dir_name = os.path.splitext(upload_file[0].name)[0]
        output_frame_names = [file.name for file in upload_file]
        output_frame_names = output_frame_names[2:-2]

        ss.video_fps = 20


    # Start Inference
    n, c, h, w = input_tensor.shape
    text = st.empty()    
    progress_bar = st.progress(0)
    col1, col2 = st.columns(2)
    output_frames = []
    flow_maps = []
    deblurnet.eval()

    # Saving settings
    os.makedirs(f'demo_output/{output_dir_name}', exist_ok = True)
    os.makedirs(f'demo_output/{output_dir_name}_flow', exist_ok = True)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fourcc2 = cv2.VideoWriter_fourcc(*'avc1')
    output_video = cv2.VideoWriter(f'demo_output/{output_dir_name}.mp4',fourcc, ss.video_fps, (w, h))
    flow_video = cv2.VideoWriter(f'demo_output/{output_dir_name}_flow.mp4',fourcc2, ss.video_fps, (w, h))


    # Calculating inference time
    torch.cuda.synchronize()
    process_start_time = time()

    with torch.no_grad():
        for i in range(n - 4):

            torch.cuda.synchronize()
            process_time = time() - process_start_time

            text.text(f'Processing frame : {i + 1} / {n - 4}  Inference time : {process_time} [s]')
            progress_bar.progress((i + 1)/(n - 4))
            
            input_frame = input_tensor[i:i+5, :, :, :]
            input_frame = input_frame.unsqueeze(dim = 0)
            # input_frame (1, 5, c, h, w)

            # Inference
            output_dict = deblurnet(input_frame)

            # Transform tensor to numpy
            output_image = output_dict['out'].cpu().detach()*255
            output_image = output_image[0].permute(1,2,0).numpy().copy()
            output_image = np.clip(output_image, 0, 255).astype(np.uint8)
            

            output_flow = ((output_dict['flow_forwards'])[-1])[0][1].permute(1,2,0).cpu().detach().numpy()   
            flow_map = visualize_flow(output_flow, None)
            flow_map = cv2.resize(flow_map, (w, h), interpolation = cv2.INTER_NEAREST)
            # Show output and flow
            with col1:
                st.write(output_frame_names[i])
                st.image(output_image)
            with col2:
                st.write(output_frame_names[i])
                st.image(flow_map)

            # Saving images and video
            cv2.imwrite(f'demo_output/{output_dir_name}/{output_frame_names[i]}', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'demo_output/{output_dir_name}_flow/{output_frame_names[i]}', cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR))            

            output_video.write(cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            flow_video.write(cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR))  

            output_frames.append(output_image)    
            flow_maps.append(flow_map)    


    output_video.release()
    flow_video.release()
    ss.flow_maps = flow_maps
    return output_frames, output_dir_name, output_frame_names

    

def show_video_or_images():
    # Show video or images depending on input type
    if ss.input_type == 'video':
        st.write(ss.uploaded_file.name)
        st.video(ss.uploaded_file)

    elif ss.input_type == 'images':
        ss.uploaded_file = sorted(ss.uploaded_file, key=lambda x: x.name)

        for i, image in enumerate(ss.uploaded_file):
            st.write(ss.uploaded_file[i].name)
            st.image(image, use_column_width=True)


def show_result_video():
    # Show result video
    if ss.input_type == 'video':
    
        st.header('Input')
        st.video(ss.uploaded_file)
        st.header('Output')
        st.video(f'demo_output/{ss.output_dir_name}.mp4')
        st.header('Flow map')
        st.video(f'demo_output/{ss.output_dir_name}_flow.mp4')


def show_result_multi_images(type_list):
    # Show image columns
    if ss.input_type == 'images':
        num_column = len(type_list)
        if num_column != 0:
            column_list = st.columns(num_column)
            for col, image_type in zip(column_list, type_list):
                
                if image_type == 'Input':
                    frames = sorted(ss.uploaded_file, key=lambda x: x.name)[2:-2]
                    frame_names = [frame.name for frame in frames]
                elif image_type == 'Output':
                    frames = ss.output_frames
                    frame_names = ss.output_frame_names
                elif image_type == 'Flow map':
                    frames = ss.flow_maps
                    frame_names = ss.output_frame_names

                # display each type of images
                with col:
                    st.header(image_type)
                    for image_name, image in zip(frame_names, frames):
                        st.write(image_name)
                        st.image(image, use_column_width=True)
            


def set_session_start():
    # Jump to processing page
    if (ss.input_type == 'video') or (ss.input_type == 'images' and len(ss.uploaded_file) >= 5):
        ss.disabled = True
        ss.page_control = 'processing'
        ss.show_result_disable = True

def set_session_exit():
    # Jump to start page
    ss.disabled = False
    ss.page_control = 'start'

def set_session_finish():
    # Jump to finish page
    ss.page_control = 'finished'

def upload_wiget():
    if ss.input_type == 'video':
        uploaded_file = st.file_uploader(
                            label = 'Upload your video file. (demo_input/*)',
                            type = 'mp4',
                            disabled = ss.disabled
                            )
        if (uploaded_file is not None):
            ss.uploaded_file = uploaded_file
    
    if ss.input_type == 'images':
        uploaded_file = st.file_uploader(
                            label = 'Upload your image files. (demo_input/*) (more than 5 images.)',
                            accept_multiple_files = True,
                            type = ['png', 'jpg', 'jpeg'],
                            disabled = ss.disabled
                            )
        if (uploaded_file != []):
            ss.uploaded_file = uploaded_file



def start_page():
    # Show input image
    if (ss.uploaded_file is not None) and (ss.uploaded_file != []):
        st.button('Start Deblurring', type = 'primary', on_click = set_session_start)
        show_video_or_images()
        
        

def processing_page():
    # Processing deblurring page
    col1, col2 = st.columns(2)
    with col1:
        st.button('Stop', on_click = set_session_exit)
    with col2:
        result = st.empty()

    output_frames, output_dir_name, output_frame_names = video_inference(
        weight = ss.weight,
        network = ss.network,
        upload_file = ss.uploaded_file
    )
    ss.output_frames = output_frames
    ss.output_dir_name = output_dir_name
    ss.output_frame_names = output_frame_names
    with col2:
        result.button('View results', type = 'primary', on_click = set_session_finish)


def finished_page():
    # Show result images
    st.button('End', on_click = set_session_exit)
    st.write(f'Results saved to "demo_output/{ss.output_dir_name}".')

    if ss.input_type == 'video':
        show_result_video()
    
    elif ss.input_type == 'images':
        type_list = st.multiselect(
            label = 'Type of image to display.',
            options = ['Input', 'Output', 'Flow map'],
            default = ['Input', 'Output', 'Flow map']
            )
        show_result_multi_images(type_list)



def main():

    # Initialize
    if 'disabled' not in ss:
        ss.disabled = False
    
    # Initialize
    if 'page_control' not in ss:
        ss.page_control = 'start'
    
    init_page()
    select_weight()
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