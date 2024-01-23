import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import importlib
import torch
import torchvision


def init_page():
    st.set_page_config(page_title = 'STDAN modified demo')
    st.title('Modified STDANet demo')
    st.sidebar.title('Options')


def select_weight():

    model_options = [file for file in os.listdir('./weights') if '.pth.tar' in file]
    selected_weight = st.sidebar.selectbox(
                            label = 'Choose your ckpt file. (weights/*.pth.tar)',
                            options = model_options,
                            disabled = st.session_state.disabled
                            )
    if selected_weight is not None:
        st.session_state.weight = selected_weight


def select_network():

    selected_network = st.sidebar.radio(
                            label = 'Choose network.', 
                            options = ('STDAN_Stack', 'STDAN_RAFT_Stack'),
                            disabled = st.session_state.disabled
                            )
    
    st.session_state.network = selected_network


def select_input_type():

    input_type = st.sidebar.radio(
                            label = 'Choose video format.',
                            options = ('video', 'images'),
                            disabled = st.session_state.disabled
                            )
    
    if 'input_type' not in st.session_state or st.session_state.input_type != input_type:
        st.session_state.input_type = input_type
        st.session_state.uploaded_file = None



def load_model(selected_weight, selected_network):

    module = importlib.import_module('models.' + selected_network)
    deblurnet = module.__dict__[selected_network]()

    checkpoint = torch.load(os.path.join('weights', selected_weight), map_location='cpu')    
    deblurnet.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['deblurnet_state_dict'].items()})
    deblurnet = torch.nn.DataParallel(deblurnet).cuda()
    st.text(f'Wights: {selected_weight}, Model: {selected_network} loaded.')
    
    return deblurnet


def video_inference(deblurnet, video_path):
    
    # video_tensor (n, h, w, c)
    video_tensor = torchvision.io.read_video(os.path.join('demo_input', video_path))[0]
    video_tensor = (video_tensor.float() / 255).cuda()
    n, h, w, c = video_tensor.shape

    text = st.empty()    
    progress_bar = st.progress(0)

    output_frames = []
    deblurnet.eval()
    
    with torch.no_grad():
        for i in range(n - 4):

            text.text(f'Processing frame: {i + 2} / {n - 3}')
            progress_bar.progress((i + 1)/(n - 4))
            
            input_frame = video_tensor[i:i+5, :, :, :].permute(0, 3, 1, 2)
            input_frame = input_frame.unsqueeze(dim = 0)
            # input_frame (1, 5, c, h, w)

            output_dict = deblurnet(input_frame)

            output_image = output_dict['out'].cpu().detach()*255
            output_image = output_image[0].permute(1,2,0).numpy().copy()
            output_image_bgr = cv2.cvtColor(np.clip(output_image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            output_frames.append(output_image_bgr)        

    return output_frames

def image2video(frames):

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # output file name, encoder, fps, size(fit to image size)
    h, w, c = frames[0].shape
    
    video = cv2.VideoWriter('demo_input/video.mp4',fourcc, 20.0, (h, w), True)
    i = 0
    for frame in frames:
        # add
        cv2.imwrite(f'demo_input/img_{i}.png', frame)
        video.write(frame)
        i += 1

    video.release()
    st.write('Converted.')


def show_video(video):
    if st.session_state.input_type == 'video':
        st.video(video)


def set_session_start():
    st.session_state.disabled = True
    st.session_state.page_control = 'processing'

def set_session_exit():
    st.session_state.disabled = False
    st.session_state.page_control = 'start'

def set_session_finish():
    st.session_state.page_control = 'finished'

def choose_wiget():
    if st.session_state.input_type == 'video':
        uploaded_file = st.file_uploader(
                            label = 'Upload your video file. (demo_input/*)',
                            type = 'mp4',
                            disabled = st.session_state.disabled
                            )
    
    if st.session_state.input_type == 'images':
        uploaded_file = st.file_uploader(
                            label = 'Upload your image files. (demo_input/*)',
                            accept_multiple_files = True,
                            type = ['png', 'jpg', 'jpeg'],
                            disabled = st.session_state.disabled
                            )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file


def start_page():

    if (st.session_state.uploaded_file is not None) and (st.session_state.uploaded_file != []):
        st.write(st.session_state.uploaded_file.name)
        show_video(st.session_state.uploaded_file)
        st.button('Start Deblurring', type = 'primary', on_click = set_session_start)


def processing_page():

    st.write(st.session_state.uploaded_file.name)
    show_video(st.session_state.uploaded_file)
    st.button('Stop', on_click = set_session_exit)

    deblurnet = load_model(
                    selected_weight = st.session_state.weight,
                    selected_network = st.session_state.network
                    )
    
    output_frames = video_inference(
                            deblurnet = deblurnet,
                            video_path = st.session_state.uploaded_file.name
                            )
    
    output_video = image2video(output_frames)

    st.button('View results', type = 'primary', on_click = set_session_finish)
    st.session_state.output = output_video


def finished_page():

    input_col, output_col = st.columns([1,1])
    with input_col:
        st.header('Input')
        show_video(st.session_state.uploaded_file)
    with output_col:
        st.header('Output')
        show_video(st.session_state.output)

    st.button('End', on_click = set_session_exit)


def main():
    if 'disabled' not in st.session_state:
        # Initialize
        st.session_state.disabled = False
    
    if 'page_control' not in st.session_state:
        st.session_state.page_control = 'start'
    
    
    init_page()
    select_weight()
    select_network()
    select_input_type()
    choose_wiget()

    if st.session_state.page_control == 'start':
        start_page()
    elif st.session_state.page_control == 'processing':
        processing_page()
    elif st.session_state.page_control == 'finished':
        finished_page()


            




if __name__ == '__main__':
    main()