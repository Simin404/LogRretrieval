import os
import torch
import numpy as np
import cv2
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

def remove_description_file(folder_path):
    # data_path="data/video/"
    # remove_description_file(data_path)
    file_name = [str(i).zfill(6) for i in range(0, 1473)]
    for f in file_name:
        spath = folder_path + f + '/camera_front_blur/'
        file_path = os.path.join(spath, 'description.txt')
        # Check if the file exists
        if os.path.exists(file_path):
            try:
                # Remove the file
                os.remove(file_path)
                print(f"{file_path} has been removed successfully.")
            except Exception as e:
                print(f"Error occurred while deleting {file_path}: {e}")
        else:
            print(f"{file_path} does not exist.")


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)
    return device



def video2txt(conversation, data_path, out_path, device):
    # Functionality:
    # This function processes video recordings and samples uniformly 32 frames from the video files within each record.
    # And generates descriptive text based on a given conversation prompt and video images.
    #
    # Arguments:
    # - conversation: A conversation input that will be passed to LLaVA's processor to generate a prompt for description generation.
    # - data_path: The directory path containing folders with video recordings to be processed.
    # - out_path: The directory where the generated descriptions based on the conversation prompt will be saved.
    # - device: The hardware (e.g., GPU or CPU) where the model will be loaded and run.
    
    model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(device)
    processor = LlavaNextVideoProcessor.from_pretrained(model_id)
    prompt = processor.apply_chat_template(conversation)
    file_name = [str(i).zfill(6) for i in range(0, 1473)]
    for f in file_name:
        if (int(f)+1) % 500 == 0:
            print('Processed: {}'.format(int(f)+1))
        spath = data_path + f + '/camera_front_blur/'
        videos=os.listdir(spath)
        sample_step=int(len(videos)/32)
        sampled_video=videos[0::sample_step]
        sampled_video=sampled_video[:32]
        clip  = np.stack([cv2.imread(spath+file) for file in sampled_video])
        inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)
        output = model.generate(**inputs_video, max_new_tokens=300, do_sample=False)
        learned_text=processor.decode(output[0][2:], skip_special_tokens=True)
        with open(f"{out_path}{f}.txt", 'w+') as f:
            f.write(learned_text)