import json
import re
import os
import requests
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd

def save_list_to_file(my_list, file_path):
    """Saves the list to a file in JSON format."""
    with open(file_path, 'w') as file:
        json.dump(my_list, file)
    print("List saved to file.")

def load_list_from_file(file_path):
    """Loads the list from a file in JSON format."""
    with open(file_path, 'r') as file:
        return json.load(file)
    


# directory = 'data/description/prompt_1/'
# search_phrase = "Describe the driving conditions and environments can be seen in the video."
# all_prompt1 = extract_info_from_file(directory, search_phrase)
# save_list_to_file(all_prompt1, 'data/description/text_per_group/prompt_6.json')

def extract_info_from_file(file_path, search_phrase):
    file_name = [str(i).zfill(6) for i in range(0, 1473)]
    all_text = []
    for f in file_name:
        with open(file_path+f+'.txt', 'r') as file:
            content = file.read()

        # Use regex to find the search phrase and extract text after it
        # \s* matches any whitespace characters (spaces, newlines, tabs)
        pattern = re.compile(re.escape(search_phrase) + r'\s*(.*)', re.DOTALL)
        match = pattern.search(content)

        if match:
            extracted_info = match.group(1).strip()
            all_text.append(extracted_info)
        else:
            print('Empty text for {}'.format(f))
            all_text.append('')
    return all_text


def print_curve(one_list, query):
    sorted_data = sorted(one_list)
    plt.figure(figsize=(5, 5))
    plt.plot(range(len(one_list)), sorted_data)
    
    plt.title(query)
    plt.xlabel('Index')
    plt.ylabel('Similarity')

    plt.legend()
    plt.show()


def print_curve_scenario(all_lists):
    sorted_data = [sorted(sub_list) for sub_list in all_lists]
    scenarios = [sorted_data[i:i + 6] for i in range(0, len(sorted_data), 6)]

    n_scenarios = len(scenarios)  
    n_columns = 3  
    n_rows = n_scenarios // n_columns + (n_scenarios % n_columns > 0) 

    fig, axs = plt.subplots(n_rows, n_columns, figsize=(15, 9))
    axs = axs.flatten() 

    for index, group in enumerate(scenarios):
        for sub_index, sub_list in enumerate(group):
            axs[index].plot(sub_list, label=f'Prompt {sub_index + 1}')

        # Adding title and labels for each subplot
        axs[index].set_title(f'Scenario {index + 1}')
        axs[index].set_xlabel('Index')
        axs[index].set_ylabel('Value')
        axs[index].legend(loc = 'lower right', fontsize = 'xx-small')
        axs[index].grid(True)

    for i in range(n_scenarios, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()  
    plt.show()

def print_curve_prompt(all_lists):
    sorted_data = [sorted(sub_list) for sub_list in all_lists]
    prompts = [sorted_data[i:i + 9] for i in range(0, len(sorted_data), 9)]

    n_prompts = len(prompts)  
    n_columns = 3  
    n_rows = n_prompts // n_columns + (n_prompts % n_columns > 0) 

    fig, axs = plt.subplots(n_rows, n_columns, figsize=(15, 6))
    axs = axs.flatten()  

    for index, group in enumerate(prompts):
        for sub_index, sub_list in enumerate(group):
            axs[index].plot(sub_list, label=f'Scenario {sub_index + 1}')

        # Adding title and labels for each subplot
        axs[index].set_title(f'Prompt {index + 1}')
        axs[index].set_xlabel('Index')
        axs[index].set_ylabel('Value')
        axs[index].legend(loc = 'lower right', fontsize = 'xx-small')
        axs[index].grid(True)

    for i in range(n_prompts, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout() 
    plt.show()


def show_images_from_folders(indexes, base_path, outpath, n_c = 6):
    """
    Display the first image from each folder specified by the list of indexes.
    
    :param indexes: List of folder names or indices.
    :param base_path: Base directory where the folders are stored.
    """
    images = []
    ts = time.time()
    for index in indexes:
        folder_path = os.path.join(base_path, str(index))
        folder_path = folder_path + '/camera_front_blur/'
        if os.path.isdir(folder_path ):
            files = os.listdir(folder_path)
            image_files = [f for f in files if f.endswith(('.jpg'))]
            
            if image_files:
                first_image_path = os.path.join(folder_path, image_files[0])
                try:
                    img = Image.open(first_image_path)
                    images.append(img)
                except Exception as e:
                    print(f"Error opening image: {first_image_path}, {e}")
            else:
                print(f"No image files found in folder: {folder_path}")
        else:
            print(f"Folder not found: {folder_path}")
    # Show images using matplotlib
    if images:
        n_cols = n_c 
        n_rows = (len(images) + n_cols -1) // n_cols

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 9))
        axs = axs.ravel()
        if len(images) == 1:
            axs = [axs]  # Make axs iterable for a single image
            
        for i in range(len(images)):
            axs[i].imshow(images[i])
            axs[i].axis('off')  # Hide the axis
        # plt.savefig(outpath+ str(ts) +'.jpg')
        plt.show()
    else:
        print("No images to display.")


def query_api(payload):
    with open('data/hf_token.txt', 'r') as file:
        hf_token = file.read().strip()
    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.post(API_URL, headers=headers, json=payload)

    return response.json()

def search_topN(query, describe_text, top_n = 5):
    retries = 0
    while retries < 5:
        try:
            # Attempt to query the API
            output = query_api({
                "inputs": {
                    "source_sentence": query,
                    "sentences": describe_text
                },
            })
            
            # If the output is not a list, handle it as an error
            if not isinstance(output, list):
                raise ValueError("API response is not a list.")
            result = output
            top_info = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:top_n]
            record_id = [f'{num:06d}' for num in top_info]
            show_images_from_folders(record_id, 'data/video/', 'output/m1/')
            id_str = ', '.join(record_id)
            print(f'The top {top_n} matching records are: {id_str}')
            for i in range(top_n):
                print (record_id[i], ':', describe_text[top_info[i]])
            return result


        except (ValueError, json.JSONDecodeError) as e:
            # Handle specific errors, such as ValueError or invalid JSON format
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            retries += 1
            time.sleep(10)     




def search_topN_combined(query, prompt_text, signal_text, top_n = 5):
    
    describe_text = [a + b for a, b in zip(prompt_text, signal_text)]
    
    retries = 0
    while retries < 5:
        try:
            # Attempt to query the API
            output = query_api({
                "inputs": {
                    "source_sentence": query,
                    "sentences": describe_text
                },
            })
            
            # If the output is not a list, handle it as an error
            if not isinstance(output, list):
                raise ValueError("API response is not a list.")
            result = output
            top_info = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:top_n]
            record_id = [f'{num:06d}' for num in top_info]
            show_images_from_folders(record_id, 'data/video/', 'output/m2/')
            id_str = ', '.join(record_id)
            print(f'The top {top_n} matching records are: {id_str}')
            # for i in range(top_n):
            #     print (record_id[i], ':', describe_text[top_info[i]])
            return result


        except (ValueError, json.JSONDecodeError) as e:
            # Handle specific errors, such as ValueError or invalid JSON format
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            retries += 1
            time.sleep(10)     


def search_topN_seperate(query, prompt_text, signal_text, top_n = 5):

    retries = 0
    sleep_sec = 500
    while retries < 5:
        try:
            # Attempt to query the API
            output_prompt = query_api({
                "inputs": {
                "source_sentence": query,
                "sentences": prompt_text
                },
            })
            output_signal = query_api({
                "inputs": {
                "source_sentence": query,
                "sentences": signal_text
            },
            })
            if not isinstance(output_prompt, list):
                raise ValueError("API response is not a list.")
            if not isinstance(output_signal, list):
                raise ValueError("API response is not a list.")

            output = [a + b for a, b in zip(output_prompt, output_signal)]
            # If the output is not a list, handle it as an error

            result = output
            top_info = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:top_n]
            record_id = [f'{num:06d}' for num in top_info]
            show_images_from_folders(record_id, 'data/video/', 'output/m3/')
            id_str = ', '.join(record_id)
            print(f'The top {top_n} matching records are: {id_str}')
            return result


        except (ValueError, json.JSONDecodeError) as e:
            # Handle specific errors, such as ValueError or invalid JSON format
            print(f"Error occurred: {e}. Retrying in {sleep_sec} seconds...")
            retries += 1
            time.sleep(sleep_sec)
            sleep_sec += 500     




def calculate_statistics(numbers):
    if len(numbers) < 2:
        return {
            "Largest Gap": np.nan,
            "Min Distance": np.nan,
            "Max Distance": np.nan,
            "Range": np.nan,
            "Standard Deviation": np.nan,
            "Relative Largest Gap": np.nan
        }

    # Calculate Largest Gap (max gap between numbers)
    largest_gap = max(numbers[i+1] - numbers[i] for i in range(len(numbers) - 1))

    # Calculate Standard Deviation
    std_dev = np.std(numbers)
    
    # Calculate Maximum Distance (distance between max and min values)
    max_distance = max(numbers)

    # Calculate Minimum Distance (min gap between adjacent sorted numbers)
    min_distance = min(numbers)
    
    # Calculate Range (difference between max and min values)
    range_value = max_distance - min_distance

    
    # Calculate Relative Largest Gap (largest gap / range)
    if range_value == 0:
        relative_largest_gap = 0  # To avoid division by zero
    else:
        relative_largest_gap = largest_gap / range_value
    
    # Return all the calculated values as a dictionary
    return {
        "Largest Gap": largest_gap,
        "Min Distance": min_distance,
        "Max Distance": max_distance,
        "Range": range_value,
        "Standard Deviation": std_dev,
        "Relative Largest Gap": relative_largest_gap
    }

def calculate_multiple_lists(lists):
    results = []
    
    for i, lst in enumerate(lists):
        stats = calculate_statistics(lst)
        stats['Scenario'] = f"Scenario {i+1}"  # Label each list
        results.append(stats)
    
    # Convert the results to a DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns so 'List' comes first
    df = df[['Scenario', 'Largest Gap', 'Min Distance', 'Max Distance', 'Range', 'Standard Deviation', 'Relative Largest Gap']]
    
    return df



def search(query):
    dict_text = {}
    dict_text[4] = load_list_from_file('data/description/text_per_group/prompt_4.json')
    dict_text[0] = load_list_from_file('data/description/text_per_group/signal.json')
    output = search_topN_seperate(query, dict_text[4], dict_text[0], top_n= 6)
    df_metric = calculate_statistics(output)
    print(df_metric)
    print_curve(output, query)