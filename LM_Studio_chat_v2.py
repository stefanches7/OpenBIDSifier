import requests
import json
import os

url = "http://localhost:1234/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}

# Initialize conversation history with the system message
conversation_history = [
    {
        "role": "system",
        
        "content": (
    "You are an assistant responsible for constructing a REAL BIDS-compliant "
    "dataset_description.json file. This is not a theoretical exercise. "
    "You must:\n"
    "1. Build correct JSON based on user-provided information.\n"
    "2. Ask for missing information if needed—never assume.\n"
    "3. When processing files, analyze only what you know or can infer and "
    "request more details if information is insufficient.\n"
    "4. Never invent fields, values, metadata, or file contents.\n"
    "5. Always output structured JSON or a clear description of missing "
    "required information.\n"
    )

    }
]

# Step 1: Get basic dataset information
def get_basic_dataset_info():
    print("Please provide the basic details of your dataset:")
    # dataset_name = input("Dataset Name: ")
    # dataset_version = input("Dataset Version: ")
    # dataset_description = input("Dataset Description: ")


    dataset_name = "Brain Tumor Segmentation(BraTS2020)";
    dataset_version = "7.06"
    dataset_description = """About Dataset
Context

BraTS has always been focusing on the evaluation of state-of-the-art methods for the segmentation of brain tumors in multimodal magnetic resonance imaging (MRI) scans. BraTS 2020 utilizes multi-institutional pre-operative MRI scans and primarily focuses on the segmentation (Task 1) of intrinsically heterogeneous (in appearance, shape, and histology) brain tumors, namely gliomas. Furthemore, to pinpoint the clinical relevance of this segmentation task, BraTS’20 also focuses on the prediction of patient overall survival (Task 2), and the distinction between pseudoprogression and true tumor recurrence (Task 3), via integrative analyses of radiomic features and machine learning algorithms. Finally, BraTS'20 intends to evaluate the algorithmic uncertainty in tumor segmentation (Task 4).
Tasks' Description and Evaluation Framework

In this year's challenge, 4 reference standards are used for the 4 tasks of the challenge:

    Manual segmentation labels of tumor sub-regions,
    Clinical data of overall survival,
    Clinical evaluation of progression status,
    Uncertainty estimation for the predicted tumor sub-regions.

Imaging Data Description

All BraTS multimodal scans are available as NIfTI files (.nii.gz) and describe a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes, and were acquired with different clinical protocols and various scanners from multiple (n=19) institutions, mentioned as data contributors here.

All the imaging datasets have been segmented manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuro-radiologists. Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1), as described both in the BraTS 2012-2013 TMI paper and in the latest BraTS summarizing paper. The provided data are distributed after their pre-processing, i.e., co-registered to the same anatomical template, interpolated to the same resolution (1 mm^3) and skull-stripped.
Dataset Description

All the slices of volumes have been converted to hdf5 format for saving memory. Metadata contains volume_no, slice_no , and target of that slice.
Use of Data Beyond BraTS

Participants are allowed to use additional public and/or private data (from their own institutions) for data augmentation, only if they also report results using only the BraTS'20 data and discuss any potential difference in their papers and results. This is due to our intentions to provide a fair comparison among the participating methods.
Data Usage Agreement / Citations:

You are free to use and/or refer to the BraTS datasets in your own research, provided that you always cite the following three manuscripts:

[1] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[2] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

[3] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)

In addition, if there are no restrictions imposed from the journal/conference you submit your paper about citing "Data Citations", please be specific and also cite the following:

[4] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q

[5] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF"
    
"""
    return {
        "name": dataset_name,
        "version": dataset_version,
        "description": dataset_description
    }

# Step 2: Get the root folder where the files are located
def get_root_folder():
    folder = r"C:\Users\lulky\Desktop\AI-assisted-Neuroimaging-harmonization\Non_Bids_Dataset\archive\BraTS2020_training_data\content"
    # input("Please provide the root folder containing the dataset files: ")
    
    while not os.path.isdir(folder):
        print("Invalid folder. Please provide a valid path.")
        folder = input("Please provide the root folder containing the dataset files: ")
    return folder

# # Step 3: Process the files in the root folder
# def process_files_in_folder(folder):
#     files = os.listdir(folder)
#     print(folder);
#     print("-----------------------------");
#     relevant_files = []
    
#     for file in files:
#         file_path = os.path.join(folder, file)
#         if os.path.isfile(file_path):
#             relevant_files.append(file_path)
    
#     return relevant_files

class Color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def scan_dataset_tree(root_folder):
    file_paths = []
    tree_lines = []

    print(f"{Color.BOLD}{Color.BLUE}\nScanning dataset folder recursively...{Color.END}")

    for current_path, dirs, files in os.walk(root_folder):
        depth = current_path.replace(root_folder, "").count(os.sep)
        indent = "    " * depth

        folder_name = os.path.basename(current_path)
        tree_lines.append(f"{indent}{folder_name}/")

        print(f"{Color.GREEN}{indent}{folder_name}/{Color.END}")

        for f in files:
            file_full = os.path.join(current_path, f)
            file_paths.append(file_full)
            tree_lines.append(f"{indent}    {f}")
            print(f"{Color.YELLOW}{indent}    {f}{Color.END}")

    tree_string = "\n".join(tree_lines)
    print(f"{Color.BOLD}{Color.CYAN}\nCompleted folder scan.\n{Color.END}")

    return file_paths, tree_string


# Step 4: Process each file with AI and update dataset_description.json
def process_and_build_json(files, tree_summary, basic_info):
    dataset_description = {
        "Name": basic_info["name"],
        "BIDSVersion": "1.0.0",
        "DatasetType": "raw",
        "License": "CC0",
        "Authors": ["Author1"],
        "DatasetDescription": basic_info["description"]
    }

    print(f"{Color.BOLD}{Color.HEADER}\n=== Sending Dataset Tree to LLM ===\n{Color.END}")
    print(tree_summary)
    print(f"{Color.BOLD}{Color.HEADER}\n===================================\n{Color.END}")

    # Step 1: Ask the LLM to analyze the entire dataset structure
    conversation_history.append({
        "role": "user",
        "content": (
            "We are constructing a REAL BIDS dataset_description.json.\n"
            "Below is the dataset directory structure:\n\n"
            f"{tree_summary}\n\n"
            "Please identify which files are relevant for dataset_description.json.\n"
            "Also list any missing metadata you need.\n"
            "Do NOT guess missing information."
        )
    })

    data = {
        "model": "deepseek/deepseek-r1-0528-qwen3-8b",
        "messages": conversation_history,
        "temperature": 0.2,
        "max_tokens": 700,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    llm_response = response.json()['choices'][0]['message']['content']

    print(f"{Color.BOLD}{Color.CYAN}\n=== LLM Analysis of Dataset Tree ==={Color.END}")
    print(llm_response)
    print(f"{Color.BOLD}{Color.CYAN}\n==================================={Color.END}")

    conversation_history.append({"role": "assistant", "content": llm_response})

    # Step 2: Now process each file individually
    for file in files:

        print(f"{Color.BOLD}{Color.BLUE}\n\n--- Processing file with LLM ---{Color.END}")
        print(f"{Color.YELLOW}FILE:{Color.END} {file}")

        conversation_history.append({
            "role": "user",
            "content": (
                f"Here is a NonBids dataset, Based on the dataset structure:\n\n{tree_summary}\n\n"
                f"Process this file: {file}\n"
                f"Tell us:\n"
                f"1. Whether the file is relevant to BIDS dataset_description.json\n"
                f"2. What metadata it provides\n"
                f"3. What metadata is missing and needs user input\n"
                f"Do NOT assume missing information."
            )
        })

        data["messages"] = conversation_history

        response = requests.post(url, headers=headers, data=json.dumps(data))
        model_reply = response.json()['choices'][0]['message']['content']

        # Print LLM output for this file
        print(f"{Color.GREEN}\nLLM Response for file:{Color.END} {file}")
        print(f"{Color.CYAN}{model_reply}{Color.END}")
        print(f"{Color.GREEN}------------------------------------{Color.END}")

        conversation_history.append({"role": "assistant", "content": model_reply})

        dataset_description[file] = model_reply

    return dataset_description




# Step 5: Save the dataset_description.json
def save_json(dataset_description):
    output_file = "dataset_description.json"
    with open(output_file, "w") as json_file:
        json.dump(dataset_description, json_file, indent=4)
    print(f"Dataset description saved as {output_file}")

# Main logic to execute the workflow
def main():
    basic_info = get_basic_dataset_info()
    root_folder = get_root_folder()

    # NEW: recursive scan + visual print
    files, tree_summary = scan_dataset_tree(root_folder)

    dataset_description = process_and_build_json(files, tree_summary, basic_info)

    save_json(dataset_description)


# Start the process
if __name__ == "__main__":
    main()
