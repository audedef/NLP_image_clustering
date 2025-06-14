import random
import ollama
import base64
from PIL import Image
import io
import json
import os
import re
import csv
import logging
from datetime import datetime

# Configuration
# The paper uses LLaVA NeXT Video 7B for MLLM but we use a lighter model for LLM than in the paper
MLLM_MODEL = 'ManishThota/llava_next_video:latest'
LLM_MODEL = 'llama3.1:8b'

# For Caption-based Proposer: process captions in subsets to avoid LLM context limits
# Paper mentions 400 for 128k context and llama3 has 128k context
CAPTIONS_PER_SUBSET = 400


# Helper function

def image_to_base64(image_path):
    # Converts an image to a base64 encoded string
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_byte = buffered.getvalue()
            return base64.b64encode(img_byte).decode('utf-8')
    except FileNotFoundError:
        logger.error(f"Error: Image not found at {image_path}")
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
    return None


def ollama_chat_call(model_name, system_prompt, user_content, images=None, format_json=False):
    # Make calls to Ollama chat endpoint
    messages = [{'role': 'system', 'content': system_prompt}]
    message_user = {'role': 'user', 'content': user_content}
    if images:
        message_user['images'] = images
    messages.append(message_user)

    try:
        if format_json:
            response = ollama.chat(
                model=model_name, messages=messages, format='json')
            content = response['message']['content']
            return json.loads(content) if isinstance(content, str) else content
        else:
            response = ollama.chat(model=model_name, messages=messages)
            return response['message']['content']
    except Exception as e:
        print(f"Error calling Ollama model {model_name}: {e}")
        if "model not found" in str(e).lower():
            logger.error(
                f"Please ensure model '{model_name}' is pulled in Ollama.")
        elif "Connection refused" in str(e):
            logger.error(
                "Ollama server may not be running? Please start Ollama.")
        return None


def parse_bulleted_list(text_content):
    # Parses a string containing a bulleted list (lines starting with '*') into a list of strings.
    if not text_content:
        return []
    # Regex to find lines starting with '*' or '-' followed by a space
    items = re.findall(r"^[*-]\s*(.*)", text_content, re.MULTILINE)
    return [item.strip() for item in items if item.strip()]


# Set up logging for the projet with date-versioned log files
def setup_logger(log_folder="logs"):
    # Create log folder if doesn't exist
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    now = datetime.now()
    log_filename = now.strftime(
        f"{log_folder}/xcluster_run_%Y-%m-%d_%H-%M-%S.log")

    logger = logging.getLogger("XCluster")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(log_filename, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


# Prompts (taken from Paper in pages 17 - 22)
# Caption-based Proposer Prompts
# Table 9 in the paper
PROMPT_CAPTION_PROPOSER_MLLM_SYSTEM = "You are a helpful AI assistant."
PROMPT_CAPTION_PROPOSER_MLLM_USER = "Describe the following image in detail."

# Table 10 in the paper
PROMPT_CAPTION_PROPOSER_LLM_CRITERIA_SYSTEM = "You are a helpful assistant."


def get_prompt_caption_proposer_llm_criteria_user(captions_str):
    return f"""Input Explanation: The following are the result of captioning a set of images:
{captions_str}
Goal Explanation: I am a machine learning researcher trying to figure out the potential clustering or grouping criteria that exist in these images. So I can better understand my data and group them into different clusters based on different criteria.
Task Instruction: Come up with ten distinct clustering criteria that exist in this set of images.
Output Instruction: Please write a list of clustering criteria (separated by bullet points “*”).
Task Reinforcement: Again I want to figure out what are the potential clustering/grouping criteria that I can use to group these images into different clusters. List ten clustering or grouping criteria that often exist in this set of images based on the captioning results. Answer with a list (separated by bullet points “*”).
Your response:"""


# Table 11 in the paper
PROMPT_CAPTION_PROPOSER_LLM_REFINE_CRITERIA_SYSTEM = "You are a helpful assistant."


def get_prompt_caption_proposer_llm_refine_criteria_user(criteria_list_str):
    return f"""Input Explanation: I am a machine learning researcher working with a set of images. I aim to cluster this set of images based on the various clustering criteria present within them. Below is a preliminary list of clustering criteria that I’ve discovered to group these images:
{criteria_list_str}
Goal Explanation: My goal is to refine this list by merging similar criteria and rephrasing them using more precise and informative terms. This will help create a set of distinct, optimized clustering criteria.
Task Instruction: Your task is to first review and understand the initial list of clustering criteria provided. Then, assist me in refining this list by:
* Merging similar criteria.
* Expressing each criterion more clearly and informatively.
Output Instruction: Please respond with the cleaned and optimized list of clustering criteria, formatted as bullet points (using “*”).
Your response:"""


# Caption-based Grouper Prompts
# Table 16 in the paper
PROMPT_CAPTION_GROUPER_MLLM_CRITERION_CAPTION_SYSTEM = "You are a helpful AI assistant."


def get_prompt_caption_grouper_mllm_criterion_caption_user(criterion):
    return f"""Analyze the image focusing specifically on the "{criterion}". Provide a detailed description of the "{criterion}" depicted in the image. Highlight key elements and interactions relevant to the "{criterion}" that enhance the understanding of the scene.
Your response:"""


# Table 17 in the paper
PROMPT_CAPTION_GROUPER_LLM_INITIAL_NAMING_SYSTEM = "You are a helpful assistant."


def get_prompt_caption_grouper_llm_initial_naming_user(criterion, criterion_specific_caption):
    return f"""Input Explanation: The following is the description about the "{criterion}" of an image:
Caption Embedding: "{criterion_specific_caption}"
Goal Explanation: I am a machine learning researcher trying to assign a label to this image based on what is the "{criterion}" depicted in this image.
Task Instruction: Understand the provided description carefully and assign a label to this image based on what is the "{criterion}" depicted in this image.
Output Instruction: Please respond in the following format within five words: ”*Answer*”. Do not talk about the description and do not respond long sentences. The answer should be within five words.
Task Reinforcement: Again, your job is to understand the description and assign a label to this image based on what is the "{criterion}" shown in this image.
Your response:"""


# Table 18 in the paper
PROMPT_CAPTION_GROUPER_LLM_MULTIGRANULARITY_REFINE_SYSTEM = "You are a helpful assistant."


def get_prompt_caption_grouper_llm_multigranularity_refine_user(criterion, initial_names_str):
    return f"""Input Explanation: The following is an initial list of "{criterion}" categories. These categories might not be at the same semantic granularity level. For example, category 1 could be “cutting vegetables”, while category 2 is simply “cutting”. In this case, category 1 is more specific than category 2.
Category Embedding:
{initial_names_str}
Task Instruction: These categories might not be at the same semantic granularity level. For example, category 1 could be “cutting vegetables”, while category 2 is simply “cutting”. In this case, category 1 is more specific than category 2. Your job is to generate a three-level class hierarchy (class taxonomy, where the first level contains more abstract or general coarse-grained classes, the third level contains more specific fine-grained classes, and the second level contains intermediate mid-grained classes) of "{criterion}" based on the provided list of "{criterion}" categories. Follow these steps to generate the hierarchy.
Sub-task Instruction: Follow these steps to generate the hierarchy:
Step 1 - Understand the provided initial list of "{criterion}" categories. The following three-level class hierarchy generation steps are all based on the provided initial list.
Step 2 - Generate a list of abstract or general "{criterion}" categories as the first level of the class hierarchy, covering all the concepts present in the initial list.
Step 3 - Generate a list of middle-grained "{criterion}" categories as the second level of the class hierarchy, in which the middle-grained categories are the subcategories of the categories in the first level. The categories in the second-level are more specific than the first level but should still cover and reflect all the concepts present in the initial list.
Step 4 - Generate a list of more specific fine-grained "{criterion}" categories as the third level of the class hierarchy, in which the categories should reflect more specific "{criterion}" concepts that you can infer from the initial list. The categories in the third-level are subcategories of the second-level.
Step 5 - Output the generated three-level class hierarchy as a JSON object where the keys are the level numbers and the values are a flat list of generated categories at each level, structured like:
{{
  "level 1": ["categories"],
  "level 2": ["categories"],
  "level 3": ["categories"]
}}
Output Instruction: Please only output the JSON object in your response and simply use a flat list to store the generated categories at each level.
Your response:"""


# Table 19 in the paper
PROMPT_CAPTION_GROUPER_LLM_FINAL_ASSIGNMENT_SYSTEM = "You are a helpful assistant."


def get_prompt_caption_grouper_llm_final_assignment_user(criterion, criterion_specific_caption, candidate_categories_str):
    return f"""Input Explanation: The following is a detailed description about the "{criterion}" of an image.
Caption Embedding: "{criterion_specific_caption}"
Task Instruction: Based on the content and details provided in the description, classify the image into one of the specified "{criterion}" categories listed below:
Candidate Category Embedding:
"{criterion}" categories:
{candidate_categories_str}
Output Instruction: Ensure that your classification adheres to the details mentioned in the image description. Respond with the classification result in the following format: “*category name*”.
Your response:"""


# X-CLUSTER IMPLEMENTATION

# Part 1 - Caption-based Criteria Proposer
class CaptionBasedCriteriaProposer:
    def __init__(self, mllm_model, llm_model):
        self.mllm_model = mllm_model
        self.llm_model = llm_model
        logger.info(
            f"CriteriaProposer initialized with MLLM: {self.mllm_model}, LLM: {self.llm_model}")

    # General description for each image
    def generate_initial_captions(self, image_paths):
        logger.info("\nGenerating Initial Captions...")
        captions_dict = {}
        for i, img_path in enumerate(image_paths):
            logger.info(
                f"Processing image {i+1}/{len(image_paths)}: {img_path}")
            img_base64 = image_to_base64(img_path)
            if not img_base64:
                continue

            user_prompt = PROMPT_CAPTION_PROPOSER_MLLM_USER
            caption = ollama_chat_call(self.mllm_model,
                                       PROMPT_CAPTION_PROPOSER_MLLM_SYSTEM,
                                       user_prompt,
                                       images=[img_base64])
            if caption:
                captions_dict[img_path] = caption
                # Print only the first 100 characters
                logger.info(f"Caption: {caption[:100]}...")
            else:
                logger.error(f"Failed to generate caption for {img_path}")
        return captions_dict

    # Proposes clustering criteria from subsets of captions
    def propose_criteria_from_captions(self, all_captions_dict):
        logger.info("\nProposing Criteria from Captions...")
        all_captions_list = list(all_captions_dict.values())
        if not all_captions_list:
            logger.info("No captions to process.")
            return []

        proposed_criteria = set()
        num_subsets = (len(all_captions_list) +
                       CAPTIONS_PER_SUBSET - 1) // CAPTIONS_PER_SUBSET

        for i in range(num_subsets):
            logger.info(f"Processing caption subset {i+1}/{num_subsets}")
            subset_captions = all_captions_list[i *
                                                CAPTIONS_PER_SUBSET: (i+1)*CAPTIONS_PER_SUBSET]

            # Format captions for the prompt like in Table 10
            captions_str_for_prompt = "\n".join(
                [f"Image {j+1}: \"{cap}\"" for j, cap in enumerate(subset_captions)])

            user_prompt = get_prompt_caption_proposer_llm_criteria_user(
                captions_str_for_prompt)
            criteria_text = ollama_chat_call(self.llm_model,
                                             PROMPT_CAPTION_PROPOSER_LLM_CRITERIA_SYSTEM,
                                             user_prompt)
            if criteria_text:
                parsed = parse_bulleted_list(criteria_text)
                logger.info(f"Subset {i+1} proposed criteria: {parsed}")
                proposed_criteria.update(parsed)
            else:
                logger.error(f"Failed to get criteria from subset {i+1}")
        return list(proposed_criteria)

    # Refines the list of proposed criteria using an LLM
    def refine_criteria(self, proposed_criteria_list):
        logger.info("\nRefining Criteria...")
        if not proposed_criteria_list:
            logger.info("No criteria to refine.")
            return []

        # Format criteria for prompt like in Table 11
        criteria_list_str_for_prompt = "\n".join(
            [f"* Criterion {i+1}: \"{crit}\"" for i, crit in enumerate(proposed_criteria_list)])

        user_prompt = get_prompt_caption_proposer_llm_refine_criteria_user(
            criteria_list_str_for_prompt)
        refined_criteria_text = ollama_chat_call(self.llm_model,
                                                 PROMPT_CAPTION_PROPOSER_LLM_REFINE_CRITERIA_SYSTEM,
                                                 user_prompt)
        if refined_criteria_text:
            parsed = parse_bulleted_list(refined_criteria_text)
            logger.info(f"Refined criteria: {parsed}")
            return parsed
        else:
            logger.error("Failed to refine criteria.")
            return proposed_criteria_list  # Return original if refinement fails

    # Runs the full pipeline
    def run(self, image_paths):
        if not image_paths:
            logger.info("No image paths provided to CriteriaProposer.")
            return []
        initial_captions = self.generate_initial_captions(image_paths)
        if not initial_captions:
            logger.error(
                "CriteriaProposer: Failed to generate initial captions.")
            return []
        self.initial_captions_dict_ = initial_captions  # Store for returning
        proposed_criteria = self.propose_criteria_from_captions(
            initial_captions)
        if not proposed_criteria:
            logger.error(
                "CriteriaProposer: Failed to propose criteria from captions.")
            return [], self.initial_captions_dict_
        refined_criteria = self.refine_criteria(proposed_criteria)
        return refined_criteria, self.initial_captions_dict_


# Part 2 - Caption-based Grouper
class CaptionBasedSemanticGrouper:
    def __init__(self, mllm_model, llm_model):
        self.mllm_model = mllm_model
        self.llm_model = llm_model
        logger.info(
            f"SemanticGrouper initialized with MLLM: {self.mllm_model}, LLM: {self.llm_model}")

    # Generates captions for each image, specific to the given criterion
    def generate_criterion_specific_captions(self, image_paths, criterion):
        logger.info(
            f"\nGenerating Criterion-Specific Captions for Criterion '{criterion}'...")
        criterion_captions_dict = {}  # image_path: caption
        for i, img_path in enumerate(image_paths):
            logger.info(
                f"Processing image {i+1}/{len(image_paths)} for criterion '{criterion}': {img_path}")
            img_base64 = image_to_base64(img_path)
            if not img_base64:
                continue

            user_prompt = get_prompt_caption_grouper_mllm_criterion_caption_user(
                criterion)
            caption = ollama_chat_call(self.mllm_model,
                                       PROMPT_CAPTION_GROUPER_MLLM_CRITERION_CAPTION_SYSTEM,
                                       user_prompt,
                                       images=[img_base64])
            if caption:
                criterion_captions_dict[img_path] = caption
                logger.info(f"Criterion-specific caption: {caption[:100]}...")
            else:
                logger.error(
                    f"Failed to generate criterion-specific caption for {img_path}")
        return criterion_captions_dict

    # Assigns an initial class name to each criterion-specific caption
    def initial_naming(self, criterion_specific_captions_dict, criterion):
        logger.info(f"\nInitial Naming for Criterion '{criterion}'...")
        initial_names_map = {}  # image_path: initial_name
        raw_initial_names_list = []

        count = 0
        total_captions = len(criterion_specific_captions_dict)
        for img_path, caption in criterion_specific_captions_dict.items():
            count += 1
            logger.info(
                f"Performing initial naming for caption {count}/{total_captions} of image {img_path}")
            user_prompt = get_prompt_caption_grouper_llm_initial_naming_user(
                criterion, caption)
            name_text = ollama_chat_call(self.llm_model,
                                         PROMPT_CAPTION_GROUPER_LLM_INITIAL_NAMING_SYSTEM,
                                         user_prompt)
            if name_text:
                # Expected format = "*Answer*"
                parsed_name = name_text.replace("*", "").strip()
                initial_names_map[img_path] = parsed_name
                raw_initial_names_list.append(parsed_name)
                logger.info(f"Initial name for {img_path}: {parsed_name}")
            else:
                logger.error(f"Failed to get initial name for {img_path}")

        return list(set(raw_initial_names_list))

    # Refines initial names into three structured granularity levels (coarse, middle, fine)
    def refine_cluster_names_multigranularity(self, initial_names_list, criterion):
        logger.info(
            f"\nMulti-granularity Cluster Refinement for Criterion '{criterion}'...")
        if not initial_names_list:
            logger.info("No initial names to refine.")
            return None

        # Format initial names for prompt like in Table 18
        initial_names_str_for_prompt = "\n".join(
            [f"* \"{name}\"" for name in initial_names_list])

        user_prompt = get_prompt_caption_grouper_llm_multigranularity_refine_user(
            criterion, initial_names_str_for_prompt)

        # prompt returning a JSON object
        refined_names_json = ollama_chat_call(self.llm_model,
                                              PROMPT_CAPTION_GROUPER_LLM_MULTIGRANULARITY_REFINE_SYSTEM,
                                              user_prompt,
                                              format_json=True)
        if refined_names_json and isinstance(refined_names_json, dict):
            logger.info(f"Refined cluster names (JSON): {refined_names_json}")
            if all(key in refined_names_json for key in ["level 1", "level 2", "level 3"]):
                return refined_names_json
            else:
                logger.error(
                    "Error: Refined names JSON does not have expected 'level 1/2/3' keys.")
                return None
        else:
            logger.error(
                f"Failed to refine cluster names or received invalid JSON for criterion '{criterion}'. Response: {refined_names_json}")
            return None

    # FINAL : Assigns each image to a cluster at different granularity levels
    def final_assignment(self, criterion_specific_captions_dict, refined_cluster_names_by_granularity, criterion):
        logger.info(f"\nFinal Assignment for Criterion '{criterion}'...")
        if not refined_cluster_names_by_granularity:
            logger.info("No refined cluster names for final assignment.")
            return {}
        # Mapping :
        granularity_map = {"level 1": "coarse",
                           "level 2": "middle", "level 3": "fine"}
        # Initialize results structure :
        # assignments_by_granularity = { "coarse": {"cluster_name_1": [img_path1, ...], ...}, "middle": ..., "fine": ... }
        assignments_by_granularity = {gran_name: {}
                                      for gran_name in granularity_map.values()}
        total_captions = len(criterion_specific_captions_dict)

        for paper_level_key, gran_name in granularity_map.items():
            logger.info(
                f"Assigning for granularity: {gran_name} (from {paper_level_key})")
            candidate_categories = refined_cluster_names_by_granularity.get(
                paper_level_key, [])
            if not candidate_categories:
                logger.info(
                    f"No candidate categories found for {gran_name} level.")
                continue

            # Format candidate categories for prompt like in Table 19
            candidate_categories_str_for_prompt = "\n".join(
                [f"* \"{cat}\"" for cat in candidate_categories])

            img_count = 0
            for img_path, caption in criterion_specific_captions_dict.items():
                img_count += 1
                logger.info(
                    f"Assigning image {img_count}/{total_captions} ({img_path}) for {gran_name} level...")
                user_prompt = get_prompt_caption_grouper_llm_final_assignment_user(
                    criterion, caption, candidate_categories_str_for_prompt)
                assigned_cluster_text = ollama_chat_call(self.llm_model,
                                                         PROMPT_CAPTION_GROUPER_LLM_FINAL_ASSIGNMENT_SYSTEM,
                                                         user_prompt)
                if assigned_cluster_text:
                    # Expected format in the paper : "*category name*"
                    parsed_cluster_name = assigned_cluster_text.replace(
                        "*", "").strip()

                    if parsed_cluster_name not in assignments_by_granularity[gran_name]:
                        assignments_by_granularity[gran_name][parsed_cluster_name] = [
                        ]
                    assignments_by_granularity[gran_name][parsed_cluster_name].append(
                        img_path)
                    logger.info(
                        f"Image {img_path} assigned to '{parsed_cluster_name}' at {gran_name} level.")
                else:
                    logger.error(
                        f"Failed to assign cluster for {img_path} at {gran_name} level.")

        return assignments_by_granularity

    # Runs the full semantic grouping pipeline for a single criterion

    def run_for_criterion(self, image_paths, criterion):
        if not image_paths:
            logger.info(
                f"No image paths provided for SemanticGrouper criterion '{criterion}'.")
            return {}

        criterion_specific_captions = self.generate_criterion_specific_captions(
            image_paths, criterion)
        if not criterion_specific_captions:
            logger.error(
                f"SemanticGrouper: Failed to generate criterion-specific captions for '{criterion}'.")
            return {}

        initial_names_list = self.initial_naming(
            criterion_specific_captions, criterion)
        if not initial_names_list:
            logger.error(
                f"SemanticGrouper: Failed in initial naming for '{criterion}'.")
            return {}

        refined_names_multigran = self.refine_cluster_names_multigranularity(
            initial_names_list, criterion)
        if not refined_names_multigran:
            logger.error(
                f"SemanticGrouper: Failed to refine cluster names for '{criterion}'.")
            return {}

        final_assignments = self.final_assignment(
            criterion_specific_captions, refined_names_multigran, criterion)
        return final_assignments

    # Runs the semantic grouping for all discovered criteria

    def run_all(self, image_paths, all_discovered_criteria):
        all_results = {}  # criterion: assignments_by_granularity
        if not all_discovered_criteria:
            logger.info("No criteria provided to SemanticGrouper.")
            return {}
        for criterion in all_discovered_criteria:
            logger.info(f"\n!!! Processing Criterion: {criterion} !!!")
            assignments_for_criterion = self.run_for_criterion(
                image_paths, criterion)
            all_results[criterion] = assignments_for_criterion
        return all_results

# Formats a string to be used as a CSV header


def sanitize_for_header(name):
    return name.replace(" ", "_").replace("/", "_").replace(":", "_").lower()


def export_results_to_csv(image_files_list,
                          initial_captions_map,
                          cluster_assignments_map,
                          globally_discovered_criteria,
                          csv_filename="xcluster_output.csv"):
    logger.info(f"\nExporting results to {csv_filename}...")

    if not image_files_list:
        logger.warning("No image files to process for CSV export.")
        return

    # Prepare headers
    headers = ['image_id', 'image_path', 'initial_general_caption']
    # Same granularity as used in SemanticGrouper
    granularity_levels = ['coarse', 'middle', 'fine']

    # Dynamically create headers for each criterion and granularity
    # sort criteria to ensure consistent column order if script is run multiple times
    sorted_criteria = sorted(list(globally_discovered_criteria))

    for criterion in sorted_criteria:
        criterion_header_safe = sanitize_for_header(criterion)
        for gran_level in granularity_levels:
            headers.append(f'cluster_{criterion_header_safe}_{gran_level}')

    # Create a reverse map for easier lookup: image_path -> {criterion: {granularity: cluster_name}}
    image_to_cluster_lookup = {img_path: {} for img_path in image_files_list}

    for criterion, assignments_by_granularity in cluster_assignments_map.items():
        for gran_level, clusters in assignments_by_granularity.items():
            for cluster_name, images_in_cluster in clusters.items():
                for img_path in images_in_cluster:
                    if img_path not in image_to_cluster_lookup:
                        image_to_cluster_lookup[img_path] = {}
                    if criterion not in image_to_cluster_lookup[img_path]:
                        image_to_cluster_lookup[img_path][criterion] = {}
                    image_to_cluster_lookup[img_path][criterion][gran_level] = cluster_name

    # Write to CSV
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for img_path in image_files_list:
                image_id = os.path.basename(img_path)
                initial_caption = initial_captions_map.get(img_path, "N/A")
                row = [image_id, img_path, initial_caption]
                img_clusters = image_to_cluster_lookup.get(img_path, {})

                for criterion in sorted_criteria:
                    criterion_clusters = img_clusters.get(criterion, {})
                    for gran_level in granularity_levels:
                        cluster_name = criterion_clusters.get(
                            gran_level, "N/A")
                        row.append(cluster_name)
                writer.writerow(row)
        logger.info(f"Successfully exported results to {csv_filename}")
    except IOError as e:
        logger.error(f"Error writing to CSV file {csv_filename}: {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during CSV export: {e}", exc_info=True)


# MAIN EXECUTION BLOCK
if __name__ == "__main__":
    logger = setup_logger()
    IMAGE_ROOT_DIRECTORY = "./data/food-101/images"

    # Check if root directory exists
    if not os.path.isdir(IMAGE_ROOT_DIRECTORY):
        logger.error(
            f"Error: Image root directory '{IMAGE_ROOT_DIRECTORY}' not found.")
        exit()

    all_image_files = []
    logger.info(
        f"Scanning for images in '{IMAGE_ROOT_DIRECTORY}' and its subdirectories...")
    for root, _, files in os.walk(IMAGE_ROOT_DIRECTORY):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_files.append(os.path.join(root, file))

    if not all_image_files:
        logger.error(
            f"No images found in '{IMAGE_ROOT_DIRECTORY}' or its subdirectories. Please check the path and add some images.")
        exit()

    logger.info(f"Found {len(all_image_files)} images to process.")

    # Select a random subset of images
    # The whole dataset is too large to run on Telecom GPUs in a reasonable time period.
    SUBSET_SIZE = 100
    if len(all_image_files) > SUBSET_SIZE:
        image_files = random.sample(all_image_files, SUBSET_SIZE)
        logger.info(
            f"Selected a random subset of {SUBSET_SIZE} images for processing.")
    else:
        image_files = all_image_files
        logger.info(
            f"Using all {len(image_files)} found images (less than {SUBSET_SIZE}).")

    logger.info(f"Images to process: {image_files}")

    # 1. Discover Criteria
    logger.info("\nStep 1: Criteria Proposal")
    proposer = CaptionBasedCriteriaProposer(
        mllm_model=MLLM_MODEL, llm_model=LLM_MODEL)
    discovered_criteria, initial_captions_dict = proposer.run(image_files)
    logger.info("\n\nDiscovered Criteria !")
    if discovered_criteria:
        for crit in discovered_criteria:
            logger.info(f"- {crit}")
    else:
        logger.warning(
            "No criteria were discovered. CSV export will be limited.")
        # If no criteria, we still want in the CSV the images and their initial captions
        if not initial_captions_dict:
            initial_captions_dict = {}
        export_results_to_csv(image_files, initial_captions_dict, {
        }, [], "xcluster_output_no_criteria.csv")
        exit()

    # 2. Semantic Grouping based on discovered criteria
    logger.info("\n\nStep 2: Semantic Grouping")
    grouper = CaptionBasedSemanticGrouper(
        mllm_model=MLLM_MODEL, llm_model=LLM_MODEL)
    all_cluster_assignments = grouper.run_all(image_files, discovered_criteria)

    logger.info("\n\nStep 3: Final X-Cluster Results")
    if all_cluster_assignments:
        for criterion, assignments_by_granularity in all_cluster_assignments.items():
            logger.info(f"\nCriterion: {criterion}")
            if not assignments_by_granularity:
                logger.info("No clusters formed for this criterion.")
                continue
            for granularity_level, clusters in assignments_by_granularity.items():
                logger.info(
                    f"Granularity: {granularity_level} - Found {len(clusters)} clusters.")
                if clusters:
                    first_cluster_name = list(clusters.keys())[0]
                    num_images_in_first_cluster = len(
                        clusters[first_cluster_name])
                    logger.info(
                        f"Example Cluster: \"{first_cluster_name}\" ({num_images_in_first_cluster} images)")
    else:
        logger.info("No cluster assignments were made.")

    # 3. Export results to CSV
    csv_output_filename = "xcluster_food101_results.csv"
    export_results_to_csv(image_files_list=image_files,
                          initial_captions_map=initial_captions_dict,
                          cluster_assignments_map=all_cluster_assignments,
                          globally_discovered_criteria=discovered_criteria,
                          csv_filename=csv_output_filename)

    logger.info("\n\n!!! X-Cluster Processing Complete !!!")
