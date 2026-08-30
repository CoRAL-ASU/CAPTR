import argparse
import copy
import gc
import json
import multiprocessing
import os
from pathlib import Path

import pandas
from utils.runtime_env import configure_runtime_environment



# Set multiprocessing start method to 'spawn' to avoid CUDA re-initialization errors
multiprocessing.set_start_method('spawn', force=True)

configure_runtime_environment()

from dotenv import load_dotenv
from pandas import DataFrame
from PIL import Image
import pandas
print(f"Using pandas version: {pandas.__version__}")
import torch

from baselines_utils import (
    DATASET_PATHS,
    evaluate,
    get_metadata,
    get_vllm_generator,
    sample_dataset,
    store_generations,
)
from mmtabqa.load_mmtabqa_utils import load_mmtabqa_dataset
from utils.utils import build_passage_context

load_dotenv(".env")

section_title_maps = {}

######## Prompts (from https://github.com/MMTabQA/mmtabqa/blob/main/src/modelling/interleaved/gpt4_prompt_creator.py) - slight variations to make the qwen2.5-omni model follow the instruction format better ########

TEXT_PROMPT_MMTABREAL = """You will be provided a table where some cells are images.
Your task is to:

Step 1: UNDERSTAND THE TABLE CONTEXT - Carefully analyze the table structure and understand the intricate relationship between image and text.

Step 2: ANALYZE THE QUESTIONS - Read all the questions provided and explore ALL TYPES OF REASONING to find answers.

Step 3: PROVIDE ANSWERS IN FORMAT - Ensure that all answers adhere strictly to the FORMAT specified. Avoid deviating from this format or including unnecessary explanations. DO NOT add any extra text beyond what is required.

Step 4: WHEN THE ANSWER IS AN ENTITY, RETURN THE ENTITY NAME EXACTLY AS IT APPEARS IN THE TABLE OR ITS CLEAR REAL-WORLD NAME.

ANSWER FORMATTING GUIDELINES:   

Sentences must be in string format without any bullet points or numbering.
Offer no explanations or justifications for your answers.
if a number is required, provide it in numeric format without words.
YOU HAVE TO ANSWER. YOU CANNOT RETURN BLANK RESPONSES.
ALWAYS PROVIDE YOUR ANSWERS IN THIS FORMAT.

IMPORTANT: ALL answers are in the table/images.
Now I will provide you with the table.
"""
TEXT_PROMPT_FETA_8_SHOTS = """Answer in a sentence, using the table data given. The table consists of data in the form of text and images. Each row of the table has been represented using [] with data for each column in the row separated by a semi-colon.
In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

Your answer must always include "Step 2:". After you have written "Step 2:", you should only state the actual answer and nothing else.

You are also provided with some question-answer examples for better understanding the format of providing the answer:

Example 1:
Table context: Table related to NHL awards in context of 2013–14 NHL season.\n\nQuestion: Which teams were competing for the Stanley Cup in the 2013-14 NHL season?\nStep 1: The Stanley Cup is the silver-coloured cup represented in the first row, Award column. In the same row under the reciepient's column, we can see a Black-coloured logo with LA written on it, which is the logo for <>.Also in the runners-up column, we can see a Blue-coloured logo with "New-York Rangers" written on it. Thus, we can conclude that The Los Angeles Kings won the Stanley Cup, defeating the New York Rangers.\n\nStep 2:\nThe Los Angeles Kings won the Stanley Cup, defeating the New York Rangers.

Example 2:
Table context: Table related to International competitions in context of Debbie Marti.\n\nQuestion: In which city was the 1991 World Championships held and what distance did Debbie Marti achieve to qualify?\nStep 1: As we can see in the 5th row, the Competiton represented in the Competitions column by a Blue-coloured logo is the World Championships. In the same row, under the venue column, we can see a collage of pictures of the prominent buildings from Tokyo. Thus We can conclude that the venue of the competiton was Tokyo. Also, in the column "Notes", we can see that Debbie Marti qualified with 1.86m.\n\nStep 2: At the 1991 World Championships in Tokyo, Debbie Marti qualified with 1.86 m.

Example 3:
Table context: Table related to Awards and nominations in context of Project Gutenberg (film).\n\nQuestion: What awards did Project Gutenberg win at the 38th Hong Kong Film Awards?\nStep 1: The answer to the question can be found by looking at the column titled "Award" in the table. We can infer that there are seven rows in the table, each corresponding to an award won by Project Gutenberg. The categories listed are (Best Film, Best Director, Best Screenplay, Best Cinematography, Best Film Editing, Best Art Direction, and Best Costume Make Up Design) exactly match up to the categories listed in the "Award" column. So, to find the answer, you would need to look for each of these categories in the "Award" column and see which movie title is listed next to it.\n\nStep 2:Project Gutenberg won seven awards at the 38th Hong Kong Film Awards, in the categories Best Film, Best Director, Best Screenplay, Best Cinematography, Best Film Editing, Best Art Direction, and Best Costume Make Up Design.

Example 4:
Table context: Table related to Awards and nominations in context of Mike Cahill (director).\n\nQuestion: What film won the Alfred P. Sloan Prize at the Sundance Film Festival in 2014?\nStep 1: Look at the "Year" column and find the year 2014. Then, look at the "Award" column for that row. If it says "Alfred P. Sloan Prize", then the movie title in the "Film" column for that row is the answer. In the table you described, on the row where "Year" is 2014, "Award" is "Alfred P. Sloan Prize", and "Film" is "I Origins".\n\nStep 2:\nCahill's film I Origins again won the Alfred P. Sloan Prize at the 2014 Sundance Film Festival, his second time receiving the award.

Example 5:
Table context: Table related to Home attendances in context of 2012–13 Everton F.C. season.\n\nQuestion: How did Everton F.C. do against Manchester United and Tottenham Hotspur during their 2012-13 season?\nStep 1: Look for Manchester United and Tottenham Hotspur on the "Opponent" column.  Look at the corresponding "Score" for each team. For Manchester United, the score is  1-0 in favor of Everton. For Tottenham Hotspur, the score is 2-1 in favor of Everton. Therefore, Everton won against both Manchester United and Tottenham Hotspur.\n\nStep 2:\nEverton F.C. won over Manchester United in the first game of the season with 1–0, defeated Tottenham Hotspur 2–1, and defeated Manchester City 2–0 in the Premier League.

Example 6:
Table context: Table related to International competitions in context of Süreyya Ayhan.\n\nQuestion: How did Sureyya Ayhan fare at the 2003 World Championships?\nStep 1: Look for "2003" in the "Year" column. Look across that row to the "Competition" column. It should say "World Championships". In the "Event" column, it shows "1500 m". Finally, under the "Position" column, it shows "2nd", indicating that Süreyya Ayhan won a silver medal.\n\nStep 2:\nSüreyya Ayhan won a silver medal in the 1500 m of the 2003 World Championships.

Example 7:
Table context: Table related to Grammy Awards in context of Roberta Flack.\n\nQuestion: When and for which songs did the singer Roberta Flack win Grammy Awards for Record of they Year?\nStep 1: Looking at the table under the "Year" column, you can see 1973 listed twice.  In the corresponding rows under "Award" it says "Record of the Year" each time.  Looking at the "Nominee / work" column for those two rows, it shows "The First Time Ever I Saw Your Face" in 1973 and "Killing Me Softly With His Song" in 1974.  This confirms that Flack won the award for these two songs in consecutive years.\n\nStep 2:\nFlack won the Grammy Award for Record of the Year on two consecutive years: "The First Time Ever I Saw Your Face" won at the 1973 Grammys as did "Killing Me Softly with His Song" at the 1974 Grammys.

Example 8:
Table context: Table related to Television series in context of Kim Jung-hyun (actor, born 1990).\n\nQuestion: What did Kim Jung-hyun do in KBS2 in 2017?\nStep 1: Look for the year "2017" in the "Year" column. Look across that row to the "Network" column. It should say "KBS2". In the "Title" column, it shows "School 2017". This indicates that Kim Jung-hyun played in that drama in 2017 on KBS2.\n\nStep 2:\\In 2017, Kim Jung-hyun played in KBS2's School 2017.

Now, based upon the examples given above, you must understand the text and images given in the table and follow the steps 1-2 to answer the question corresponding to the table represented bt the data. Try to keep the answer in active voice. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.
"""

TEXT_PROMPT_WTQ_8_SHOTS = """Answer in a sentence, using the table data given. The table consists of data in the form of text and images. Each row of the table has been represented using [] with data for each column in the row separated by a semi-colon.
In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

Your answer must always include "Step 2:". After you have written "Step 2:", you should only state the actual answer and nothing else.

You are also provided with some question-answer examples for better understanding the format of providing the answer:

Example 1:
Table context: Table related to Fifth round proper in context of 1975–76 FA Cup.\n\nQuestion: how many games played by sunderland are listed here?\nStep 1: We can conclude Sunderland played in 2 games. The table shows teams listed under "Home team" and "Away team" columns [column headers provide this information].  Looking across the rows, Sunderland's logo, which comprises of 2 horses to the side and a Black&White sheild in between, is listed under one of these columns twice [in the 2nd and 3rd row]. Therefore, Sunderland participated in two games.\n\nStep 2:\n2

Example 2:
Table context: Table related to Complete Formula One World Championship results in context of Playlife.\n\nQuestion: when was the benetton b198 chassis used?\nStep 1: The table shows Formula One results with a context of Playlife, possibly a constructor. As we can see, the Benetton b198 Chassis is the blue coloured supporting structure, as seen in the chassis column of the table. In the same row, there is the column year, which gives us the answer as 1998.\n\nStep 2: 1998.

Example 3:
Table context: Table related to Defunct railroads in context of List of Washington, D.C., railroads.\n\nQuestion: was the pennsylvania railroad under the prr or the rf&p?\nStep 1: The table shows defunct railroads in Washington D.C. The "Pennsylvania Railroad" is the golden background picture represented in the 11th row with trains visible in it. In the same row, another column named "Mark" has the abbreviation as "PRR". Thus, the Pennsylvania Railroad operated under PRR since "PRR" is its short name.\n\nStep 2:PRR

Example 4:
Table context: Table related to Schedule and results in context of 2013–14 Chicago State Cougars women's basketball team.\n\nQuestion: how many games were played against grand canyon?\nStep 1: We can see that there are 2 instances of the grand canyon in the opponent column. One in the 20th row, where there is a purple coloured logo which says GCC, which refers to the Grand Canyon College. Another is in the 26th row. Thus, we can conclude that 2 matches were played against the grand canyon.\n\nStep 2:\n2

Example 5:
Table context: Table related to Roster|Letter winners in context of 1915 Michigan Wolverines football team.\n\nQuestion: how many players were taller and weighed more than frank millard?\nStep 1: Frank Millard is the clean-shaved, short haired guy visible in the 5th row. His height is 5'7 and weight is 212. Thus clearly, there are only 2 players whose height and weight is more than his, one in the 2nd row and other in the 8th row.\n\nStep 2:\n2

Example 6:
Table context: Table related to Racing record|Career summary in context of Conor Daly.\n\nQuestion: the two teams who raced in 2011 are carlin motorsport and what other team?\nStep 1: In the year column there are 2 rows which have a mention of 2011. Apart from Carlin motorsport, the other one has a green car with the logo Schmidt Motorsports on it.\n\nStep 2:\nSchmidt Motorsports

Example 7:
Table context: Table related to Regular season|Schedule in context of 1995 New York Jets season.\n\nQuestion: team that scored more than 40 points against the jets that is not the miami dolphins\nStep 1: As clearly visible, the opponent mentioned in the 4th row, which has a Black-coloured logo written as "RAIDERS" on it scored 47 goals against the jets. The logo is of the team Oakland Raiders. Thus, We can conclude that Oakland Raiders is the other team that scored 47 goals against the jets.\n\nStep 2:\nOakland Raiders

Example 8:
Table context: Table related to Winners|By Country in context of EHF Cup Winners' Cup.\n\nQuestion: did france or croatia have a larger finals total?\nStep 1: Under the country column, in the 5th row we can see a Blue-coloured chicken logo, with FFHANDBALL written under it. That is the logo for France's handball federation. In the 8th row, we can see and Red-Blue coloured handall logo, with the Croatia Handball federation written underneath it, which represents Croatia. Thus, we can conclude that France had more Finals Total, 4, than Croatia, 1.\n\nStep 2:\nFrance

Now, based upon the examples given above, you must understand the text and images given in the table and follow the steps 1-2 to answer the question corresponding to the table represented bt the data. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.
"""

TEXT_PROMPT_WIKISQL_8_SHOTS = """Answer in a sentence, using the table data given. The table consists of data in the form of text and images. Each row of the table has been represented using [] with data for each column in the row separated by a semi-colon.
In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

Your answer must always include "Step 2:". After you have written "Step 2:", you should only state the actual answer and nothing else.

You are also provided with some question-answer examples for better understanding the format of providing the answer:

Example 1:
Table related to Teams and venues in context of 2004 Belarusian Premier League.\n\nQuestion:  What is the Location for Belshina?\nStep 1: The table shows the teams and their corresponding locations. To find the location for Belshina, we need to locate Belshina in the 'Team' column. Belshina is located in row 9, and its location is given in the 'Location' column as 'Bobruisk'. Therefore, the location for Belshina is Bobruisk.\n\nStep 2: bobruisk

Example 2:
Table related to Made the cut in context of 2009 U.S. Open (golf).\n\nQuestion:  what is the to par for retief goosen?\nStep 1: The question asks for the "to par" for Retief Goosen. Looking at the table, we can see that Retief Goosen is listed on row 1, and the "To par" column for row 1 is +3. Therefore, the answer is +3.\n\nStep 2: +3

Example 3:
Table related to Round 16 in context of 1969 VFL season.\n\nQuestion:  If the Venue was kardinia park what was the highest Crowd attended?\nStep 1: The question asks for the highest crowd that attended a game at Kardinia Park. Looking at the table, we see that Kardinia Park is listed as the venue for round 3.  The crowd for that game was 16,211. The question asked for the highest crowd.  Looking at the rest of the table, the only higher crowd is 21,025.  Therefore, 21,025 is the answer.\n\nStep 2: 21,025

Example 4:
Table related to Prime ministers in context of Interwar unemployment and poverty in the United Kingdom.\n\nQuestion:  What is the birth place of the prime minister who served George V and entered office on 23 October 1922?\nStep 1: The answer can be arrived by locating the row where the prime minister entered office on 23 October 1922. This is the row corresponding to Andrew Bonar Law. Looking at the 'Birth Place' column for this row, the answer is given as "Rexton, Kent County, New Brunswick, Canada".\n\nStep 2: rexton, kent county, new brunswick, canada

Example 5:
Table related to Round 12 in context of 1976 VFL season.\n\nQuestion:  What date did North Melbourne score 22.14 (146) as the home team?\nStep 1: The question asks what date North Melbourne scored 22.14 (146) as the home team. We can find the row that corresponds to North Melbourne by finding the row with their logo. Row 5 shows their logo, which shows a Blue-coloured kangaroo with "North Melbourne" written on it, and their home score, which is 22.14 (146). In the last column, the date is 19 June 1976. Therefore, the answer to the question is 19 June 1976.\n\nStep 2: 19 june 1976

Example 6:
Table related to Game Log in context of 1982 Atlanta Braves season.\n\nQuestion:  Where was the game on Thursday, April 29, and they played the Chicago Cubs?\nStep 1: The table shows the 1982 Atlanta Braves game log. We need to find the game on Thursday, April 29. We see this date on row 19. Looking at the "Opponent" column, we see the Chicago Cubs. It is represented by a Purple background logo with a Cwritten in Red colour. Therefore, the game was played at the location listed in the "Site" column for row 19: Atlanta-Fulton County Stadium.\n\nStep 2: atlanta-fulton county stadium

Example 7:
Table related to Medal table in context of 2007 Military World Games.\n\nQuestion:  Which nation has a Silver of 1, a Gold of 0, and a Total of 1?\nStep 1: The question asks for the nation with 1 silver medal, 0 gold medals, and a total of 1 medal.Looking at the table, we can see that Bulgaria, Cameroon and Hungary, which are represented by a White-Green_Red striped flag, Green-Red-Tellow flag with a star, and a Red-White Green striped flag respectively, have 1 silver medal, 0 gold medals and a total of 1 medal.\n\nStep 2: bulgaria, cameroon, hungary

Example 8:
Table related to Round 4 in context of 1956 VFL season.\n\nQuestion:  Which home team scored 12.15 (87)?\nStep 1: The table shows the scores of different teams in round 4 of the 1956 VFL season. The row with the home team score of 12.15 (87) is the second row which corresponds to Collingwood. Therefore, the home team that scored 12.15 (87) is Collingwood.\n\nStep 2: collingwood

Now, based upon the examples given above, you must understand the text and images given in the table and follow the steps 1-2 to answer the question corresponding to the table represented bt the data. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.
"""

TEXT_PROMPT_DIRECT_GENERIC = """Answer the question using the table data.
Do not explain your reasoning.
Respond only in the format:
Step 2:
<answer>
"""

TEXT_PROMPT_DIRECT_MMTABREAL = """You will be provided a table where some cells are images.
Answer the question using only the table and images.
Think silently and do not reveal your reasoning.
Respond with exactly one line containing only the final answer.
If the answer is an entity, return only the entity name as written in the table.
If multiple entities are required, return only the entity names separated by commas.
Do not include any prefixes (for example, do not output 'Step 1', 'Step 2', 'Answer:', or 'Final answer:').
Do not output uncertainty text such as "cannot determine", "not enough information", or placeholders.
All answers are in the table/images.
Now I will provide you with the table.
"""


def is_qwen35_model(model_name: str) -> bool:
    model_name = (model_name or "").lower()
    return "qwen3.5" in model_name or "qwen3_5" in model_name


def get_prompt_text(dataset_name, disable_cot=False):
    dataset_name = (dataset_name or "").strip()
    if disable_cot:
        return TEXT_PROMPT_DIRECT_MMTABREAL if dataset_name.lower() in {"mmtabreal", "mmtbench"} else TEXT_PROMPT_DIRECT_GENERIC

    few_shot_example_prompts_map = {
        "WikiTQ": TEXT_PROMPT_WTQ_8_SHOTS,
        "WikiSQL": TEXT_PROMPT_WIKISQL_8_SHOTS,
        "FetaQA": TEXT_PROMPT_FETA_8_SHOTS,
        "HybridQA": TEXT_PROMPT_WTQ_8_SHOTS,
        "MMTabReal": TEXT_PROMPT_MMTABREAL,
        "mmtbench": TEXT_PROMPT_MMTABREAL,
    }
    if dataset_name in few_shot_example_prompts_map:
        return few_shot_example_prompts_map[dataset_name]

    normalized_lookup = {key.lower(): value for key, value in few_shot_example_prompts_map.items()}
    return normalized_lookup[dataset_name.lower()]


def create_prompt_for_example(example, few_shot_example_text, selected_rows=None):
    table_metadata, question = get_metadata(example)
    example["table_with_metadata"] = copy.deepcopy(example["table"])
    passage_context_text = build_passage_context(example, selected_rows=selected_rows)

    header = example["table"]["header"]
    rows = example["table"]["rows"]
    table_array = [row["content"] for row in rows]  # "content" may contain images and text
    table_types = [row["type"] for row in rows]  # "type" is a list of "text" or "image"

    prompt_content = []
    # prompt_content will look like this:
    # [
    #     {
    #         "type": "image",
    #         "image": <Image.Image>,
    #     }, # could also set min_pixels and max_pixels, but since the images are already at reasonable size, this isn't necessary
    #     {"type": "text", "text": question},
    # ],

    # 1. Add the few-shot examples
    prompt_content.append({"type": "text", "text": few_shot_example_text})

    # 2. Add the table context and question as text
    prompt_content.append({"type": "text", "text": f"Table context: {table_metadata}\n"})

    if passage_context_text:
        prompt_content.append({"type": "text", "text": passage_context_text})

    prompt_content.append({"type": "text", "text": "Table:\n"})

    # 2.1 add header
    current_table_string = ""
    for cell in header:
        cell = cell.replace("\t", " ").replace("\n", " ").replace("\\n", " ").replace("\\t", " ").replace("|", " ")
        current_table_string = current_table_string + cell + " ; "
    current_table_string = current_table_string + "\n"

    # 3. add the main example (with images)
    for row_idx, row in enumerate(table_array):
        current_table_string += "["
        for cell_idx, cell in enumerate(row):
            cell_type = table_types[row_idx][cell_idx]

            if cell_type == "image":
                # Case 1: We have an image: add current_table_string & add image & re-init current_table_string
                # Handle both Image.Image objects and string paths
                if isinstance(cell, str):
                    # Check if there are multiple images separated by "|"
                    if "|" in cell:
                        image_paths = cell.split("|")
                        for img_path_str in image_paths:
                            img_path_str = img_path_str.strip().replace("\\", "/")
                            try:
                                loaded_img = Image.open(img_path_str).convert("RGB")
                                loaded_img = loaded_img.resize((128, 128), Image.Resampling.LANCZOS)
                                prompt_content.append({
                                    "type": "text",
                                    "text": current_table_string,
                                })
                                prompt_content.append({
                                    "type": "image",
                                    "image": loaded_img,
                                })
                                current_table_string = " ; "
                            except Exception as e:
                                print(f"Warning: Failed to load image {img_path_str}: {e}")
                        continue
                    else:
                        # Single image path
                        try:
                            img_path = cell.replace("\\", "/")
                            cell = Image.open(img_path).convert("RGB")
                            cell = cell.resize((128, 128), Image.Resampling.LANCZOS)
                        except Exception as e:
                            print(f"Warning: Failed to load image {cell}: {e}")
                            # Use a placeholder or skip
                            current_table_string += "[IMAGE] ; "
                            continue
                
                if not isinstance(cell, Image.Image):
                    print(f"Warning: Cell {cell_idx} in row {row_idx} is not an image, skipping")
                    current_table_string += "[IMAGE] ; "
                    continue
                    
                prompt_content.append(
                    {
                        "type": "text",
                        "text": current_table_string,
                    }
                )
                prompt_content.append(
                    {
                        "type": "image",
                        "image": cell,
                    }
                )
                current_table_string = " ; "
            else:
                # Case 2: text -> simply append to current_table_string
                # Handle text content - clean and add to table string (like we did for the partial baseline)
                cell = (str(cell).replace("\t", " ").replace("\n", " ").replace("\\n", " ").replace("\\t", " ").replace("|", " "))  # fmt: skip
                current_table_string += cell + " ; "
        current_table_string += "]\n"

    prompt_content.append(
        {
            "type": "text",
            "text": current_table_string + f"\nQuestion: {question}",
        }
    )

    return prompt_content


# def get_section_title(data_item, dataset_name):
#     """
#     Get the section title for a data item.
#     This function uses a global cache `section_title_maps` to avoid reloading the dataset and to store a mapping from item ID to section title.
#     """
#     if dataset_name not in section_title_maps:
#         id_to_title_map = {}
#         paths_dataset_name = (
#             "WikiTQ"
#             if dataset_name == "wikitq"
#             else "WikiSQL"
#             if dataset_name == "wikisql"
#             else "FetaQA"
#             if dataset_name == "fetaqa"
#             else None
#         )
#         for split_name in DATASET_PATHS[paths_dataset_name].keys():
#             dataset_path = DATASET_PATHS[paths_dataset_name][split_name]
#             mmtabqa_dataset = load_mmtabqa_dataset(dataset_path, load_images=False, partial_input_baseline=False)

#             id_to_title_map.update({example["id"]: example["table"]["section_title"] for example in mmtabqa_dataset})
#         section_title_maps[dataset_name] = id_to_title_map

#     title_map = section_title_maps[dataset_name]
#     item_id = data_item["id"]

#     return title_map[item_id]


def create_prompt_for_our_pipeline(data_item, dataset_name, selected_rows, selected_cols, col_values=None, disable_cot=False):
    """
    Create a prompt for the CAPTR pipeline.
    selected rows should be a list of the row integers that got selected
    """

    df: DataFrame = data_item["table"]

    question = data_item["question"]
    headers = df.columns.tolist()
    page_title = data_item["table_with_metadata"]["page_title"]
    section_title = data_item["table_with_metadata"]["section_title"]
    table_metadata = f"Table related to {section_title} in context of {page_title}."

    selected_rows_with_row_ids = [f"row {idx}" for idx in selected_rows]
    passage_context_text = build_passage_context(
        data_item, selected_rows=selected_rows_with_row_ids, selected_cols=selected_cols
    )

    table_array = df.values.tolist()

    few_shot_example_text = get_prompt_text(dataset_name, disable_cot=disable_cot)

    # 1. Add the few-shot examples
    prompt = few_shot_example_text

    # col_values is almost always None but for WikiSQL we set it in the config to true because it improves performance in the text-only WikiSQL as well as MMTabQA WikiSQL
    if col_values is not None:
        prompt = prompt[:-1] if prompt.endswith("\n") else prompt
        prompt += f" The table contains {len(col_values)} columns, namely: {', '.join(col_values.keys())}. The table that you see has been filtered to only contain relevant information to answer the question. For context and to see how the text in the table cells is structured, these are the other values in the columns:\n"
        for col, values in col_values.items():
            prompt += f"{col} contains the following values: {', '.join(map(str, values))}\n"

        prompt += "Now answer the question:\n"

    prompt += f"Table context: {table_metadata}\n"
    if passage_context_text:
        prompt += passage_context_text
    prompt += "Table:\n"

    # 2.1 add header
    for cell in headers:
        cell = cell.replace("\t", " ").replace("\n", " ").replace("\\n", " ").replace("\\t", " ").replace("|", " ")
        prompt += cell + " ; "
    prompt += "\n"

    # 3. add the main example (images are already replaced by text placeholders)
    for row in table_array:
        prompt += "["
        for cell in row:
            cell = (str(cell).replace("\t", " ").replace("\n", " ").replace("\\n", " ").replace("\\t", " ").replace("|", " "))  # fmt: skip
            prompt += cell + " ; "
        prompt += "]\n"

    prompt += f"\nQuestion: {question}"
    return prompt


def create_prompts(mmtabqa_dataset, dataset_name, start_index=0, is_retrieval_hybrid_qa=False, disable_cot=False):
    prompts = {}
    for i, example in enumerate(mmtabqa_dataset):
        few_shot_example_text = get_prompt_text(dataset_name, disable_cot=disable_cot)

        if is_retrieval_hybrid_qa:
            retrieved_row_indices = example["retrieved_row_indices"]
            retrieved_row_indices = [f"row {idx}" for idx in retrieved_row_indices]
            prompt_content = create_prompt_for_example(example=example, few_shot_example_text=few_shot_example_text, selected_rows=retrieved_row_indices)  # fmt: skip
        else:
            prompt_content = create_prompt_for_example(example=example, few_shot_example_text=few_shot_example_text)

        prompt_data = {
            "content": prompt_content,
            "answer_text": example["answer_text"],
            "question": example["question"],
            "page_title": example["table"]["page_title"],
            "section_title": example["table"]["section_title"],
        }
        prompts[str(i + start_index)] = prompt_data

    return prompts


def load_mmtabreal_dataset(dataset_path):
    """
    Load MMTabReal dataset from HuggingFace dataset format.
    
    Args:
        dataset_path: Path to the saved dataset directory
    
    Returns:
        List of examples compatible with create_prompt_for_example
    """
    import datasets
    
    dataset = datasets.load_from_disk(dataset_path)
    
    examples = []
    for item in dataset:
        example = {
            "id": item["id"],
            "question": item["question"],
            "answer_text": item["answer_text"],
            "table_id": item["table_id"],
            "table": item["table"]  
        }
        
        examples.append(example)
    
    return examples


######################################################## Main Functions ########################################################
def main_for_mmtabqa(vllm_generator, model_name, args, disable_cot=False):
    for dataset_name, dataset_splits in DATASET_PATHS.items():
        for split_name, dataset_path in dataset_splits.items():
            results_file = f"results/{model_name}/{dataset_name}_{split_name}_interleaved.json"
            
            # Check if results already exist
            if os.path.exists(results_file):
                print(f"\n{'='*60}")
                print(f"Found existing results for {dataset_name}-{split_name}")
                print(f"Skipping generation, running evaluation only...")
                print(f"{'='*60}")
                
                # Load saved results
                with open(results_file, 'r') as f:
                    all_results = json.load(f)
                
                # Run evaluation
                evaluate(
                    all_results,
                    vllm_generator,
                    dataset_name,
                    model_name_short=model_name,
                    dataset_split=split_name,
                    mode="interleaved",
                    use_llm_as_judge=(dataset_name != "FetaQA"),
                )
                continue
            
            print(f"Processing {dataset_name}-{split_name} dataset...")

            current_index = 0
            shard_size = 50  # Reduced to avoid CUDA OOM with sampled datasets

            mmtabqa_dataset = load_mmtabqa_dataset(
                dataset_path,
                image_base_path=os.getenv("MMTABQA_IMAGE_BASE_PATH"),
                load_images=True,
                partial_input_baseline=False,
            )

            # Apply sampling if n_examples is specified
            if args.n_examples is not None:
                mmtabqa_dataset = sample_dataset(mmtabqa_dataset, n_examples=args.n_examples, seed=42)

            if args.debug:
                mmtabqa_dataset = mmtabqa_dataset.select(range(min(100, len(mmtabqa_dataset))))

            all_results = {}

            while current_index < len(mmtabqa_dataset):
                # We will load A LOT of images into memory. So we will process the dataset in shards to not run OOM.
                end_index = min(current_index + shard_size, len(mmtabqa_dataset))
                print(
                    f"\nProcessing shard {current_index}:{end_index}. Total: {len(mmtabqa_dataset)}. Progress: {current_index / len(mmtabqa_dataset):.2%}"
                )

                shard_dataset = mmtabqa_dataset.select(range(current_index, end_index))
                prompts = create_prompts(shard_dataset, dataset_name, start_index=current_index, disable_cot=disable_cot)
                vllm_generator.generate_batch_pass(prompts)
                all_results.update(prompts)
                
                # Clean up images from all_results to prevent accumulation
                for eid in prompts.keys():
                    if eid in all_results and "content" in all_results[eid]:
                        content_items = all_results[eid]["content"]
                        if isinstance(content_items, list):
                            for item in content_items:
                                if isinstance(item, dict) and item.get("type") == "image" and "image" in item:
                                    del item["image"]  # Remove the actual image object
                
                # Clean up memory after each shard
                del prompts, shard_dataset
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                current_index = end_index

            # store generations
            store_generations(
                all_results,
                model_name_short=model_name,
                mode="interleaved",
                dataset_name=dataset_name,
                dataset_split_name=split_name,
            )
            evaluate(
                all_results,
                vllm_generator,
                dataset_name,
                model_name_short=model_name,
                dataset_split=split_name,
                mode="interleaved",
                use_llm_as_judge=(dataset_name != "FetaQA"),
            )


def main_for_retrieval(vllm_generator, model_name, args, disable_cot=False):
    # 1. parse which datasets we have in the args.retrieval_output_dir
    retrieval_output_dir = Path(args.retrieval_output_dir)
    retrieval_mode = retrieval_output_dir.name
    datasets_list = {}
    for dir in retrieval_output_dir.iterdir():
        if dir.is_dir():
            dataset_name = dir.stem.split("_")[0]
            split_name = dir.stem.split("_")[1]
            if dataset_name not in datasets_list:
                datasets_list[dataset_name] = {}
            datasets_list[dataset_name][split_name] = dir

    print(f"Datasets list: {datasets_list}")

    # 2. Process each dataset split
    for dataset_name, dataset_splits in datasets_list.items():
        for split_name, dataset_path in dataset_splits.items():
            print(f"Processing {dataset_name}-{split_name} dataset; path: {dataset_path}")

            current_index = 0
            shard_size = 50  # Reduced to avoid CUDA OOM with sampled datasets

            mmtabqa_dataset = load_mmtabqa_dataset(
                dataset_path,
                image_base_path=os.getenv("MMTABQA_IMAGE_BASE_PATH"),
                load_images=True,
                partial_input_baseline=False,
            )

            # Apply sampling if n_examples is specified
            if args.n_examples is not None:
                mmtabqa_dataset = sample_dataset(mmtabqa_dataset, n_examples=args.n_examples, seed=42)

            if args.debug:
                mmtabqa_dataset = mmtabqa_dataset.select(range(min(100, len(mmtabqa_dataset))))

            all_results = {}

            while current_index < len(mmtabqa_dataset):
                end_index = min(current_index + shard_size, len(mmtabqa_dataset))
                print(f"\nProcessing shard {current_index}:{end_index}. Total: {len(mmtabqa_dataset)}. Progress: {current_index / len(mmtabqa_dataset):.2%}")  # fmt: skip

                shard_dataset = mmtabqa_dataset.select(range(current_index, end_index))
                if dataset_name == "HybridQA":
                    prompts = create_prompts(shard_dataset, dataset_name, start_index=current_index, is_retrieval_hybrid_qa=True, disable_cot=disable_cot)  # fmt: skip
                else:
                    prompts = create_prompts(shard_dataset, dataset_name, start_index=current_index, disable_cot=disable_cot)
                vllm_generator.generate_batch_pass(prompts)
                all_results.update(prompts)
                
                # Clean up images from all_results to prevent accumulation
                for eid in prompts.keys():
                    if eid in all_results and "content" in all_results[eid]:
                        content_items = all_results[eid]["content"]
                        if isinstance(content_items, list):
                            for item in content_items:
                                if isinstance(item, dict) and item.get("type") == "image" and "image" in item:
                                    del item["image"]  # Remove the actual image object
                
                # Clean up memory after each shard
                del prompts, shard_dataset
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                current_index = end_index

            # store generations
            store_generations(
                all_results,
                model_name_short=model_name,
                mode=f"interleaved_{retrieval_mode}",
                dataset_name=dataset_name,
                dataset_split_name=split_name,
            )
            evaluate(
                all_results,
                vllm_generator,
                dataset_name,
                model_name_short=model_name,
                dataset_split=split_name,
                mode=f"interleaved_{retrieval_mode}",
                use_llm_as_judge=(dataset_name != "FetaQA"),
            )


def main_for_mmtabreal(vllm_generator, model_name, args, disable_cot=False):
    """
    Process MMTabReal datasets with two-phase approach: generation then evaluation.
    """
    mmtabreal_base_path = os.getenv("MMT_BENCH_BASE_PATH")
    if not mmtabreal_base_path:
        raise ValueError("MMT_BENCH_BASE_PATH must be set for MMTabReal evaluation.")
    mmtabreal_path = os.path.join(mmtabreal_base_path, "hf_dataset")  # path to MMTabReal HF dataset
    print(f"\n=== MMTabReal Dataset Mode ===")
    print(f"Model: {model_name}")
    print(f"Dataset Path: {mmtabreal_path}")
    
    from tqdm import tqdm
    
    question_types = ["EQ", "VQ", "AQ", "IQ"]
    processed_types = []
    
    # Check which results already exist
    existing_results = []
    for q_type in question_types:
        results_file = f"results/{model_name}/MMTabReal_{q_type}_interleaved.json"
        if os.path.exists(results_file):
            print(f"Found existing results for {q_type}: {results_file}")
            existing_results.append(q_type)
            processed_types.append(q_type)
    
    # Phase 1: Generate all results (skip if all exist)
    if len(existing_results) == len(question_types):
        print(f"\n{'='*60}")
        print("All results already exist - skipping generation phase")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("PHASE 1: Generating results for missing question types")
        print(f"{'='*60}")
    
    for q_type in question_types:
        if q_type in existing_results:
            print(f"Skipping {q_type} - results already exist")
            continue
            
        dataset_path = os.path.join(mmtabreal_path, f"mmtabreal_{q_type}")
        
        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing MMTabReal-{q_type} dataset...")
        print(f"{'='*60}")
        
        mmtabreal_dataset = load_mmtabreal_dataset(dataset_path)
        
        if args.n_examples is not None:
            mmtabreal_dataset = mmtabreal_dataset[:args.n_examples]
        
        if args.debug:
            mmtabreal_dataset = mmtabreal_dataset[:5]
        
        current_index = 0
        shard_size = 50  # Reduced to avoid CUDA OOM with sampled datasets
        all_results = {}
        
        while current_index < len(mmtabreal_dataset):
            end_index = min(current_index + shard_size, len(mmtabreal_dataset))
            print(f"\nProcessing shard {current_index}:{end_index}. Total: {len(mmtabreal_dataset)}. Progress: {current_index / len(mmtabreal_dataset):.2%}")
            
            shard_dataset = mmtabreal_dataset[current_index:end_index]
            prompts = create_prompts(shard_dataset, "MMTabReal", start_index=current_index, disable_cot=disable_cot)
            vllm_generator.generate_batch_pass(prompts)
            all_results.update(prompts)
            
            # Clean up images from all_results to prevent accumulation
            for eid in prompts.keys():
                if eid in all_results and "content" in all_results[eid]:
                    content_items = all_results[eid]["content"]
                    if isinstance(content_items, list):
                        for item in content_items:
                            if isinstance(item, dict) and item.get("type") == "image" and "image" in item:
                                del item["image"]  # Remove the actual image object
            
            # Clean up memory after each shard
            del prompts, shard_dataset
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            current_index = end_index
        
        store_generations(
            all_results,
            model_name_short=model_name,
            mode="interleaved",
            dataset_name="MMTabReal",
            dataset_split_name=q_type,
        )
        
        processed_types.append(q_type)
        
        del all_results, mmtabreal_dataset
        gc.collect()
        
        print(f"Completed generating results for MMTabReal-{q_type}")
    
    # Phase 2: Evaluate all results with LLM-as-a-judge
    print(f"\n{'='*60}")
    print("PHASE 2: Evaluating results with LLM-as-a-judge")
    print(f"{'='*60}")
    
    # Evaluate all saved results
    for q_type in processed_types:
        print(f"\n{'='*60}")
        print(f"Evaluating MMTabReal-{q_type}...")
        print(f"{'='*60}")
        
        # Load saved results
        results_file = f"results/{model_name}/MMTabReal_{q_type}_interleaved.json"
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        
        # Use LLM-as-a-judge for all question types
        evaluate(
            all_results,
            vllm_generator,
            "MMTabReal",
            q_type,
            model_name_short=model_name,
            mode="interleaved",
            use_llm_as_judge=True,
        )
        
        print(f"Completed evaluation for MMTabReal-{q_type}")


######################################################## Main Function ########################################################
def main():
    os.makedirs("results", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--retrieval_output_dir",
        type=str,
        required=False,
        help="Path to the directory that we got through the retrieval pipeline, e.g.: data/row_wise_retrieval/",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("CAPTR_INTERLEAVED_MODEL", "google/gemma-3-27b-it"),
        help="Model name (set CAPTR_INTERLEAVED_MODEL in the environment or pass an explicit repo id / snapshot path)",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=700,
        help="Number of examples to sample from each dataset split. If None, use all examples.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory reserved for vLLM (e.g., 0.75).",
    )
    parser.add_argument(
        "--mmtabreal",
        action="store_true",
        help="Process MMTabReal datasets",
    )
    args = parser.parse_args()

    vllm_generator, model_name = get_vllm_generator(
        model_name=args.model,
        prompt_mode="text-image",
        limit_mm_per_prompt={"image": 324},
        context_window_size=90000,
        number_of_gpus=args.num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    disable_cot = (
        args.mmtabreal
        or is_qwen35_model(args.model)
        or is_qwen35_model(getattr(vllm_generator, "model_name", ""))
    )
    if args.mmtabreal:
        print("MMTabReal mode - forcing direct-answer prompts (no CoT, answer-only output).")
    elif disable_cot:
        print("Detected Qwen3.5 model - using direct-answer prompts without CoT instructions.")

    if args.mmtabreal:
        main_for_mmtabreal(vllm_generator, model_name, args, disable_cot=disable_cot)
    elif args.retrieval_output_dir:
        main_for_retrieval(vllm_generator, model_name, args, disable_cot=disable_cot)
    else:
        main_for_mmtabqa(vllm_generator, model_name, args, disable_cot=disable_cot)


if __name__ == "__main__":
    main()
