import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

from utils.runtime_env import configure_runtime_environment

configure_runtime_environment()

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

import cv2
import dotenv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from transformers import AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
#sys.path.append(str(Path(__file__).parent.parent))
from baselines.baselines_utils import (
    DATASET_PATHS,
    evaluate,
    get_metadata,
    get_vllm_generator,
    sample_dataset,
    store_generations,
)
from mmtabqa.load_mmtabqa_utils import load_mmtabqa_dataset
from utils.utils import build_passage_context

from baselines.sparsevlm.dynamic_sparsifier import (
    DynamicTokenSparsifier, 
    select_text_raters_exact,
    PRUNING_LAYERS,
    TOKEN_BUDGETS
)

TEXT_PROMPT_MMTABREAL = """You will be provided a table where some cells are images.
Your task is to:

Step 1: UNDERSTAND THE TABLE CONTEXT - Carefully analyze the table structure and understand the intricate relationship between image and text.

Step 2: ANALYZE THE QUESTIONS - Read all the questions provided and explore ALL TYPES OF REASONING to find answers.

Step 3: PROVIDE ANSWERS IN FORMAT - Ensure that all answers adhere strictly to the FORMAT specified. Avoid deviating from this format or including unnecessary explanations. DO NOT add any extra text beyond what is required.

ANSWER FORMATTING GUIDELINES:   

Sentences must be in string format without any bullet points or numbering.
Offer no explanations or justifications for your answers.
if a number is required, provide it in numeric format without words.
YOU HAVE TO ANSWER. YOU CANNOT RETURN BLANK RESPONSES.
ALWAYS PROVIDE YOUR ANSWERS IN THIS FORMAT.

IMPORTANT: ALL answers are there in the table/images.
Now I will provide you with the table.
"""

x="""
Answer Format Rules:

1. Single Entity: Return a single string representing one entity such as a name, country, company, object, or similar. The answer should be concise and written in one line without extra text.
   Examples:
   - Name: Elon Musk
   - Country: China
   - Company: Google
   - Color: Red

2. Single Number: If the answer is a whole number, write it without decimals. If it has decimals, round to two decimal places. If the last digit after rounding is 0 (e.g., 23.40), remove the trailing zero (→ 23.4). Units should only be included if explicitly mentioned in the question.
   Examples:
   - Whole Number: 45
   - Decimal: 12.36
   - Trimmed Decimal: 23.4

3. Multiple Entities: Provide a list of strings, each following the same rules as the Single Entity format. Use comma-separated values enclosed in square brackets.
   Example: ["Apple", "Microsoft", "Google"]

4. Multiple Numbers: Provide a list of numbers, each following the Single Number formatting rule. Use comma-separated values enclosed in square brackets.
   Example: [23, 45.67, 89.4]

5. Image Locations: When the answer involves identifying a location within a visual or tabular structure, specify it using the format: row_num_col_num
   Example: row_2_col_3

Now I will provide you with the table.
"""

TEXT_PROMPT_FETA_8_SHOTS = """Answer in a sentence, using the table data given. The table consists of data in the form of text and images. Each row of the table has been represented using [] with data for each column in the row separated by a semi-colon.
In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

Your answer must always include "Step 2:". After you have written "Step 2:", you should only state the actual answer and nothing else.

You are also provided with some question-answer examples for better understanding the format of providing the answer:

Example 1:
Table context: Table related to NHL awards in context of 2013–14 NHL season.\n\nQuestion: Which teams were competing for the Stanley Cup in the 2013-14 NHL season?
Step 1: The Stanley Cup is the silver-coloured cup represented in the first row, Award column. In the same row under the reciepient's column, we can see a Black-coloured logo with LA written on it, which is the logo for <>.Also in the runners-up column, we can see a Blue-coloured logo with "New-York Rangers" written on it. Thus, we can conclude that The Los Angeles Kings won the Stanley Cup, defeating the New York Rangers.\n\nStep 2:\nThe Los Angeles Kings won the Stanley Cup, defeating the New York Rangers.

Example 2:
Table context: Table related to International competitions in context of Debbie Marti.\n\nQuestion: In which city was the 1991 World Championships held and what distance did Debbie Marti achieve to qualify?
Step 1: As we can see in the 5th row, the Competiton represented in the Competitions column by a Blue-coloured logo is the World Championships. In the same row, under the venue column, we can see a collage of pictures of the prominent buildings from Tokyo. Thus We can conclude that the venue of the competiton was Tokyo. Also, in the column "Notes", we can see that Debbie Marti qualified with 1.86m.\n\nStep 2: At the 1991 World Championships in Tokyo, Debbie Marti qualified with 1.86 m.

Example 3:
Table context: Table related to Awards and nominations in context of Project Gutenberg (film).\n\nQuestion: What awards did Project Gutenberg win at the 38th Hong Kong Film Awards?
Step 1: The answer to the question can be found by looking at the column titled "Award" in the table. We can infer that there are seven rows in the table, each corresponding to an award won by Project Gutenberg. The categories listed are (Best Film, Best Director, Best Screenplay, Best Cinematography, Best Film Editing, Best Art Direction, and Best Costume Make Up Design) exactly match up to the categories listed in the "Award" column. So, to find the answer, you would need to look for each of these categories in the "Award" column and see which movie title is listed next to it.\n\nStep 2:Project Gutenberg won seven awards at the 38th Hong Kong Film Awards, in the categories Best Film, Best Director, Best Screenplay, Best Cinematography, Best Film Editing, Best Art Direction, and Best Costume Make Up Design.

Example 4:
Table context: Table related to Awards and nominations in context of Mike Cahill (director).\n\nQuestion: What film won the Alfred P. Sloan Prize at the Sundance Film Festival in 2014?
Step 1: Look at the "Year" column and find the year 2014. Then, look at the "Award" column for that row. If it says "Alfred P. Sloan Prize", then the movie title in the "Film" column for that row is the answer. In the table you described, on the row where "Year" is 2014, "Award" is "Alfred P. Sloan Prize", and "Film" is "I Origins".\n\nStep 2:\nCahill's film I Origins again won the Alfred P. Sloan Prize at the 2014 Sundance Film Festival, his second time receiving the award.

Example 5:
Table context: Table related to Home attendances in context of 2012–13 Everton F.C. season.\n\nQuestion: How did Everton F.C. do against Manchester United and Tottenham Hotspur during their 2012-13 season?
Step 1: Look for Manchester United and Tottenham Hotspur on the "Opponent" column.  Look at the corresponding "Score" for each team. For Manchester United, the score is  1-0 in favor of Everton. For Tottenham Hotspur, the score is 2-1 in favor of Everton. Therefore, Everton won against both Manchester United and Tottenham Hotspur.\n\nStep 2:\nEverton F.C. won over Manchester United in the first game of the season with 1–0, defeated Tottenham Hotspur 2–1, and defeated Manchester City 2–0 in the Premier League.

Example 6:
Table context: Table related to International competitions in context of Süreyya Ayhan.\n\nQuestion: How did Sureyya Ayhan fare at the 2003 World Championships?
Step 1: Look for "2003" in the "Year" column. Look across that row to the "Competition" column. It should say "World Championships". In the "Event" column, it shows "1500 m". Finally, under the "Position" column, it shows "2nd", indicating that Süreyya Ayhan won a silver medal.\n\nStep 2:\nSüreyya Ayhan won a silver medal in the 1500 m of the 2003 World Championships.

Example 7:
Table context: Table related to Grammy Awards in context of Roberta Flack.\n\nQuestion: When and for which songs did the singer Roberta Flack win Grammy Awards for Record of they Year?
Step 1: Looking at the table under the "Year" column, you can see 1973 listed twice.  In the corresponding rows under "Award" it says "Record of the Year" each time.  Looking at the "Nominee / work" column for those two rows, it shows "The First Time Ever I Saw Your Face" in 1973 and "Killing Me Softly With His Song" in 1974.  This confirms that Flack won the award for these two songs in consecutive years.\n\nStep 2:\nFlack won the Grammy Award for Record of the Year on two consecutive years: "The First Time Ever I Saw Your Face" won at the 1973 Grammys as did "Killing Me Softly with His Song" at the 1974 Grammys.

Example 8:
Table context: Table related to Television series in context of Kim Jung-hyun (actor, born 1990).\n\nQuestion: What did Kim Jung-hyun do in KBS2 in 2017?
Step 1: Look for the year "2017" in the "Year" column. Look across that row to the "Network" column. It should say "KBS2". In the "Title" column, it shows "School 2017". This indicates that Kim Jung-hyun played in that drama in 2017 on KBS2.\n\nStep 2:\\In 2017, Kim Jung-hyun played in KBS2's School 2017.

Now, based upon the examples given above, you must understand the text and images given in the table and follow the steps 1-2 to answer the question corresponding to the table represented bt the data. Try to keep the answer in active voice. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.
"""

TEXT_PROMPT_WTQ_8_SHOTS = """Answer in a sentence, using the table data given. The table consists of data in the form of text and images. Each row of the table has been represented using [] with data for each column in the row separated by a semi-colon.
In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

Your answer must always include "Step 2:". After you have written "Step 2:", you should only state the actual answer and nothing else.

You are also provided with some question-answer examples for better understanding the format of providing the answer:

Example 1:
Table context: Table related to Fifth round proper in context of 1975–76 FA Cup.\n\nQuestion: how many games played by sunderland are listed here?
Step 1: We can conclude Sunderland played in 2 games. The table shows teams listed under "Home team" and "Away team" columns [column headers provide this information].  Looking across the rows, Sunderland's logo, which comprises of 2 horses to the side and a Black&White sheild in between, is listed under one of these columns twice [in the 2nd and 3rd row]. Therefore, Sunderland participated in two games.\n\nStep 2:\n2

Example 2:
Table context: Table related to Complete Formula One World Championship results in context of Playlife.\n\nQuestion: when was the benetton b198 chassis used?
Step 1: The table shows Formula One results with a context of Playlife, possibly a constructor. As we can see, the Benetton b198 Chassis is the blue coloured supporting structure, as seen in the chassis column of the table. In the same row, there is the column year, which gives us the answer as 1998.\n\nStep 2: 1998.

Example 3:
Table context: Table related to Defunct railroads in context of List of Washington, D.C., railroads.\n\nQuestion: was the pennsylvania railroad under the prr or the rf&p?
Step 1: The table shows defunct railroads in Washington D.C. The "Pennsylvania Railroad" is the golden background picture represented in the 11th row with trains visible in it. In the same row, another column named "Mark" has the abbreviation as "PRR". Thus, the Pennsylvania Railroad operated under PRR since "PRR" is its short name.\n\nStep 2:PRR

Example 4:
Table context: Table related to Schedule and results in context of 2013–14 Chicago State Cougars women's basketball team.\n\nQuestion: how many games were played against grand canyon?
Step 1: We can see that there are 2 instances of the grand canyon in the opponent column. One in the 20th row, where there is a purple coloured logo which says GCC, which refers to the Grand Canyon College. Another is in the 26th row. Thus, we can conclude that 2 matches were played against the grand canyon.\n\nStep 2:\n2

Example 5:
Table context: Table related to Roster|Letter winners in context of 1915 Michigan Wolverines football team.\n\nQuestion: how many players were taller and weighed more than frank millard?
Step 1: Frank Millard is the clean-shaved, short haired guy visible in the 5th row. His height is 5'7 and weight is 212. Thus clearly, there are only 2 players whose height and weight is more than his, one in the 2nd row and other in the 8th row.\n\nStep 2:\n2

Example 6:
Table context: Table related to Racing record|Career summary in context of Conor Daly.\n\nQuestion: the two teams who raced in 2011 are carlin motorsport and what other team?
Step 1: In the year column there are 2 rows which have a mention of 2011. Apart from Carlin motorsport, the other one has a green car with the logo Schmidt Motorsports on it.\n\nStep 2:\nSchmidt Motorsports

Example 7:
Table context: Table related to Regular season|Schedule in context of 1995 New York Jets season.\n\nQuestion: team that scored more than 40 points against the jets that is not the miami dolphins
Step 1: As clearly visible, the opponent mentioned in the 4th row, which has a Black-coloured logo written as "RAIDERS" on it scored 47 goals against the jets. The logo is of the team Oakland Raiders. Thus, We can conclude that Oakland Raiders is the other team that scored 47 goals against the jets.\n\nStep 2:\nOakland Raiders

Example 8:
Table context: Table related to Winners|By Country in context of EHF Cup Winners' Cup.\n\nQuestion: did france or croatia have a larger finals total?
Step 1: Under the country column, in the 5th row we can see a Blue-coloured chicken logo, with FFHANDBALL written under it. That is the logo for France's handball federation. In the 8th row, we can see and Red-Blue coloured handall logo, with the Croatia Handball federation written underneath it, which represents Croatia. Thus, we can conclude that France had more Finals Total, 4, than Croatia, 1.\n\nStep 2:\nFrance

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


PRUNING_LAYERS = [3, 9, 18]  # changed to match gemma layers
TOKEN_BUDGETS = {
    192: [300, 200, 118],  # same budget as llava in sparsevlm repo
    128: [238, 108, 60],
    96: [246, 54, 28],
    64: [66, 34, 20]
}

base_dir = os.path.dirname(os.path.abspath(__file__))
dotenv.load_dotenv("../.env")


def find_image_blocks(input_ids: torch.Tensor, processor) -> List[Tuple[int, int]]:
    start_tok = processor.tokenizer.convert_tokens_to_ids("<start_of_image>")
    image_tok = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
    ids = input_ids.tolist()
    starts = [i for i, v in enumerate(ids) if v == start_tok]
    blocks = []
    for s in starts:
        i = s + 1
        while i < len(ids) and ids[i] == image_tok:
            i += 1
        if i > s + 1:
            blocks.append((s + 1, i - 1))  
        else:
            blocks.append((s, s))
    return blocks

def create_prompt_for_example(example, few_shot_example_text, selected_rows=None):
    """
    Create a prompt for a single MMTabQA example with images.
    Similar to interleaved_baseline.py's create_prompt_for_example.
    Supports cell types: "text", "image", and "image/text" (mixed).
    """
    table_metadata, question = get_metadata(example)
    example["table_with_metadata"] = copy.deepcopy(example["table"])
    passage_context_text = build_passage_context(example, selected_rows=selected_rows)

    header = example["table"]["header"]
    rows = example["table"]["rows"]
    
    table_array = []
    table_types = []
    for row in rows:
        if isinstance(row, dict) and "content" in row and "type" in row:
            table_array.append(row["content"])
            table_types.append(row["type"])
        elif isinstance(row, list) and len(row) > 0 and isinstance(row[0], str):
            row_data = [json.loads(cell) if isinstance(cell, str) else cell for cell in row]
            table_array.append([cell["content"] for cell in row_data])
            table_types.append([cell.get("type", "text") for cell in row_data])
        elif isinstance(row, str):
            row_data = json.loads(row)
            table_array.append(row_data)
            table_types.append([json.loads(cell)["type"] if isinstance(cell, str) else cell.get("type", "text") 
                               for cell in row_data])
        else:
            table_array.append(row["content"] if isinstance(row, dict) else row)
            table_types.append(row["type"] if isinstance(row, dict) else ["text"] * len(row))

    prompt_content = []

    prompt_content.append({"type": "text", "text": few_shot_example_text})

    prompt_content.append({"type": "text", "text": f"Table context: {table_metadata}\n"})

    if passage_context_text:
        prompt_content.append({"type": "text", "text": passage_context_text})

    prompt_content.append({"type": "text", "text": "Table:\n"})

    current_table_string = ""
    for cell in header:
        cell = cell.replace("\t", " ").replace("\n", " ").replace("\\n", " ").replace("\\t", " ").replace("|", " ")
        current_table_string = current_table_string + cell + " ; "
    current_table_string = current_table_string + "\n"

    for row_idx, row in enumerate(table_array):
        current_table_string += "["
        for cell_idx, cell in enumerate(row):
            cell_type = table_types[row_idx][cell_idx] if row_idx < len(table_types) and cell_idx < len(table_types[row_idx]) else "text"
            cell_content = cell
            
            if isinstance(cell, str):
                try:
                    cell_data = json.loads(cell)
                    if isinstance(cell_data, dict):
                        cell_type = cell_data.get("type", cell_type)
                        cell_content = cell_data.get("content", cell)
                except (json.JSONDecodeError, ValueError):
                    pass

            if cell_type == "image":
                # Check if there are multiple images separated by "|"
                if isinstance(cell_content, str) and "|" in cell_content:
                    image_paths = cell_content.split("|")
                    for img_path in image_paths:
                        img_path = img_path.strip()
                        if img_path:
                            if current_table_string.strip():
                                prompt_content.append({"type": "text", "text": current_table_string})
                                current_table_string = ""
                            prompt_content.append({"type": "image", "image": img_path})
                    current_table_string += " ; "
                else:
                    if current_table_string.strip():
                        prompt_content.append({"type": "text", "text": current_table_string})
                        current_table_string = ""
                    
                    prompt_content.append({"type": "image", "image": cell_content})
                    current_table_string += " ; "
            elif cell_type == "image/text":
                if isinstance(cell_content, str) and "TEXT:" in cell_content and "IMAGES:" in cell_content:
                    parts = cell_content.split("|IMAGES:")
                    text_part = parts[0].replace("TEXT:", "").strip()
                    images_part = parts[1].strip() if len(parts) > 1 else ""
                    
                    if text_part:
                        text_part = text_part.replace("\t", " ").replace("\n", " ").replace("\\n", " ").replace("\\t", " ").replace("|", " ")
                        current_table_string += text_part + " "
                    
                    if images_part:
                        image_paths = images_part.split("|")
                        for img_path in image_paths:
                            img_path = img_path.strip()
                            if img_path:
                                if current_table_string.strip():
                                    prompt_content.append({"type": "text", "text": current_table_string})
                                    current_table_string = ""
                                prompt_content.append({"type": "image", "image": img_path})
                    
                    current_table_string += " ; "
                else:
                    content_str = str(cell_content).replace("\t", " ").replace("\n", " ").replace("\\n", " ").replace("\\t", " ").replace("|", " ")
                    current_table_string += content_str + " ; "
            else: 
                content_str = str(cell_content).replace("\t", " ").replace("\n", " ").replace("\\n", " ").replace("\\t", " ").replace("|", " ")
                current_table_string += content_str + " ; "
                
        current_table_string += "]\n"

    prompt_content.append(
        {
            "type": "text",
            "text": current_table_string + f"\nQuestion: {question}",
        }
    )

    return prompt_content


def create_prompts(mmtabqa_dataset, dataset_name, start_index=0, is_retrieval_hybrid_qa=False):
    """
    Create prompts for all examples in a dataset.
    Returns a dict mapping example index to prompt data.
    """
    prompts = {}
    for i, example in enumerate(mmtabqa_dataset):
        few_shot_example_prompts_map = {
            "WikiTQ": TEXT_PROMPT_WTQ_8_SHOTS,
            "WikiSQL": TEXT_PROMPT_WIKISQL_8_SHOTS,
            "FetaQA": TEXT_PROMPT_FETA_8_SHOTS,
            "HybridQA": TEXT_PROMPT_WTQ_8_SHOTS,
            "MMTabReal": TEXT_PROMPT_MMTABREAL,
        }
        few_shot_example_text = few_shot_example_prompts_map[dataset_name]

        if is_retrieval_hybrid_qa:
            retrieved_row_indices = example["retrieved_row_indices"]
            retrieved_row_indices = [f"row {idx}" for idx in retrieved_row_indices]
            prompt_content = create_prompt_for_example(example=example, few_shot_example_text=few_shot_example_text, selected_rows=retrieved_row_indices)
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


def batched_vision_encoding(model, pixel_values, batch_size=3):
    """
    Process images through vision tower + projector in batches to avoid OOM.
    Reduces batch size to 3 images at a time for 896x896 resolution.
    
    Args:
        model: Gemma3ForConditionalGeneration model
        pixel_values: Tensor of shape (num_images, 3, H, W)
        batch_size: Number of images to process at once (default 3 for memory safety)
    
    Returns:
        Concatenated projected image features tensor (num_images, 256, hidden_dim)
    """
    num_images = pixel_values.shape[0]
    image_features_list = []
    
    for i in range(0, num_images, batch_size):
        batch_end = min(i + batch_size, num_images)
        batch_pixels = pixel_values[i:batch_end]
        
        with torch.no_grad():
            vision_features = model.model.vision_tower(pixel_values=batch_pixels).last_hidden_state
            projected_features = model.model.multi_modal_projector(vision_features)
            image_features_list.append(projected_features)
        
        del batch_pixels, vision_features, projected_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if num_images > 20 and (i + batch_size) % 15 == 0:
            print(f"  Encoded {min(i + batch_size, num_images)}/{num_images} images...")
    
    all_image_features = torch.cat(image_features_list, dim=0)
    del image_features_list
    
    return all_image_features


def process_single_example_with_sparsevlm(
    example_data,
    model,
    processor,
    device,
    topk_ratio=0.7,
):
    """
    Process a single example with dynamic SparseVLM pruning and generate output.
    Uses forward hooks for multi-layer progressive sparsification at layers [3, 9, 18].
    
    Args:
        example_data: Dict with "content" (prompt), "question", etc.
        model: Gemma model
        processor: AutoProcessor
        device: torch device
        topk_ratio: Ratio of tokens to keep (0.7 = keep 70%)
    
    Returns:
        Generated text output
    """
    prompt_content = example_data["content"]
    question = example_data["question"]
    
    prompt_text = ""
    images = []
    
    for item in prompt_content:
        if item["type"] == "text":
            prompt_text += item["text"]
        elif item["type"] == "image":
            img = item["image"]
            if isinstance(img, str):
                try:
                    img_path = img.replace("\\", "/")
                    loaded_img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"Warning: Failed to load image {img}: {e}")
                    loaded_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            elif isinstance(img, Image.Image):
                loaded_img = img
            else:
                loaded_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            
            images.append(loaded_img)
            prompt_text += "<start_of_image>"
    
    image_token_count = prompt_text.count("<start_of_image>")
    print(f"Debug: Prompt has {image_token_count} <start_of_image> tokens, {len(images)} images loaded")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    try:
        # Handle case where there are no images (text-only table or images only in header)
        if len(images) == 0:
            # Process as text-only
            model_inputs = processor(text=prompt_text, return_tensors="pt")
        else:
            model_inputs = processor(text=prompt_text, images=images, return_tensors="pt")
    except Exception as e:
        print(f"Error during processor call with {len(images)} images: {e}")
        raise
        
    if "pixel_values" in model_inputs:
        pv_shape = model_inputs["pixel_values"].shape
        pv_size_gb = model_inputs["pixel_values"].element_size() * model_inputs["pixel_values"].nelement() / (1024**3)
        #print(f"Pixel values shape: {pv_shape}, size: {pv_size_gb:.2f} GB")
    
    pixel_values = model_inputs.pop("pixel_values", None)  # Use pop with default None
    for k, v in model_inputs.items():
        if isinstance(v, torch.Tensor):
            model_inputs[k] = v.to(device)
    
    # Only process vision features if we have images
    if pixel_values is not None and len(images) > 6:
        pixel_values = pixel_values.to(device)
        image_features = batched_vision_encoding(model, pixel_values, batch_size=3)
        model_inputs["pixel_values"] = pixel_values
        model_inputs["_precomputed_image_features"] = image_features
        #print(f"Vision encoding complete: {image_features.shape}")
    elif pixel_values is not None:
        model_inputs["pixel_values"] = pixel_values.to(device)
    
    input_ids = model_inputs["input_ids"][0]
    blocks = find_image_blocks(input_ids, processor)
    
    # Only use dynamic sparsifier if there are images
    sparsifier = None
    original_get_image_features = None
    
    if len(images) > 0:
        # dynamic token averaging
        from baselines.sparsevlm.dynamic_sparsifier import DynamicTokenSparsifier, select_text_raters_exact
        
        question_token_ids = processor.tokenizer(question, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        question_indices = select_text_raters_exact(input_ids, question_token_ids, blocks)
        
        sparsifier = DynamicTokenSparsifier(
            model=model,
            processor=processor,
            pruning_layers=PRUNING_LAYERS,
            topk_ratio=topk_ratio
        )
        sparsifier.set_visual_token_info(blocks, question_indices)
        sparsifier.register_hooks()
        
        #print(f"Dynamic sparsifier ENABLED at layers {PRUNING_LAYERS}")
        #print(f"Text raters: {len(question_indices)} question tokens")
        #print(f"Visual blocks: {len(blocks)} image blocks")
        
        if "_precomputed_image_features" in model_inputs:
            precomputed_features = model_inputs.pop("_precomputed_image_features")
            original_get_image_features = model.model.get_image_features
            
            def patched_get_image_features(pixel_values):
                return precomputed_features
            
            model.model.get_image_features = patched_get_image_features
    
    try:
        with torch.no_grad():
            generation = model.generate(
                **model_inputs,
                max_new_tokens=100,
                do_sample=False,  
                use_cache=True, 
            )
            input_len = model_inputs["input_ids"].shape[-1]
            generation_text = generation[0][input_len:]
    finally:
        if sparsifier is not None:
            sparsifier.remove_hooks()
        
        if original_get_image_features is not None:
            model.model.get_image_features = original_get_image_features
            if "_precomputed_image_features" in locals():
                del precomputed_features
    
    decoded = processor.decode(generation_text, skip_special_tokens=True)
    
    del model_inputs, generation, generation_text
    del images, input_ids, blocks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    return decoded


def load_mmtabreal_dataset(dataset_path):
    """
    Load MMTabReal dataset from HuggingFace dataset format.
    
    Args:
        dataset_path: Path to the saved dataset directory
        image_base_path: Base path for resolving image paths (optional)
    
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


def main_for_mmtabqa(model, processor, device, model_name, args, judge_generator=None):
    """
    Process all MMTabQA datasets similar to interleaved_baseline.py.
    Phase 1: Generate all results
    Phase 2: Unload model, load judge, evaluate all results
    """
    from tqdm import tqdm
    
    # Track which datasets have been processed
    processed_datasets = []
    
    # Phase 1: Generate all results
    print(f"\n{'='*60}")
    print("PHASE 1: Generating results for all datasets")
    print(f"{'='*60}")
    
    for dataset_name, dataset_splits in DATASET_PATHS.items():
        for split_name, dataset_path in dataset_splits.items():
            dataset_key = (dataset_name, split_name)
            
            # Check if results already exist
            results_file = f"results/{model_name}/{dataset_name}_{split_name}_sparsevlm.json"
            if os.path.exists(results_file):
                print(f"Results already exist for {dataset_name}-{split_name} - skipping generation")
                processed_datasets.append(dataset_key)
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}-{split_name} dataset...")
            print(f"{'='*60}")

            mmtabqa_dataset = load_mmtabqa_dataset(
                dataset_path,
                image_base_path=os.getenv("MMTABQA_IMAGE_BASE_PATH"),
                load_images=True,
                partial_input_baseline=False,
            )

            if args.n_examples is not None:
                mmtabqa_dataset = sample_dataset(mmtabqa_dataset, n_examples=args.n_examples)

            if args.debug:
                mmtabqa_dataset = mmtabqa_dataset.select(range(min(5, len(mmtabqa_dataset))))

            prompts = create_prompts(mmtabqa_dataset, dataset_name)
            
            all_results = {}
            for idx_str, prompt_data in tqdm(prompts.items(), desc=f"Processing {dataset_name}-{split_name}"):
                try:
                    generated_text = process_single_example_with_sparsevlm(
                        prompt_data,
                        model,
                        processor,
                        device,
                        topk_ratio=args.topk_ratio,
                    )
                    
                    all_results[idx_str] = {
                        "question": prompt_data["question"],
                        "answer_text": prompt_data["answer_text"],
                        "generations": [generated_text],
                        "page_title": prompt_data["page_title"],
                        "section_title": prompt_data["section_title"],
                    }
                except Exception as e:
                    print(f"Error processing example {idx_str}: {e}")
                    all_results[idx_str] = {
                        "question": prompt_data["question"],
                        "answer_text": prompt_data["answer_text"],
                        "generations": ["ERROR"],
                        "page_title": prompt_data.get("page_title", ""),
                        "section_title": prompt_data.get("section_title", ""),
                    }
                
                if int(idx_str) % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

            store_generations(
                all_results,
                model_name_short=model_name,
                mode="sparsevlm",
                dataset_name=dataset_name,
                dataset_split_name=split_name,
            )
            
            processed_datasets.append(dataset_key)
            
            del prompts, all_results, mmtabqa_dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Completed generating results for {dataset_name}-{split_name}")
    
    # Phase 2: Evaluate all results (only if judge_generator is provided)
    if judge_generator is not None:
        print(f"\n{'='*60}")
        print("PHASE 2: Evaluating all results with LLM-as-a-judge")
        print(f"{'='*60}")
        
        for dataset_name, split_name in processed_datasets:
            print(f"\n{'='*60}")
            print(f"Evaluating {dataset_name}-{split_name}...")
            print(f"{'='*60}")
            
            # Load saved results
            results_file = f"results/{model_name}/{dataset_name}_{split_name}_sparsevlm.json"
            with open(results_file, 'r') as f:
                all_results = json.load(f)
            
            evaluate(
                all_results,
                judge_generator, 
                dataset_name,
                model_name_short=model_name,
                dataset_split=split_name,
                mode="sparsevlm",
                use_llm_as_judge=(dataset_name != "FetaQA"),
            )
            
            print(f"Completed evaluation for {dataset_name}-{split_name}")


def main():
    """
    Main function with CLI argument parsing.
    Supports single-table mode (--single_table), MMTabQA dataset mode, and MMTabReal dataset mode (--mmtabreal).
    """
    os.makedirs("results", exist_ok=True)
    parser = argparse.ArgumentParser(description="SparseVLM for multimodal table QA")
    parser.add_argument("--debug", action="store_true", help="Debug mode (process only 5 examples)")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-27b-it",
        help="Model name (e.g., 'google/gemma-3-27b-it')",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=None,
        help="Number of examples to sample from each dataset split. If None, use all examples.",
    )
    parser.add_argument(
        "--topk_ratio",
        type=float,
        default=0.7,
        help="Ratio of visual tokens to keep (0.7 = keep 70%)",
    )

    parser.add_argument(
        "--mmtabreal",
        action="store_true",
        help="Process MMTabReal datasets",
    )

    args = parser.parse_args()

    model_id = args.model
    print(f"Loading model: {model_id}")
    print(f"Cache directory: {HF_CACHE_DIR}")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPU(s) (after CUDA_VISIBLE_DEVICES filtering)")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU detected, using CPU")
    
    use_flash_attn = True

    if use_flash_attn:
        try:
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id, 
                device_map="cuda:0",  # Changed from "auto" to explicitly use first visible GPU (which is GPU 5)
                dtype=torch.bfloat16, 
                trust_remote_code=True, 
                attn_implementation="flash_attention_2",
                cache_dir=HF_CACHE_DIR,
                token=HF_TOKEN,
            ).eval()
        except Exception as e:
            print("FlashAttention failed:", e)

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled (reduces memory usage)")
    
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=HF_CACHE_DIR, token=HF_TOKEN)
    device = next(model.parameters()).device
    model_name = model_id.split("/")[-1]
    
    if args.mmtabreal:
        mmtabreal_base_path = os.getenv("MMT_BENCH_BASE_PATH")
        if not mmtabreal_base_path:
            raise ValueError("MMT_BENCH_BASE_PATH must be set for MMTabReal evaluation.")
        mmtabreal_path = os.path.join(mmtabreal_base_path, "hf_dataset") #path to MMTabReal HF dataset 
        print(f"\n=== MMTabReal Dataset Mode ===")
        print(f"Model: {model_name}")
        print(f"TopK Ratio: {args.topk_ratio}")
        print(f"Dataset Path: {mmtabreal_path}")
        
        from tqdm import tqdm
        
        question_types = ["EQ", "VQ", "AQ", "IQ"]
        processed_types = []
        
        # Check which results already exist
        existing_results = []
        for q_type in question_types:
            results_file = f"results/{model_name}/MMTabReal_{q_type}_sparsevlm.json"
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
            
            prompts = create_prompts(mmtabreal_dataset, "MMTabReal", start_index=0)
            
            all_results = {}
            for idx_str, prompt_data in tqdm(prompts.items(), desc=f"Processing MMTabReal-{q_type}"):
                try:
                    print(f"\n{'='*60}")
                    print(f"Example {idx_str}")
                    print(f"Question: {prompt_data['question']}")
                    print(f"Answer: {prompt_data['answer_text']}")
                    print(f"{'='*60}")
                    
                    generated_text = process_single_example_with_sparsevlm(
                        prompt_data,
                        model,
                        processor,
                        device,
                        topk_ratio=args.topk_ratio,
                    )
                    
                    print(f"Generated: {generated_text}")
                    print(f"{'='*60}\n")
                    
                    all_results[idx_str] = {
                        "question": prompt_data["question"],
                        "answer_text": prompt_data["answer_text"],
                        "generations": [generated_text],
                        "page_title": prompt_data.get("page_title", ""),
                        "section_title": prompt_data.get("section_title", ""),
                    }
                except Exception as e:
                    print(f"Error processing example {idx_str}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results[idx_str] = {
                        "question": prompt_data["question"],
                        "answer_text": prompt_data["answer_text"],
                        "generations": ["ERROR"],
                        "page_title": prompt_data.get("page_title", ""),
                        "section_title": prompt_data.get("section_title", ""),
                    }
                
                if int(idx_str) % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            
            store_generations(
                all_results,
                model_name_short=model_name,
                mode="sparsevlm",
                dataset_name="MMTabReal",
                dataset_split_name=q_type,
            )
            
            processed_types.append(q_type)
            
            del prompts, all_results, mmtabreal_dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Completed generating results for MMTabReal-{q_type}")
        
        # Phase 2: Unload SparseVLM and evaluate all results
        print(f"\n{'='*60}")
        print("PHASE 2: Evaluating results with LLM-as-a-judge")
        print(f"{'='*60}")
        
        if 'model' in locals():
            del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Load judge model for evaluation with reduced memory usage
        print("\nLoading judge model for evaluation (GPU memory utilization: 0.6)...")
        
        # Create generator with reduced memory
        from generation.generator_vllm_gemma3 import VLLMGeneratorGemma3
        
        judge_args = argparse.Namespace()
        judge_args.engine = "vllm"
        judge_args.vllm_model_name = model_id
        judge_args.number_of_gpus = 1
        judge_args.max_api_total_tokens = 90000
        judge_args.prompt_mode = "text-image"
        judge_args.prompt_style = ""
        judge_args.seed = 42
        
        # Create generator with reduced GPU memory
        generator = VLLMGeneratorGemma3(judge_args, system_prompt=None, gpu_memory_utilization=0.6)
        
        # Evaluate all saved results
        for q_type in processed_types:
            print(f"\n{'='*60}")
            print(f"Evaluating MMTabReal-{q_type}...")
            print(f"{'='*60}")
            
            # Load saved results
            results_file = f"results/{model_name}/MMTabReal_{q_type}_sparsevlm.json"
            with open(results_file, 'r') as f:
                all_results = json.load(f)
            
            # Use LLM-as-a-judge for all question types
            evaluate(
                all_results,
                generator,
                "MMTabReal",
                q_type,
                model_name_short=model_name,
                mode="sparsevlm",
                use_llm_as_judge=True,
            )
            
            print(f"Completed evaluation for MMTabReal-{q_type}")
        
        # Clean up judge model
        del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\n{'='*60}")
            print("Reloading SparseVLM model for next question type...")
            print(f"{'='*60}")
                
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
                
            use_flash_attn = True
            if use_flash_attn:
                model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_id,
                    cache_dir=HF_CACHE_DIR,
                    token=HF_TOKEN,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    attn_implementation="flash_attention_2",
                )
                
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=HF_CACHE_DIR, token=HF_TOKEN)
            device = next(model.parameters()).device
    else:
        print(f"\n=== MMTabQA Dataset Mode ===")
        print(f"Model: {model_name}")
        print(f"TopK Ratio: {args.topk_ratio}")
        print(f"Debug: {args.debug}")
        print(f"N Examples: {args.n_examples}")
        
        # Phase 1: Generate all results with SparseVLM model
        main_for_mmtabqa(model, processor, device, model_name, args, judge_generator=None)
        
        # Unload SparseVLM model
        print(f"\n{'='*60}")
        print("Unloading SparseVLM model to free memory...")
        print(f"{'='*60}")
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Phase 2: Load judge generator and evaluate all results
        print("\nLoading judge generator for evaluation...")
        from generation.generator_vllm_gemma3 import VLLMGeneratorGemma3
        
        judge_args = argparse.Namespace()
        judge_args.engine = "vllm"
        judge_args.vllm_model_name = model_id
        judge_args.number_of_gpus = 1
        judge_args.max_api_total_tokens = 90000
        judge_args.prompt_mode = "text-image"
        judge_args.prompt_style = ""
        judge_args.seed = 42
        
        judge_generator = VLLMGeneratorGemma3(judge_args, system_prompt=None, gpu_memory_utilization=0.6)
        
        # Run evaluation phase
        main_for_mmtabqa(None, None, None, model_name, args, judge_generator)
        
        # Clean up judge generator
        del judge_generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



if __name__ == "__main__":
    main()
