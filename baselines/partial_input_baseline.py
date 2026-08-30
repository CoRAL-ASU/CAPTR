import argparse
import copy
import gc
import json
import os
import pathlib
import random

import datasets
import pandas as pd
from tqdm import tqdm
import multiprocessing

from utils.runtime_env import configure_runtime_environment

multiprocessing.set_start_method('spawn', force=True)

configure_runtime_environment()


from baselines_utils import (
    DATASET_PATHS,
    evaluate,
    get_metadata,
    get_vllm_generator,
    sample_dataset,
    store_generations,
)
from datasets import concatenate_datasets, load_dataset
from mmtabqa.load_mmtabqa_utils import load_mmtabqa_dataset
from utils.utils import build_passage_context

load_dotenv(".env")


# prompts copied from MMTabQA repo with slight variations to make the qwen2.5-omni model follow the instruction format better

FEW_SHOT_EXAMPLES_PROMPT = """Table context: {table_metadata}

Table:
{table}

Question: {question}
Step 1: {reason}
Step 2: {answer}"""

FINAL_EXAMPLE_PROMPT = """Table context: {table_metadata}

Table:
{table}

Question: {question}
Step 1: """

TEXT_PROMPT_MMTABREAL="""
You will be provided a table in a pipeseparated table where all the entities have
been removed. Your task is to:

Step 1: UNDERSTAND THE TABLE CONTEXT - 
Carefully analyze the table structure and identify its purpose and what it mentions.

Step 2: FILL IN THE GAPS - 
Use the table context and your real-world knowledge to deduce the missing entities logically.

Step 3: ANALYZE THE QUESTIONS -
Read all the questions provided and explore **ALL TYPES OF REASONING** to find answers, including but not limited to Numerical reasoning(relationships, totals, and comparisons), Visual reasoning (Colors, shapes, or patterns), Contextual reasoning, (Real-world connections or logic), etc

Step 4: PROVIDE ANSWERS IN FORMAT - Ensure that all answers adhere strictly to the FORMAT specified. Avoid deviating from this format or including unnecessary explanations.
Answer Format Rules:

1. Single Entity: Return a single string representing one entity such as a name, country, company, object, or similar. The answer should be concise and written in one line without extra text.
   Examples:
   - Question: Who is the founder of Tesla?: Elon Musk
   - Question: Which country’s flag has a red background and five yellow stars? China
   - Question: Which company is known for its search engine? Google
   - Question: What color is the sky on a clear day? Red

2. Single Number: If the answer is a whole number, write it without decimals. If it has decimals, round to two decimal places. If the last digit after rounding is 0 (e.g., 23.40), remove the trailing zero (→ 23.4). Units should only be included if explicitly mentioned in the question.
   Examples:
   - Question: How many planets are in the solar system? → 8
   - Question: What is the value of π rounded to two decimals? → 3.14
   - Question: What is the average temperature in °C if stated as 23.40? → 23.4


3. Multiple Entities: Provide a list of strings, each following the same rules as the Single Entity format. Use comma-separated values enclosed in square brackets.
   - Question: Name three major technology companies → ["Apple", "Microsoft", "Google"]
   - Question: List the primary colors → ["Red", "Blue", "Yellow"]
   - Question: Which countries are permanent members of the UN Security Council? → ["United States", "United Kingdom", "China", "France", "Russia"]

4. Multiple Numbers: Provide a list of numbers, each following the Single Number formatting rule. Use comma-separated values enclosed in square brackets.
   Question: List the first five prime numbers → [2, 3, 5, 7, 11]
   Question: What are the recorded temperatures today in °C? → [23.5, 25, 21.4]
   Question: Provide the ages of the participants → [18, 24, 31]

5. Image Locations: When the answer involves identifying a location within a visual or tabular structure, specify it using the format: row_num_col_num
   Question: Provide the row and column number of the image which has a cat → row_2_col_3
   Question: Provide the row and column number of the image which has a red car → row_5_col_1
   Question: Provide the row and column number of the image which has a warning sign → row_1_col_4

ALWAYS PROVIDE YOUR ANSWERS IN THIS FORMAT.
If you are unable to answer it, simply answer UNKNOWN

I will provide one example to show you:

Example:
|Name |  |  | Salary|
|A | Electrician | New York | 65000|
|B | Carpenter |  | 58000|
|C |  | California | 62000|
|| Plumber | Florida | 60000|
|| Mechanic | Ohio | |


Inference:
Electrician, Carpenter, Plumber, Mechanic are jobs, so the second column is Job.
New York, California, Florida, Ohio are locations, so the third column is Location.


Sample Question:
Question: What does A do? → Electrician


Now I will provide you with the table and question

"""
TEXT_PROMPT_8_examples = """You are given a table in which some entities in various table cells have been replaced by tokens of the type '{{ENTITY-<entity_id>}}. Each row of the table is in separate lines, and the columns are separated by '|'. Based upon the context of the table and using real-world knowledge, your task is to answer a question based upon the table by guessing the replaced entities of the table. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question based upon the context of the table. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question and guessing various entities involved in finding the answer. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

Your answer must always include "Step 2:". After you have written "Step 2:", you should only state the actual answer and nothing else.

You are given some question-answer samples to better format for providing the answer. IMPORTANT: You must give the answer in the format "Step 2: <answer>".:

Example 1:
{example_1}

Example 2:
{example_2}

Example 3:
{example_3}

Example 4:
{example_4}

Example 5:
{example_5}

Example 6:
{example_6}

Example 7:
{example_7}

Example 8:
{example_8}

Now, using the above examples as context, answer the question given:
{main_part}"""

TEXT_PROMPT_5_examples = """You are given a table in which some entities in various table cells have been replaced by tokens of the type '{{ENTITY-<entity_id>}}. Each row of the table is in separate lines, and the columns are separated by '|'. Based upon the context of the table and using real-world knowledge, your task is to answer a question based upon the table by guessing the replaced entities of the table. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question based upon the context of the table. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question and guessing various entities involved in finding the answer. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

Your answer must always include "Step 2:". After you have written "Step 2:", you should only state the actual answer and nothing else.

You are given some question-answer samples to better format for providing the answer. IMPORTANT: You must give the answer in the format "Step 2: <answer>".:

Example 1:
{example_1}

Example 2:
{example_2}

Example 3:
{example_3}

Example 4:
{example_4}

Example 5:
{example_5}

Now, using the above examples as context, answer the question given:
{main_part}"""


TEXT_PROMPT_3_examples = """You are given a table in which some entities in various table cells have been replaced by tokens of the type '{{ENTITY-<entity_id>}}. Each row of the table is in separate lines, and the columns are separated by '|'. Based upon the context of the table and using real-world knowledge, your task is to answer a question based upon the table by guessing the replaced entities of the table. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question based upon the context of the table. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question and guessing various entities involved in finding the answer. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

Your answer must always include "Step 2:". After you have written "Step 2:", you should only state the actual answer and nothing else.

You are given some question-answer samples to better format for providing the answer. IMPORTANT: You must give the answer in the format "Step 2: <answer>".:

Example 1:
{example_1}

Example 2:
{example_2}

Example 3:
{example_3}

Now, using the above examples as context, answer the question given:
{main_part}"""

WIKI_TQ_FEW_SHOT_EXAMPLES_QUESTION_IDS = [
    "nt-5463",
    "nt-1454",
    "nt-8539",
    "nt-10664",
    "nt-5933",
    "nu-2806",
    "nt-1555",
    "nt-3613",
]

WIKISQL_FEW_SHOT_EXAMPLES_QUESTION_IDS = [
    "WSQL-74751",
    "WSQL-29267",
    "WSQL-56037",
    "WSQL-26020",
    "WSQL-23863",
    "WSQL-61633",
    "WSQL-67435",
    "WSQL-78329",
]

FetaQA_FEW_SHOT_EXAMPLES_QUESTION_IDS = [
    20919,
    15854,
    18189,
    8341,
    16849,
    10446,
    16422,
    2282,
]


QID_TO_REASON = {
    "nt-5463": "The table shows the medal count for various countries in the 1973 Asian Athletics Championships. We need to find the country/countries that have the same total medals as Thailand. We identify {ENTITY-7} as Thailand, which has a total of 4 medals (2 gold, 2 silver, and 0 bronze). Looking at the table, we see that {ENTITY-9} which is Iran and {ENTITY-2} which is Malayasia also have a total of 4 medals. Therefore, the answer is Iran and Malaysia.",
    "nu-1628": "The question asks for the player who earned less than $200 but more than $100 besides Ben Hogan. We know Ben Hogan is {ENTITY-9} from the table. Looking at the table, we see that the only player besides Ben Hogan who earned less than $200 but more than $100 is the player in the 8th place. The player in the 8th place is {ENTITY-8}, who is Henry Picard.",
    "nu-2495": "The table shows the schedule of the 1972 Minnesota Vikings season. We need to find out how many times the Vikings played at Three Rivers Stadium. To do this, we need to identify which team plays at Three Rivers Stadium. Three Rivers Stadium was the home stadium of the Pittsburgh Steelers. We need to find the rows in the table where the opponent is the Pittsburgh Steelers. The opponent on November 26 is Pittsburg Steelers, which is {ENTITY-2} and {ENTITY-3} represents the Three Rivers Stadium. Therefore, the answer is 1.",
    "nt-1454": 'The table shows the number of goals scored by different players in the 2010–11 PFC Levski Sofia season, categorized by competition. We need to find the row corresponding to Ismail Isa and then look the goals in the "Total" column. The row corresponding to Ismail Isa is the 3rd row and the total goals scored by him is 8. Therefore, the answer is 8.',
    "nt-8539": "The table shows the winners and runners-up of the men's and women's tournaments in the Old Four tournament.  We are looking for the winner of the women's tournament in 2003.  The table shows that the women's winner in 2003 was {ENTITY-2}.  We need to figure out what school {ENTITY-2} represents.  Looking at the table, we see that {ENTITY-2} is the winner of the men's tournament in 2003, 2004, 2006 and 2007 and runner-up in 2012 and 2013.  We also see that {ENTITY-2} is the runner-up in the women's tournament in 2004, 2006 and 2012.  This suggests that {ENTITY-2} is a school that is consistently competitive in the Old Four tournament.  Based on this information, we can guess that {ENTITY-2} represents the University of Western Ontario. We answer 'Western' based upon other entires of 'Toronto' and 'London' in the table.",
    "nt-10664": """ The table shows the release history of the album "Fables of the Reconstruction". We need to find the number of releases in compact disc format and the number of releases in cassette tape format. Then we need to subtract the number of cassette tape releases from the number of compact disc releases to find the difference. Looking at the table, we can see only {ENTITY-1} and {ENTITY-2} as the entity tokens in the Format column, which correspond to LP and cassette tape respectively. Looking at the format column, we can see 6 releases in Compact Disc format and 1 release in Cassette Tape format. Therefore, the answer is 6-1 = 5.""",
    "nt-5933": """The table shows the schedule of the 1974 Kansas City Chiefs season. We need to find the date when they played the Broncos and lost. Looking at the table, we see that the Chiefs played the Broncos in week 4, on October 6, 1974. The result column shows that they lost the game. Thus, the date is October 6, 1974.""",
    "nu-2806": """The question asks for the manufacturer(s) that appear the least on the chart. To answer this, we need to count how many times each manufacturer appears in the "Manufacturer" column. {ENTITY-1} only occurs twice and it corresponds to "New Flyer", while {ENTITY-4} also occurs twice and it corresponds to "Gillig". Therefore, the answer is New Flyer and Gillig.""",
    "nt-1555": """The question asks for the number of games played against the team whose logo features a red cardinal. We need to identify the team with a red cardinal logo from the table.  Real-world knowledge tells us that the Chicago Cardinals have a red cardinal logo.  We need to find the team name in the table that corresponds to the Chicago Cardinals.  Looking at the table, we see that the team name "Chicago Cardinals" is not explicitly mentioned. However, we can infer that the team with the red cardinal logo is the team that plays in Chicago. From real-world knowledge of game schedules and results in the 1952 season, we can determine that {ENTITY-6} is the Chicago Cardinals and is mentioned twice in the table.""",
    "nt-3613": """he question asks for the total medals won by the country whose flag is composed of white and yellow stripes. Looking at the table, we need to identify the country with a flag of white and yellow stripes.  Real-world knowledge tells us that the flag of Argentina is composed of white and yellow stripes.  We need to find the row corresponding to Argentina in the table and then find the value in the "Total" column for that row. From real-world knowledge of individual gold/silver/bronze medal wins in 2011 Pan American Games, we can identify that Argentina is represented by {ENTITY-6} in the table.  The table shows that Argentina won a total of 7 medals in the 2011 Pan American Games.""",
    "WSQL-74751": """The question asks for the home team that scored 12.6 (78). Looking at the table, we need to find the row where the "Home team score" column has the value "12.6 (78)".  We can then identify the corresponding "Home team" from that row. This corresponds to the sixth row, where the Home Team is {ENTITY-9}, which corresponds to hawthorn. Therefore, the answer is hawthorn.""",
    "WSQL-29267": """The table shows Carlo Simionato's achievements in various competitions. We need to find the Time for the European Cup competition held in Moscow. Looking at the table, we see that there are three rows with "European Cup" as the Competition. The first row has London as the Venue, and the second row has {ENTITY-4} as the Venue. We need to figure out what {ENTITY-4} represents. From the year column, we can see that this European Cup happened in 1985. From the real-world knowledge, we know that this European Cup was held in Moscow. Therefore, the time corresponding to this is 38.88.""",
    "WSQL-56037": """The table shows the Members of Parliament for the North Staffordshire constituency in the UK Parliament. The columns are Election, 1st Member, 1st Party, 2nd Member, and 2nd Party. The question asks for the 1st Party in the election of 1865. Looking at the row for 1865, we see that the 1st Party is listed as {ENTITY-7}. We need to guess what party this entity represents.  Since the table shows various political parties like Whig, Conservative, and {ENTITY-4}, it's likely that {ENTITY-7} also represents a political party. From real-world knowledge and based upon the 1st Member "Sir Edward Manningham-Buller, Bt" we can conclude that {ENTITY-7} corresponds to liberal.""",
    "WSQL-26020": """The table shows Margarita Ponomaryova's achievements in various competitions. We need to find the competition where she achieved 1st position with a note of 57.03. Looking at the table, we can see that the only row with a 1st position and a note of 57.03 is the one for the year 1989. The competition in that row is '{ENTITY-8}' and the venue is '{ENTITY-9} , {ENTITY-10}'. From real-world knowledge, we know that Margarita Ponomaryova won the 1989 World Student Games (Universiade) in the 400m hurdles event. Therefore, the answer is World Student Games (Universiade).""",
    "WSQL-23863": """The table shows the results of the promotion round for the 2nd Bundesliga. The first column shows the season, the second column shows the 16th placed team in the 2nd Bundesliga, the third column shows the 3rd placed team in the 3rd Liga, and the last two columns show the results of the two games played between the two teams. We are asked to find the 3rd Liga team from the game 1 of 0-1 between the years 2008-09. Looking at the table, we can see that the game 1 of 0-1 occurred in the 2008-09 season. The 3rd Liga team in that season is {ENTITY-2}. From real-world knowledge, we know that the 3rd Liga team in the 2008-09 season with score 0-1 in both game 1 and game 2 is SC Paderborn 07. Therefore, the answer is SC Paderborn 07.""",
    "WSQL-61633": """The table shows Cristina Fink's achievements in various competitions, including the Olympic Games. We need to find the venue where she achieved a DNQ (Did Not Qualify) position in the Olympic Games. Looking at the table, we see that she achieved DNQ in the 1992 Olympic Games. The venue for the 1992 Olympic Games is listed as {ENTITY-6}.  Based on real-world knowledge, we know that the 1992 Summer Olympics were held in Barcelona, Spain. Therefore, we can guess that {ENTITY-6} represents "Barcelona, Spain".""",
    "WSQL-67435": """The table shows information about the 1935 VFL season, specifically Round 13. We need to find the venue where South Melbourne played as the away team. Using real-world knowledge, we can guess that South Melbourne played Footscray in this round and corresponds to {ENTITY-2}. The corresponding venue from the table is Western Oval.""",
    "WSQL-78329": """The table shows the 2000 NFL Draft for the Baltimore Ravens.  We are looking for the round where Southern Mississippi is listed as the school/club team.  The table shows that Jamal Lewis was drafted in the first round from the University of Tennessee.  The next player listed is Travis Taylor from Florida.  The third round shows that the player was drafted from the Louisville.  The fifth round shows Miami (FL) as the school/club team.  The sixth round shows that the Linebacker player was drafted from the South Mississippi, which is listed as {ENTITY-5} in the table.  Therefore, Southern Mississippi must be the school/club team for the player drafted in the sixth round. Therefore, the answer is 6.""",
    20919: """The table shows the Prime Ministers of Qatar from 1970 to present.  The fourth row shows that {ENTITY-2} was appointed as Prime Minister on 3 April 2007.  The table does not provide a reason for his appointment, but we can infer that he was appointed because the previous Prime Minister, Abdullah bin Khalifa Al Thani, resigned. From real-world knowledge, we know that {ENTITY-2} who is the Prime Minister of Qatar after 2007 is Hamad bin Jassim bin Jaber Al Thani.""",
    15854: """The table shows the results of the United Bowl (IFL) games. We need to find the row corresponding to the game between Sioux Falls Storm and Tri-Cities Fever on July 14, 2012.  The table shows that the Sioux Falls Storm played against the Tri-Cities Fever on July 14, 2012, and the Sioux Falls Storm won with a score of 59 to 32.""",
    18189: """The table shows Joshua Grommen's club history, including the season, league, and number of appearances and goals. In 2018, he played for {ENTITY-3} in the {ENTITY-2} league. We can infer that {ENTITY-3} is a club and {ENTITY-2} is a league based on the table's structure and the context of the question.  We can also infer that {ENTITY-2} is likely a professional league, as it is listed alongside other leagues like NPL Queensland. Based on real-world knowledge, we can guess that {ENTITY-3} is Davao Aguilas FC and {ENTITY-2} is the Philippines Football League (PFL).""",
    8341: """The table lists Gwen Verdon's filmography, including the year, title, role, and notes. To answer the question, we need to find the row corresponding to the movie "Walking Across Egypt" and the row corresponding to the year 2000.
* **"Walking Across Egypt":** Based upon real-world knowledge, we know that this movie came out in 1999. We need to find the row where the "Title" column is "Walking Across Egypt" and the "Year" column is 1999. This row will tell us the role Gwen Verdon played in this movie. From the table, this role is Alora.
* **2000:** We need to find the row where the "Year" column is 2000. This row will tell us the title of the movie she appeared in that year. The movie corresponds to {ENTITY-14} where she played the role of MRs. Drago. Based upon real-world knowledge, we know that this is the movie Bruno, released in 2000.""",
    16849: """The question asks for the number of goals scored by Simon against a team whose flag has blue and white stripes at the 2011 FIFA Women's World Cup. Based upon real-world knowledge we know that this country is Norway. Looking at the table, we can see that Simon scored two goals against Norway, which is the team from the country of {ENTITY-28}. The table also shows that these goals were scored in the 2011 FIFA Women's World Cup, which is represented by {ENTITY-29}.""",
    10446: """The question asks which country scored better between Canada and the country whose flag has blue stripes. Looking at the table, Canada is listed as the third-place finisher with a total score of 97.357. The country whose flag has blue stripes is likely France, which corresponds to 4th rank due to French swimmers like Cinthia Bouhier, Charlotte Fabre, Myriam Glez. Canada scored better than France, finishing ahead of France by almost a full point (96.467).""",
    16422: """The table shows Peter McKennan's career statistics, including the number of appearances and goals scored for various clubs. The question asks for the total number of goals scored in 121 appearances for Patrick Thistle. We can find this information by looking at the rows for Patrick Thistle, which is represented by the token {ENTITY-4}. The table shows the number of appearances and goals scored for each season, and the total for all seasons. We need to find the total goals scored in 121 appearances. Finding the sum, we see that the total number of goals scored in 121 appearances for Patrick Thistle is 70.""",
    2282: """The table shows the career statistics of Yohan Betsch, a French professional footballer. The table lists the clubs he played for, the seasons, and the leagues. We need to find the clubs he played for during the 2011-2013 seasons. Looking at the table, we see that he played for {ENTITY-5} during the 2011-12 season and {ENTITY-1} during the 2012-13 season. From real-world knowledge we know that {ENTITY-5} is FC Metz and {ENTITY-1} is Ligue 2 side Laval. Therefore, Betsch was at Metz in the 2011–12 season, then he joined Ligue 2 side Laval in the 2012-13 season.""",
}

def convert_table_to_prompt(example, reason=None):
    """
    Convert a table example to a formatted prompt string.

    This function takes a table example and converts it into a text prompt format suitable for
    model input. The table is formatted with pipe-separated values, and metadata about the
    table (title and section) is included.

    Args:
        example (dict): A dictionary containing the table data with the following structure:
            - "question" (str): The question to be answered based on the table
            - "table" (dict): Table data containing:
                - "header" (list): List of header values
                - "page_title" (str): Page title of the table
                - "section_title" (str): Section of the table
                - "rows" (list): List of row dictionaries, each containing:
                    - "content" (list): List of cell values for that row
            - "answer_text" (list, optional): List of answer values (only used when reason is provided)
        reason (str, optional): If provided, formats as a few-shot example with reasoning.
                               If None, formats as the final example to be answered.

    Returns:
        str: A formatted prompt string containing:
            - Table metadata (title and section context)
            - Pipe-separated table content with rows on separate lines
            - The question to be answered
            - If reason is provided: includes the answer and reasoning as a few-shot example
            - If reason is None: formats as the main question to be answered

    Note:
        - Table cells are cleaned by removing tabs, newlines, and pipe characters
        - Uses FEW_SHOT_EXAMPLES_PROMPT when reason is provided
        - Uses FINAL_EXAMPLE_PROMPT when reason is None
    """
    table_metadata, question = get_metadata(example)
    example["table_with_metadata"] = copy.deepcopy(example["table"])
    passage_context_text = build_passage_context(example)
    header = example["table"]["header"]
    rows = example["table"]["rows"]
    table_array = [row["content"] for row in rows]
    table_string = ""

    for row in [header] + table_array:
        for cell in row:
            cell = cell.replace("\t", " ").replace("\n", " ").replace("\\n", " ").replace("\\t", " ").replace("|", " ")
            table_string = table_string + cell + " | "
        table_string = table_string + "\n"

    if reason is not None:
        answer_text = ", ".join([str(_) for _ in example["answer_text"]])
        return FEW_SHOT_EXAMPLES_PROMPT.format(
            table_metadata=table_metadata, table=table_string, question=question, answer=answer_text, reason=reason
        )

    if passage_context_text:
        return FINAL_EXAMPLE_PROMPT.format(
            table_metadata=table_metadata + "\n" + passage_context_text, table=table_string, question=question
        )
    else:
        return FINAL_EXAMPLE_PROMPT.format(table_metadata=table_metadata, table=table_string, question=question)


def get_few_shot_examples(dataset_name: str, num_few_shot_examples: int):
    if dataset_name in ["WikiTQ", "HybridQA"]:
        default_wikitq_dataset = load_dataset("stanfordnlp/wikitablequestions", trust_remote_code=True)
        default_wikitq_dataset = concatenate_datasets(
            [default_wikitq_dataset["train"], default_wikitq_dataset["test"], default_wikitq_dataset["validation"]]
        )
        MMTABQA_BASE_PATH = pathlib.Path(os.environ["MMTABQA_BASE_PATH"])
        meta_data_df = pd.read_csv(
            MMTABQA_BASE_PATH / "WikiTableQuestions" / "table-metadata.tsv", sep="\t"
        ).set_index("contextId")

        examples = []
        few_shot_examples_used = WIKI_TQ_FEW_SHOT_EXAMPLES_QUESTION_IDS[:num_few_shot_examples]
        for qid in few_shot_examples_used:
            print(f"qid: {qid}")
            example = default_wikitq_dataset.filter(lambda x: x["id"] == qid)[0]

            meta_data_id = example["table"]["name"].replace("tsv", "csv")
            metadata = meta_data_df.loc[meta_data_id]
            page_title = metadata["title"]
            section_title = metadata["headers"]

            # We need to bring it into the format of our HF dataset:
            example["table"] = {
                "header": example["table"]["header"],
                "page_title": page_title,
                "section_title": section_title,
                "rows": [{"content": row} for row in example["table"]["rows"]],
            }
            example["answer_text"] = example["answers"]

            prompt = convert_table_to_prompt(example, reason=QID_TO_REASON[qid])
            examples.append(prompt)
        random.shuffle(examples)
        return examples

    elif dataset_name == "WikiSQL":
        MMTabQA_BASE_PATH = pathlib.Path(os.environ["MMTABQA_BASE_PATH"])
        image_id_to_original_string = json.load(
            open(MMTabQA_BASE_PATH / "WikiSQL" / "image_id_to_original_string.json")
        )
        tables = []
        with open(MMTabQA_BASE_PATH / "WikiSQL" / "tables.jsonl", "r") as f:
            for line in f:
                tables.append(json.loads(line))

        all_examples = []
        for split_file in ["explicit_ans_mention.jsonl", "explicit_questions.jsonl", "implicit_questions.jsonl", "visual_questions.jsonl"]:  # fmt: skip
            with open(MMTabQA_BASE_PATH / "WikiSQL" / split_file, "r") as f:
                for line in f:
                    all_examples.append(json.loads(line))

        examples = []
        few_shot_examples_used = WIKISQL_FEW_SHOT_EXAMPLES_QUESTION_IDS[:num_few_shot_examples]
        for qid in few_shot_examples_used:
            print(f"qid: {qid}")
            # 1. Get the right question
            question_example = [example for example in all_examples if example["question_id"] == qid][0]

            # 2. Get the right table
            table_example = [table for table in tables if table["table_id"] == question_example["table_id"]][0]

            # 3. Prepare the example
            example = {
                "question": question_example["question"],
                "answer_text": question_example["answer"],
                "table": {
                    "header": table_example["table_array"][0],
                    "page_title": table_example["page_title"],
                    "section_title": table_example["section_title"],
                    "rows": [],
                },
            }

            for row in table_example["table_array"][1:]:
                row_content = []
                for value in row:
                    if value == "{IMG-{WSQ-2-15489384-1-6-2-0}} , {IMG-{WSQ-2-15489384-1-6-2-1}}":
                        value = f"{image_id_to_original_string['{IMG-{WSQ-2-15489384-1-6-2-0}}']} , {image_id_to_original_string['{IMG-{WSQ-2-15489384-1-6-2-1}}']}"
                    elif value == "{IMG-{WSQ-2-14617261-1-2-2-0}} , {IMG-{WSQ-2-14617261-1-2-2-1}}":
                        value = f"{image_id_to_original_string['{IMG-{WSQ-2-14617261-1-2-2-0}}']} , {image_id_to_original_string['{IMG-{WSQ-2-14617261-1-2-2-1}}']}"
                    elif value == "{IMG-{WSQ-2-14617261-1-5-2-0}} , {IMG-{WSQ-2-14617261-1-5-2-1}}":
                        value = f"{image_id_to_original_string['{IMG-{WSQ-2-14617261-1-5-2-0}}']} , {image_id_to_original_string['{IMG-{WSQ-2-14617261-1-5-2-1}}']}"
                    elif value.startswith("{IMG-"):
                        # replace image id with original string
                        value = image_id_to_original_string[value]
                        row_content.append(value)
                    else:
                        row_content.append(value)
                example["table"]["rows"].append({"content": row_content})

            # 4. Convert the table to a prompt
            prompt = convert_table_to_prompt(example, reason=QID_TO_REASON[qid])
            examples.append(prompt)
        random.shuffle(examples)
        return examples

    elif dataset_name == "FetaQA":
        fetaqa_dataset = load_dataset("DongfuJiang/FeTaQA")
        fetaqa_dataset = concatenate_datasets(
            [fetaqa_dataset["train"], fetaqa_dataset["test"], fetaqa_dataset["validation"]]
        )

        examples = []
        few_shot_examples_used = FetaQA_FEW_SHOT_EXAMPLES_QUESTION_IDS[:num_few_shot_examples]
        for qid in few_shot_examples_used:
            print(f"qid: {qid}")
            example = fetaqa_dataset.filter(lambda x: x["feta_id"] == qid)[0]
            example["table"] = {}
            example["table"]["header"] = example["table_array"][0]
            example["table"]["page_title"] = example["table_page_title"]
            example["table"]["section_title"] = example["table_section_title"]
            example["table"]["rows"] = [{"content": row} for row in example["table_array"][1:]]
            example["answer_text"] = [example["answer"]]
            prompt = convert_table_to_prompt(example, reason=QID_TO_REASON[qid])
            examples.append(prompt)
        random.shuffle(examples)
        return examples
    else:
        # Placeholder for other datasets
        raise NotImplementedError(f"Few-shot example retrieval for {dataset_name} is not yet implemented.")


def create_prompts(mmtabqa_wikitq_dataset, num_few_shot_examples, dataset_name):
    few_shot_examples = get_few_shot_examples(dataset_name, num_few_shot_examples)

    prompts = {}
    for i, example in enumerate(mmtabqa_wikitq_dataset):
        if example["id"] in WIKI_TQ_FEW_SHOT_EXAMPLES_QUESTION_IDS:
            continue

        # Bring example into few-shot template
        template_map = {8: TEXT_PROMPT_8_examples, 5: TEXT_PROMPT_5_examples, 3: TEXT_PROMPT_3_examples}
        template = template_map[num_few_shot_examples]
        example_kwargs = {f"example_{i + 1}": few_shot_examples[i] for i in range(num_few_shot_examples)}
        prompt = convert_table_to_prompt(example)
        final_prompt = template.format(main_part=prompt, **example_kwargs)

        # bring into format that our vllm_generator can handle (like the g_dicts in the H-STAR scripts)
        prompt_data = {
            "content": [{"type": "text", "text": final_prompt}],
            "answer_text": example["answer_text"],
            "question": example["question"],
            "page_title": example["table"]["page_title"],
            "section_title": example["table"]["section_title"],
        }
        prompts[str(i)] = prompt_data

    return prompts


def load_mmtabreal_dataset(dataset_path):
    """
    Load MMTabReal dataset from HuggingFace dataset format.
    
    Args:
        dataset_path: Path to the saved dataset directory
    
    Returns:
        List of examples compatible with convert_table_to_prompt
    """
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


def main_for_mmtabreal(vllm_generator, model_name, args):
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
    
    question_types = ["EQ", "VQ", "AQ", "IQ"]
    processed_types = []
    
    # Check which results already exist
    existing_results = []
    for q_type in question_types:
        results_file = f"results/{model_name}/MMTabReal_{q_type}_partial_input.json"
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
        
        # Create prompts using TEXT_PROMPT_MMTABREAL (no images for partial input)
        prompts = {}
        for i, example in enumerate(mmtabreal_dataset):
            # Use convert_table_to_prompt without reason (final example format)
            prompt = convert_table_to_prompt(example, reason=None)
            
            # Prepend the MMTabReal-specific prompt
            final_prompt = TEXT_PROMPT_MMTABREAL + prompt
            
            # Format as expected by vllm_generator
            prompt_data = {
                "content": [{"type": "text", "text": final_prompt}],
                "answer_text": example["answer_text"],
                "question": example["question"],
                "table_id": example.get("table_id", ""),
                "page_title": "",
                "section_title": "",
            }
            prompts[str(i)] = prompt_data
        
        vllm_generator.generate_batch_pass(prompts)
        
        store_generations(
            prompts,
            model_name_short=model_name,
            mode="partial_input",
            dataset_name="MMTabReal",
            dataset_split_name=q_type,
        )
        
        processed_types.append(q_type)
        
        del prompts, mmtabreal_dataset
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
        results_file = f"results/{model_name}/MMTabReal_{q_type}_partial_input.json"
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        
        # Use LLM-as-a-judge for all question types
        evaluate(
            all_results,
            vllm_generator,
            "MMTabReal",
            q_type,
            model_name_short=model_name,
            mode="partial_input",
            use_llm_as_judge=True,
        )
        
        print(f"Completed evaluation for MMTabReal-{q_type}")


def main():
    os.makedirs("results", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_few_shot_examples", type=int, default=8, choices=[8, 5, 3])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("CAPTR_PARTIAL_INPUT_MODEL", "google/gemma-3-27b-it"),
        help="Model name (set CAPTR_PARTIAL_INPUT_MODEL in the environment or pass an explicit repo id / snapshot path)",
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
        "--mmtabreal",
        action="store_true",
        help="Process MMTabReal datasets",
    )
    args = parser.parse_args()

    vllm_generator, model_name = get_vllm_generator(
        model_name=args.model,
        prompt_mode="text-image",
        limit_mm_per_prompt={"image": 1},
        number_of_gpus=args.num_gpus,
    )

    if args.mmtabreal:
        main_for_mmtabreal(vllm_generator, model_name, args)
        return

    for dataset_name, dataset_splits in DATASET_PATHS.items():
        for split_name, dataset_path in dataset_splits.items():
            print(f"Processing {dataset_name}-{split_name} dataset...")
            
            # Check if results already exist
            results_file = f"results/{model_name}/{dataset_name}_{split_name}_partial_input.json"
            if os.path.exists(results_file):
                print(f"\n{'='*60}")
                print(f"Results already exist for {dataset_name}-{split_name}")
                print(f"Skipping generation phase, proceeding to evaluation...")
                print(f"{'='*60}\n")
                
                # Load existing results and evaluate
                with open(results_file, 'r') as f:
                    prompts = json.load(f)
                
                evaluate(
                    outputs=prompts,
                    generator=vllm_generator,
                    dataset_name=dataset_name,
                    model_name_short=model_name,
                    dataset_split=split_name,
                    mode="partial_input",
                    use_llm_as_judge=(dataset_name != "FetaQA"),
                )
                continue

            mmtabqa_dataset = load_mmtabqa_dataset(
                dataset_path,
                load_images=False,
                partial_input_baseline=True,
            )

            # Apply sampling if n_examples is specified
            if args.n_examples is not None:
                mmtabqa_dataset = sample_dataset(mmtabqa_dataset, n_examples=args.n_examples, seed=42)

            if args.debug:
                mmtabqa_dataset = mmtabqa_dataset.select(range(min(100, len(mmtabqa_dataset))))

            prompts = create_prompts(mmtabqa_dataset, args.num_few_shot_examples, dataset_name)
            vllm_generator.generate_batch_pass(prompts)
            store_generations(
                prompts,
                model_name_short=model_name,
                dataset_name=dataset_name,
                dataset_split_name=split_name,
                mode="partial_input",
            )
            evaluate(
                outputs=prompts,
                generator=vllm_generator,
                dataset_name=dataset_name,
                model_name_short=model_name,
                dataset_split=split_name,
                mode="partial_input",
                use_llm_as_judge=(dataset_name != "FetaQA"),
            )


if __name__ == "__main__":
    main()
