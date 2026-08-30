# WikiTQ & TabFact Statistics

Below you see the output of the `analyse_tokens_per_ds.py`script. The cost estimate _($12.57 for WikiTQ; $32.13 for Tabfact)_ is for gpt-4o-mini

```
Processing wikitq: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4344/4344 [00:06<00:00, 685.56it/s]
wikitq has #4344 examples with a total of 4043521 tokens

Step 1: Col Select SQL
Prompt tokens: 1,547
Input tokens: 10,763,689    per example: 2,477.8289594843463
Output tokens: 1,303,200
Step 1 cost: $2.40

Step 2: Col Select Text
Prompt tokens: 656
Input tokens: 6,893,185    per example: 1,586.8289594843461
Output tokens: 1,303,200
Step 2 cost: $1.82

Step 3: Row Select SQL
Prompt tokens: 1,145
Input tokens: 9,017,401    per example: 2,075.8289594843463
Output tokens: 1,303,200
Step 3 cost: $2.13

Step 4: Row Select Text
Prompt tokens: 851
Input tokens: 7,740,265    per example: 1,781.8289594843461
Output tokens: 1,303,200
Step 4 cost: $1.94

Step 5: Reason Text
Prompt tokens: 1,157
Input tokens: 9,069,529    per example: 2,087.8289594843463
Output tokens: 1,303,200
Step 5 cost: $2.14

Step 6: Reason SQL
Prompt tokens: 1,157
Input tokens: 9,069,529    per example: 2,087.8289594843463
Output tokens: 1,303,200
Step 6 cost: $2.14

Total cost: $12.57
total tokens per example: 12097.973756906078
Processing tab_fact: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12779/12779 [00:02<00:00, 5003.53it/s]
tab_fact has #12779 examples with a total of 6498707 tokens

Step 1: Col Select SQL
Prompt tokens: 1,547
Input tokens: 26,267,820    per example: 2,055.5458173566008
Output tokens: 3,833,700
Step 1 cost: $6.24

Step 2: Col Select Text
Prompt tokens: 656
Input tokens: 14,881,731    per example: 1,164.5458173566008
Output tokens: 3,833,700
Step 2 cost: $4.53

Step 3: Row Select SQL
Prompt tokens: 1,145
Input tokens: 21,130,662    per example: 1,653.5458173566008
Output tokens: 3,833,700
Step 3 cost: $5.47

Step 4: Row Select Text
Prompt tokens: 851
Input tokens: 17,373,636    per example: 1,359.5458173566008
Output tokens: 3,833,700
Step 4 cost: $4.91

Step 5: Reason Text
Prompt tokens: 1,157
Input tokens: 21,284,010    per example: 1,665.5458173566008
Output tokens: 3,833,700
Step 5 cost: $5.49

Step 6: Reason SQL
Prompt tokens: 1,157
Input tokens: 21,284,010    per example: 1,665.5458173566008
Output tokens: 3,833,700
Step 6 cost: $5.49

Total cost: $32.13
total tokens per example: 9564.274904139604
```
