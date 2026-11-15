# RAGnarok

A Discord bot that answers Ark: Survival Ascended questions. You type `/ask`, it searches the ARK wiki and writes an answer from what it found, with links to the pages it used.

**[Join the Discord and try it](https://discord.gg/SZ9HXsxMn)**

![RAGnarok answering questions in Discord](docs/demo.png)

| At a glance | |
|---|---|
| Knowledge base | 791 wiki pages: every creature, weapon, armor piece and saddle, plus core mechanics |
| Answer time | 1 to 2 seconds once warm |
| Retrieval accuracy | 38 of 38 test questions, 9 of them deliberately misspelled |
| Cost | about $0.0027 per question |
| Running cost when idle | $0 |

## Why

I built this for myself. I wanted to know how to tame certain dinos and what to feed them. ChatGPT on the free plan kept hitting the limit, and when it did answer I had no way to tell which numbers to trust. Googling meant digging through links, half of them written for the older game. So I made a bot that only reads the wiki, shows where each answer came from, and says so when the wiki doesn't have it.

## How it works

It's a RAG pipeline on two AWS Lambdas.

```mermaid
flowchart LR
    D[Discord] -->|1. signed POST| H[Handler Lambda<br/>verify, defer]
    H -->|2. async invoke| K[Worker Lambda]
    K -->|6. edit the reply| D
    K <-->|3. daily count| Q[(DynamoDB)]
    K <-->|4. hybrid search| P[(Pinecone)]
    K <-->|5. embed, answer| O[OpenAI]
    P ---|filled offline by scripts/ingest.py| W[ARK wiki API]
```

1. Discord sends a signed request to the handler Lambda. The handler checks the Ed25519 signature and replies "thinking..." right away. Discord wants a reply within 3 seconds and a cold worker takes about 7, so the slow part has to run somewhere else.
2. Before replying, the handler starts the worker Lambda with an async invoke.
3. The worker checks your daily limit in DynamoDB.
4. It embeds the question and runs a hybrid search in Pinecone. Dense vectors match meaning. BM25 matches exact Ark terms like "engram".
5. The top 8 chunks go to gpt-5.4-mini as numbered passages. It has to answer from those passages only, name the ones it used, and return a fixed marker when none of them answer the question.
6. The worker edits the "thinking..." message into the answer and adds links to the source pages.

The knowledge base is built separately. `scripts/ingest.py` runs on my laptop, never on Lambda, and fills Pinecone from the wiki's API.

## Things I measured instead of guessing

### How much to trust keywords

Hybrid search has a knob, alpha, that sets how much weight goes to meaning versus exact keywords. I guessed keyword-heavy would win, since Ark is full of strange names. Then I measured it.

`scripts/eval_retrieval.py` runs 38 questions and checks two things for each: did the right page come back, and is the answer itself in the retrieved text.

| alpha | Right page | Misspelled questions | Answer in text |
|---|---|---|---|
| 0.0, keywords only | 84% | 56% | 79% |
| 0.4 | 95% | 89% | 97% |
| 0.6 to 0.75 | 97% | 100% | 100% |
| **0.8 to 0.9** | **100%** | **100%** | **100%** |
| 1.0, meaning only | 97% | 100% | 100% |

My guess was wrong, for two reasons. BM25 can't match "gigantoraptr" at all, so typos need the dense vector. And keywords lose value as the corpus grows: once "saddle" appears in hundreds of chunks it stops narrowing anything down. The best range was 0.6 to 0.95 at 2,371 chunks and shrank to 0.8 to 0.9 at 3,876. The bot uses 0.85.

### Which model

gpt-5-mini costs a third as much per token as gpt-5.4-mini. `scripts/eval_answers.py` runs 19 questions through the full pipeline and checks each answer for the right fact.

| Model | Correct | Cost per question | Average wait |
|---|---|---|---|
| **gpt-5.4-mini, no reasoning** | **19/19** | **$0.0027** | **1.0 s** |
| gpt-5-mini, minimal reasoning | 17/19 | $0.0010 | 1.5 s |
| gpt-5-mini, low reasoning | 19/19 | $0.0013 | 3.3 s |

The cheaper model only matches the accuracy with reasoning turned on, and then it bills reasoning tokens and takes three times as long. I kept gpt-5.4-mini.

### What the scraper was throwing away

A retrieval score can be perfect while the bot still can't answer, if the fact never made it into the database. That happened twice.

The first version only kept prose. The wiki keeps its numbers in infoboxes and tables, so the right page came back every time and the answer wasn't on it. Previewing one page's extracted text showed the gap. The ingest now turns the infobox into a "key facts" line and rewrites each data table as text rows.

| Question | Before | After |
|---|---|---|
| What saddle does an Ankylosaurus need? | I don't know | Ankylo Saddle, level 36 |
| What does a Rex Saddle cost to craft? | I don't know | 380 Hide, 200 Fiber, 50 Metal Ingot |
| Which armor gives the most cold protection? | I don't know | Fur Armor, 249 |
| How much damage does an Assault Rifle do? | I don't know | 47 |

That's why the eval checks for the fact now, and not only the page.

## Knowledge base

`scripts/ingest.py` runs locally and never gets deployed. It lists every page in the wiki's Creatures, Weapons and Armor categories through the MediaWiki API, plus 17 mechanics pages like Taming, Breeding and Imprinting. Over half the Creatures category is redirects, such as "Aberrant Rex" pointing at "Rex", so dropping redirects removes the variant duplicates without a hand-written list.

Each page is cleaned to plain text, cut into 500-token chunks with 50 tokens of overlap, embedded, and uploaded to Pinecone with both vector types.

791 pages, 3,876 chunks, about $0.035 and 35 minutes per full run. The run stops before touching the live index if more than 2% of pages fail to scrape.

## Daily limits by Discord role

Each user gets a number of questions per day, counted in DynamoDB with an atomic `ADD`. Discord includes a member's role IDs in every slash-command request, so a server can give its ranks different limits without any accounts or logins. The table that maps roles to limits is in `src/worker/quota.py`.

![Three answers, then the daily limit message](docs/demo-limit.png)

## Cost

About $0.0027 per question, measured. OpenAI is 99% of that, and the input context is 94%. Lambda, API Gateway, DynamoDB and Pinecone stay inside their free tiers at this volume, and nothing is billed while nobody is asking.

## Tech stack

- Python 3.13, FastAPI + Mangum
- AWS SAM: two Lambdas on arm64, API Gateway, DynamoDB, IAM
- OpenAI text-embedding-3-small for embeddings, gpt-5.4-mini for answers
- Pinecone for hybrid dense + sparse search
- BM25 via pinecone-text
- requests, BeautifulSoup and pandas for ingestion
- PyNaCl for Discord's Ed25519 signature check
- pytest for the handler and quota logic

## Project structure

```text
.
├── src/app          handler Lambda: verifies, defers, invokes worker
├── src/worker       worker Lambda
│   ├── worker.py        quota check, reply, usage log
│   ├── retriever.py     hybrid search
│   ├── generator.py     prompt, answer, source links
│   └── quota.py         daily limits by Discord role
├── scripts
│   ├── ingest.py             builds the knowledge base
│   ├── eval_retrieval.py     scores retrieval across blend values
│   ├── eval_answers.py       scores final answers, cost and speed per model
│   ├── register_commands.py  registers /ask with Discord
│   └── deploy.sh             build and deploy
├── tests            handler and quota tests, run offline
└── template.yaml    AWS SAM infrastructure
```

## Run it yourself

You need Python 3.13, the AWS SAM CLI, Docker, AWS credentials, and accounts for OpenAI, Pinecone and a Discord application.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # fill in your keys
```

Build the knowledge base. This also writes the fitted BM25 encoder into `src/worker/`, where the worker loads it.

```bash
python scripts/ingest.py
```

Deploy, register the command, then paste the endpoint URL from the deploy output into the Discord developer portal as the Interactions Endpoint URL.

```bash
scripts/deploy.sh
python scripts/register_commands.py
```

Check your work:

```bash
pytest                              # handler and quota tests, no network
python scripts/eval_retrieval.py    # retrieval accuracy at each alpha
python scripts/eval_answers.py      # answer accuracy, cost and speed
```

If your OpenAI project limits which models it can use, allow both text-embedding-3-small and gpt-5.4-mini. I only allowed the chat model once and every question failed with a 403.

## Limitations

- No memory. Every `/ask` is standalone, so follow-up questions don't work yet.
- Answers are only as fresh as the last ingest. After a game patch, re-run it.
- It covers creatures, weapons, armor and core mechanics. Structures, resources, consumables and maps aren't in yet.
- It only knows the vanilla game. Modded servers change stats and add items the wiki doesn't cover.

## Roadmap

- Structures, resources, consumables and tools
- Maps, caves and boss requirements
- A private knowledge base per server for its own rules, rates and mods
- A queue between the two Lambdas to smooth out bursts

## Credits and license

Game data comes from the [ARK Official Community Wiki](https://ark.wiki.gg), whose content is licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0). Every answer links back to the pages it used. This project isn't affiliated with Studio Wildcard.

The code is MIT licensed. See [LICENSE](LICENSE).
