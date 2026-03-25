# Retrieval-Augmented Generation (RAG) for Corporate Wikis/Documentation

The scope of this project is to build an AI-powered chatbot LLM-based chatbot that provides 
accurate answers to user queries related to computer science using Retrieval-Augmented 
Generation (RAG). RAG allows us to incorporate information retrieval before response generation, 
thereby creating contextually supported answers backed by real, traceable sources. By utilizing RAG, 
we can reduce AI hallucinations and the need to retrain LLMs with new data. 

### Methods

The dataset used is an English Wikipedia dataset dump snapshot retrieved from [this download link](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2) and extracted and cleaned using [WikiExtractor](https://github.com/attardi/wikiextractor). It was then pre-processed using [RegEx](https://www.w3schools.com/python/python_regex.asp), filtered by specific keywords related to the "Computer Science" subject field to reduce the very large source dataset size and memory costs and saved as a `.jsonl` file.

**Cleaned Data Description:**
| Variable | Description |
| --- | --- |
| id | Unique identifier for a Wikipedia article|
| revid | Article revision ID for the particular version of the article text |
| url | URL link to the article page |
| title | Title of the article |
| text | All text located on the page, including subheadings and line breaks and excluding tables and images |

**Corpus Data Description per chunk:**
| Variable | Description |
| --- | --- |
| chunk_id | A unique, deterministic identifier for the chunk |
| page_id | Wikipedia’s unique identifier for the article |
| title | Title of the article |
| section_path | A list representing the hierarchical section structure of the article |
| paragraph_index | The index of the paragraph within the article |
| text | Text contained within the chunk |
| source | Origin of dataset |
| url | URL link of the specific Wikipedia article |
| revid | Article revision ID for the particular version of the article text |

# Credits

Project by Yasmine Huo & Tracy Ling.
