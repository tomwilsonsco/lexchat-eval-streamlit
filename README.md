# lexchat-eval-streamlit
Testing streamlit dashboard showing lexchat evaluation results

## Current evaluations

| Metric | Description |
|--------|-------------|
| Tool Usage | Are all of delegate research, search legislation, get legislation text used. |
| Research Output Structure | Does the worker agent return the findings to the manager with the requested headers. |
| Reference Links | Are reference links included in the answer provided to the user. |
| Consistency (Cosine) | Compare the answers provided when the same question is asked multiple times using TF cosine similarity. |
| Consistency (AI Judge) | AI as a judge metric: Decide if multiple answers to the same question have contradictions, omissions, or additional irrelevant information. |
| Answer Relevancy | AI as a judge metric: Measures how directly and completely the response addresses the user's question, penalising vague answers and irrelevant content. |
| Research Groundedness | AI as a judge metric: Measures whether the research summary is grounded exclusively in the legal text retrieved from the Lex API, penalising any external inferences or factual distortions. |
| Response Groundedness | AI as a judge metric: Evaluates whether the final response is strictly grounded in the research worker's summary, ensuring no new information or contradictions have been introduced. |
