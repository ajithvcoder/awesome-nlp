import pandas as pd
from datasets import Dataset
import openai
from ragas import evaluate,aevaluate
import os
import pdb
# from datasets import Dataset 
# from ragas.metrics.collections  import (
#     Faithfulness,
#     AnswerRelevancy,
#     ContextPrecision,
#     ContextRecall,
#     ContextEntityRecall,
#     SemanticSimilarity,
#     FactualCorrectness
# )

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)
from dotenv import load_dotenv
from ragas.embeddings.base import embedding_factory
from openai import AsyncOpenAI, OpenAI
from ragas.llms import llm_factory
from ragas.dataset_schema import EvaluationDataset
import ast
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key

client = AsyncOpenAI()
ragas_llm = llm_factory("gpt-4o-mini", client=client)
ragas_embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)


# metrics_eval = [
#     Faithfulness(llm=ragas_llm),
#     # AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
#     # ContextPrecision(llm=ragas_llm),
#     # ContextRecall(llm=ragas_llm),
#     # ContextEntityRecall(llm=ragas_llm),
#     # SemanticSimilarity(llm=ragas_llm, embeddings=ragas_embeddings),
#     # FactualCorrectness(llm=ragas_llm),
# ]

# for m in metrics_eval:
#     print(type(m))
    
# exit()

# read excel
df = pd.read_excel("result_1.xlsx")
df['contexts'] = df['contexts'].apply(ast.literal_eval)
df.rename(columns={"question": "user_input",
                   "answer": "response",
                   "ground_truth": "reference",
                   "contexts": "retrieved_contexts"}, inplace=True)
# convert back to HF Dataset
dataset = Dataset.from_pandas(df)
pdb.set_trace()

dataset = EvaluationDataset.from_pandas(df)

metrics = [
    faithfulness,
    context_recall
    # ContextRecall(llm=ragas_llm),
    # Faithfulness(llm=ragas_llm),
    # AnswerRelevancy(llm=ragas_llm),
]
result = evaluate(dataset=dataset, metrics=metrics)
print(result)

score_df = result.to_pandas()
print(score_df)
# score_df
score_df.to_csv("EvaluationScores2.csv", encoding="utf-8", index=False)

exit()
# Convert results to a pandas DataFrame and print
# df = result.to_pandas()
# print(df)

# print(dataset.to_pandas())

# dataset_str = (
#     df.to_dict(orient="records")
# )

# # import pdb
# # pdb.set_trace()

# score = evaluate(dataset_str, metrics=[faithfulness])

# # async def save_eval():
# #     score = await aevaluate(
# #         dataset, 
# #         # metrics=metrics_eval,
# #         # llm=ragas_llm,  # Global override
# #         # embeddings=ragas_embeddings,
# #         # show_progress=True,
# #         # batch_size=8,  # For efficiency on large datasets
# #         # raise_exceptions=False,  # Continues on metric failures
# #         # allow_nest_asyncio=False
# #     )
# score_df = score.to_pandas()
# print(score_df)
# # score_df
# score_df.to_csv("EvaluationScores2.csv", encoding="utf-8", index=False)

# # import asyncio
# # asyncio.run(save_eval())