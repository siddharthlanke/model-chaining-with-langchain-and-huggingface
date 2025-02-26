from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error

# suppressing unnecessary warnings and errors
set_verbosity_error()

# summarization pipeline and wrapper
summarization_model = "facebook/bart-large-cnn"
summarization_pipeline = pipeline("summarization", model=summarization_model, device=0)
summarizer = HuggingFacePipeline(pipeline=summarization_pipeline)

# refinement pipeline and wrapper
refinement_model = "facebook/bart-large"
refinement_pipeline = pipeline("summarization", model=refinement_model, device=0)
refiner = HuggingFacePipeline(pipeline=refinement_pipeline)

# question-answering pipeline
qa_model = "deepset/roberta-base-squad2"
qa_pipeline = pipeline("question-answering", model=qa_model, device=0)

summary_template = PromptTemplate.from_template("Summarize the following text in a {length} way:\n\n{text}") # prompt template
summarization_chain = summary_template | summarizer | refiner # summarization and refinement chain

text_to_summarize = input("\nEnter the text to summarize:\n")
length = input("\nEnter the summary length (short/medium/long): ")
summary = summarization_chain.invoke({"text": text_to_summarize, "length": length})

print("\n **Generated Summary:**")
print(summary)

# question-answering loop
while True:
    question = input("\nAsk a question about the summary (or type 'exit' to stop):\n")
    if question.strip().lower() == "exit":
        print("\nGoodbye!")
        break

    qa_result = qa_pipeline(question=question, context=summary)

    print("\n **Answer:**")
    print(qa_result["answer"])
