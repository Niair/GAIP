from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader

# text loader
# loader = PyPDFLoader("7_RAG/attention_is_all_you_need.pdf")

# docs = loader.load()

text = """
The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU
[20], ByteNet [15] and ConvS2S [8], all of which use convolutional neural networks as basic building
block, computing hidden representations in parallel for all input and output positions. In these models,
the number of operations required to relate signals from two arbitrary input or output positions grows
in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes
it more difficult to learn dependencies between distant positions [11]. In the Transformer this is
reduced to a constant number of operations, albeit at the cost of reduced effective resolution due
to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as
described in section 3.2.

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions
of a single sequence in order to compute a representation of the sequence. Self-attention has been
used successfully in a variety of tasks including reading comprehension, abstractive summarization,
textual entailment and learning task-independent sentence representations [4, 22, 23, 19].
End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-
aligned recurrence and have been shown to perform well on simple-language question answering and
language modeling tasks [28].

To the best of our knowledge, however, the Transformer is the first transduction model relying
entirely on self-attention to compute representations of its input and output without using sequence-
aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate
self-attention and discuss its advantages over models such as [14, 15] and [8].
"""

# text splitter
splitter = RecursiveCharacterTextSplitter(
      chunk_size = 1000,
      chunk_overlap = 0
)

# chunks = splitter.split_documents(docs)
chunks = splitter.split_text(text)

print(chunks)  # the chunk we get is also an document object