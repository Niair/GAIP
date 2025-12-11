from langchain_text_splitters import CharacterTextSplitter

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
splitter = CharacterTextSplitter(
      chunk_size = 500,
      chunk_overlap = 0,
      separator = ''
)
# chunk overlap hepls us to set the overlaping common characters between the two chunks, 
# we do that so that our context dont't get cut and the meaning will not get cut
# the best value we can set is 10% or 20% of the chunk_size

chunks = splitter.split_text(text)

print(chunks)