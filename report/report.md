---
documentclass: article
fontsize: 11pt
bibliography: ml-2019.bib
---

<!-- Pure latex to handle the title -->
\title{An Astonishing Title}
\author{
  Emanuele Cosenza \\ 
  \href{mailto:e.cosenza@studenti.unipi.it}{e.cosenza@studenti.unipi.it} 
  \and Riccardo Massidda \\ 
  \href{mailto:r.massidda@studenti.unipi.it}{r.massidda@studenti.unipi.it}
}
\maketitle
\begin{center}
  ML course, 2019/2020. \\
  \today \\
  Type A project.
\end{center}
\begin{abstract}
A marvellous abstract, about an artificial neural network based on the multilayer perceptron architecture.
\end{abstract}

# Introduction

# Method
The numerical computational needs are solved by using NumPy[@noauthor_numpy_nodate].
The topology of the network is represented using a list of layers.
The $i$-th weight matrix represents the weights connecting the nodes of the layer $i$ to the nodes in the layer $i+1$, that is the outgoing edges for the layer $i$ and the ingoing edges for $i+1$.

The prediction is coded by simply forwarding the input in the network.
During the training phase the input is shuffled to avoid ordering bias[@unknown], after this the scan of the data is done in a minibatching fashion.
The learning of the weights in the network is done by using the backpropagation algorithm[@rumelhart_parallel_1986].

# Experiments

## Monk's Results

## Cup Results

# Conclusions

# References
