#!/usr/bin/env python3
import random
import networkx as nx


def generate_edges_ind_of_random_graph_given_degree_seq(degree_seq):
    G=nx.configuration_model(degree_seq)
    return [(i,j) for i,j,k in G.edges]

if __name__ == "__main__":
    # generate_random_graph_given_degree_dist(10)
    sequence = nx.random_powerlaw_tree_sequence(100, tries=5000, seed=0)
    edges = generate_edges_ind_of_random_graph_given_degree_seq(sequence)
    print(edges)
