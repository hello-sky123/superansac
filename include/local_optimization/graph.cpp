/* graph.cpp */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "graph.h"


template <typename captype, typename tcaptype, typename flowtype>
	Graph<captype, tcaptype, flowtype>::Graph(int node_num_max, int edge_num_max, void (*err_function)(const char *))
	: node_num(0),
	  nodeptr_block(NULL),
	  error_function(err_function)
{
	if (node_num_max < 16) node_num_max = 16;
	if (edge_num_max < 16) edge_num_max = 16;

	nodes = (node*) malloc(node_num_max*sizeof(node));
	arcs = (arc*) malloc(2*edge_num_max*sizeof(arc));
	if (!nodes || !arcs) { if (error_function) (*error_function)("Not enough memory!"); exit(1); }

	this->node_num_max = node_num_max;
	arc_num = 0;
	arc_num_max = 2*edge_num_max;

	maxflow_iteration = 0;
	flow = 0;
}

template <typename captype, typename tcaptype, typename flowtype>
	Graph<captype,tcaptype,flowtype>::~Graph()
{
	if (nodeptr_block)
	{
		delete nodeptr_block;
		nodeptr_block = NULL;
	}
	free(nodes);
	free(arcs);
}

template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::reset()
{
	node_num = 0;
	arc_num = 0;

	if (nodeptr_block)
	{
		delete nodeptr_block;
		nodeptr_block = NULL;
	}

	maxflow_iteration = 0;
	flow = 0;
}

template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::reallocate_nodes(int num)
{
	node_num_max += node_num_max / 2;
	if (node_num_max < node_num + num) node_num_max = node_num + num;
	nodes = (node*) realloc(nodes, node_num_max*sizeof(node));
	if (!nodes) { if (error_function) (*error_function)("Not enough memory!"); exit(1); }

	/* with index-based links no fix-up of the existing nodes/arcs is needed */
}

template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::reallocate_arcs()
{
	arc_num_max += arc_num_max / 2; if (arc_num_max & 1) arc_num_max ++;
	arcs = (arc*) realloc(arcs, arc_num_max*sizeof(arc));
	if (!arcs) { if (error_function) (*error_function)("Not enough memory!"); exit(1); }

	/* with index-based links no fix-up of the existing nodes/arcs is needed */
}
