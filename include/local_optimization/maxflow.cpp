/* maxflow.cpp */


#include <stdio.h>
#include "graph.h"


/*
	special constants for node parent arc indices
	(see INDEX_TERMINAL / INDEX_ORPHAN in graph.h):
	INDEX_TERMINAL - arc to terminal
	INDEX_ORPHAN   - orphan
*/


#define INFINITE_D (static_cast<int>(((unsigned)-1)/2))		/* infinite distance to the terminal */

/***********************************************************************/

/*
	Functions for processing active list.
	nodes[i].next is the index of the next node in the list
	(or i itself, if i is the last node in the list).
	nodes[i].next is INDEX_NONE iff i is not in the list.

	There are two queues. Active nodes are added
	to the end of the second queue and read from
	the front of the first queue. If the first queue
	is empty, it is replaced by the second queue
	(and the second queue becomes empty).
*/


template <typename captype, typename tcaptype, typename flowtype>
	inline void Graph<captype,tcaptype,flowtype>::set_active(int i)
{
	if (nodes[i].next == INDEX_NONE)
	{
		/* it's not in the list yet */
		if (queue_last[1] != INDEX_NONE) nodes[queue_last[1]].next = i;
		else                             queue_first[1]            = i;
		queue_last[1] = i;
		nodes[i].next = i;
	}
}

/*
	Returns the next active node.
	If it is connected to the sink, it stays in the list,
	otherwise it is removed from the list
*/
template <typename captype, typename tcaptype, typename flowtype>
	inline int Graph<captype,tcaptype,flowtype>::next_active()
{
	int i;

	while ( 1 )
	{
		if ((i=queue_first[0]) == INDEX_NONE)
		{
			queue_first[0] = i = queue_first[1];
			queue_last[0]  = queue_last[1];
			queue_first[1] = INDEX_NONE;
			queue_last[1]  = INDEX_NONE;
			if (i == INDEX_NONE) return INDEX_NONE;
		}

		/* remove it from the active list */
		if (nodes[i].next == i) queue_first[0] = queue_last[0] = INDEX_NONE;
		else                    queue_first[0] = nodes[i].next;
		nodes[i].next = INDEX_NONE;

		/* a node in the list is active iff it has a parent */
		if (nodes[i].parent != INDEX_NONE) return i;
	}
}

/***********************************************************************/

template <typename captype, typename tcaptype, typename flowtype>
	inline void Graph<captype,tcaptype,flowtype>::set_orphan_front(int i)
{
	nodeptr *np;
	nodes[i].parent = INDEX_ORPHAN;
	np = nodeptr_block -> New();
	np -> ptr = i;
	np -> next = orphan_first;
	orphan_first = np;
}

template <typename captype, typename tcaptype, typename flowtype>
	inline void Graph<captype,tcaptype,flowtype>::set_orphan_rear(int i)
{
	nodeptr *np;
	nodes[i].parent = INDEX_ORPHAN;
	np = nodeptr_block -> New();
	np -> ptr = i;
	if (orphan_last) orphan_last -> next = np;
	else             orphan_first        = np;
	orphan_last = np;
	np -> next = NULL;
}

/***********************************************************************/

template <typename captype, typename tcaptype, typename flowtype>
	inline void Graph<captype,tcaptype,flowtype>::add_to_changed_list(int i)
{
	if (changed_list && !nodes[i].is_in_changed_list)
	{
		node_id* ptr = changed_list->New();
		*ptr = i;
		nodes[i].is_in_changed_list = true;
	}
}

/***********************************************************************/

template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::maxflow_init()
{
	int i;

	queue_first[0] = queue_last[0] = INDEX_NONE;
	queue_first[1] = queue_last[1] = INDEX_NONE;
	orphan_first = NULL;

	TIME = 0;

	for (i=0; i<node_num; i++)
	{
		node* n = nodes + i;
		n -> next = INDEX_NONE;
		n -> is_marked = 0;
		n -> is_in_changed_list = 0;
		n -> TS = TIME;
		if (n->tr_cap > 0)
		{
			/* i is connected to the source */
			n -> is_sink = 0;
			n -> parent = INDEX_TERMINAL;
			set_active(i);
			n -> DIST = 1;
		}
		else if (n->tr_cap < 0)
		{
			/* i is connected to the sink */
			n -> is_sink = 1;
			n -> parent = INDEX_TERMINAL;
			set_active(i);
			n -> DIST = 1;
		}
		else
		{
			n -> parent = INDEX_NONE;
		}
	}
}

template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::maxflow_reuse_trees_init()
{
	int i;
	int j;
	int queue = queue_first[1];
	int a;
	nodeptr* np;

	queue_first[0] = queue_last[0] = INDEX_NONE;
	queue_first[1] = queue_last[1] = INDEX_NONE;
	orphan_first = orphan_last = NULL;

	TIME ++;

	while ((i=queue) != INDEX_NONE)
	{
		queue = nodes[i].next;
		if (queue == i) queue = INDEX_NONE;
		nodes[i].next = INDEX_NONE;
		nodes[i].is_marked = 0;
		set_active(i);

		if (nodes[i].tr_cap == 0)
		{
			if (nodes[i].parent != INDEX_NONE) set_orphan_rear(i);
			continue;
		}

		if (nodes[i].tr_cap > 0)
		{
			if (nodes[i].parent == INDEX_NONE || nodes[i].is_sink)
			{
				nodes[i].is_sink = 0;
				for (a=nodes[i].first; a!=INDEX_NONE; a=arcs[a].next)
				{
					j = arcs[a].head;
					if (!nodes[j].is_marked)
					{
						if (nodes[j].parent == arcs[a].sister) set_orphan_rear(j);
						if (nodes[j].parent != INDEX_NONE && nodes[j].is_sink && arcs[a].r_cap > 0) set_active(j);
					}
				}
				add_to_changed_list(i);
			}
		}
		else
		{
			if (nodes[i].parent == INDEX_NONE || !nodes[i].is_sink)
			{
				nodes[i].is_sink = 1;
				for (a=nodes[i].first; a!=INDEX_NONE; a=arcs[a].next)
				{
					j = arcs[a].head;
					if (!nodes[j].is_marked)
					{
						if (nodes[j].parent == arcs[a].sister) set_orphan_rear(j);
						if (nodes[j].parent != INDEX_NONE && !nodes[j].is_sink && arcs[arcs[a].sister].r_cap > 0) set_active(j);
					}
				}
				add_to_changed_list(i);
			}
		}
		nodes[i].parent = INDEX_TERMINAL;
		nodes[i].TS = TIME;
		nodes[i].DIST = 1;
	}

	//test_consistency();

	/* adoption */
	while ((np=orphan_first))
	{
		orphan_first = np -> next;
		i = np -> ptr;
		nodeptr_block -> Delete(np);
		if (!orphan_first) orphan_last = NULL;
		if (nodes[i].is_sink) process_sink_orphan(i);
		else                  process_source_orphan(i);
	}
	/* adoption end */

	//test_consistency();
}

template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::augment(int middle_arc)
{
	int i;
	int a;
	tcaptype bottleneck;


	/* 1. Finding bottleneck capacity */
	/* 1a - the source tree */
	bottleneck = arcs[middle_arc].r_cap;
	for (i=arcs[arcs[middle_arc].sister].head; ; i=arcs[a].head)
	{
		a = nodes[i].parent;
		if (a == INDEX_TERMINAL) break;
		if (bottleneck > arcs[arcs[a].sister].r_cap) bottleneck = arcs[arcs[a].sister].r_cap;
	}
	if (bottleneck > nodes[i].tr_cap) bottleneck = nodes[i].tr_cap;
	/* 1b - the sink tree */
	for (i=arcs[middle_arc].head; ; i=arcs[a].head)
	{
		a = nodes[i].parent;
		if (a == INDEX_TERMINAL) break;
		if (bottleneck > arcs[a].r_cap) bottleneck = arcs[a].r_cap;
	}
	if (bottleneck > - nodes[i].tr_cap) bottleneck = - nodes[i].tr_cap;


	/* 2. Augmenting */
	/* 2a - the source tree */
	arcs[arcs[middle_arc].sister].r_cap += bottleneck;
	arcs[middle_arc].r_cap -= bottleneck;
	for (i=arcs[arcs[middle_arc].sister].head; ; i=arcs[a].head)
	{
		a = nodes[i].parent;
		if (a == INDEX_TERMINAL) break;
		arcs[a].r_cap += bottleneck;
		arcs[arcs[a].sister].r_cap -= bottleneck;
		if (!arcs[arcs[a].sister].r_cap)
		{
			set_orphan_front(i); // add i to the beginning of the adoption list
		}
	}
	nodes[i].tr_cap -= bottleneck;
	if (!nodes[i].tr_cap)
	{
		set_orphan_front(i); // add i to the beginning of the adoption list
	}
	/* 2b - the sink tree */
	for (i=arcs[middle_arc].head; ; i=arcs[a].head)
	{
		a = nodes[i].parent;
		if (a == INDEX_TERMINAL) break;
		arcs[arcs[a].sister].r_cap += bottleneck;
		arcs[a].r_cap -= bottleneck;
		if (!arcs[a].r_cap)
		{
			set_orphan_front(i); // add i to the beginning of the adoption list
		}
	}
	nodes[i].tr_cap += bottleneck;
	if (!nodes[i].tr_cap)
	{
		set_orphan_front(i); // add i to the beginning of the adoption list
	}


	flow += bottleneck;
}

/***********************************************************************/

template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::process_source_orphan(int i)
{
	int j;
	int a0, a0_min = INDEX_NONE, a;
	int d, d_min = INFINITE_D;

	/* trying to find a new parent */
	for (a0=nodes[i].first; a0!=INDEX_NONE; a0=arcs[a0].next)
	if (arcs[arcs[a0].sister].r_cap)
	{
		j = arcs[a0].head;
		if (!nodes[j].is_sink && (a=nodes[j].parent) != INDEX_NONE)
		{
			/* checking the origin of j */
			d = 0;
			while ( 1 )
			{
				if (nodes[j].TS == TIME)
				{
					d += nodes[j].DIST;
					break;
				}
				a = nodes[j].parent;
				d ++;
				if (a==INDEX_TERMINAL)
				{
					nodes[j].TS = TIME;
					nodes[j].DIST = 1;
					break;
				}
				if (a==INDEX_ORPHAN) { d = INFINITE_D; break; }
				j = arcs[a].head;
			}
			if (d<INFINITE_D) /* j originates from the source - done */
			{
				if (d<d_min)
				{
					a0_min = a0;
					d_min = d;
				}
				/* set marks along the path */
				for (j=arcs[a0].head; nodes[j].TS!=TIME; j=arcs[nodes[j].parent].head)
				{
					nodes[j].TS = TIME;
					nodes[j].DIST = d --;
				}
			}
		}
	}

	if ((nodes[i].parent = a0_min) != INDEX_NONE)
	{
		nodes[i].TS = TIME;
		nodes[i].DIST = d_min + 1;
	}
	else
	{
		/* no parent is found */
		add_to_changed_list(i);

		/* process neighbors */
		for (a0=nodes[i].first; a0!=INDEX_NONE; a0=arcs[a0].next)
		{
			j = arcs[a0].head;
			if (!nodes[j].is_sink && (a=nodes[j].parent) != INDEX_NONE)
			{
				if (arcs[arcs[a0].sister].r_cap) set_active(j);
				if (a!=INDEX_TERMINAL && a!=INDEX_ORPHAN && arcs[a].head==i)
				{
					set_orphan_rear(j); // add j to the end of the adoption list
				}
			}
		}
	}
}

template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::process_sink_orphan(int i)
{
	int j;
	int a0, a0_min = INDEX_NONE, a;
	int d, d_min = INFINITE_D;

	/* trying to find a new parent */
	for (a0=nodes[i].first; a0!=INDEX_NONE; a0=arcs[a0].next)
	if (arcs[a0].r_cap)
	{
		j = arcs[a0].head;
		if (nodes[j].is_sink && (a=nodes[j].parent) != INDEX_NONE)
		{
			/* checking the origin of j */
			d = 0;
			while ( 1 )
			{
				if (nodes[j].TS == TIME)
				{
					d += nodes[j].DIST;
					break;
				}
				a = nodes[j].parent;
				d ++;
				if (a==INDEX_TERMINAL)
				{
					nodes[j].TS = TIME;
					nodes[j].DIST = 1;
					break;
				}
				if (a==INDEX_ORPHAN) { d = INFINITE_D; break; }
				j = arcs[a].head;
			}
			if (d<INFINITE_D) /* j originates from the sink - done */
			{
				if (d<d_min)
				{
					a0_min = a0;
					d_min = d;
				}
				/* set marks along the path */
				for (j=arcs[a0].head; nodes[j].TS!=TIME; j=arcs[nodes[j].parent].head)
				{
					nodes[j].TS = TIME;
					nodes[j].DIST = d --;
				}
			}
		}
	}

	if ((nodes[i].parent = a0_min) != INDEX_NONE)
	{
		nodes[i].TS = TIME;
		nodes[i].DIST = d_min + 1;
	}
	else
	{
		/* no parent is found */
		add_to_changed_list(i);

		/* process neighbors */
		for (a0=nodes[i].first; a0!=INDEX_NONE; a0=arcs[a0].next)
		{
			j = arcs[a0].head;
			if (nodes[j].is_sink && (a=nodes[j].parent) != INDEX_NONE)
			{
				if (arcs[a0].r_cap) set_active(j);
				if (a!=INDEX_TERMINAL && a!=INDEX_ORPHAN && arcs[a].head==i)
				{
					set_orphan_rear(j); // add j to the end of the adoption list
				}
			}
		}
	}
}

/***********************************************************************/

template <typename captype, typename tcaptype, typename flowtype>
	flowtype Graph<captype,tcaptype,flowtype>::maxflow(bool reuse_trees, Block<node_id>* _changed_list)
{
	int i, j, current_node = INDEX_NONE;
	int a;
	nodeptr *np, *np_next;

	if (!nodeptr_block)
	{
		nodeptr_block = new DBlock<nodeptr>(NODEPTR_BLOCK_SIZE, error_function);
	}

	changed_list = _changed_list;
	if (maxflow_iteration == 0 && reuse_trees) { if (error_function) (*error_function)("reuse_trees cannot be used in the first call to maxflow()!"); exit(1); }
	if (changed_list && !reuse_trees) { if (error_function) (*error_function)("changed_list cannot be used without reuse_trees!"); exit(1); }

	if (reuse_trees) maxflow_reuse_trees_init();
	else             maxflow_init();

	// main loop
	while ( 1 )
	{
		// test_consistency(current_node);

		if ((i=current_node) != INDEX_NONE)
		{
			nodes[i].next = INDEX_NONE; /* remove active flag */
			if (nodes[i].parent == INDEX_NONE) i = INDEX_NONE;
		}
		if (i == INDEX_NONE)
		{
			if ((i = next_active()) == INDEX_NONE) break;
		}

		/* growth */
		if (!nodes[i].is_sink)
		{
			/* grow source tree */
			for (a=nodes[i].first; a!=INDEX_NONE; a=arcs[a].next)
			if (arcs[a].r_cap)
			{
				j = arcs[a].head;
				if (nodes[j].parent == INDEX_NONE)
				{
					nodes[j].is_sink = 0;
					nodes[j].parent = arcs[a].sister;
					nodes[j].TS = nodes[i].TS;
					nodes[j].DIST = nodes[i].DIST + 1;
					set_active(j);
					add_to_changed_list(j);
				}
				else if (nodes[j].is_sink) break;
				else if (nodes[j].TS <= nodes[i].TS &&
				         nodes[j].DIST > nodes[i].DIST)
				{
					/* heuristic - trying to make the distance from j to the source shorter */
					nodes[j].parent = arcs[a].sister;
					nodes[j].TS = nodes[i].TS;
					nodes[j].DIST = nodes[i].DIST + 1;
				}
			}
		}
		else
		{
			/* grow sink tree */
			for (a=nodes[i].first; a!=INDEX_NONE; a=arcs[a].next)
			if (arcs[arcs[a].sister].r_cap)
			{
				j = arcs[a].head;
				if (nodes[j].parent == INDEX_NONE)
				{
					nodes[j].is_sink = 1;
					nodes[j].parent = arcs[a].sister;
					nodes[j].TS = nodes[i].TS;
					nodes[j].DIST = nodes[i].DIST + 1;
					set_active(j);
					add_to_changed_list(j);
				}
				else if (!nodes[j].is_sink) { a = arcs[a].sister; break; }
				else if (nodes[j].TS <= nodes[i].TS &&
				         nodes[j].DIST > nodes[i].DIST)
				{
					/* heuristic - trying to make the distance from j to the sink shorter */
					nodes[j].parent = arcs[a].sister;
					nodes[j].TS = nodes[i].TS;
					nodes[j].DIST = nodes[i].DIST + 1;
				}
			}
		}

		TIME ++;

		if (a != INDEX_NONE)
		{
			nodes[i].next = i; /* set active flag */
			current_node = i;

			/* augmentation */
			augment(a);
			/* augmentation end */

			/* adoption */
			while ((np=orphan_first))
			{
				np_next = np -> next;
				np -> next = NULL;

				while ((np=orphan_first))
				{
					orphan_first = np -> next;
					i = np -> ptr;
					nodeptr_block -> Delete(np);
					if (!orphan_first) orphan_last = NULL;
					if (nodes[i].is_sink) process_sink_orphan(i);
					else                  process_source_orphan(i);
				}

				orphan_first = np_next;
			}
			/* adoption end */
		}
		else current_node = INDEX_NONE;
	}
	// test_consistency();

	if (!reuse_trees || (maxflow_iteration % 64) == 0)
	{
		delete nodeptr_block;
		nodeptr_block = NULL;
	}

	maxflow_iteration ++;
	return flow;
}

/***********************************************************************/


template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::test_consistency(int current_node)
{
	int i;
	int a;
	int r;
	int num1 = 0, num2 = 0;

	// test whether all nodes i with i->next!=INDEX_NONE are indeed in the queue
	for (i=0; i<node_num; i++)
	{
		if (nodes[i].next != INDEX_NONE || i==current_node) num1 ++;
	}
	for (r=0; r<3; r++)
	{
		i = (r == 2) ? current_node : queue_first[r];
		if (i != INDEX_NONE)
		for ( ; ; i=nodes[i].next)
		{
			num2 ++;
			if (nodes[i].next == i)
			{
				if (r<2) assert(i == queue_last[r]);
				else     assert(i == current_node);
				break;
			}
		}
	}
	assert(num1 == num2);

	for (i=0; i<node_num; i++)
	{
		// test whether all edges in seach trees are non-saturated
		if (nodes[i].parent == INDEX_NONE) {}
		else if (nodes[i].parent == INDEX_ORPHAN) {}
		else if (nodes[i].parent == INDEX_TERMINAL)
		{
			if (!nodes[i].is_sink) assert(nodes[i].tr_cap > 0);
			else                   assert(nodes[i].tr_cap < 0);
		}
		else
		{
			if (!nodes[i].is_sink) assert (arcs[arcs[nodes[i].parent].sister].r_cap > 0);
			else                   assert (arcs[nodes[i].parent].r_cap > 0);
		}
		// test whether passive nodes in search trees have neighbors in
		// a different tree through non-saturated edges
		if (nodes[i].parent != INDEX_NONE && nodes[i].next == INDEX_NONE)
		{
			if (!nodes[i].is_sink)
			{
				assert(nodes[i].tr_cap >= 0);
				for (a=nodes[i].first; a!=INDEX_NONE; a=arcs[a].next)
				{
					if (arcs[a].r_cap > 0) assert(nodes[arcs[a].head].parent != INDEX_NONE && !nodes[arcs[a].head].is_sink);
				}
			}
			else
			{
				assert(nodes[i].tr_cap <= 0);
				for (a=nodes[i].first; a!=INDEX_NONE; a=arcs[a].next)
				{
					if (arcs[arcs[a].sister].r_cap > 0) assert(nodes[arcs[a].head].parent != INDEX_NONE && nodes[arcs[a].head].is_sink);
				}
			}
		}
		// test marking invariants
		if (nodes[i].parent != INDEX_NONE && nodes[i].parent != INDEX_ORPHAN && nodes[i].parent != INDEX_TERMINAL)
		{
			assert(nodes[i].TS <= nodes[arcs[nodes[i].parent].head].TS);
			if (nodes[i].TS == nodes[arcs[nodes[i].parent].head].TS) assert(nodes[i].DIST > nodes[arcs[nodes[i].parent].head].DIST);
		}
	}
}

template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::Copy(Graph<captype, tcaptype, flowtype>* g0)
{
	reset();

	if (node_num_max < g0->node_num)
	{
		free(nodes);
		nodes = (node*) malloc(g0->node_num*sizeof(node));
		node_num_max = g0->node_num;
	}
	if (arc_num_max < g0->arc_num)
	{
		free(arcs);
		arcs = (arc*) malloc(g0->arc_num*sizeof(arc));
		arc_num_max = g0->arc_num;
	}

	node_num = g0->node_num;
	memcpy(nodes, g0->nodes, node_num*sizeof(node));

	arc_num = g0->arc_num;
	memcpy(arcs, g0->arcs, arc_num*sizeof(arc));

	error_function = g0->error_function;
	flow = g0->flow;
	maxflow_iteration = g0->maxflow_iteration;

	queue_first[0] = g0->queue_first[0];
	queue_first[1] = g0->queue_first[1];
	queue_last[0] = g0->queue_last[0];
	queue_last[1] = g0->queue_last[1];
	TIME = g0->TIME;
}
